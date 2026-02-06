"""ParallelismPlanner - 第一层 chip 间切分

负责：
1. 根据 PP 配置划分 stage
2. 根据 pattern 为 op 选择 ParallelSpec
3. 插入通信算子
4. 输出 DistributedModel
"""

from dataclasses import dataclass
import math
import copy

from math_model.L3_mapping.parallelism.parallel_spec import ParallelSpec, ParallelType
from math_model.L3_mapping.parallelism.pattern_rules import (
    get_default_spec,
    get_pattern_spec,
)
from math_model.L3_mapping.plan.distributed_model import (
    CommType,
    DistributedModel,
    DistributedOp,
    NodeRole,
)
from math_model.L2_arch.topology import TopologySpecImpl
from math_model.L1_workload.graph import NodeKind
from math_model.L1_workload.ir import WorkloadIR
from math_model.L1_workload.layer import Layer
from math_model.L1_workload.op import Op
from math_model.L1_workload.tensor import TensorDesc


@dataclass
class DeploymentSpec:
    """部署规格

    Attributes:
        tp: Tensor Parallelism 并行度
        pp: Pipeline Parallelism 并行度
        ep: Expert Parallelism 并行度 (MoE)
        moe_tp: MoE Expert 内部 TP 并行度
        dp: Data Parallelism 并行度
        seq_len: 序列长度
        batch_size: 批次大小
        enable_tp_sp: 是否启用 TP sequence parallelism
        embed_tp: Embedding 层 TP 并行度
        lmhead_tp: LMHead 层 TP 并行度
        comm_protocol: 通信协议 (0=基础, 1=优化, 2/3=扩展)
        kv_cache_rate: KV Cache 比例
        is_prefill: 是否为 Prefill 阶段
    """

    tp: int = 1
    pp: int = 1
    ep: int = 1
    moe_tp: int = 1
    dp: int = 1
    seq_len: int = 2048
    batch_size: int = 1
    enable_tp_sp: bool = False
    embed_tp: int = 1
    lmhead_tp: int = 1
    comm_protocol: int = 1
    kv_cache_rate: float = 0.0
    is_prefill: bool = False


@dataclass
class BoardSpec:
    """板卡规格

    Attributes:
        num_chips: 芯片数量
        chip_memory_gb: 每个芯片的显存 (GB)
        inter_chip_bw_gbps: 芯片间带宽 (GB/s)
    """

    num_chips: int = 8
    chip_memory_gb: int = 16
    inter_chip_bw_gbps: float = 400.0


class ParallelismPlanner:
    """第一层：chip 间切分

    计算流程:
    1. 遍历 model.graph 的 op
    2. 根据 PP 参数确定 op 在哪个 stage 执行
    3. 根据 pattern 为 op 选择 ParallelSpec
    4. 插入通信节点
    5. 输出 DistributedModel
    """

    def __init__(
        self,
        deployment: DeploymentSpec,
        board: BoardSpec | None = None,
        topology: TopologySpecImpl | None = None,
    ) -> None:
        """初始化 ParallelismPlanner

        Args:
            deployment: 部署规格
            board: 板卡规格
            topology: 拓扑规格
        """
        self.deployment = deployment
        self.board = board or BoardSpec()
        self.topology = topology
        self._validate()

    def _validate(self) -> None:
        """验证配置"""
        total_chips = self.deployment.tp * self.deployment.pp * self.deployment.dp
        if total_chips > self.board.num_chips:
            raise ValueError(
                f"TP×PP={total_chips} 超过可用 chip 数量 {self.board.num_chips}"
            )
        if self.deployment.moe_tp > self.deployment.tp:
            raise ValueError(
                f"moe_tp={self.deployment.moe_tp} 不能超过 tp={self.deployment.tp}"
            )
        if self.deployment.moe_tp * self.deployment.ep > self.deployment.tp * self.deployment.dp:
            raise ValueError(
                "moe_tp×ep 不能超过 tp×dp（当前实现约束）"
            )

    def _validate_model_parallelism(self, layers: list[Layer]) -> None:
        """验证模型维度与并行配置一致性"""
        tp = self.deployment.tp
        ep = self.deployment.ep

        for layer in layers:
            layer_type = layer.attrs.get("layer_type", layer.op_type)
            params = layer.params

            num_heads = params.get("num_heads")
            if num_heads is not None and num_heads % tp != 0:
                raise ValueError(f"num_heads={num_heads} 必须能被 tp={tp} 整除")

            kv_heads = params.get("kv_heads", params.get("num_kv_heads"))
            if kv_heads is not None and kv_heads % tp != 0:
                raise ValueError(f"kv_heads={kv_heads} 必须能被 tp={tp} 整除")

            if layer_type in {"ffn", "mlp"}:
                inter = params.get("intermediate_size")
                if inter is not None and inter % tp != 0:
                    raise ValueError(
                        f"intermediate_size={inter} 必须能被 tp={tp} 整除"
                    )

            if layer_type == "moe":
                experts = params.get("n_routed_experts")
                if experts is not None and experts % ep != 0:
                    raise ValueError(
                        f"n_routed_experts={experts} 必须能被 ep={ep} 整除"
                    )

    def plan(self, model: WorkloadIR) -> DistributedModel:
        """执行 chip 间切分

        Args:
            model: WorkloadIR 对应的 Model

        Returns:
            DistributedModel: 分布式模型
        """
        layers = model.get_layers()
        layer_index = {layer.name: idx for idx, layer in enumerate(layers)}
        graph = model.get_graph()
        layer_nodes = {
            node.ref: node
            for node in graph.iter_nodes()
            if node.kind == NodeKind.LAYER
        }
        num_layers = len(layers)

        dist_model = DistributedModel(
            tp=self.deployment.tp,
            pp=self.deployment.pp,
            ep=self.deployment.ep,
            num_chips=self.deployment.tp * self.deployment.pp * self.deployment.dp,
        )

        # 初始化 parallel_groups
        dist_model.parallel_groups, dist_model.rank_map = self._build_parallel_groups()

        # 初始化 stages
        dist_model.stages = [[] for _ in range(self.deployment.pp)]

        # 计算每个 stage 包含的层范围
        layers_per_stage = num_layers // self.deployment.pp

        self._validate_model_parallelism(layers)

        last_op_by_layer: dict[str, str] = {}

        for layer in layers:
            node = layer_nodes.get(layer.name)
            if node is None:
                raise ValueError(f"找不到 layer 对应的 graph node: {layer.name}")
            layer_deps = [
                last_op_by_layer[pred.ref]
                for pred in graph.get_predecessors(node.node_id)
                if pred.kind == NodeKind.LAYER and pred.ref in last_op_by_layer
            ]

            # 1. 确定 stage
            stage_id = self._get_stage_id(layer_index[layer.name], layers_per_stage)

            layer_ops = layer.ops if layer.ops else [None]
            prev_op_id: str | None = None

            op_spec_cache: dict[str, ParallelSpec] = {}

            for op in layer_ops:
                op_id = op.name if op is not None else layer.name
                op_type = op.op_type if op is not None else layer.op_type
                op_role = self._infer_op_role(layer, op)
                layer_type = layer.attrs.get("layer_type", layer.op_type)

                # 2. 选择 ParallelSpec
                spec = self._select_parallel_spec(layer, op_role, op_type)
                if op is not None:
                    op_spec_cache[op.name] = spec

                # 2.1 对齐 layout_signature 中的 split_factor
                spec.layout_signature = self._build_layout_signature(
                    spec, stage_id, layer_type, op_role
                )

                # 3. 计算 local_shape
                shard_factor = self._resolve_split_factor(spec, layer_type, op_role)
                local_shape = spec.get_local_shape(
                    self._infer_op_shape(layer, op), shard_factor
                )
                local_shape = self._apply_data_parallel_shard(local_shape)
                local_shape = self._apply_moe_expert_shard(
                    local_shape, layer, layer_type, op_role
                )

                # 4. 创建分布式 op
                dtype_attrs = self._extract_dtype_attrs(op)
                weight_attrs = self._extract_weight_attrs(
                    op, shard_factor, op_role, layer_type
                )
                dist_op = DistributedOp(
                    op_id=op_id,
                    op_type=op_type,
                    role=NodeRole.COMPUTE,
                    local_shape=local_shape,
                    parallel_spec=spec,
                    stage_id=stage_id,
                    chip_ids=self._get_chip_ids(stage_id),
                    deps=[],
                    attrs={
                        "layer_type": layer_type,
                        "op_role": op_role,
                        "layer_name": layer.name,
                        "op_name": op_id,
                        **dtype_attrs,
                        **weight_attrs,
                    },
                )
                base_deps = [prev_op_id] if prev_op_id else list(layer_deps)

                pre_comm_ops = self._maybe_insert_moe_dispatch(
                    layer,
                    layer_type,
                    op_role,
                    op,
                    dist_op,
                    base_deps,
                )
                for comm_op in pre_comm_ops:
                    dist_model.add_op(comm_op)
                    dist_model.stages[stage_id].append(comm_op.op_id)
                    base_deps = [comm_op.op_id]

                dist_op.deps = base_deps
                dist_model.add_op(dist_op)
                dist_model.stages[stage_id].append(op_id)
                last_op_id = op_id

                if op is None:
                    layer_comm_ops = self._maybe_insert_layer_comm(
                        layer,
                        dist_op,
                        stage_id,
                        layer_type,
                    )
                    for comm_op in layer_comm_ops:
                        dist_model.add_op(comm_op)
                        dist_model.stages[stage_id].append(comm_op.op_id)
                        last_op_id = comm_op.op_id
                else:
                    # 4.1 写入 layout_signature
                    self._apply_layout_signature(op, spec, stage_id, layer_type, op_role)

                    last_op_id = op_id
                    # 5. 根据布局不一致插入通信节点
                    comm_ops = self._maybe_insert_comm_ops(
                        layer,
                        op,
                        dist_op,
                        stage_id,
                        op_spec_cache,
                        layer_type=layer_type,
                        op_role=op_role,
                        is_terminal=(op is not None and op == layer_ops[-1]),
                    )
                    for comm_op in comm_ops:
                        dist_model.add_op(comm_op)
                        dist_model.stages[stage_id].append(comm_op.op_id)
                        last_op_id = comm_op.op_id

                prev_op_id = last_op_id

            if prev_op_id is not None:
                last_op_by_layer[layer.name] = prev_op_id

        return dist_model

    def _get_stage_id(self, layer_idx: int, layers_per_stage: int) -> int:
        """根据层索引计算 stage id"""
        if layers_per_stage <= 0:
            return 0
        stage_id = layer_idx // layers_per_stage
        return min(stage_id, self.deployment.pp - 1)

    def _get_chip_ids(self, stage_id: int) -> list[int]:
        """获取指定 stage 使用的 chip id 列表"""
        tp = self.deployment.tp
        pp = self.deployment.pp
        dp = self.deployment.dp
        chips: list[int] = []
        for dp_rank in range(dp):
            base = dp_rank * pp * tp
            start = base + stage_id * tp
            chips.extend(range(start, start + tp))
        return chips

    def _select_parallel_spec(
        self, layer: Layer, op_role: str, op_type: str
    ) -> ParallelSpec:
        """为 op 选择 ParallelSpec

        优先级:
        1. Pattern 匹配 (layer_type + op_role)
        2. 默认 spec (op_type)
        3. REPLICATE
        """
        # 1. 尝试 pattern 匹配
        layer_type = layer.attrs.get("layer_type", layer.op_type)
        if layer_type and op_role:
            spec = get_pattern_spec(layer_type, op_role)
            if spec is not None:
                return copy.deepcopy(spec)

        # 2. 使用默认 spec
        return copy.deepcopy(get_default_spec(op_type))

    def _get_layer(
        self, layer_name: str, layer_index: dict[str, int], layers: list[Layer]
    ) -> Layer:
        """根据 layer 名称获取 Layer"""
        index = layer_index.get(layer_name)
        if index is None:
            raise ValueError(f"找不到 layer: {layer_name}")
        return layers[index]

    def _maybe_insert_moe_dispatch(
        self,
        layer: Layer,
        layer_type: str,
        op_role: str,
        op: Op | None,
        compute_op: DistributedOp,
        deps: list[str],
    ) -> list[DistributedOp]:
        """在 MoE routed experts 前插入 dispatch 通信"""
        if layer_type != "moe" or op is None:
            return []
        if op_role != "gate" or "expert_gate" not in op.name:
            return []
        if self.deployment.ep <= 1:
            return []
        if not op.inputs and not op.outputs:
            return []

        tensor = op.inputs[0] if op.inputs else op.outputs[0]
        return [
            self._create_custom_comm_op(
                base_op_id=compute_op.op_id,
                suffix="dispatch",
                compute_op=compute_op,
                tensor=tensor,
                comm_type=CommType.ALL2ALL,
                cause="moe_dispatch",
                reason="moe_expert_dispatch",
                deps=deps,
                comm_bytes_override=self._moe_dispatch_comm_bytes(layer),
                extra_attrs={
                    "bs": self._moe_batch_local(layer),
                    "moe_tp": self.deployment.moe_tp,
                    "ep": self.deployment.ep,
                },
            )
        ]

    def _moe_dispatch_comm_bytes(self, layer: Layer) -> int:
        tokens = self._moe_tokens_per_ep_group(layer)
        hidden = self._get_layer_hidden(layer)
        dtype_bytes = layer.inputs[0].dtype.bytes if layer.inputs else 2
        return tokens * hidden * dtype_bytes // max(1, self.deployment.moe_tp)

    def _maybe_insert_layer_comm(
        self,
        layer: Layer,
        compute_op: DistributedOp,
        stage_id: int,
        layer_type: str,
    ) -> list[DistributedOp]:
        if layer_type not in {"embedding", "lmhead"}:
            return []
        if not layer.outputs:
            return []
        tensor = layer.outputs[0]
        comm_op = self._create_custom_comm_op(
            base_op_id=compute_op.op_id,
            suffix="allreduce",
            compute_op=compute_op,
            tensor=tensor,
            comm_type=CommType.ALLREDUCE,
            cause="layout_mismatch",
            reason=f"{layer_type}_allreduce",
            deps=[compute_op.op_id],
            comm_bytes_override=self._scale_comm_bytes(tensor.bytes),
        )
        comm_op.stage_id = stage_id
        return [comm_op]

    def _maybe_insert_comm_ops(
        self,
        layer: Layer,
        op: Op | None,
        compute_op: DistributedOp,
        stage_id: int,
        op_spec_cache: dict[str, ParallelSpec],
        *,
        layer_type: str,
        op_role: str,
        is_terminal: bool,
    ) -> list[DistributedOp]:
        """根据布局不一致插入通信节点"""
        if op is None:
            return []
        outputs = op.outputs
        if not outputs:
            return []

        comm_ops: list[DistributedOp] = []
        for tensor in outputs:
            producer_sig = tensor.layout_signature
            if producer_sig is None:
                continue

            for consumer_op in self._get_tensor_consumer_ops(layer, tensor):
                consumer_role = self._infer_op_role(layer, consumer_op)
                consumer_spec = op_spec_cache.get(consumer_op.name)
                if consumer_spec is None:
                    consumer_spec = self._select_parallel_spec(
                        layer, consumer_role, consumer_op.op_type
                    )
                    op_spec_cache[consumer_op.name] = consumer_spec

                consumer_sig = self._build_layout_signature(
                    consumer_spec, stage_id, layer_type, consumer_role
                )
                if producer_sig != consumer_sig:
                    comm_ops.append(
                        self._create_comm_op(
                            compute_op=compute_op,
                            tensor=tensor,
                            cause="layout_mismatch",
                            deps=[compute_op.op_id],
                        )
                    )
        if comm_ops:
            return self._chain_comm_ops(comm_ops, compute_op.op_id)

        if (
            layer_type == "moe"
            and op_role == "shared_down"
            and compute_op.parallel_spec is not None
            and compute_op.parallel_spec.parallel_type == ParallelType.TP_ROW
        ):
                comm_ops.append(
                    self._create_custom_comm_op(
                        base_op_id=compute_op.op_id,
                        suffix="shared_allreduce",
                        compute_op=compute_op,
                        tensor=outputs[0],
                        comm_type=CommType.ALLREDUCE,
                        cause="layout_mismatch",
                        reason="moe_shared_allreduce",
                        deps=[compute_op.op_id],
                        comm_bytes_override=self._moe_shared_comm_bytes(layer, outputs[0]),
                        extra_attrs={
                            "bs": self._moe_batch_local(layer),
                            "moe_tp": self.deployment.moe_tp,
                            "ep": self.deployment.ep,
                        },
                    )
                )

        if is_terminal and self._needs_terminal_collective(compute_op.parallel_spec):
            if layer_type == "moe" and op_role == "down":
                comm_ops.append(
                    self._create_custom_comm_op(
                        base_op_id=compute_op.op_id,
                        suffix="routed_allreduce",
                        compute_op=compute_op,
                        tensor=outputs[0],
                        comm_type=CommType.ALLREDUCE,
                        cause="layout_mismatch",
                        reason="moe_routed_allreduce",
                        deps=[compute_op.op_id],
                        comm_bytes_override=self._moe_routed_comm_bytes(layer, outputs[0]),
                        extra_attrs={
                            "bs": self._moe_batch_local(layer),
                            "moe_tp": self.deployment.moe_tp,
                            "ep": self.deployment.ep,
                        },
                    )
                )
            else:
                comm_ops.append(
                    self._create_comm_op(
                        compute_op=compute_op,
                        tensor=outputs[0],
                        cause="layout_mismatch",
                        reason="terminal_op_requires_collective",
                        deps=[compute_op.op_id],
                    )
                )

        if (
            is_terminal
            and layer_type in {"embedding", "lmhead"}
            and compute_op.parallel_spec is not None
            and compute_op.parallel_spec.parallel_type == ParallelType.TP_COL
        ):
            comm_ops.append(
                self._create_comm_op(
                    compute_op=compute_op,
                    tensor=outputs[0],
                    cause="layout_mismatch",
                    reason=f"{layer_type}_allreduce",
                    deps=[compute_op.op_id],
                )
            )
        if (
            layer_type == "moe"
            and op_role == "down"
            and op is not None
            and "expert_down" in op.name
            and self.deployment.ep > 1
        ):
            comm_ops.append(
                self._create_custom_comm_op(
                    base_op_id=compute_op.op_id,
                    suffix="combine",
                    compute_op=compute_op,
                    tensor=outputs[0],
                    comm_type=CommType.ALL2ALL,
                    cause="moe_combine",
                    reason="moe_expert_combine",
                    deps=[compute_op.op_id],
                    comm_bytes_override=self._moe_combine_comm_bytes(layer, outputs[0]),
                    extra_attrs={
                        "bs": self._moe_batch_local(layer),
                        "moe_tp": self.deployment.moe_tp,
                        "ep": self.deployment.ep,
                    },
                )
            )
        return self._chain_comm_ops(comm_ops, compute_op.op_id)

    def _create_comm_op(
        self,
        compute_op: DistributedOp,
        tensor: TensorDesc,
        cause: str,
        reason: str | None = None,
        deps: list[str] | None = None,
        extra_attrs: dict[str, int | str | bool] | None = None,
    ) -> DistributedOp:
        """创建通信节点"""
        comm_bytes = self._scale_comm_bytes(tensor.bytes)
        comm_type = self._infer_comm_type(compute_op.parallel_spec)
        path_key = self._resolve_path_key(compute_op.chip_ids)
        attrs = self._build_comm_attrs(extra_attrs)

        return DistributedOp(
            op_id=f"{compute_op.op_id}_comm",
            op_type=comm_type.name.lower(),
            role=NodeRole.COMM,
            comm_type=comm_type,
            comm_bytes=comm_bytes,
            scope="inter_chip",
            cause=cause,
            topology_path_key=path_key,
            participants=list(compute_op.chip_ids),
            algo_hint=comm_type.name.lower(),
            stage_id=compute_op.stage_id,
            chip_ids=compute_op.chip_ids,
            deps=deps or [compute_op.op_id],
            trigger_edge_id=tensor.producer_id,
            reason=reason or "layout_signature_mismatch",
            attrs=attrs,
        )

    def _create_custom_comm_op(
        self,
        base_op_id: str,
        suffix: str,
        compute_op: DistributedOp,
        tensor: TensorDesc,
        comm_type: CommType,
        cause: str,
        reason: str,
        deps: list[str],
        comm_bytes_override: int | None = None,
        extra_attrs: dict[str, int | str | bool] | None = None,
    ) -> DistributedOp:
        """创建自定义通信节点"""
        path_key = self._resolve_path_key(compute_op.chip_ids)
        comm_bytes = (
            comm_bytes_override
            if comm_bytes_override is not None
            else self._scale_comm_bytes(tensor.bytes)
        )
        attrs = self._build_comm_attrs(extra_attrs)
        return DistributedOp(
            op_id=f"{base_op_id}_{suffix}",
            op_type=comm_type.name.lower(),
            role=NodeRole.COMM,
            comm_type=comm_type,
            comm_bytes=comm_bytes,
            scope="inter_chip",
            cause=cause,
            topology_path_key=path_key,
            participants=list(compute_op.chip_ids),
            algo_hint=comm_type.name.lower(),
            stage_id=compute_op.stage_id,
            chip_ids=compute_op.chip_ids,
            deps=deps,
            trigger_edge_id=tensor.producer_id,
            reason=reason,
            attrs=attrs,
        )

    def _chain_comm_ops(
        self, comm_ops: list[DistributedOp], base_dep: str
    ) -> list[DistributedOp]:
        """顺序串联通信节点依赖"""
        prev_id = base_dep
        for comm_op in comm_ops:
            comm_op.deps = [prev_id]
            prev_id = comm_op.op_id
        return comm_ops

    def _infer_comm_type(self, spec: ParallelSpec | None) -> CommType:
        """根据并行规格推断通信类型"""
        if spec is None:
            return CommType.P2P
        if spec.parallel_type in {ParallelType.TP_ROW, ParallelType.TP_HEAD}:
            return CommType.ALLREDUCE
        if spec.parallel_type == ParallelType.EP_EXPERT:
            return CommType.ALL2ALL
        return CommType.ALLGATHER

    def _needs_terminal_collective(self, spec: ParallelSpec | None) -> bool:
        """判断末尾算子是否需要聚合通信"""
        if spec is None:
            return False
        return spec.parallel_type in {
            ParallelType.TP_ROW,
            ParallelType.TP_HEAD,
        }

    def _needs_expert_alltoall(self, spec: ParallelSpec | None) -> bool:
        """判断是否需要 MoE 专家通信"""
        if spec is None:
            return False
        return spec.parallel_type == ParallelType.EP_EXPERT

    def _resolve_path_key(self, chip_ids: list[int]) -> str:
        """解析 topology_path_key"""
        if self.topology is None or not chip_ids:
            return "intra_board"
        if len(chip_ids) <= 1:
            return "intra_board"
        src = chip_ids[0]
        dst = chip_ids[-1]
        path_key, _ = self.topology.resolve_path(src, dst)
        return path_key

    def _scale_comm_bytes(self, comm_bytes: int) -> int:
        """根据 DP 缩放通信字节数"""
        if self.deployment.dp <= 1:
            return comm_bytes
        return comm_bytes // self.deployment.dp

    def _build_comm_attrs(
        self, extra_attrs: dict[str, int | str | bool] | None = None
    ) -> dict[str, str]:
        """构建通信节点的附加属性"""
        attrs: dict[str, str] = {
            "tp": str(self.deployment.tp),
            "pp": str(self.deployment.pp),
            "dp": str(self.deployment.dp),
            "ep": str(self.deployment.ep),
            "moe_tp": str(self.deployment.moe_tp),
            "enable_tp_sp": str(self.deployment.enable_tp_sp),
            "embed_tp": str(self.deployment.embed_tp),
            "lmhead_tp": str(self.deployment.lmhead_tp),
            "comm_protocol": str(self.deployment.comm_protocol),
            "kv_cache_rate": str(self.deployment.kv_cache_rate),
            "is_prefill": str(self.deployment.is_prefill),
        }
        if extra_attrs:
            for key, value in extra_attrs.items():
                attrs[key] = str(value)
        return attrs

    def _layer_param(self, layer: Layer, keys: tuple[str, ...], default: int) -> int:
        for key in keys:
            value = layer.params.get(key)
            if value is not None:
                return int(value)
        return default

    def _get_layer_batch(self, layer: Layer) -> int:
        return self._layer_param(layer, ("batch", "batch_size"), 1)

    def _get_layer_seq(self, layer: Layer) -> int:
        return self._layer_param(layer, ("seq_len", "q_seq_len"), 1)

    def _get_layer_hidden(self, layer: Layer) -> int:
        return self._layer_param(layer, ("hidden_size", "hidden_dim"), 1)

    def _get_layer_activated_experts(self, layer: Layer) -> int:
        return self._layer_param(layer, ("n_activated_experts",), 1)

    def _moe_shared_comm_bytes(self, layer: Layer, tensor: TensorDesc) -> int:
        batch = self._get_layer_batch(layer)
        seq_len = self._get_layer_seq(layer)
        hidden = self._get_layer_hidden(layer)
        batch_local = batch // max(1, self.deployment.dp)
        return batch_local * seq_len * hidden * tensor.dtype.bytes

    def _moe_tokens_per_ep_group(self, layer: Layer) -> int:
        batch = self._get_layer_batch(layer)
        seq_len = self._get_layer_seq(layer)
        activated = self._get_layer_activated_experts(layer)
        tokens = batch * seq_len * activated
        return math.ceil(tokens / max(1, self.deployment.ep))

    def _moe_tokens_per_ep_group_local(self, layer: Layer) -> int:
        batch_local = self._moe_batch_local(layer)
        seq_len = self._get_layer_seq(layer)
        activated = self._get_layer_activated_experts(layer)
        tokens = batch_local * seq_len * activated
        return math.ceil(tokens / max(1, self.deployment.ep))

    def _moe_batch_local(self, layer: Layer) -> int:
        batch = self._get_layer_batch(layer)
        return batch // max(1, self.deployment.dp)

    def _moe_routed_comm_bytes(self, layer: Layer, tensor: TensorDesc) -> int:
        tokens = self._moe_tokens_per_ep_group(layer)
        hidden = self._get_layer_hidden(layer)
        return tokens * hidden * tensor.dtype.bytes

    def _moe_combine_comm_bytes(self, layer: Layer, tensor: TensorDesc) -> int:
        tokens = self._moe_tokens_per_ep_group(layer)
        hidden = self._get_layer_hidden(layer)
        return tokens * hidden * tensor.dtype.bytes

    def _get_tensor_consumer_ops(self, layer: Layer, tensor: TensorDesc) -> list[Op]:
        """获取张量的消费者算子集合"""
        consumers: list[Op] = []
        for op in layer.ops:
            for input_tensor in op.inputs:
                if input_tensor.name == tensor.name:
                    consumers.append(op)
        return consumers

    def _apply_layout_signature(
        self, op: Op, spec: ParallelSpec, stage_id: int, layer_type: str, op_role: str
    ) -> None:
        """为 op 的输出写入布局签名"""
        for tensor in op.outputs:
            tensor.layout_signature = self._build_layout_signature(
                spec, stage_id, layer_type, op_role
            )

    def _build_layout_signature(
        self, spec: ParallelSpec, stage_id: int, layer_type: str, op_role: str
    ) -> dict[str, str | int]:
        """生成布局签名"""
        split_factor = self._resolve_split_factor(spec, layer_type, op_role)
        return {
            "parallel_type": spec.parallel_type.name,
            "split_dim": spec.split_dim,
            "split_factor": split_factor,
            "replica_group_id": f"pp{stage_id}",
        }

    def _resolve_split_factor(
        self, spec: ParallelSpec, layer_type: str, op_role: str
    ) -> int:
        """根据并行类型解析 split_factor"""
        if spec.parallel_type == ParallelType.REPLICATE:
            return 1
        if layer_type == "moe":
            if op_role in {"gate", "up", "down"}:
                return max(1, self.deployment.moe_tp)
            if op_role in {"shared_gate", "shared_up", "shared_down"}:
                return max(1, self.deployment.tp)
        if spec.parallel_type == ParallelType.EP_EXPERT:
            return max(1, self.deployment.ep)
        return max(1, self.deployment.tp)

    def _build_parallel_groups(self) -> tuple[dict[str, list[list[int]]], dict[int, dict[str, int]]]:
        """构建并行分组与 rank_map"""
        tp = self.deployment.tp
        pp = self.deployment.pp
        dp = self.deployment.dp
        total = tp * pp * dp
        if total <= 0:
            return ({}, {})
        chips = list(range(total))

        dp_groups: list[list[int]] = []
        if dp > 1:
            per_dp = total // dp
            for i in range(dp):
                dp_groups.append(chips[i * per_dp : (i + 1) * per_dp])
        else:
            dp_groups = [chips]

        pp_stages: list[list[int]] = []
        for group in dp_groups:
            per_stage = len(group) // pp if pp > 0 else len(group)
            for i in range(pp):
                pp_stages.append(group[i * per_stage : (i + 1) * per_stage])

        tp_groups: list[list[int]] = []
        for stage in pp_stages:
            for i in range(0, len(stage), tp):
                tp_groups.append(stage[i : i + tp])

        ep_groups: list[list[int]] = []
        if self.deployment.ep > 1:
            if self.deployment.ep > dp:
                raise ValueError("ep 不能超过 dp（当前实现约束）")
            for dp_rank in range(self.deployment.ep):
                ep_groups.append(dp_groups[dp_rank])

        rank_map: dict[int, dict[str, int]] = {}
        for chip_id in chips:
            dp_rank = chip_id // (pp * tp)
            within_dp = chip_id % (pp * tp)
            pp_rank = within_dp // tp
            tp_rank = within_dp % tp
            ep_rank = dp_rank if self.deployment.ep > 1 else 0
            rank_map[chip_id] = {
                "dp": dp_rank,
                "pp": pp_rank,
                "tp": tp_rank,
                "ep": ep_rank,
            }

        return {
            "dp_groups": dp_groups,
            "pp_stages": pp_stages,
            "tp_groups": tp_groups,
            "ep_groups": ep_groups,
        }, rank_map

    def _infer_op_role(self, layer: Layer, op: Op | None) -> str:
        """根据 op 名称推断角色"""
        if op is None:
            return layer.attrs.get("op_role", "")

        name = op.name.lower()
        layer_type = layer.attrs.get("layer_type", layer.op_type)

        if layer_type in {"mlp", "ffn"}:
            if "gate" in name and "proj" in name:
                return "gate"
            if "up" in name and "proj" in name:
                return "up"
            if "down" in name and "proj" in name:
                return "down"

        if layer_type == "moe":
            if "shared" in name and "gate" in name and "proj" in name:
                return "shared_gate"
            if "shared" in name and "up" in name and "proj" in name:
                return "shared_up"
            if "shared" in name and "down" in name and "proj" in name:
                return "shared_down"
            if "gate_proj" in name:
                return "router"
            if "expert_gate" in name:
                return "gate"
            if "expert_up" in name:
                return "up"
            if "expert_down" in name:
                return "down"

        if layer_type in {"mla", "mla_absorb", "attention"}:
            if "_q_a" in name:
                return "q_a"
            if "_q_b" in name:
                return "q_b"
            if "_kv_a" in name:
                return "kv_a"
            if "k_compact" in name:
                return "k_compact"
            if "v_compact" in name:
                return "v_compact"
            if "attn_score" in name:
                return "attn_score"
            if "attn_out" in name:
                return "attn_out"
            if "out_proj" in name or "o_proj" in name:
                return "o_proj"
            if "_q_" in name or name.endswith("_q") or "q_proj" in name:
                return "q_proj"
            if "_kv_" in name or "kv_proj" in name:
                return "kv_proj"
            if "k_proj" in name:
                return "k_proj"
            if "v_proj" in name:
                return "v_proj"

        if layer_type == "lmhead":
            if "lm_head" in name or "lmhead" in name or "proj" in name:
                return "proj"

        if layer_type == "embedding":
            if "embed" in name:
                return "embed"

        return ""

    def _extract_dtype_attrs(self, op: Op | None) -> dict[str, str]:
        if op is None:
            return {}
        input_bytes: int | None = None
        weight_bytes: int | None = None
        output_bytes: int | None = None
        for tensor in op.inputs:
            if tensor.is_weight:
                if weight_bytes is None:
                    weight_bytes = tensor.dtype.bytes
            else:
                if input_bytes is None:
                    input_bytes = tensor.dtype.bytes
        for tensor in op.outputs:
            if output_bytes is None:
                output_bytes = tensor.dtype.bytes
        attrs: dict[str, str] = {}
        if input_bytes is not None:
            attrs["input_dtype_bytes"] = str(input_bytes)
        if weight_bytes is not None:
            attrs["weight_dtype_bytes"] = str(weight_bytes)
        if output_bytes is not None:
            attrs["output_dtype_bytes"] = str(output_bytes)
        return attrs

    def _extract_weight_attrs(
        self,
        op: Op | None,
        shard_factor: int,
        op_role: str,
        layer_type: str,
    ) -> dict[str, str]:
        """提取权重占用属性（bytes）"""
        if op is None or op_role == "":
            return {}
        # Attention 分解中的 attn_score/attn_out 是 activation @ activation，
        # 第二输入不是参数权重，不应计入 dram_occupy。
        if layer_type in {"mla", "mla_absorb", "attention"} and op_role in {
            "attn_score",
            "attn_out",
        }:
            return {}
        total_weight_bytes = 0
        for tensor in op.inputs:
            if tensor.is_weight and tensor.producer_id is None:
                total_weight_bytes += tensor.bytes
        if total_weight_bytes <= 0:
            return {}
        local_weight_bytes = math.ceil(total_weight_bytes / max(1, shard_factor))
        return {
            "weight_bytes": str(total_weight_bytes),
            "local_weight_bytes": str(local_weight_bytes),
        }

    def _infer_op_shape(self, layer: Layer, op: Op | None) -> dict[str, int]:
        """推断 op 的 shape (用于 local_shape 计算)"""
        if op is None:
            return {}
        if op.op_type == "matmul":
            return self._infer_matmul_shape(op.inputs)
        outputs = op.outputs
        if outputs:
            return self._shape_from_output(outputs[0])
        return {}

    def _infer_matmul_shape(self, inputs: list[TensorDesc]) -> dict[str, int]:
        """从 MatMul 输入推断 M/K/N/G"""
        if len(inputs) < 2:
            return {}
        a_shape = inputs[0].shape
        b_shape = inputs[1].shape
        if len(a_shape) == 2 and len(b_shape) == 2:
            m, k = a_shape
            _, n = b_shape
            return {"M": m, "K": k, "N": n}
        if len(a_shape) == 3 and len(b_shape) == 3:
            g, m, k = a_shape
            _, _, n = b_shape
            return {"G": g, "M": m, "K": k, "N": n}
        return {}

    def _shape_from_output(self, output: TensorDesc) -> dict[str, int]:
        """从输出张量形状推断 M/N/G"""
        shape = output.shape
        if len(shape) == 2:
            return {"M": shape[0], "N": shape[1]}
        if len(shape) == 3:
            return {"G": shape[0], "M": shape[1], "N": shape[2]}
        return {}

    def _get_output_bytes(self, layer: Layer, op: Op | None) -> int:
        """计算输出通信字节数"""
        outputs = op.outputs if op is not None else layer.outputs
        return sum(tensor.bytes for tensor in outputs)

    def _apply_data_parallel_shard(self, local_shape: dict[str, int]) -> dict[str, int]:
        """应用 DP 维度切分

        输入:
            - local_shape: 并行 spec 切分后的 shape（单位: elements）。
        输出:
            - 按 dp 再切分后的 shape（单位: elements）。
        关键步骤:
            - 优先切分 batch/token 维（M/B），避免重复计入每个 DP 副本。
        """
        if self.deployment.dp <= 1:
            return local_shape
        shaped = dict(local_shape)
        if "M" in shaped and shaped["M"] > 0:
            shaped["M"] = math.ceil(shaped["M"] / self.deployment.dp)
        if "B" in shaped and shaped["B"] > 0:
            shaped["B"] = math.ceil(shaped["B"] / self.deployment.dp)
        # 某些 batched-matmul 将 batch 折叠进 G（如 [B*H, M, K]），需同步按 DP 切分。
        if "G" in shaped and shaped["G"] > 0 and shaped.get("M", 0) == 1:
            shaped["G"] = math.ceil(shaped["G"] / self.deployment.dp)
        return shaped

    def _apply_moe_expert_shard(
        self,
        local_shape: dict[str, int],
        layer: Layer,
        layer_type: str,
        op_role: str,
    ) -> dict[str, int]:
        """应用 MoE expert token 维切分

        输入:
            - local_shape: 已完成 TP/DP 切分的 shape（单位: elements）。
            - layer/layer_type/op_role: 当前算子上下文。
        输出:
            - 对 expert matmul 的 M 维按激活专家与 EP 再切分后的 shape。
        关键步骤:
            - 仅对 moe 的 gate/up/down 生效，M=ceil(tokens_per_ep_group_local)。
        """
        if layer_type != "moe" or op_role not in {"gate", "up", "down"}:
            return local_shape
        if "M" not in local_shape or local_shape["M"] <= 0:
            return local_shape
        shaped = dict(local_shape)
        # 对齐 DS_TPU 的 MoE_TP 口径：routed expert matmul 的 group 维按 moe_tp 放大。
        if "G" in shaped and shaped["G"] > 0 and self.deployment.moe_tp > 1:
            shaped["G"] = shaped["G"] * self.deployment.moe_tp
        shaped["M"] = max(1, self._moe_tokens_per_ep_group_local(layer))
        return shaped
