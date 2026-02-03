"""
依赖图模块

构建和管理算子之间的依赖关系，用于事件驱动调度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set
from enum import Enum


class DependencyType(Enum):
    """依赖类型"""
    DATA = "data"              # 数据依赖：后续操作需要前序操作的输出
    RESOURCE = "resource"      # 资源依赖：需要等待资源空闲
    SYNC = "sync"              # 同步依赖：需要等待多个操作都完成
    COMM = "comm"              # 通信依赖：PP 前向传递需要等上一 stage


@dataclass
class OperatorNode:
    """算子节点

    表示依赖图中的一个算子，包含执行所需的所有信息。
    """

    # 唯一标识
    name: str
    chip_id: str
    layer_index: int
    micro_batch: int = 0

    # 算子类型
    op_type: str = ""
    is_compute: bool = True  # True: 计算算子, False: 通信算子

    # PP 阶段
    pp_stage: int = 0

    # 执行时间（由评估器预先计算）
    duration_us: float = 0.0

    # 计算算子的性能数据
    flops: float = 0.0
    dram_traffic_bytes: float = 0.0
    compute_time_us: float = 0.0
    memory_time_us: float = 0.0

    # GEMM 优化结果
    best_tile: Optional[dict] = None
    best_partition: Optional[dict] = None
    gemm_shape: Optional[dict] = None

    # 通信算子的数据
    comm_size_bytes: float = 0.0
    participating_chips: list[str] = field(default_factory=list)

    # 依赖关系
    predecessors: Set[tuple] = field(default_factory=set)  # 前驱算子的 key
    successors: Set[tuple] = field(default_factory=set)    # 后继算子的 key

    # 状态
    completed: bool = False
    pending_predecessors: int = 0  # 未完成的前驱数量

    @property
    def key(self) -> tuple:
        """返回算子的唯一标识 key"""
        return (self.chip_id, self.layer_index, self.name, self.micro_batch)

    def is_ready(self) -> bool:
        """检查算子是否可以开始执行"""
        return self.pending_predecessors == 0 and not self.completed


@dataclass
class DependencyEdge:
    """依赖边"""
    source: tuple  # 源算子的 key
    target: tuple  # 目标算子的 key
    dep_type: DependencyType


class DependencyGraph:
    """依赖图

    管理算子之间的依赖关系，支持：
    - 添加算子节点
    - 添加依赖边
    - 查询就绪的算子
    - 标记算子完成
    """

    def __init__(self):
        # 算子节点 {key: OperatorNode}
        self._nodes: dict[tuple, OperatorNode] = {}

        # 依赖边
        self._edges: list[DependencyEdge] = []

        # 层完成追踪 {(layer_index, micro_batch): set of completed op keys}
        self._layer_completed_ops: dict[tuple[int, int], set] = {}
        self._layer_total_ops: dict[tuple[int, int], int] = {}

    def add_node(self, node: OperatorNode) -> None:
        """添加算子节点

        Args:
            node: 算子节点
        """
        self._nodes[node.key] = node

        # 更新层的算子计数
        layer_key = (node.layer_index, node.micro_batch)
        self._layer_total_ops[layer_key] = self._layer_total_ops.get(layer_key, 0) + 1

    def add_edge(
        self,
        source_key: tuple,
        target_key: tuple,
        dep_type: DependencyType = DependencyType.DATA,
    ) -> None:
        """添加依赖边

        Args:
            source_key: 源算子的 key
            target_key: 目标算子的 key
            dep_type: 依赖类型
        """
        # 添加边
        self._edges.append(DependencyEdge(
            source=source_key,
            target=target_key,
            dep_type=dep_type,
        ))

        # 更新节点的前驱/后继关系
        source_node = self._nodes.get(source_key)
        target_node = self._nodes.get(target_key)

        if source_node:
            source_node.successors.add(target_key)

        if target_node:
            target_node.predecessors.add(source_key)
            target_node.pending_predecessors += 1

    def get_node(self, key: tuple) -> Optional[OperatorNode]:
        """获取算子节点

        Args:
            key: 算子的 key

        Returns:
            算子节点，不存在返回 None
        """
        return self._nodes.get(key)

    def mark_completed(self, key: tuple) -> None:
        """标记算子完成

        Args:
            key: 算子的 key
        """
        node = self._nodes.get(key)
        if node is None or node.completed:
            return

        node.completed = True

        # 更新层完成追踪
        layer_key = (node.layer_index, node.micro_batch)
        if layer_key not in self._layer_completed_ops:
            self._layer_completed_ops[layer_key] = set()
        self._layer_completed_ops[layer_key].add(key)

        # 更新后继节点的前驱计数
        for succ_key in node.successors:
            succ_node = self._nodes.get(succ_key)
            if succ_node:
                succ_node.pending_predecessors -= 1

    def get_ready_successors(self, key: tuple) -> list[OperatorNode]:
        """获取可以开始执行的后继算子

        Args:
            key: 刚完成的算子的 key

        Returns:
            就绪的后继算子列表
        """
        node = self._nodes.get(key)
        if node is None:
            return []

        ready = []
        for succ_key in node.successors:
            succ_node = self._nodes.get(succ_key)
            if succ_node and succ_node.is_ready():
                ready.append(succ_node)

        return ready

    def get_ready_nodes(self) -> list[OperatorNode]:
        """获取所有就绪的算子

        Returns:
            就绪的算子列表
        """
        return [node for node in self._nodes.values() if node.is_ready()]

    def is_layer_complete(self, layer_index: int, micro_batch: int = 0) -> bool:
        """检查某层是否完成

        Args:
            layer_index: 层索引
            micro_batch: 微批次索引

        Returns:
            该层是否完成
        """
        layer_key = (layer_index, micro_batch)
        total = self._layer_total_ops.get(layer_key, 0)
        completed = len(self._layer_completed_ops.get(layer_key, set()))
        return completed >= total and total > 0

    def get_first_op_in_layer(
        self,
        layer_index: int,
        chip_id: str,
        micro_batch: int = 0,
    ) -> Optional[OperatorNode]:
        """获取某层的第一个算子

        Args:
            layer_index: 层索引
            chip_id: 芯片ID
            micro_batch: 微批次索引

        Returns:
            该层的第一个算子
        """
        # 找到该层该芯片上没有前驱的算子
        for node in self._nodes.values():
            if (
                node.layer_index == layer_index
                and node.chip_id == chip_id
                and node.micro_batch == micro_batch
                and node.pending_predecessors == 0
                and not node.completed
            ):
                return node
        return None

    def get_entry_nodes(self, micro_batch: int = 0) -> list[OperatorNode]:
        """获取入口节点（没有前驱的节点）

        Args:
            micro_batch: 微批次索引

        Returns:
            入口节点列表
        """
        return [
            node for node in self._nodes.values()
            if node.micro_batch == micro_batch
            and len(node.predecessors) == 0
            and not node.completed
        ]

    def reset(self) -> None:
        """重置所有状态"""
        for node in self._nodes.values():
            node.completed = False
            node.pending_predecessors = len(node.predecessors)

        self._layer_completed_ops.clear()

    def get_stats(self) -> dict:
        """获取依赖图统计信息"""
        total = len(self._nodes)
        completed = sum(1 for n in self._nodes.values() if n.completed)
        compute_ops = sum(1 for n in self._nodes.values() if n.is_compute)
        comm_ops = total - compute_ops

        return {
            "total_nodes": total,
            "completed_nodes": completed,
            "pending_nodes": total - completed,
            "compute_ops": compute_ops,
            "comm_ops": comm_ops,
            "total_edges": len(self._edges),
        }

    def to_dict(self) -> dict:
        """序列化为字典（用于调试）"""
        return {
            "nodes": [
                {
                    "key": node.key,
                    "name": node.name,
                    "op_type": node.op_type,
                    "is_compute": node.is_compute,
                    "layer_index": node.layer_index,
                    "chip_id": node.chip_id,
                    "duration_us": node.duration_us,
                    "completed": node.completed,
                    "predecessors": list(node.predecessors),
                    "successors": list(node.successors),
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.dep_type.value,
                }
                for edge in self._edges
            ],
        }


class DependencyGraphBuilder:
    """依赖图构建器

    从层定义构建依赖图。
    """

    def __init__(self):
        self.graph = DependencyGraph()

    def build_from_layers(
        self,
        layers: list,  # List of BaseLayer
        chip_assignments: dict[int, str],  # layer_index -> chip_id
        pp_stage_map: dict[int, int],  # layer_index -> pp_stage
        micro_batch: int = 0,
    ) -> DependencyGraph:
        """从层列表构建依赖图

        Args:
            layers: 层列表
            chip_assignments: 层到芯片的映射
            pp_stage_map: 层到 PP 阶段的映射
            micro_batch: 微批次索引

        Returns:
            构建好的依赖图
        """
        prev_layer_last_op = None

        for layer_idx, layer in enumerate(layers):
            chip_id = chip_assignments.get(layer_idx, "chip_0")
            pp_stage = pp_stage_map.get(layer_idx, 0)

            # 该层内的算子
            layer_ops = []

            # 添加计算算子
            for op in layer.comp_ops:
                node = OperatorNode(
                    name=op.name,
                    chip_id=chip_id,
                    layer_index=layer_idx,
                    micro_batch=micro_batch,
                    op_type=op.operator_type,
                    is_compute=True,
                    pp_stage=pp_stage,
                    duration_us=op.elapse,
                    flops=getattr(op, 'flops', 0.0),
                    dram_traffic_bytes=getattr(op, 'dram_traffic', 0.0),
                    compute_time_us=getattr(op, 'comp_elapse', 0.0),
                    memory_time_us=getattr(op, 'dma_elapse', 0.0),
                )
                self.graph.add_node(node)
                layer_ops.append(node)

            # 添加通信算子
            for op in layer.comm_ops:
                node = OperatorNode(
                    name=op.name,
                    chip_id=chip_id,
                    layer_index=layer_idx,
                    micro_batch=micro_batch,
                    op_type=op.comm_kind,
                    is_compute=False,
                    pp_stage=pp_stage,
                    duration_us=op.comm_elapse,
                    comm_size_bytes=getattr(op, 'comm_size', 0.0),
                )
                self.graph.add_node(node)
                layer_ops.append(node)

            # 层内依赖：顺序执行
            for i in range(1, len(layer_ops)):
                self.graph.add_edge(
                    layer_ops[i - 1].key,
                    layer_ops[i].key,
                    DependencyType.DATA,
                )

            # 层间依赖
            if prev_layer_last_op and layer_ops:
                prev_pp_stage = pp_stage_map.get(layer_idx - 1, 0)
                curr_pp_stage = pp_stage

                if prev_pp_stage == curr_pp_stage:
                    # 同一 PP 阶段，数据依赖
                    self.graph.add_edge(
                        prev_layer_last_op.key,
                        layer_ops[0].key,
                        DependencyType.DATA,
                    )
                else:
                    # 跨 PP 阶段，通信依赖
                    self.graph.add_edge(
                        prev_layer_last_op.key,
                        layer_ops[0].key,
                        DependencyType.COMM,
                    )

            if layer_ops:
                prev_layer_last_op = layer_ops[-1]

        return self.graph

    def add_tp_comm_dependencies(
        self,
        layer_index: int,
        comm_op_name: str,
        tp_chips: list[str],
        micro_batch: int = 0,
    ) -> None:
        """添加 TP 通信的同步依赖

        在 TP AllReduce 前，所有 TP 组的芯片需要完成对应的计算。

        Args:
            layer_index: 层索引
            comm_op_name: 通信算子名称
            tp_chips: TP 组的芯片列表
            micro_batch: 微批次索引
        """
        # 找到该通信算子
        for chip_id in tp_chips:
            key = (chip_id, layer_index, comm_op_name, micro_batch)
            node = self.graph.get_node(key)
            if node:
                node.participating_chips = tp_chips.copy()

    def build_moe_layer_with_tbo(
        self,
        layer_idx: int,
        layer,  # MoELayer
        chip_id: str,
        pp_stage: int,
        micro_batch: int = 0,
        enable_tbo: bool = True,
    ) -> list[OperatorNode]:
        """构建 MoE 层的依赖图（支持 TBO 优化）

        TBO (Tensor-Bus Overlap) 优化允许：
        - Dispatch 通信和 Expert 计算并行（使用不同资源）
        - Combine 通信和下一层的 Attention 计算并行

        MoE 层结构：
        Gate -> Dispatch -> Expert -> Combine

        TBO 优化后：
        Gate -> Dispatch -----> Combine
                    \         /
                     Expert --

        Args:
            layer_idx: 层索引
            layer: MoE 层对象
            chip_id: 芯片 ID
            pp_stage: PP 阶段
            micro_batch: 微批次索引
            enable_tbo: 是否启用 TBO 优化

        Returns:
            该层的所有算子节点
        """
        layer_ops = []

        # 分类算子
        gate_ops = []
        dispatch_ops = []
        expert_ops = []
        combine_ops = []
        other_comp_ops = []
        other_comm_ops = []

        for op in layer.comp_ops:
            op_type = op.operator_type.lower() if hasattr(op, 'operator_type') else ""
            op_name = op.name.lower() if hasattr(op, 'name') else ""

            if "gate" in op_type or "gate" in op_name:
                gate_ops.append(op)
            elif "expert" in op_type or "expert" in op_name:
                expert_ops.append(op)
            else:
                other_comp_ops.append(op)

        for op in layer.comm_ops:
            comm_kind = op.comm_kind.lower() if hasattr(op, 'comm_kind') else ""
            op_name = op.name.lower() if hasattr(op, 'name') else ""

            if "dispatch" in comm_kind or "dispatch" in op_name:
                dispatch_ops.append(op)
            elif "combine" in comm_kind or "combine" in op_name:
                combine_ops.append(op)
            else:
                other_comm_ops.append(op)

        # 创建节点
        def create_compute_node(op):
            node = OperatorNode(
                name=op.name,
                chip_id=chip_id,
                layer_index=layer_idx,
                micro_batch=micro_batch,
                op_type=op.operator_type,
                is_compute=True,
                pp_stage=pp_stage,
                duration_us=op.elapse,
                flops=getattr(op, 'flops', 0.0),
                dram_traffic_bytes=getattr(op, 'dram_traffic', 0.0),
                compute_time_us=getattr(op, 'comp_elapse', 0.0),
                memory_time_us=getattr(op, 'dma_elapse', 0.0),
            )
            self.graph.add_node(node)
            return node

        def create_comm_node(op):
            node = OperatorNode(
                name=op.name,
                chip_id=chip_id,
                layer_index=layer_idx,
                micro_batch=micro_batch,
                op_type=op.comm_kind,
                is_compute=False,
                pp_stage=pp_stage,
                duration_us=op.comm_elapse,
                comm_size_bytes=getattr(op, 'comm_size', 0.0),
            )
            self.graph.add_node(node)
            return node

        # 构建节点
        gate_nodes = [create_compute_node(op) for op in gate_ops]
        dispatch_nodes = [create_comm_node(op) for op in dispatch_ops]
        expert_nodes = [create_compute_node(op) for op in expert_ops]
        combine_nodes = [create_comm_node(op) for op in combine_ops]

        # 构建依赖关系
        if enable_tbo:
            # TBO 模式：允许并行

            # Gate -> Dispatch（顺序）
            for i, gate_node in enumerate(gate_nodes):
                layer_ops.append(gate_node)
                if i > 0:
                    self.graph.add_edge(gate_nodes[i-1].key, gate_node.key, DependencyType.DATA)

            if gate_nodes and dispatch_nodes:
                self.graph.add_edge(gate_nodes[-1].key, dispatch_nodes[0].key, DependencyType.DATA)

            for dispatch_node in dispatch_nodes:
                layer_ops.append(dispatch_node)

            # Dispatch 和 Expert 可以并行（TBO 核心：不建立强依赖）
            # Expert 只需要等待 Gate 完成（而非 Dispatch）
            if gate_nodes and expert_nodes:
                self.graph.add_edge(gate_nodes[-1].key, expert_nodes[0].key, DependencyType.DATA)

            for i, expert_node in enumerate(expert_nodes):
                layer_ops.append(expert_node)
                if i > 0:
                    self.graph.add_edge(expert_nodes[i-1].key, expert_node.key, DependencyType.DATA)

            # Combine 需要等待 Expert 和 Dispatch 都完成
            if expert_nodes and combine_nodes:
                self.graph.add_edge(expert_nodes[-1].key, combine_nodes[0].key, DependencyType.DATA)
            if dispatch_nodes and combine_nodes:
                # 同步依赖：Combine 需要等待 Dispatch 完成
                self.graph.add_edge(dispatch_nodes[-1].key, combine_nodes[0].key, DependencyType.SYNC)

            for combine_node in combine_nodes:
                layer_ops.append(combine_node)

        else:
            # 非 TBO 模式：严格顺序执行
            all_ops = gate_nodes + dispatch_nodes + expert_nodes + combine_nodes

            for i, node in enumerate(all_ops):
                layer_ops.append(node)
                if i > 0:
                    self.graph.add_edge(all_ops[i-1].key, node.key, DependencyType.DATA)

        # 添加其他算子
        for op in other_comp_ops:
            node = create_compute_node(op)
            layer_ops.append(node)

        for op in other_comm_ops:
            node = create_comm_node(op)
            layer_ops.append(node)

        return layer_ops
