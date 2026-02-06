"""DistributedModel - 分布式模型

chip 间切分后的模型表示，包含计算节点和通信节点的统一 DAG。
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class NodeRole(Enum):
    """节点角色"""

    COMPUTE = auto()  # 计算节点
    COMM = auto()  # 通信节点


class CommType(Enum):
    """通信类型"""

    ALLREDUCE = auto()
    ALLGATHER = auto()
    ALL2ALL = auto()
    P2P = auto()


@dataclass
class DistributedOp:
    """分布式 op

    Attributes:
        op_id: 原始 op id
        op_type: op 类型
        role: 节点角色 (COMPUTE 或 COMM)
        local_shape: 切分后的 shape
        parallel_spec: 并行规格 (仅 COMPUTE 节点)
        comm_type: 通信类型 (仅 COMM 节点)
        comm_bytes: 通信数据量 (仅 COMM 节点)
        scope: 通信范围 (inter_chip/intra_chip)
        cause: 触发原因 (layout_mismatch/tiling_reduce/tiling_barrier/tiling_relayout)
        topology_path_key: 路径键 (如 intra_board/inter_board/inter_node/intra_noc)
        participants: 参与者列表
        src: 源端 (P2P)
        dst: 目的端 (P2P)
        algo_hint: 通信算法提示 (如 ring/tree)
        trigger_edge_id: 触发通信的依赖边标识
        reason: 触发摘要
        stage_id: PP 阶段 id
        chip_ids: 参与的 chip id 列表
        deps: 依赖的 op_id 列表
        attrs: 其他属性
    """

    op_id: str
    op_type: str
    role: NodeRole
    local_shape: dict[str, int] = field(default_factory=dict)
    parallel_spec: "ParallelSpec | None" = None  # type: ignore
    comm_type: CommType | None = None
    comm_bytes: int = 0
    scope: str = ""
    cause: str = ""
    topology_path_key: str = ""
    participants: list[int] = field(default_factory=list)
    src: int | None = None
    dst: int | None = None
    algo_hint: str | None = None
    trigger_edge_id: str | None = None
    reason: str | None = None
    stage_id: int = 0
    chip_ids: list[int] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class DistributedModel:
    """分布式模型

    chip 间切分后的模型表示。

    Attributes:
        ops: 所有 op 列表 (计算 + 通信)
        op_map: op_id -> DistributedOp 映射
        tp: TP 并行度
        pp: PP 并行度
        ep: EP 并行度 (MoE)
        num_chips: 总 chip 数量
        stages: PP 阶段列表，每个阶段包含的 op_id 列表
        parallel_groups: 并行分组 (tp/dp/ep/pp)
        rank_map: chip_id -> rank 信息 (dp/tp/pp/ep)
        graph_nodes: 计算图节点列表
        graph_edges: 计算图边列表 (src, dst)
        chip_assignments: op_id -> chip_ids
        op_parallel_specs: op_id -> ParallelSpec
    """

    ops: list[DistributedOp] = field(default_factory=list)
    op_map: dict[str, DistributedOp] = field(default_factory=dict)
    tp: int = 1
    pp: int = 1
    ep: int = 1
    num_chips: int = 1
    stages: list[list[str]] = field(default_factory=list)
    parallel_groups: dict[str, list[list[int]]] = field(default_factory=dict)
    rank_map: dict[int, dict[str, int]] = field(default_factory=dict)
    graph_nodes: list[str] = field(default_factory=list)
    graph_edges: list[tuple[str, str]] = field(default_factory=list)
    _graph_node_set: set[str] = field(default_factory=set, repr=False)
    _graph_edge_set: set[tuple[str, str]] = field(default_factory=set, repr=False)
    chip_assignments: dict[str, list[int]] = field(default_factory=dict)
    op_parallel_specs: dict[str, "ParallelSpec"] = field(default_factory=dict)  # type: ignore

    def add_op(self, op: DistributedOp) -> None:
        """添加 op"""
        self.ops.append(op)
        self.op_map[op.op_id] = op
        if op.op_id not in self._graph_node_set:
            self._graph_node_set.add(op.op_id)
            self.graph_nodes.append(op.op_id)
        for dep in op.deps:
            edge = (dep, op.op_id)
            if edge not in self._graph_edge_set:
                self._graph_edge_set.add(edge)
                self.graph_edges.append(edge)
        if op.chip_ids:
            self.chip_assignments[op.op_id] = list(op.chip_ids)
        if op.parallel_spec is not None:
            self.op_parallel_specs[op.op_id] = op.parallel_spec

    def get_op(self, op_id: str) -> DistributedOp | None:
        """获取 op"""
        return self.op_map.get(op_id)

    def get_compute_ops(self) -> list[DistributedOp]:
        """获取所有计算节点"""
        return [op for op in self.ops if op.role == NodeRole.COMPUTE]

    def get_comm_ops(self) -> list[DistributedOp]:
        """获取所有通信节点"""
        return [op for op in self.ops if op.role == NodeRole.COMM]

    def get_stage_ops(self, stage_id: int) -> list[DistributedOp]:
        """获取指定阶段的所有 op"""
        return [op for op in self.ops if op.stage_id == stage_id]

    def summary(self) -> str:
        """生成摘要信息"""
        compute_ops = self.get_compute_ops()
        comm_ops = self.get_comm_ops()
        return (
            f"DistributedModel(\n"
            f"  tp={self.tp}, pp={self.pp}, ep={self.ep}, num_chips={self.num_chips}\n"
            f"  compute_ops={len(compute_ops)}, comm_ops={len(comm_ops)}\n"
            f"  stages={len(self.stages)}\n"
            f"  parallel_groups={list(self.parallel_groups.keys())}\n"
            f")"
        )
