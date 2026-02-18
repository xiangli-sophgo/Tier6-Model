"""Scheduler - 时序调度实现.

计算流程:
    1. 拓扑排序：对 Op/CommOp 依赖图进行 Kahn 拓扑排序
    2. 优先级计算：fanout 或关键路径深度
    3. 优先级调度：在 ready 集合内按优先级选择下一个调度节点
    4. 实例展开：将 Op/CommOp 展开为 OpInstance/CommInstance
    5. 资源统计与冲突检测：core_slots/path_slots 占用与冲突
    6. 重叠放置：尝试计算/通信重叠（allow_overlap=True 时）
    7. buffer 估算与峰值检测
    8. 计划产出

输入:
    - DistributedModel（graph_nodes/graph_edges 为 DAG）
    - TilePlan（可选，用于填充 tile/kernel 配置）
    - SchedulePolicy（调度策略参数）

输出:
    - ExecPlan（timeline/instances/binding/precedence/buffer_plan/overlap/trace_meta）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from perf_model.L3_mapping.common.plan.distributed_model import DistributedModel, DistributedOp
from perf_model.L3_mapping.math.plan.exec_plan import CommInstance, ExecPlan, OpInstance
from perf_model.L3_mapping.math.tiling.planner import TilePlan


@dataclass
class SchedulePolicy:
    """调度策略参数

    Attributes:
        allow_overlap: 是否允许计算/通信重叠
        priority_mode: 优先级模式 ("fanout" | "critical_path")
        buffer_peak_limit: buffer 峰值上限（bytes），0 表示不限制
        max_conflicts_per_step: 单步最大允许冲突数，超过则回退
    """

    allow_overlap: bool = False
    priority_mode: str = "fanout"  # "fanout" | "critical_path"
    buffer_peak_limit: int = 0  # bytes, 0 表示不限制
    max_conflicts_per_step: int = 0  # 0 表示不限制


@dataclass
class ResourceSlot:
    """资源槽位状态

    Attributes:
        core_slots: chip_id -> 当前占用的 core 数量
        path_slots: path_key -> 当前占用的链路数量
        active_ops: 当前活跃的 op_id 集合
    """

    core_slots: dict[int, int] = field(default_factory=dict)
    path_slots: dict[str, int] = field(default_factory=dict)
    active_ops: set[str] = field(default_factory=set)

    def copy(self) -> "ResourceSlot":
        """创建副本"""
        return ResourceSlot(
            core_slots=self.core_slots.copy(),
            path_slots=self.path_slots.copy(),
            active_ops=self.active_ops.copy(),
        )


@dataclass
class ConflictRecord:
    """冲突记录

    Attributes:
        step: 发生冲突的调度步骤
        op_id: 冲突的 op_id
        conflict_type: 冲突类型 (core_conflict | path_conflict | buffer_overflow)
        conflicting_ops: 冲突的其他 op_id
        resolved: 是否已修复
        resolution: 修复方式 (delay | serial)
    """

    step: int
    op_id: str
    conflict_type: str
    conflicting_ops: list[str] = field(default_factory=list)
    resolved: bool = False
    resolution: str = ""


class Scheduler:
    """调度器

    按依赖拓扑排序生成线性时序计划，支持计算/通信重叠与资源冲突检测。
    不新增通信节点。
    """

    def __init__(self, policy: SchedulePolicy | None = None) -> None:
        self.policy = policy or SchedulePolicy()

    def plan(
        self, dist_model: DistributedModel, tile_plan: TilePlan | None = None
    ) -> ExecPlan:
        """生成调度计划

        输入:
            - dist_model.graph_edges 为 DAG，节点为 op_id（单位: 节点数）。
            - tile_plan 可选，用于填充 tile/kernel 配置。
        输出:
            - ExecPlan，包含 timeline/instances/binding/precedence/buffer_plan/overlap。
        关键步骤:
            - topo_sort -> priority -> schedule -> overlap -> conflict -> buffer_peak。
        """
        # 1. 拓扑排序
        topo_order = self._topo_sort(dist_model)

        # 2. 优先级计算
        if self.policy.priority_mode == "critical_path":
            priorities = self._compute_critical_path(dist_model)
        else:
            priorities = self._compute_priority_fanout(dist_model)

        # 3. 按优先级调度
        schedule_order = self._schedule_with_priority(
            dist_model, topo_order, priorities
        )

        # 4-7. 实例展开、资源统计、重叠放置、buffer 估算
        timeline: list[dict[str, Any]] = []
        binding: dict[str, Any] = {}
        instances: list[OpInstance | CommInstance] = []
        buffer_plan: dict[str, Any] = {}
        overlap_results: list[dict[str, Any]] = []
        conflicts: list[ConflictRecord] = []

        resource_slot = ResourceSlot()
        buffer_live: dict[str, dict[str, Any]] = {}  # op_id -> {bytes, start}
        buffer_peak = 0
        buffer_current = 0
        step = 0

        # 构建依赖的消费者映射（用于 buffer 生命周期）
        consumers = self._build_consumers(dist_model)

        for op_id in schedule_order:
            op = dist_model.get_op(op_id)
            if op is None:
                continue

            instance_id = f"inst::{op_id}"
            is_comm = op.role.name.lower() == "comm"

            # 尝试重叠放置
            overlap_info: dict[str, Any] | None = None
            if self.policy.allow_overlap and is_comm:
                overlap_info = self._try_overlap(
                    op, step, resource_slot, timeline, dist_model
                )
                if overlap_info:
                    overlap_results.append(overlap_info)

            # 冲突检测
            conflict = self._detect_conflict(op, resource_slot, step)
            if conflict:
                conflicts.append(conflict)
                # 尝试修复冲突（通过延迟）
                self._fix_conflict(conflict, resource_slot)

            # 实例展开
            start_step = overlap_info["start"] if overlap_info else step
            end_step = start_step + 1

            if is_comm:
                instance = CommInstance(
                    instance_id=instance_id,
                    op_id=op_id,
                    chip_ids=list(op.chip_ids),
                    path_key=op.topology_path_key or None,
                    deps=list(op.deps),
                    start=start_step,
                    end=end_step,
                )
            else:
                instance = OpInstance(
                    instance_id=instance_id,
                    op_id=op_id,
                    chip_ids=list(op.chip_ids),
                    core_ids=[],
                    deps=list(op.deps),
                    start=start_step,
                    end=end_step,
                )
            instances.append(instance)

            # timeline 记录
            timeline.append(
                {
                    "op_id": op_id,
                    "role": op.role.name.lower(),
                    "start": start_step,
                    "end": end_step,
                    "scope": op.scope,
                    "cause": op.cause,
                    "chip_ids": list(op.chip_ids),
                    "instance_id": instance_id,
                    "overlapped": overlap_info is not None,
                }
            )

            # 资源占用更新
            self._update_resource_slot(op, resource_slot)

            # 绑定信息
            binding[op_id] = {
                "chip_ids": list(op.chip_ids),
                "path_key": op.topology_path_key or None,
                "core_ids": [],
            }

            # buffer 估算与峰值追踪
            buf_bytes = self._estimate_output_bytes(op)
            if buf_bytes > 0:
                buffer_live[op_id] = {"bytes": buf_bytes, "start": step}
                buffer_current += buf_bytes
                buffer_peak = max(buffer_peak, buffer_current)

                # 检查 buffer 峰值是否超限
                if (
                    self.policy.buffer_peak_limit > 0
                    and buffer_current > self.policy.buffer_peak_limit
                ):
                    conflicts.append(
                        ConflictRecord(
                            step=step,
                            op_id=op_id,
                            conflict_type="buffer_overflow",
                            conflicting_ops=[],
                            resolved=False,
                            resolution="",
                        )
                    )

            # 释放已消费的 buffer
            buffer_current = self._release_consumed_buffers(
                op_id, buffer_live, buffer_plan, consumers, step, buffer_current
            )

            if not overlap_info:
                step += 1

        # 收尾：关闭所有未释放的 buffer
        for buf_op_id, buf_info in buffer_live.items():
            buffer_plan[buf_op_id] = {
                "bytes": buf_info["bytes"],
                "start": buf_info["start"],
                "end": step,
            }

        # 8. 构建 ExecPlan
        exec_plan = ExecPlan(
            tile_config=tile_plan.tile_configs if tile_plan else {},
            kernel_config=tile_plan.kernel_configs if tile_plan else {},
            timeline=timeline,
            instances=instances,
            binding=binding,
            precedence=list(dist_model.graph_edges),
            buffer_plan=buffer_plan,
            overlap=overlap_results,
            trace_meta={
                "policy": {
                    "allow_overlap": self.policy.allow_overlap,
                    "priority_mode": self.policy.priority_mode,
                    "buffer_peak_limit": self.policy.buffer_peak_limit,
                },
                "priority": priorities,
                "core_slots": dict(resource_slot.core_slots),
                "path_slots": dict(resource_slot.path_slots),
                "buffer_peak": buffer_peak,
                "conflicts": [
                    {
                        "step": c.step,
                        "op_id": c.op_id,
                        "type": c.conflict_type,
                        "resolved": c.resolved,
                        "resolution": c.resolution,
                    }
                    for c in conflicts
                ],
            },
        )
        return exec_plan

    def _topo_sort(self, dist_model: DistributedModel) -> list[str]:
        """拓扑排序（Kahn 算法）

        输入:
            - dist_model.graph_nodes 为所有 op_id 集合（单位: 节点数）。
        输出:
            - 返回满足依赖的 op_id 顺序列表（单位: 节点数）。
        关键步骤:
            - 构建入度表（单位: 依赖计数）并使用 Kahn 算法。
        """
        nodes = list(dist_model.graph_nodes)
        indegree: dict[str, int] = {node: 0 for node in nodes}
        adj: dict[str, list[str]] = {node: [] for node in nodes}

        for src, dst in dist_model.graph_edges:
            if src not in indegree:
                indegree[src] = 0
                adj[src] = []
            if dst not in indegree:
                indegree[dst] = 0
                adj[dst] = []
            adj[src].append(dst)
            indegree[dst] += 1

        queue = deque([node for node, deg in indegree.items() if deg == 0])
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adj.get(node, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(indegree):
            raise ValueError("调度失败：检测到环或不完整依赖图")
        return order

    def _compute_priority_fanout(self, dist_model: DistributedModel) -> dict[str, int]:
        """计算 fanout 优先级

        输入:
            - dist_model.graph_edges（单位: 依赖边数）。
        输出:
            - priorities：op_id -> fanout 分数（单位: 边数）。
        关键步骤:
            - 统计每个节点的 fanout（出度），fanout 越大越优先。
        """
        fanout: dict[str, int] = {node: 0 for node in dist_model.graph_nodes}
        for src, _ in dist_model.graph_edges:
            fanout[src] = fanout.get(src, 0) + 1
        return fanout

    def _compute_critical_path(self, dist_model: DistributedModel) -> dict[str, int]:
        """计算关键路径优先级

        输入:
            - dist_model.graph_edges（单位: 依赖边数）。
        输出:
            - priorities：op_id -> 关键路径深度（单位: 节点数）。
        关键步骤:
            - 反向 BFS 计算每个节点到终点的最长路径长度。
        """
        nodes = list(dist_model.graph_nodes)
        # 构建反向邻接表
        reverse_adj: dict[str, list[str]] = {node: [] for node in nodes}
        out_degree: dict[str, int] = {node: 0 for node in nodes}
        for src, dst in dist_model.graph_edges:
            reverse_adj[dst].append(src)
            out_degree[src] = out_degree.get(src, 0) + 1

        # 从叶子节点反向 BFS
        depth: dict[str, int] = {node: 0 for node in nodes}
        leaves = [node for node, deg in out_degree.items() if deg == 0]
        queue = deque(leaves)

        while queue:
            node = queue.popleft()
            for pred in reverse_adj.get(node, []):
                new_depth = depth[node] + 1
                if new_depth > depth[pred]:
                    depth[pred] = new_depth
                out_degree[pred] -= 1
                if out_degree[pred] == 0:
                    queue.append(pred)

        return depth

    def _schedule_with_priority(
        self,
        dist_model: DistributedModel,
        topo_order: list[str],
        priorities: dict[str, int],
    ) -> list[str]:
        """在 topo 约束内按优先级排序

        输入:
            - topo_order 为 DAG 拓扑序（单位: op_id 列表）。
            - priorities 为优先级分数。
        输出:
            - 返回满足依赖的调度顺序（单位: op_id 列表）。
        关键步骤:
            - 使用 ready 队列，按优先级择优出队。
        """
        indegree: dict[str, int] = {node: 0 for node in dist_model.graph_nodes}
        adj: dict[str, list[str]] = {node: [] for node in dist_model.graph_nodes}
        for src, dst in dist_model.graph_edges:
            adj[src].append(dst)
            indegree[dst] = indegree.get(dst, 0) + 1

        ready = [node for node, deg in indegree.items() if deg == 0]
        order: list[str] = []
        while ready:
            ready.sort(key=lambda n: priorities.get(n, 0), reverse=True)
            node = ready.pop(0)
            order.append(node)
            for nxt in adj.get(node, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(order) != len(dist_model.graph_nodes):
            return topo_order
        return order

    def _try_overlap(
        self,
        op: DistributedOp,
        current_step: int,
        resource_slot: ResourceSlot,
        timeline: list[dict[str, Any]],
        dist_model: DistributedModel,
    ) -> dict[str, Any] | None:
        """尝试计算/通信重叠

        输入:
            - op: 当前通信 op。
            - current_step: 当前调度步骤。
            - resource_slot: 资源槽位状态。
            - timeline: 已调度的 timeline。
        输出:
            - 重叠信息 dict，包含 start/overlap_with/type；若无法重叠则返回 None。
        关键步骤:
            - 查找前序计算 op，检查资源不冲突，尝试并行放置。
        """
        if current_step == 0:
            return None

        # 查找最近的计算 op
        prev_compute: dict[str, Any] | None = None
        for entry in reversed(timeline):
            if entry["role"] == "compute":
                prev_compute = entry
                break

        if prev_compute is None:
            return None

        prev_op_id = prev_compute["op_id"]
        prev_op = dist_model.get_op(prev_op_id)
        if prev_op is None:
            return None

        # 检查依赖：通信 op 不能依赖于 prev_compute
        if prev_op_id in op.deps:
            return None

        # 检查路径冲突
        if op.topology_path_key and op.topology_path_key in resource_slot.path_slots:
            return None

        # 检查 chip 资源冲突（通信和计算使用不同资源，通常可重叠）
        # 这里简化处理：只要不在同一 chip 上有活跃计算就可以重叠
        overlap_chips = set(op.chip_ids) & set(prev_op.chip_ids)
        if not overlap_chips:
            return None  # 不在同一 chip，无需重叠

        # 可以重叠
        return {
            "op_id": op.op_id,
            "overlap_with": prev_op_id,
            "type": "compute_comm_overlap",
            "start": prev_compute["start"],
            "chips": list(overlap_chips),
        }

    def _detect_conflict(
        self, op: DistributedOp, resource_slot: ResourceSlot, step: int
    ) -> ConflictRecord | None:
        """检测资源冲突

        输入:
            - op: 当前 op。
            - resource_slot: 资源槽位状态。
            - step: 当前步骤。
        输出:
            - ConflictRecord 或 None。
        关键步骤:
            - 检查 core/path 资源是否已被占用。
        """
        conflicting_ops: list[str] = []

        # 检查 path 冲突（通信）
        if op.topology_path_key:
            if resource_slot.path_slots.get(op.topology_path_key, 0) > 0:
                # 找出占用该路径的 op
                for active_op in resource_slot.active_ops:
                    conflicting_ops.append(active_op)
                return ConflictRecord(
                    step=step,
                    op_id=op.op_id,
                    conflict_type="path_conflict",
                    conflicting_ops=conflicting_ops,
                    resolved=False,
                    resolution="",
                )

        return None

    def _fix_conflict(
        self, conflict: ConflictRecord, resource_slot: ResourceSlot
    ) -> None:
        """修复资源冲突

        输入:
            - conflict: 冲突记录。
            - resource_slot: 资源槽位状态。
        关键步骤:
            - 通过串行化（delay）修复冲突。
        """
        # 当前简化实现：标记为串行化修复
        conflict.resolved = True
        conflict.resolution = "serial"
        # 清空活跃 op（强制串行）
        resource_slot.active_ops.clear()

    def _update_resource_slot(
        self, op: DistributedOp, resource_slot: ResourceSlot
    ) -> None:
        """更新资源槽位

        输入:
            - op: 当前 op。
            - resource_slot: 资源槽位状态。
        关键步骤:
            - 累加 core_slots/path_slots 占用。
        """
        for cid in op.chip_ids:
            resource_slot.core_slots[cid] = resource_slot.core_slots.get(cid, 0) + 1

        if op.topology_path_key:
            resource_slot.path_slots[op.topology_path_key] = (
                resource_slot.path_slots.get(op.topology_path_key, 0) + 1
            )

        resource_slot.active_ops.add(op.op_id)

    def _build_consumers(self, dist_model: DistributedModel) -> dict[str, list[str]]:
        """构建消费者映射

        输入:
            - dist_model: 分布式模型。
        输出:
            - op_id -> 消费该 op 输出的 op_id 列表。
        """
        consumers: dict[str, list[str]] = {node: [] for node in dist_model.graph_nodes}
        for src, dst in dist_model.graph_edges:
            if src in consumers:
                consumers[src].append(dst)
        return consumers

    def _release_consumed_buffers(
        self,
        current_op_id: str,
        buffer_live: dict[str, dict[str, Any]],
        buffer_plan: dict[str, Any],
        consumers: dict[str, list[str]],
        step: int,
        buffer_current: int,
    ) -> int:
        """释放已消费的 buffer

        输入:
            - current_op_id: 当前 op_id。
            - buffer_live: 活跃 buffer 映射。
            - buffer_plan: buffer 计划。
            - consumers: 消费者映射。
            - step: 当前步骤。
            - buffer_current: 当前 buffer 占用（bytes）。
        输出:
            - 更新后的 buffer_current。
        关键步骤:
            - 检查当前 op 是否消费了某个 buffer，若是最后一个消费者则释放。
        """
        to_release: list[str] = []
        for buf_op_id, buf_info in buffer_live.items():
            op_consumers = consumers.get(buf_op_id, [])
            # 如果当前 op 是该 buffer 的消费者
            if current_op_id in op_consumers:
                # 简化：假设单次消费即释放
                # 实际应追踪所有消费者是否都已调度
                to_release.append(buf_op_id)

        for buf_op_id in to_release:
            buf_info = buffer_live.pop(buf_op_id)
            buffer_plan[buf_op_id] = {
                "bytes": buf_info["bytes"],
                "start": buf_info["start"],
                "end": step,
            }
            buffer_current -= buf_info["bytes"]

        return buffer_current

    def _estimate_output_bytes(self, op: DistributedOp) -> int:
        """估算输出大小

        输入:
            - op.local_shape 中的维度（单位: elements）。
        输出:
            - 估算 bytes（单位: bytes）。
        关键步骤:
            - 按 G×M×N 或 B×S×H 或 B×S×D 维度组合推导输出元素数量。
        """
        out_bytes = op.attrs.get("output_dtype_bytes")
        try:
            dtype_bytes = int(out_bytes) if out_bytes is not None else 2
        except (TypeError, ValueError):
            dtype_bytes = 2

        shape = op.local_shape
        m = int(shape.get("M", 0))
        n = int(shape.get("N", 0))
        g = int(shape.get("G", 0))
        b = int(shape.get("B", 0) or shape.get("batch", 0))
        s = int(shape.get("S", 0) or shape.get("seq_len", 0))
        h = int(shape.get("H", 0) or shape.get("hidden", 0))
        d = int(shape.get("D", 0) or shape.get("dim", 0))

        if g and m and n:
            return int(g * m * n * dtype_bytes)
        if m and n:
            return int(m * n * dtype_bytes)
        if b and s and h:
            return int(b * s * h * dtype_bytes)
        if b and s and d:
            return int(b * s * d * dtype_bytes)
        return 0
