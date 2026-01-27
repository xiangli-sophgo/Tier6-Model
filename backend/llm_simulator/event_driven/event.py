"""
事件定义模块

定义事件驱动仿真系统中的所有事件类型。
借鉴 Vidur 的设计，事件按照 (timestamp, event_type, event_id) 排序。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .resource import ResourceManager
    from ..core.gantt import GanttChartBuilder


class EventType(IntEnum):
    """事件类型枚举

    数值越小优先级越高（同一时刻先处理）。
    END 事件优先于 START 事件，确保资源正确释放。
    """
    # 结束事件（高优先级）
    COMPUTE_END = 1
    COMM_END = 2
    MEMORY_END = 3

    # 完成事件（中优先级）
    LAYER_COMPLETE = 10
    STAGE_READY = 11
    BATCH_COMPLETE = 12

    # 开始事件（低优先级）
    COMPUTE_START = 20
    COMM_START = 21
    MEMORY_START = 22

    # 调度事件（最低优先级，确保其他事件先处理）
    SCHEDULE = 30


class ResourceType(IntEnum):
    """资源类型"""
    COMPUTE = 1      # 计算资源（Tensor Core）
    NETWORK = 2      # 网络资源（NVLink/IB）
    MEMORY_BUS = 3   # 内存总线


@dataclass
class BaseEvent(ABC):
    """事件基类

    所有事件必须实现 handle() 方法，返回新产生的事件列表。
    事件按照 (timestamp, event_type, event_id) 排序。
    """

    # 类级别的事件ID计数器，保证唯一性和确定性
    _next_id: int = 0

    # 必需字段（子类会覆盖 event_type 的默认值）
    timestamp: float = 0.0  # 事件发生时间 (us)
    chip_id: str = ""  # 发生事件的芯片

    # 元数据（带默认值）
    layer_index: int = -1
    token_index: int = -1
    micro_batch: int = 0
    pp_stage: int = 0

    # 自动分配的唯一ID
    event_id: int = field(default=-1, init=False)

    # 事件类型（子类会设置）
    event_type: EventType = field(default=EventType.COMPUTE_START, init=False)

    def __post_init__(self):
        """分配唯一的事件ID"""
        self.event_id = BaseEvent._next_id
        BaseEvent._next_id += 1

    def __lt__(self, other: BaseEvent) -> bool:
        """事件排序：时间 -> 类型 -> ID"""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        if self.event_type != other.event_type:
            return self.event_type < other.event_type
        return self.event_id < other.event_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEvent):
            return False
        return (
            self.timestamp == other.timestamp
            and self.event_type == other.event_type
            and self.event_id == other.event_id
        )

    def __hash__(self) -> int:
        return hash(self.event_id)

    @abstractmethod
    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理事件，返回新产生的事件列表

        Args:
            resource_manager: 资源管理器
            gantt_builder: Gantt 图构建器
            context: 仿真上下文（包含评估器、依赖图等）

        Returns:
            新产生的事件列表
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.name,
            "chip_id": self.chip_id,
            "layer_index": self.layer_index,
            "token_index": self.token_index,
            "micro_batch": self.micro_batch,
            "pp_stage": self.pp_stage,
        }


@dataclass
class ComputeStartEvent(BaseEvent):
    """计算开始事件"""

    event_type: EventType = field(default=EventType.COMPUTE_START, init=False)

    # 算子信息
    operator_name: str = ""
    operator_type: str = ""  # ComputeOpType
    duration_us: float = 0.0  # 预计执行时间

    # 性能数据（用于 Gantt 图）
    flops: float = 0.0
    dram_traffic_bytes: float = 0.0
    compute_time_us: float = 0.0
    memory_time_us: float = 0.0

    # GEMM 优化结果
    best_tile: Optional[dict] = None
    best_partition: Optional[dict] = None
    gemm_shape: Optional[dict] = None

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理计算开始事件"""
        # 1. 请求计算资源
        actual_start, actual_end = resource_manager.request_resource(
            chip_id=self.chip_id,
            resource_type=ResourceType.COMPUTE,
            requested_start=self.timestamp,
            duration=self.duration_us,
        )

        # 2. 记录等待时间（气泡）
        if actual_start > self.timestamp:
            bubble_duration = actual_start - self.timestamp
            resource_manager.record_bubble(
                chip_id=self.chip_id,
                start=self.timestamp,
                duration=bubble_duration,
                reason="compute_resource_busy",
            )

        # 3. 添加到 Gantt 图
        from ..config import GanttTaskType, InferencePhase

        # 映射算子类型到 GanttTaskType
        task_type = self._map_operator_to_gantt_type()
        phase = InferencePhase.PREFILL if self.token_index < 0 else InferencePhase.DECODE

        gantt_builder.add_compute_task(
            task_type=task_type,
            start=actual_start,
            duration=actual_end - actual_start,
            phase=phase,
            chip_id=self.chip_id,
            pp_stage=self.pp_stage,
            layer_index=self.layer_index,
            token_index=self.token_index if self.token_index >= 0 else None,
            name=self.operator_name,
            flops=self.flops,
            dram_traffic_bytes=self.dram_traffic_bytes,
            compute_time_us=self.compute_time_us,
            memory_time_us=self.memory_time_us,
            best_tile=self.best_tile,
            best_partition=self.best_partition,
            gemm_shape=self.gemm_shape,
        )

        # 4. 创建计算结束事件
        end_event = ComputeEndEvent(
            timestamp=actual_end,
            chip_id=self.chip_id,
            layer_index=self.layer_index,
            token_index=self.token_index,
            micro_batch=self.micro_batch,
            pp_stage=self.pp_stage,
            operator_name=self.operator_name,
            operator_type=self.operator_type,
        )

        return [end_event]

    def _map_operator_to_gantt_type(self):
        """映射算子类型到 Gantt 任务类型"""
        from ..config import GanttTaskType

        mapping = {
            "attention_qkv": GanttTaskType.ATTENTION_QKV,
            "attention_score": GanttTaskType.ATTENTION_SCORE,
            "attention_output": GanttTaskType.ATTENTION_OUTPUT,
            "ffn_gate": GanttTaskType.FFN_GATE,
            "ffn_up": GanttTaskType.FFN_UP,
            "ffn_down": GanttTaskType.FFN_DOWN,
            "layernorm": GanttTaskType.LAYERNORM,
            "rmsnorm": GanttTaskType.LAYERNORM,
            "embedding": GanttTaskType.EMBEDDING,
            "lm_head": GanttTaskType.LM_HEAD,
            # MLA 相关
            "mla_q_proj": GanttTaskType.MM_Q_LORA_A,
            "mla_kv_proj": GanttTaskType.MM_KV_LORA_A,
            # MoE 相关
            "moe_gate": GanttTaskType.MOE_GATE,
            "moe_expert": GanttTaskType.MOE_EXPERT,
            "moe_shared_expert": GanttTaskType.MOE_SHARED_EXPERT,
        }

        return mapping.get(self.operator_type, GanttTaskType.COMPUTE)


@dataclass
class ComputeEndEvent(BaseEvent):
    """计算结束事件"""

    event_type: EventType = field(default=EventType.COMPUTE_END, init=False)

    operator_name: str = ""
    operator_type: str = ""

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理计算结束事件"""
        new_events = []

        # 1. 释放计算资源（资源管理器自动处理）

        # 2. 检查依赖图，触发后续事件
        dependency_graph = context.get("dependency_graph")
        if dependency_graph:
            # 标记当前算子完成
            op_key = (self.chip_id, self.layer_index, self.operator_name, self.micro_batch)
            dependency_graph.mark_completed(op_key)

            # 获取可以开始的后续算子
            ready_ops = dependency_graph.get_ready_successors(op_key)

            for next_op in ready_ops:
                # 根据算子类型创建对应的开始事件
                if next_op.is_compute:
                    event = ComputeStartEvent(
                        timestamp=self.timestamp,
                        chip_id=next_op.chip_id,
                        layer_index=next_op.layer_index,
                        token_index=self.token_index,
                        micro_batch=self.micro_batch,
                        pp_stage=next_op.pp_stage,
                        operator_name=next_op.name,
                        operator_type=next_op.op_type,
                        duration_us=next_op.duration_us,
                        flops=next_op.flops,
                        dram_traffic_bytes=next_op.dram_traffic_bytes,
                        compute_time_us=next_op.compute_time_us,
                        memory_time_us=next_op.memory_time_us,
                        best_tile=next_op.best_tile,
                        best_partition=next_op.best_partition,
                        gemm_shape=next_op.gemm_shape,
                    )
                else:
                    event = CommStartEvent(
                        timestamp=self.timestamp,
                        chip_id=next_op.chip_id,
                        layer_index=next_op.layer_index,
                        token_index=self.token_index,
                        micro_batch=self.micro_batch,
                        pp_stage=next_op.pp_stage,
                        comm_type=next_op.op_type,
                        comm_size_bytes=next_op.comm_size_bytes,
                        duration_us=next_op.duration_us,
                        participating_chips=next_op.participating_chips,
                    )
                new_events.append(event)

        # 3. 检查是否是层的最后一个算子
        if self._is_last_op_in_layer(context):
            layer_event = LayerCompleteEvent(
                timestamp=self.timestamp,
                chip_id=self.chip_id,
                layer_index=self.layer_index,
                token_index=self.token_index,
                micro_batch=self.micro_batch,
                pp_stage=self.pp_stage,
            )
            new_events.append(layer_event)

        return new_events

    def _is_last_op_in_layer(self, context: dict[str, Any]) -> bool:
        """检查是否是层的最后一个算子"""
        dependency_graph = context.get("dependency_graph")
        if not dependency_graph:
            return False

        op_key = (self.chip_id, self.layer_index, self.operator_name, self.micro_batch)
        return dependency_graph.is_layer_complete(self.layer_index, self.micro_batch)


@dataclass
class CommStartEvent(BaseEvent):
    """通信开始事件"""

    event_type: EventType = field(default=EventType.COMM_START, init=False)

    comm_type: str = ""  # allreduce, allgather, p2p, etc.
    comm_size_bytes: float = 0.0
    duration_us: float = 0.0
    participating_chips: list[str] = field(default_factory=list)

    # 通信算法信息
    comm_algorithm: str = "ring"
    comm_group_size: int = 1

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理通信开始事件"""
        new_events = []
        arrival_times: dict[str, float] = {}

        # 1. 检查所有参与芯片是否就绪
        if self.participating_chips:
            # 集合通信需要等待所有参与者
            arrival_times = resource_manager.get_comm_arrival_times(
                self.participating_chips,
                self.comm_type,
                self.layer_index,
                self.micro_batch,
            )

            # 记录当前芯片到达
            resource_manager.record_comm_arrival(
                chip_id=self.chip_id,
                comm_type=self.comm_type,
                layer_index=self.layer_index,
                micro_batch=self.micro_batch,
                arrival_time=self.timestamp,
            )

            # 检查是否所有芯片都到达
            if not resource_manager.all_chips_ready_for_comm(
                self.participating_chips,
                self.comm_type,
                self.layer_index,
                self.micro_batch,
            ):
                # 还有芯片没准备好，等待
                return []

            # 所有芯片就绪，使用最晚到达时间作为实际开始时间
            actual_start = max(arrival_times.values()) if arrival_times else self.timestamp
        else:
            # P2P 通信
            actual_start = self.timestamp

        # 2. 为所有参与芯片请求网络资源
        chips_to_use = self.participating_chips if self.participating_chips else [self.chip_id]
        actual_end = actual_start + self.duration_us

        for chip_id in chips_to_use:
            _, chip_end = resource_manager.request_resource(
                chip_id=chip_id,
                resource_type=ResourceType.NETWORK,
                requested_start=actual_start,
                duration=self.duration_us,
            )
            actual_end = max(actual_end, chip_end)

        # 3. 记录等待时间（仅对集合通信）
        if arrival_times:
            for chip_id in chips_to_use:
                chip_arrival = arrival_times.get(chip_id, actual_start)
                if chip_arrival < actual_start:
                    wait_time = actual_start - chip_arrival
                    resource_manager.record_bubble(
                        chip_id=chip_id,
                        start=chip_arrival,
                        duration=wait_time,
                        reason="comm_sync_wait",
                    )

        # 4. 添加到 Gantt 图
        from ..config import GanttTaskType, InferencePhase

        task_type = self._map_comm_to_gantt_type()
        phase = InferencePhase.PREFILL if self.token_index < 0 else InferencePhase.DECODE

        for chip_id in chips_to_use:
            gantt_builder.add_comm_task(
                task_type=task_type,
                start=actual_start,
                duration=actual_end - actual_start,
                phase=phase,
                chip_id=chip_id,
                pp_stage=self.pp_stage,
                layer_index=self.layer_index,
                token_index=self.token_index if self.token_index >= 0 else None,
                name=f"{self.comm_type}_{self.layer_index}",
                comm_size_bytes=self.comm_size_bytes,
                comm_algorithm=self.comm_algorithm,
                comm_group_size=self.comm_group_size,
            )

        # 5. 创建通信结束事件（为主芯片创建）
        end_event = CommEndEvent(
            timestamp=actual_end,
            chip_id=self.chip_id,
            layer_index=self.layer_index,
            token_index=self.token_index,
            micro_batch=self.micro_batch,
            pp_stage=self.pp_stage,
            comm_type=self.comm_type,
            participating_chips=self.participating_chips,
        )
        new_events.append(end_event)

        return new_events

    def _map_comm_to_gantt_type(self):
        """映射通信类型到 Gantt 任务类型"""
        from ..config import GanttTaskType

        mapping = {
            "allreduce": GanttTaskType.TP_COMM,
            "allgather": GanttTaskType.SP_ALLGATHER,
            "reduce_scatter": GanttTaskType.SP_REDUCE_SCATTER,
            "p2p_send": GanttTaskType.PP_COMM,
            "p2p_recv": GanttTaskType.PP_COMM,
            "all2all": GanttTaskType.EP_COMM,
            "ep_dispatch": GanttTaskType.EP_DISPATCH,
            "ep_combine": GanttTaskType.EP_COMBINE,
        }

        return mapping.get(self.comm_type, GanttTaskType.TP_COMM)


@dataclass
class CommEndEvent(BaseEvent):
    """通信结束事件"""

    event_type: EventType = field(default=EventType.COMM_END, init=False)

    comm_type: str = ""
    participating_chips: list[str] = field(default_factory=list)

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理通信结束事件"""
        new_events = []

        # 1. 清除通信同步状态
        resource_manager.clear_comm_state(
            self.participating_chips if self.participating_chips else [self.chip_id],
            self.comm_type,
            self.layer_index,
            self.micro_batch,
        )

        # 2. 检查依赖图，触发后续事件
        dependency_graph = context.get("dependency_graph")
        if dependency_graph:
            op_key = (self.chip_id, self.layer_index, self.comm_type, self.micro_batch)
            dependency_graph.mark_completed(op_key)

            ready_ops = dependency_graph.get_ready_successors(op_key)

            for next_op in ready_ops:
                if next_op.is_compute:
                    event = ComputeStartEvent(
                        timestamp=self.timestamp,
                        chip_id=next_op.chip_id,
                        layer_index=next_op.layer_index,
                        token_index=self.token_index,
                        micro_batch=self.micro_batch,
                        pp_stage=next_op.pp_stage,
                        operator_name=next_op.name,
                        operator_type=next_op.op_type,
                        duration_us=next_op.duration_us,
                        flops=next_op.flops,
                        dram_traffic_bytes=next_op.dram_traffic_bytes,
                        compute_time_us=next_op.compute_time_us,
                        memory_time_us=next_op.memory_time_us,
                    )
                else:
                    event = CommStartEvent(
                        timestamp=self.timestamp,
                        chip_id=next_op.chip_id,
                        layer_index=next_op.layer_index,
                        token_index=self.token_index,
                        micro_batch=self.micro_batch,
                        pp_stage=next_op.pp_stage,
                        comm_type=next_op.op_type,
                        comm_size_bytes=next_op.comm_size_bytes,
                        duration_us=next_op.duration_us,
                        participating_chips=next_op.participating_chips,
                    )
                new_events.append(event)

        return new_events


@dataclass
class LayerCompleteEvent(BaseEvent):
    """层完成事件"""

    event_type: EventType = field(default=EventType.LAYER_COMPLETE, init=False)

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理层完成事件"""
        new_events = []

        # 获取配置
        num_layers = context.get("num_layers", 1)
        pp_degree = context.get("pp_degree", 1)
        layers_per_stage = num_layers // pp_degree

        # 检查是否是当前 stage 的最后一层
        stage_last_layer = (self.pp_stage + 1) * layers_per_stage - 1

        if self.layer_index == stage_last_layer:
            # 当前 stage 完成，触发 PP 通信或 batch 完成
            if self.pp_stage < pp_degree - 1:
                # 不是最后一个 stage，发送到下一个 stage
                next_stage = self.pp_stage + 1
                next_stage_chips = context.get("stage_chips", {}).get(next_stage, [])

                if next_stage_chips:
                    # 触发 StageReady 事件
                    stage_event = StageReadyEvent(
                        timestamp=self.timestamp,
                        chip_id=next_stage_chips[0],  # 下一阶段的第一个芯片
                        layer_index=self.layer_index + 1,
                        token_index=self.token_index,
                        micro_batch=self.micro_batch,
                        pp_stage=next_stage,
                        source_stage=self.pp_stage,
                    )
                    new_events.append(stage_event)
            else:
                # 最后一个 stage 完成，触发 batch 完成
                batch_event = BatchCompleteEvent(
                    timestamp=self.timestamp,
                    chip_id=self.chip_id,
                    layer_index=self.layer_index,
                    token_index=self.token_index,
                    micro_batch=self.micro_batch,
                    pp_stage=self.pp_stage,
                )
                new_events.append(batch_event)

        return new_events


@dataclass
class StageReadyEvent(BaseEvent):
    """Pipeline Stage 就绪事件

    表示上一个 stage 已完成，当前 stage 可以开始处理。
    """

    event_type: EventType = field(default=EventType.STAGE_READY, init=False)

    source_stage: int = 0  # 来源 stage

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理 Stage 就绪事件"""
        new_events = []

        # 1. 计算 PP P2P 通信延迟
        pp_comm_latency = context.get("pp_comm_latency_us", 0.0)

        # 2. 添加 P2P 通信到 Gantt 图
        from ..config import GanttTaskType, InferencePhase

        phase = InferencePhase.PREFILL if self.token_index < 0 else InferencePhase.DECODE

        gantt_builder.add_comm_task(
            task_type=GanttTaskType.PP_COMM,
            start=self.timestamp,
            duration=pp_comm_latency,
            phase=phase,
            chip_id=self.chip_id,
            pp_stage=self.pp_stage,
            layer_index=self.layer_index,
            name=f"pp_recv_from_stage_{self.source_stage}",
        )

        # 3. 触发新 stage 的第一层计算
        dependency_graph = context.get("dependency_graph")
        if dependency_graph:
            first_op = dependency_graph.get_first_op_in_layer(
                self.layer_index,
                self.chip_id,
                self.micro_batch,
            )

            if first_op:
                event = ComputeStartEvent(
                    timestamp=self.timestamp + pp_comm_latency,
                    chip_id=self.chip_id,
                    layer_index=self.layer_index,
                    token_index=self.token_index,
                    micro_batch=self.micro_batch,
                    pp_stage=self.pp_stage,
                    operator_name=first_op.name,
                    operator_type=first_op.op_type,
                    duration_us=first_op.duration_us,
                    flops=first_op.flops,
                    dram_traffic_bytes=first_op.dram_traffic_bytes,
                    compute_time_us=first_op.compute_time_us,
                    memory_time_us=first_op.memory_time_us,
                )
                new_events.append(event)

        return new_events


@dataclass
class BatchCompleteEvent(BaseEvent):
    """Batch 完成事件"""

    event_type: EventType = field(default=EventType.BATCH_COMPLETE, init=False)

    def handle(
        self,
        resource_manager: ResourceManager,
        gantt_builder: GanttChartBuilder,
        context: dict[str, Any],
    ) -> list[BaseEvent]:
        """处理 Batch 完成事件"""
        # 记录 batch 完成时间
        context.setdefault("batch_end_times", {})[self.micro_batch] = self.timestamp

        # 检查是否所有 micro-batch 都完成
        num_micro_batches = context.get("num_micro_batches", 1)
        batch_end_times = context.get("batch_end_times", {})

        if len(batch_end_times) >= num_micro_batches:
            # 所有 micro-batch 完成
            context["simulation_complete"] = True
            context["total_time"] = max(batch_end_times.values())

        return []


def reset_event_counter():
    """重置事件ID计数器（用于测试）"""
    BaseEvent._next_id = 0
