"""
资源管理模块

管理每个芯片的计算和网络资源，追踪资源占用状态，
记录等待时间（气泡）用于性能分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .event import ResourceType


@dataclass
class ResourceState:
    """单个资源的状态"""

    resource_type: ResourceType
    chip_id: str

    # 资源下次空闲的时间
    idle_at: float = 0.0

    # 当前正在执行的任务（如果有）
    current_task: Optional[str] = None

    # 统计信息
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    task_count: int = 0


@dataclass
class BubbleRecord:
    """气泡记录"""

    chip_id: str
    start: float
    duration: float
    reason: str  # compute_resource_busy, network_resource_busy, comm_sync_wait, pp_bubble


@dataclass
class CommSyncState:
    """集合通信同步状态"""

    comm_type: str
    layer_index: int
    micro_batch: int
    participating_chips: list[str]
    arrival_times: dict[str, float] = field(default_factory=dict)

    def is_ready(self) -> bool:
        """检查是否所有芯片都已到达"""
        return len(self.arrival_times) >= len(self.participating_chips)

    def get_latest_arrival(self) -> float:
        """获取最晚到达时间"""
        if not self.arrival_times:
            return 0.0
        return max(self.arrival_times.values())


class ResourceManager:
    """资源管理器

    管理所有芯片的计算和网络资源，提供：
    - 资源请求和释放
    - 等待时间（气泡）记录
    - 集合通信同步协调
    """

    def __init__(self, chip_ids: list[str]):
        """初始化资源管理器

        Args:
            chip_ids: 所有芯片的ID列表
        """
        self.chip_ids = chip_ids

        # 为每个芯片创建计算和网络资源
        self._resources: dict[tuple[str, ResourceType], ResourceState] = {}
        for chip_id in chip_ids:
            for res_type in [ResourceType.COMPUTE, ResourceType.NETWORK]:
                key = (chip_id, res_type)
                self._resources[key] = ResourceState(
                    resource_type=res_type,
                    chip_id=chip_id,
                )

        # 气泡记录
        self._bubbles: list[BubbleRecord] = []

        # 集合通信同步状态
        # key: (comm_type, layer_index, micro_batch)
        self._comm_sync_states: dict[tuple[str, int, int], CommSyncState] = {}

    def request_resource(
        self,
        chip_id: str,
        resource_type: ResourceType,
        requested_start: float,
        duration: float,
    ) -> tuple[float, float]:
        """请求资源

        Args:
            chip_id: 芯片ID
            resource_type: 资源类型
            requested_start: 请求的开始时间
            duration: 占用时长

        Returns:
            (actual_start, actual_end): 实际的开始和结束时间
        """
        key = (chip_id, resource_type)
        resource = self._resources.get(key)

        if resource is None:
            # 资源不存在，创建一个
            resource = ResourceState(
                resource_type=resource_type,
                chip_id=chip_id,
            )
            self._resources[key] = resource

        # 计算实际开始时间
        actual_start = max(requested_start, resource.idle_at)
        actual_end = actual_start + duration

        # 更新资源状态
        resource.idle_at = actual_end
        resource.total_busy_time += duration
        resource.task_count += 1

        # 记录空闲时间（如果有）
        if actual_start > requested_start:
            idle_duration = actual_start - requested_start
            resource.total_idle_time += idle_duration

        return actual_start, actual_end

    def get_resource_idle_time(
        self,
        chip_id: str,
        resource_type: ResourceType,
    ) -> float:
        """获取资源的空闲时间点

        Args:
            chip_id: 芯片ID
            resource_type: 资源类型

        Returns:
            资源下次空闲的时间
        """
        key = (chip_id, resource_type)
        resource = self._resources.get(key)
        return resource.idle_at if resource else 0.0

    def record_bubble(
        self,
        chip_id: str,
        start: float,
        duration: float,
        reason: str,
    ) -> None:
        """记录气泡（等待时间）

        Args:
            chip_id: 芯片ID
            start: 开始时间
            duration: 持续时间
            reason: 原因
        """
        if duration > 0:
            self._bubbles.append(BubbleRecord(
                chip_id=chip_id,
                start=start,
                duration=duration,
                reason=reason,
            ))

    def get_bubbles(self) -> list[BubbleRecord]:
        """获取所有气泡记录"""
        return self._bubbles.copy()

    def get_total_bubble_time(self, chip_id: Optional[str] = None) -> float:
        """获取总气泡时间

        Args:
            chip_id: 如果指定，只计算该芯片的气泡时间

        Returns:
            总气泡时间
        """
        if chip_id:
            return sum(b.duration for b in self._bubbles if b.chip_id == chip_id)
        return sum(b.duration for b in self._bubbles)

    def get_bubble_breakdown(self) -> dict[str, float]:
        """按原因分类统计气泡时间"""
        breakdown: dict[str, float] = {}
        for bubble in self._bubbles:
            breakdown[bubble.reason] = breakdown.get(bubble.reason, 0.0) + bubble.duration
        return breakdown

    # ========== 集合通信同步 ==========

    def record_comm_arrival(
        self,
        chip_id: str,
        comm_type: str,
        layer_index: int,
        micro_batch: int,
        arrival_time: float,
    ) -> None:
        """记录芯片到达通信同步点

        Args:
            chip_id: 芯片ID
            comm_type: 通信类型
            layer_index: 层索引
            micro_batch: 微批次索引
            arrival_time: 到达时间
        """
        key = (comm_type, layer_index, micro_batch)

        if key not in self._comm_sync_states:
            self._comm_sync_states[key] = CommSyncState(
                comm_type=comm_type,
                layer_index=layer_index,
                micro_batch=micro_batch,
                participating_chips=[],
            )

        self._comm_sync_states[key].arrival_times[chip_id] = arrival_time

    def get_comm_arrival_times(
        self,
        participating_chips: list[str],
        comm_type: str,
        layer_index: int,
        micro_batch: int,
    ) -> dict[str, float]:
        """获取通信同步的到达时间

        Args:
            participating_chips: 参与的芯片列表
            comm_type: 通信类型
            layer_index: 层索引
            micro_batch: 微批次索引

        Returns:
            各芯片的到达时间字典
        """
        key = (comm_type, layer_index, micro_batch)

        if key not in self._comm_sync_states:
            self._comm_sync_states[key] = CommSyncState(
                comm_type=comm_type,
                layer_index=layer_index,
                micro_batch=micro_batch,
                participating_chips=participating_chips,
            )

        return self._comm_sync_states[key].arrival_times.copy()

    def all_chips_ready_for_comm(
        self,
        participating_chips: list[str],
        comm_type: str,
        layer_index: int,
        micro_batch: int,
    ) -> bool:
        """检查所有芯片是否都已准备好进行通信

        Args:
            participating_chips: 参与的芯片列表
            comm_type: 通信类型
            layer_index: 层索引
            micro_batch: 微批次索引

        Returns:
            是否所有芯片都已就绪
        """
        key = (comm_type, layer_index, micro_batch)
        state = self._comm_sync_states.get(key)

        if state is None:
            return False

        return all(chip in state.arrival_times for chip in participating_chips)

    def clear_comm_state(
        self,
        participating_chips: list[str],
        comm_type: str,
        layer_index: int,
        micro_batch: int,
    ) -> None:
        """清除通信同步状态

        Args:
            participating_chips: 参与的芯片列表
            comm_type: 通信类型
            layer_index: 层索引
            micro_batch: 微批次索引
        """
        key = (comm_type, layer_index, micro_batch)
        if key in self._comm_sync_states:
            del self._comm_sync_states[key]

    # ========== 统计信息 ==========

    def get_resource_stats(self, chip_id: str) -> dict:
        """获取芯片的资源统计信息

        Args:
            chip_id: 芯片ID

        Returns:
            资源统计字典
        """
        stats = {}
        for res_type in [ResourceType.COMPUTE, ResourceType.NETWORK]:
            key = (chip_id, res_type)
            resource = self._resources.get(key)
            if resource:
                stats[res_type.name.lower()] = {
                    "total_busy_time": resource.total_busy_time,
                    "total_idle_time": resource.total_idle_time,
                    "task_count": resource.task_count,
                    "utilization": (
                        resource.total_busy_time /
                        (resource.total_busy_time + resource.total_idle_time)
                        if (resource.total_busy_time + resource.total_idle_time) > 0
                        else 0.0
                    ),
                }
        return stats

    def get_all_stats(self) -> dict:
        """获取所有资源的统计信息"""
        return {
            chip_id: self.get_resource_stats(chip_id)
            for chip_id in self.chip_ids
        }

    def reset(self) -> None:
        """重置所有状态"""
        for resource in self._resources.values():
            resource.idle_at = 0.0
            resource.current_task = None
            resource.total_busy_time = 0.0
            resource.total_idle_time = 0.0
            resource.task_count = 0

        self._bubbles.clear()
        self._comm_sync_states.clear()
