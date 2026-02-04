"""执行计划数据结构."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class OpInstance:
    """调度实例 - 计算节点"""

    instance_id: str
    op_id: str
    kind: Literal["compute"] = "compute"
    chip_ids: list[int] = field(default_factory=list)
    core_ids: list[int] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0


@dataclass
class CommInstance:
    """调度实例 - 通信节点"""

    instance_id: str
    op_id: str
    kind: Literal["comm"] = "comm"
    chip_ids: list[int] = field(default_factory=list)
    path_key: str | None = None
    deps: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0


@dataclass
class ExecPlan:
    """执行计划

    Attributes:
        tile_config: op_id -> TileConfig
        kernel_config: op_id -> KernelConfig
        timeline: 调度后的事件列表
        instances: 展开的 Op/Comm 实例列表
        binding: op_id -> 资源绑定信息
        precedence: 调度后依赖边 (src, dst)
        buffer_plan: buffer 生命周期与峰值
        overlap: 计算/通信重叠信息
        trace_meta: 调试/追踪信息
    """

    tile_config: dict[str, Any] = field(default_factory=dict)
    kernel_config: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    instances: list[OpInstance | CommInstance] = field(default_factory=list)
    binding: dict[str, Any] = field(default_factory=dict)
    precedence: list[tuple[str, str]] = field(default_factory=list)
    buffer_plan: dict[str, Any] = field(default_factory=dict)
    overlap: list[dict[str, Any]] = field(default_factory=list)
    trace_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def engine_schedule(self) -> list[dict[str, Any]]:
        return self.timeline
