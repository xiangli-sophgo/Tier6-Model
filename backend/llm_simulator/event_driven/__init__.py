"""
事件驱动仿真系统

基于离散事件仿真（DES）的 LLM 推理模拟器，
支持精确的计算-通信重叠建模和流水线并行调度。

核心组件：
- Event: 事件基类和具体事件类型
- EventQueue: 基于优先队列的事件调度
- ResourceManager: 计算/网络资源管理
- DependencyGraph: 算子依赖关系图
- EventDrivenSimulator: 事件驱动仿真器主类
"""

from .event import (
    EventType,
    ResourceType,
    BaseEvent,
    ComputeStartEvent,
    ComputeEndEvent,
    CommStartEvent,
    CommEndEvent,
    ChunkedCommStartEvent,
    ChunkedCommEndEvent,
    LayerCompleteEvent,
    StageReadyEvent,
    BatchCompleteEvent,
)

from .event_queue import EventQueue

from .resource import ResourceManager, ResourceState

from .dependency import DependencyGraph, OperatorNode

from .simulator import EventDrivenSimulator, EventDrivenSimConfig

__all__ = [
    # 事件类型
    "EventType",
    "ResourceType",
    "BaseEvent",
    "ComputeStartEvent",
    "ComputeEndEvent",
    "CommStartEvent",
    "CommEndEvent",
    "ChunkedCommStartEvent",
    "ChunkedCommEndEvent",
    "LayerCompleteEvent",
    "StageReadyEvent",
    "BatchCompleteEvent",
    # 事件队列
    "EventQueue",
    # 资源管理
    "ResourceManager",
    "ResourceState",
    # 依赖图
    "DependencyGraph",
    "OperatorNode",
    # 仿真器
    "EventDrivenSimulator",
    "EventDrivenSimConfig",
]
