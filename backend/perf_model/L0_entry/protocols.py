"""Tier6 核心协议接口

定义各层之间的接口契约，使用 Protocol 实现结构化类型。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from perf_model.L2_arch.protocols import ChipSpec


# ============================================================================
# L0: Entry & Orchestration Protocols
# ============================================================================


@runtime_checkable
class Configurable(Protocol):
    """可配置对象接口

    支持从配置文件或字典加载的对象。
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Configurable":
        """从字典创建实例"""
        ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Configurable":
        """从 YAML 文件创建实例"""
        ...

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        ...


# ============================================================================
# L1: Workload Representation Protocols
# ============================================================================


@runtime_checkable
class WorkloadIR(Protocol):
    """工作负载统一 IR 接口

    所有模型/算子图都需要实现此接口，提供统一的性能分析入口。
    """

    @property
    def name(self) -> str:
        """工作负载名称"""
        ...

    def get_ops_breakdown(self) -> "OpsBreakdown":
        """获取算力分解"""
        ...

    def get_memory_footprint(self) -> "MemoryFootprint":
        """获取内存占用"""
        ...

    def get_communication_pattern(self) -> "CommPattern":
        """获取通信模式"""
        ...


class OpsBreakdown(Protocol):
    """算力分解接口"""

    @property
    def total_flops(self) -> float:
        """总 FLOPs"""
        ...

    @property
    def cube_flops(self) -> float:
        """Cube 单元 FLOPs"""
        ...

    @property
    def vector_flops(self) -> float:
        """Vector 单元 FLOPs"""
        ...


class MemoryFootprint(Protocol):
    """内存占用接口"""

    @property
    def weight_bytes(self) -> int:
        """权重大小（字节）"""
        ...

    @property
    def activation_bytes(self) -> int:
        """激活大小（字节）"""
        ...

    @property
    def kv_cache_bytes(self) -> int:
        """KV Cache 大小（字节）"""
        ...

    @property
    def total_bytes(self) -> int:
        """总内存占用"""
        ...


class CommPattern(Protocol):
    """通信模式接口"""

    @property
    def allreduce_size(self) -> int:
        """AllReduce 数据量（字节）"""
        ...

    @property
    def alltoall_size(self) -> int:
        """AllToAll 数据量（字节）"""
        ...

    @property
    def p2p_size(self) -> int:
        """点对点通信数据量（字节）"""
        ...


# ============================================================================
# L3: Mapping & Scheduling Protocols
# ============================================================================


class ExecPlan(Protocol):
    """执行计划接口"""

    @property
    def tile_config(self) -> dict[str, Any]:
        """分块配置"""
        ...

    @property
    def engine_schedule(self) -> list[dict[str, Any]]:
        """引擎调度序列"""
        ...


class EventTrace(Protocol):
    """事件追踪接口"""

    @property
    def events(self) -> list["Event"]:
        """事件列表"""
        ...

    @property
    def total_cycles(self) -> int:
        """总周期数"""
        ...


class Event(Protocol):
    """单个事件接口"""

    @property
    def timestamp(self) -> int:
        """时间戳（周期）"""
        ...

    @property
    def engine(self) -> str:
        """执行引擎"""
        ...

    @property
    def duration(self) -> int:
        """持续时间（周期）"""
        ...


# ============================================================================
# L4: Evaluation Engines Protocols
# ============================================================================


@runtime_checkable
class Evaluable(Protocol):
    """可评估对象接口"""

    def evaluate(
        self,
        workload: WorkloadIR,
        arch: "ChipSpec",
        plan: ExecPlan | None = None,
    ) -> "EngineResult":
        """执行评估"""
        ...


class EngineResult(Protocol):
    """评估结果接口"""

    @property
    def total_time_ns(self) -> float:
        """总时延 (ns)"""
        ...

    @property
    def compute_time_ns(self) -> float:
        """计算时延 (ns)"""
        ...

    @property
    def memory_time_ns(self) -> float:
        """访存时延 (ns)"""
        ...

    @property
    def communication_time_ns(self) -> float:
        """通信时延 (ns)"""
        ...

    @property
    def bottleneck(self) -> str:
        """性能瓶颈 (compute/memory/comm)"""
        ...


# ============================================================================
# L5: Metrics & Output Protocols
# ============================================================================


@runtime_checkable
class Exportable(Protocol):
    """可导出对象接口"""

    def export(self, result: Any, path: str | Path) -> None:
        """导出结果到文件"""
        ...

    @property
    def format(self) -> str:
        """导出格式名称"""
        ...


class MetricsCollector(Protocol):
    """指标收集器接口"""

    def collect(self, result: EngineResult) -> dict[str, Any]:
        """收集并格式化指标"""
        ...


class Calibrator(Protocol):
    """校准器接口"""

    def calibrate(
        self,
        predicted: dict[str, float],
        measured: dict[str, float],
    ) -> dict[str, float]:
        """校准预测值"""
        ...

    def get_error(
        self,
        predicted: dict[str, float],
        measured: dict[str, float],
    ) -> dict[str, float]:
        """计算误差"""
        ...
