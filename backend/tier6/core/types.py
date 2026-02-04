"""Tier6 核心类型定义

定义全局使用的基础类型、枚举和类型别名。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType, TypeAlias

# ============================================================================
# 类型别名
# ============================================================================

#: 时延（纳秒）
Latency = NewType("Latency", float)

#: 内存大小（字节）
MemorySize = NewType("MemorySize", int)

#: 带宽（GB/s）
Bandwidth = NewType("Bandwidth", float)

#: 算力（FLOPS）
FLOPs = NewType("FLOPs", float)

#: 算力（TFLOPS）
TFLOPS = NewType("TFLOPS", float)

#: 利用率（0.0 - 1.0）
Utilization: TypeAlias = float


# ============================================================================
# 数据类型枚举
# ============================================================================


class DataType(Enum):
    """数据类型枚举

    支持的计算数据类型及其属性。
    """

    INT8 = "int8"
    INT4 = "int4"
    UINT8 = "uint8"
    UINT4 = "uint4"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    TF32 = "tf32"
    FP64 = "fp64"

    @property
    def bits(self) -> int:
        """数据类型位宽"""
        bits_map = {
            DataType.INT8: 8,
            DataType.INT4: 4,
            DataType.UINT8: 8,
            DataType.UINT4: 4,
            DataType.INT16: 16,
            DataType.UINT16: 16,
            DataType.INT32: 32,
            DataType.UINT32: 32,
            DataType.FP8_E4M3: 8,
            DataType.FP8_E5M2: 8,
            DataType.FP16: 16,
            DataType.BF16: 16,
            DataType.FP32: 32,
            DataType.TF32: 19,  # 实际存储 32，有效位 19
            DataType.FP64: 64,
        }
        return bits_map[self]

    @property
    def bytes(self) -> int:
        """数据类型字节数"""
        # TF32 实际存储为 32 位
        if self == DataType.TF32:
            return 4
        return (self.bits + 7) // 8

    @property
    def is_floating(self) -> bool:
        """是否为浮点类型"""
        return self in {
            DataType.FP8_E4M3,
            DataType.FP8_E5M2,
            DataType.FP16,
            DataType.BF16,
            DataType.FP32,
            DataType.TF32,
            DataType.FP64,
        }

    @classmethod
    def from_string(cls, s: str) -> "DataType":
        """从字符串解析数据类型"""
        s_lower = s.lower().replace("-", "_")
        if s_lower in {"fp8"}:
            return cls.FP8_E4M3
        if s_lower in {"e4m3"}:
            return cls.FP8_E4M3
        if s_lower in {"e5m2"}:
            return cls.FP8_E5M2
        for dtype in cls:
            if dtype.value == s_lower or dtype.name.lower() == s_lower:
                return dtype
        raise ValueError(f"Unknown data type: {s}")


# ============================================================================
# 并行模式枚举
# ============================================================================


class ParallelMode(Enum):
    """并行模式"""

    TENSOR_PARALLEL = auto()  # TP: 张量并行
    DATA_PARALLEL = auto()  # DP: 数据并行
    PIPELINE_PARALLEL = auto()  # PP: 流水线并行
    EXPERT_PARALLEL = auto()  # EP: 专家并行 (MoE)
    SEQUENCE_PARALLEL = auto()  # SP: 序列并行


class BottleneckType(Enum):
    """性能瓶颈类型"""

    COMPUTE_BOUND = "compute"
    MEMORY_BOUND = "memory"
    COMMUNICATION_BOUND = "comm"
    BALANCED = "balanced"


class EngineType(Enum):
    """评估引擎类型"""

    ANALYTICAL = "analytical"  # 公式估算
    DES = "des"  # 离散事件仿真
    TRACE = "trace"  # Trace 回放


# ============================================================================
# 性能指标数据类
# ============================================================================


@dataclass(frozen=True)
class LatencyBreakdown:
    """时延分解"""

    compute_ns: float = 0.0
    memory_ns: float = 0.0
    communication_ns: float = 0.0
    sync_ns: float = 0.0

    @property
    def total_ns(self) -> float:
        """总时延（纳秒）"""
        return self.compute_ns + self.memory_ns + self.communication_ns + self.sync_ns

    @property
    def total_ms(self) -> float:
        """总时延（毫秒）"""
        return self.total_ns / 1e6

    @property
    def bottleneck(self) -> BottleneckType:
        """识别瓶颈类型"""
        max_time = max(self.compute_ns, self.memory_ns, self.communication_ns)
        if max_time == 0:
            return BottleneckType.BALANCED

        # 允许 10% 的误差范围判断平衡
        threshold = max_time * 0.9
        is_compute = self.compute_ns >= threshold
        is_memory = self.memory_ns >= threshold
        is_comm = self.communication_ns >= threshold

        if is_compute and not is_memory and not is_comm:
            return BottleneckType.COMPUTE_BOUND
        elif is_memory and not is_compute and not is_comm:
            return BottleneckType.MEMORY_BOUND
        elif is_comm and not is_compute and not is_memory:
            return BottleneckType.COMMUNICATION_BOUND
        else:
            return BottleneckType.BALANCED


@dataclass
class UtilizationMetrics:
    """利用率指标"""

    cube_util: float = 0.0  # Cube 单元利用率
    vector_util: float = 0.0  # Vector 单元利用率
    dram_bw_util: float = 0.0  # DRAM 带宽利用率
    l2m_bw_util: float = 0.0  # L2M 带宽利用率
    c2c_bw_util: float = 0.0  # C2C 带宽利用率

    @property
    def compute_util(self) -> float:
        """综合计算利用率"""
        return max(self.cube_util, self.vector_util)


@dataclass
class ThroughputMetrics:
    """吞吐量指标"""

    tokens_per_second: float = 0.0  # TPS
    first_token_latency_ms: float = 0.0  # TTFT (毫秒)
    time_per_output_token_ms: float = 0.0  # TPOT (毫秒)
    model_flops_utilization: float = 0.0  # MFU
    batch_size: int = 1

    @property
    def tps(self) -> float:
        """TPS 别名"""
        return self.tokens_per_second

    @property
    def ttft(self) -> float:
        """TTFT 别名（毫秒）"""
        return self.first_token_latency_ms

    @property
    def tpot(self) -> float:
        """TPOT 别名（毫秒）"""
        return self.time_per_output_token_ms

    @property
    def mfu(self) -> float:
        """MFU 别名"""
        return self.model_flops_utilization


# ============================================================================
# 配置相关类型
# ============================================================================


@dataclass
class ParallelismConfig:
    """并行配置"""

    tp: int = 1  # Tensor Parallel
    dp: int = 1  # Data Parallel
    pp: int = 1  # Pipeline Parallel
    ep: int = 1  # Expert Parallel (for MoE)
    sp: bool = False  # Sequence Parallel

    @property
    def world_size(self) -> int:
        """总并行度"""
        return self.tp * self.dp * self.pp

    def validate(self) -> None:
        """校验并行配置"""
        if self.tp < 1 or self.dp < 1 or self.pp < 1 or self.ep < 1:
            raise ValueError("All parallel dimensions must be >= 1")


@dataclass
class BatchConfig:
    """批次配置"""

    batch_size: int = 1
    prompt_length: int = 2048  # prefill 序列长度
    output_length: int = 512  # decode 输出长度
    kv_seq_len: int | None = None  # KV cache 序列长度，None 表示等于 prompt_length

    @property
    def effective_kv_seq_len(self) -> int:
        """有效 KV 序列长度"""
        return self.kv_seq_len if self.kv_seq_len is not None else self.prompt_length

    @property
    def total_seq_len(self) -> int:
        """总序列长度"""
        return self.prompt_length + self.output_length


# ============================================================================
# 推理阶段枚举
# ============================================================================


class InferencePhase(Enum):
    """推理阶段"""

    PREFILL = "prefill"  # 预填充阶段
    DECODE = "decode"  # 解码阶段


# ============================================================================
# 通信算法枚举
# ============================================================================


class AllReduceAlgorithm(Enum):
    """AllReduce 算法"""

    RING = "ring"
    DOUBLE_BINARY_TREE = "double_binary_tree"
    HALVING_DOUBLING = "halving_doubling"
    REDUCE_BROADCAST = "reduce_broadcast"


class AllToAllAlgorithm(Enum):
    """AllToAll 算法"""

    PAIRWISE = "pairwise"
    RING = "ring"
    BRUCK = "bruck"
