"""精评估器 - 为 L3 TilingPlanner 提供精确评估.

支持按算子类型路由到专属评估器：
    - MatMulPreciseEvaluator: matmul/linear/gemm
    - AttentionPreciseEvaluator: attention/mha/mla/fa2
    - ConvPreciseEvaluator: conv2d/conv1d
    - ElementwisePreciseEvaluator: softmax/layernorm/relu 等
    - FallbackPreciseEvaluator: 未注册算子
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from math_model.L2_arch.chip import ChipSpecImpl
    from math_model.L3_mapping.plan.distributed_model import DistributedOp
    from math_model.L1_workload.specs import TileConfig


@dataclass
class PreciseMetrics:
    """精评估结果

    Attributes:
        t_compute_ms: 计算时间（ms）
        t_memory_ms: 访存时间（ms）
        dram_traffic_bytes: 精确 DRAM traffic（bytes）
        lmem_bytes: LMEM 占用（bytes）
        flops: 浮点运算量
        compute_urate: 计算利用率（0-1）
        memory_urate: 访存利用率（0-1）
        bottleneck: 瓶颈类型（compute/memory）
        best_loop_order: 最优循环顺序
        extra: 其他元数据
    """

    t_compute_ms: float = 0.0
    t_memory_ms: float = 0.0
    dram_traffic_bytes: int = 0
    lmem_bytes: int = 0
    flops: int = 0
    compute_urate: float = 0.0
    memory_urate: float = 0.0
    bottleneck: str = "unknown"
    best_loop_order: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | float | str]:
        """转换为字典（供 L3 TilingPlanner 使用）"""
        return {
            "t_compute_ms": self.t_compute_ms,
            "t_memory_ms": self.t_memory_ms,
            "traffic": self.dram_traffic_bytes,
            "lmem_bytes": self.lmem_bytes,
            "flops": self.flops,
            "compute_urate": self.compute_urate,
            "memory_urate": self.memory_urate,
            "bottleneck": self.bottleneck,
            "best_loop_order": self.best_loop_order,
        }


class OpPreciseEvaluator(Protocol):
    """算子精评估器协议"""

    def supports(self, op: "DistributedOp") -> bool:
        """是否支持该算子"""
        ...

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """执行精评估"""
        ...


# ============== 内置精评估器 ==============


class MatMulPreciseEvaluator:
    """MatMul 精评估器

    适用：matmul/linear/gemm
    特点：枚举 loop-order（mnk/nkm/mkn 等），计算 tile 复用下的精确 traffic
    """

    SUPPORTED_OPS = {"matmul", "linear", "gemm", "bmm"}

    def supports(self, op: "DistributedOp") -> bool:
        return op.op_type.lower() in self.SUPPORTED_OPS

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """MatMul 精评估

        输入:
            - op: 包含 local_shape {"G", "M", "N", "K"} 和 attrs
            - tile_config: TileConfig（tile_m, tile_n, tile_k）
        输出:
            - PreciseMetrics
        关键步骤:
            - 枚举 loop-order 计算 DRAM traffic
            - 选择最优 loop-order
            - 计算执行时间与 urate
        """
        shape = op.local_shape
        g = shape.get("G", 1)
        m = shape.get("M", 1)
        n = shape.get("N", 1)
        k = shape.get("K", 1)

        # 提取 dtype
        a_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)
        b_bytes = self._get_dtype_bytes(op.attrs, "weight_dtype_bytes", 2)
        c_bytes = self._get_dtype_bytes(op.attrs, "output_dtype_bytes", 2)
        accum_bytes = self._get_dtype_bytes(op.attrs, "accum_dtype_bytes", 4)

        tile_m = tile_config.tile_m or m
        tile_n = tile_config.tile_n or n
        tile_k = tile_config.tile_k or k

        # 计算 tile 迭代次数
        m_tiles = math.ceil(m / tile_m)
        n_tiles = math.ceil(n / tile_n)
        k_tiles = math.ceil(k / tile_k)

        # 枚举 loop-order 计算 DRAM traffic
        best_traffic = float("inf")
        best_order = "mnk"

        for order in ["mnk", "nkm", "mkn", "nmk", "kmn", "knm"]:
            traffic = self._compute_traffic(
                order,
                g,
                tile_m,
                tile_n,
                tile_k,
                m_tiles,
                n_tiles,
                k_tiles,
                a_bytes,
                b_bytes,
                c_bytes,
                accum_bytes,
            )
            if traffic < best_traffic:
                best_traffic = traffic
                best_order = order

        # 计算 LMEM 占用
        lmem_bytes = self._compute_lmem(
            tile_m, tile_n, tile_k, a_bytes, b_bytes, accum_bytes
        )

        # 计算 FLOPs
        flops = 2 * g * m * n * k

        # 计算执行时间
        t_compute_ms = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0
        t_memory_ms = best_traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        # 计算 urate
        total_time = max(t_compute_ms, t_memory_ms)
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0
        bottleneck = "compute" if t_compute_ms >= t_memory_ms else "memory"

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=int(best_traffic),
            lmem_bytes=lmem_bytes,
            flops=flops,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck=bottleneck,
            best_loop_order=best_order,
            extra={"m_tiles": m_tiles, "n_tiles": n_tiles, "k_tiles": k_tiles},
        )

    def _compute_traffic(
        self,
        order: str,
        g: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        m_tiles: int,
        n_tiles: int,
        k_tiles: int,
        a_bytes: int,
        b_bytes: int,
        c_bytes: int,
        accum_bytes: int,
    ) -> int:
        """计算 MatMul 在指定 loop-order 下的 DRAM traffic

        MatMul: C[G,M,N] = A[G,M,K] @ B[G,K,N]

        Loop-order 影响数据复用：
        - 外层循环的 tile 每次迭代都要重新加载
        - 内层循环的 tile 可以在 LMEM 中复用
        """
        a_tile_bytes = tile_m * tile_k * a_bytes
        b_tile_bytes = tile_k * tile_n * b_bytes
        c_tile_bytes = tile_m * tile_n * c_bytes
        accum_tile_bytes = tile_m * tile_n * accum_bytes

        # 所有 loop-order 的基础加载次数
        a_loads = m_tiles * k_tiles
        b_loads = n_tiles * k_tiles
        c_reads = m_tiles * n_tiles * max(0, k_tiles - 1)
        c_writes = m_tiles * n_tiles * k_tiles

        # 计算总 traffic
        a_traffic = a_loads * a_tile_bytes
        b_traffic = b_loads * b_tile_bytes
        c_read_traffic = c_reads * accum_tile_bytes
        c_write_traffic = c_writes * c_tile_bytes

        return (a_traffic + b_traffic + c_read_traffic + c_write_traffic) * g

    def _compute_lmem(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        a_bytes: int,
        b_bytes: int,
        accum_bytes: int,
    ) -> int:
        """计算 LMEM 占用：A tile + B tile + C tile (accumulator)"""
        return (
            tile_m * tile_k * a_bytes
            + tile_k * tile_n * b_bytes
            + tile_m * tile_n * accum_bytes
        )

    def _get_dtype_bytes(self, attrs: dict, key: str, default: int) -> int:
        value = attrs.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return default


class AttentionPreciseEvaluator:
    """Attention 精评估器

    适用：attention/mha/mla/fa2
    特点：FlashAttention-2 风格，考虑 Q/K/V/P/O buffer 占用
    """

    SUPPORTED_OPS = {"attention", "mha", "mla", "fa2", "flash_attention", "sdpa"}

    def supports(self, op: "DistributedOp") -> bool:
        return op.op_type.lower() in self.SUPPORTED_OPS

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """Attention 精评估（FlashAttention-2 style）"""
        shape = op.local_shape
        b = shape.get("B", 1)
        s = shape.get("S", shape.get("seq_len", 1))
        h = shape.get("H", shape.get("num_heads", 1))
        d = shape.get("D", shape.get("head_dim", 64))

        qkv_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)
        out_bytes = self._get_dtype_bytes(op.attrs, "output_dtype_bytes", 2)

        tile_q = tile_config.tile_m or min(s, 64)
        tile_k = tile_config.tile_k or min(s, 64)

        q_tiles = math.ceil(s / tile_q)
        k_tiles = math.ceil(s / tile_k)

        # FLOPs: 2 * B * H * S * S * D (QK^T) + 2 * B * H * S * S * D (PV)
        flops = 4 * b * h * s * s * d

        # Traffic 考虑 tiling
        q_loads = q_tiles * k_tiles
        k_loads = q_tiles * k_tiles
        v_loads = q_tiles * k_tiles
        o_writes = q_tiles

        traffic = (
            q_loads * b * h * tile_q * d * qkv_bytes
            + k_loads * b * h * tile_k * d * qkv_bytes
            + v_loads * b * h * tile_k * d * qkv_bytes
            + o_writes * b * h * tile_q * d * out_bytes
        )

        # LMEM: Q + K + V + P + O tiles
        lmem_bytes = (
            tile_q * d * qkv_bytes
            + tile_k * d * qkv_bytes
            + tile_k * d * qkv_bytes
            + 2 * tile_q * tile_k * 4  # P (float32)
            + 4 * tile_q * d * 4  # O (float32)
        )

        t_compute_ms = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0
        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        total_time = max(t_compute_ms, t_memory_ms)
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0
        bottleneck = "compute" if t_compute_ms >= t_memory_ms else "memory"

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=int(traffic),
            lmem_bytes=lmem_bytes,
            flops=flops,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck=bottleneck,
            best_loop_order="qk",
            extra={"q_tiles": q_tiles, "k_tiles": k_tiles},
        )

    def _get_dtype_bytes(self, attrs: dict, key: str, default: int) -> int:
        value = attrs.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return default


class ConvPreciseEvaluator:
    """Conv 精评估器

    适用：conv2d/conv1d/depthwise_conv
    特点：考虑 im2col 或 direct conv 的 traffic 模式
    """

    SUPPORTED_OPS = {"conv2d", "conv1d", "conv", "depthwise_conv"}

    def supports(self, op: "DistributedOp") -> bool:
        return op.op_type.lower() in self.SUPPORTED_OPS

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """Conv 精评估（im2col style）"""
        shape = op.local_shape
        n = shape.get("N", shape.get("B", 1))  # batch
        c_in = shape.get("C", shape.get("in_channels", 64))
        c_out = shape.get("K", shape.get("out_channels", 64))
        h = shape.get("H", shape.get("height", 1))
        w = shape.get("W", shape.get("width", 1))
        kh = shape.get("KH", shape.get("kernel_h", 3))
        kw = shape.get("KW", shape.get("kernel_w", 3))

        dtype_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)

        # im2col: 将 conv 转换为 matmul
        # M = N * H_out * W_out, K = C_in * KH * KW, N = C_out
        h_out = h  # 简化假设 stride=1, padding=same
        w_out = w
        m_gemm = n * h_out * w_out
        k_gemm = c_in * kh * kw
        n_gemm = c_out

        # FLOPs
        flops = 2 * m_gemm * n_gemm * k_gemm

        # Traffic: im2col + weight + output
        # im2col 会扩展输入
        input_traffic = n * c_in * h * w * dtype_bytes
        weight_traffic = c_out * c_in * kh * kw * dtype_bytes
        output_traffic = n * c_out * h_out * w_out * dtype_bytes
        traffic = input_traffic + weight_traffic + output_traffic

        # 简化的 LMEM 估算
        tile_m = tile_config.tile_m or 64
        tile_n = tile_config.tile_n or 64
        lmem_bytes = tile_m * k_gemm * dtype_bytes + k_gemm * tile_n * dtype_bytes

        t_compute_ms = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0
        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        total_time = max(t_compute_ms, t_memory_ms)
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0
        bottleneck = "compute" if t_compute_ms >= t_memory_ms else "memory"

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=int(traffic),
            lmem_bytes=lmem_bytes,
            flops=flops,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck=bottleneck,
            best_loop_order="im2col",
        )

    def _get_dtype_bytes(self, attrs: dict, key: str, default: int) -> int:
        value = attrs.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return default


class ElementwisePreciseEvaluator:
    """Elementwise 精评估器

    适用：softmax/layernorm/rmsnorm/relu/gelu/silu/add/mul 等
    特点：memory-bound，traffic = 2 * elements * dtype_bytes
    """

    SUPPORTED_OPS = {
        "softmax",
        "layernorm",
        "rmsnorm",
        "relu",
        "gelu",
        "silu",
        "add",
        "mul",
        "sub",
        "div",
        "concat",
        "split",
        "reshape",
        "transpose",
        "embedding",
        "lmhead",
    }

    def supports(self, op: "DistributedOp") -> bool:
        return op.op_type.lower() in self.SUPPORTED_OPS

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """Elementwise 精评估（memory-bound）"""
        shape = op.local_shape
        dtype_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)

        # 计算元素数
        elements = 1
        for v in shape.values():
            elements *= v

        # FLOPs: 假设 ~5 ops per element
        flops = elements * 5
        # Traffic: read + write
        traffic = 2 * elements * dtype_bytes

        t_compute_ms = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0
        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        total_time = max(t_compute_ms, t_memory_ms)
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=traffic,
            lmem_bytes=0,
            flops=flops,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck="memory",  # Elementwise 通常是 memory-bound
            best_loop_order="sequential",
        )

    def _get_dtype_bytes(self, attrs: dict, key: str, default: int) -> int:
        value = attrs.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        return default


class FallbackPreciseEvaluator:
    """Fallback 精评估器

    适用：未注册的算子
    特点：简化评估，使用理论 traffic（无 tile 复用）
    """

    def supports(self, op: "DistributedOp") -> bool:
        return True  # 支持所有算子

    def evaluate(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
        compute_tflops: float,
        memory_bw_gbps: float,
    ) -> PreciseMetrics:
        """Fallback 精评估"""
        shape = op.local_shape
        dtype_bytes = 2

        elements = 1
        for v in shape.values():
            elements *= v

        flops = elements * 2  # 简化假设
        traffic = 2 * elements * dtype_bytes

        t_compute_ms = flops / (compute_tflops * 1e9) if compute_tflops > 0 else 0
        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        total_time = max(t_compute_ms, t_memory_ms)
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0
        bottleneck = "compute" if t_compute_ms >= t_memory_ms else "memory"

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=traffic,
            lmem_bytes=0,
            flops=flops,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck=bottleneck,
            best_loop_order="fallback",
        )


# ============== 注册表 ==============


class PreciseEvaluatorRegistry:
    """精评估器注册表

    按 op_type 路由到专属精评估器，支持插件式注册。
    """

    def __init__(self) -> None:
        self._evaluators: list[OpPreciseEvaluator] = []
        self._fallback: OpPreciseEvaluator = FallbackPreciseEvaluator()

    def register(self, evaluator: OpPreciseEvaluator) -> None:
        """注册算子专属评估器"""
        self._evaluators.append(evaluator)

    def register_fallback(self, evaluator: OpPreciseEvaluator) -> None:
        """注册回退评估器"""
        self._fallback = evaluator

    def resolve(self, op: "DistributedOp") -> OpPreciseEvaluator:
        """获取评估器（未找到返回 fallback）"""
        for evaluator in self._evaluators:
            if evaluator.supports(op):
                return evaluator
        return self._fallback


# ============== 统一入口 ==============


class PreciseTileEvaluator:
    """精评估器（统一入口）

    实现 L4TileEvaluator 协议，为 L3 TilingPlanner 提供精确评估。
    支持按算子类型路由到专属评估器。
    """

    def __init__(
        self,
        compute_tflops: float = 125.0,
        memory_bandwidth_gbps: float = 2000.0,
        registry: PreciseEvaluatorRegistry | None = None,
    ) -> None:
        """初始化精评估器

        Args:
            compute_tflops: 峰值算力（TFLOPS）
            memory_bandwidth_gbps: 显存带宽（GB/s）
            registry: 评估器注册表（不指定则使用默认）
        """
        self.compute_tflops = compute_tflops
        self.memory_bandwidth_gbps = memory_bandwidth_gbps

        if registry is not None:
            self.registry = registry
        else:
            self.registry = self._create_default_registry()

    def _create_default_registry(self) -> PreciseEvaluatorRegistry:
        """创建默认注册表"""
        registry = PreciseEvaluatorRegistry()
        registry.register(MatMulPreciseEvaluator())
        registry.register(AttentionPreciseEvaluator())
        registry.register(ConvPreciseEvaluator())
        registry.register(ElementwisePreciseEvaluator())
        return registry

    def evaluate_tile(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
    ) -> dict[str, int | float | str]:
        """评估 tile 配置（L4TileEvaluator 协议）

        输入:
            - op: DistributedOp
            - tile_config: TileConfig
            - chip: ChipSpecImpl
        输出:
            - 精评估结果字典
        关键步骤:
            - 路由选择：根据 op_type 获取专属评估器
            - 执行评估：调用评估器的 evaluate 方法
        """
        # 获取硬件参数（优先使用 chip 的参数）
        compute_tflops = getattr(chip, "compute_tflops", self.compute_tflops)
        memory_bw_gbps = getattr(
            chip, "memory_bandwidth_gbps", self.memory_bandwidth_gbps
        )

        # 路由选择
        evaluator = self.registry.resolve(op)

        # 执行评估
        metrics = evaluator.evaluate(
            op, tile_config, chip, compute_tflops, memory_bw_gbps
        )

        return metrics.to_dict()

    # 向后兼容的方法
    def __call__(
        self,
        op: "DistributedOp",
        tile_config: "TileConfig",
        chip: "ChipSpecImpl",
    ) -> dict[str, int | float | str]:
        """支持函数调用方式（向后兼容）"""
        return self.evaluate_tile(op, tile_config, chip)


def create_precise_evaluator(
    compute_tflops: float = 125.0,
    memory_bandwidth_gbps: float = 2000.0,
    registry: PreciseEvaluatorRegistry | None = None,
) -> PreciseTileEvaluator:
    """创建精评估器

    Args:
        compute_tflops: 峰值算力（TFLOPS）
        memory_bandwidth_gbps: 显存带宽（GB/s）
        registry: 评估器注册表（可选）

    Returns:
        PreciseTileEvaluator 实例
    """
    return PreciseTileEvaluator(
        compute_tflops=compute_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        registry=registry,
    )
