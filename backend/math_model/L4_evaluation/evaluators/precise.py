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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Protocol

import numba


# ============== Numba JIT 加速的 MatMul Traffic 计算 ==============

LOOP_ORDER_MAP: Dict[str, int] = {
    "mnk": 0, "nkm": 1, "mkn": 2, "nmk": 3, "kmn": 4, "knm": 5,
}


@numba.jit(nopython=True, cache=True)
def _compute_traffic_numba(
    order_idx: int,
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
    """计算 MatMul 在指定 loop-order 下的 DRAM traffic (Numba JIT)"""
    m_blk = tile_m * m_tiles
    n_blk = tile_n * n_tiles
    k_blk = tile_k * k_tiles

    traffic = 0

    if order_idx == 0:  # mnk
        traffic = (
            (m_blk * k_blk) * a_bytes * n_tiles
            + (n_blk * k_blk) * b_bytes * m_tiles
            + (m_blk * n_blk) * c_bytes
        )
    elif order_idx == 1:  # nkm
        traffic = (
            (n_blk * k_blk) * b_bytes
            + (m_blk * k_blk) * a_bytes * n_tiles
            + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)
            + (m_blk * n_blk) * c_bytes
        )
    elif order_idx == 2:  # mkn
        traffic = (
            (m_blk * k_blk) * a_bytes
            + (n_blk * k_blk) * b_bytes * m_tiles
            + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)
            + (m_blk * n_blk) * c_bytes
        )
    elif order_idx == 3:  # nmk
        traffic = (
            (n_blk * k_blk) * b_bytes * m_tiles
            + (m_blk * k_blk) * a_bytes * n_tiles
            + (m_blk * n_blk) * c_bytes
        )
    elif order_idx == 4:  # kmn
        traffic = (
            (m_blk * k_blk) * a_bytes
            + (n_blk * k_blk) * b_bytes * m_tiles
            + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)
            + (m_blk * n_blk) * c_bytes
        )
    else:  # knm
        traffic = (
            (n_blk * k_blk) * b_bytes
            + (m_blk * k_blk) * a_bytes * n_tiles
            + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)
            + (m_blk * n_blk) * c_bytes
        )

    return int(traffic * g)


def _compute_traffic_fn(
    order: str, g: int,
    tile_m: int, tile_n: int, tile_k: int,
    m_tiles: int, n_tiles: int, k_tiles: int,
    a_bytes: int, b_bytes: int, c_bytes: int, accum_bytes: int,
) -> int:
    """Python 接口，调用 Numba 优化版本"""
    order_idx = LOOP_ORDER_MAP.get(order, 0)
    return _compute_traffic_numba(
        order_idx, g, tile_m, tile_n, tile_k,
        m_tiles, n_tiles, k_tiles,
        a_bytes, b_bytes, c_bytes, accum_bytes,
    )


if TYPE_CHECKING:
    from math_model.L2_arch.chip import ChipSpecImpl
    from math_model.L3_mapping.plan.distributed_model import DistributedOp
    from math_model.L1_workload.specs import TileConfig


def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    if alignment <= 1:
        return value
    return math.ceil(value / alignment) * alignment


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
        result: dict[str, int | float | str] = {
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
        # 将 extra 中的 P0 指标暴露给 L3/L4 链路
        for key in ("arch_urate", "active_cores", "overlap_rate"):
            if key in self.extra:
                result[key] = self.extra[key]
        return result


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
        # C tile 在 SRAM 中至少需要 BF16 精度 (2 bytes)
        # FP8 只是存储/通信格式, 片上 buffer 不能低于 BF16
        c_bytes = max(c_bytes, 2)
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
            traffic = _compute_traffic_fn(
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

        # 计算 FLOPs（real + aligned）
        flops = 2 * g * m * n * k

        # arch_urate: 对齐损耗（P0.1）
        cube_m = chip.cube_m if chip else 0
        cube_k = chip.cube_k if chip else 0
        cube_n = chip.cube_n if chip else 0
        if cube_m > 0 and cube_k > 0 and cube_n > 0:
            aligned_flops = (
                2 * g
                * _align_up(m, cube_m)
                * _align_up(k, cube_k)
                * _align_up(n, cube_n)
            )
            arch_urate = flops / aligned_flops if aligned_flops > 0 else 1.0
        else:
            aligned_flops = flops
            arch_urate = 1.0

        # 分核感知（P0.3）: 活跃核数缩放
        active_cores = chip.core_count if chip else 1
        if chip and chip.core_count > 0:
            total_parallel_tiles = g * m_tiles * n_tiles
            active_cores = min(chip.core_count, max(1, total_parallel_tiles))
            effective_tflops = compute_tflops * active_cores / chip.core_count
        else:
            effective_tflops = compute_tflops

        # 计算执行时间（用 aligned_flops，除以有效算力）
        t_compute_ms = aligned_flops / (effective_tflops * 1e9) if effective_tflops > 0 else 0
        t_memory_ms = best_traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        # compute-DMA overlap（P0.2）
        overlap_rate = chip.compute_dma_overlap_rate if chip else 0.0
        total_time = (
            min(t_compute_ms, t_memory_ms) * (1 - overlap_rate)
            + max(t_compute_ms, t_memory_ms)
        )

        # 计算 urate
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
            extra={
                "m_tiles": m_tiles,
                "n_tiles": n_tiles,
                "k_tiles": k_tiles,
                "arch_urate": arch_urate,
                "active_cores": active_cores,
                "overlap_rate": overlap_rate,
            },
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
        """计算 MatMul 在指定 loop-order 下的 DRAM traffic（精确模型）

        MatMul: C[G,M,N] = A[G,M,K] @ B[G,K,N]

        Loop-order 影响数据复用模式：
        - mnk: K 在最内层，A 复用 n_tiles 次，B 复用 m_tiles 次，C 无需累加读写
        - nkm: M 在最内层，B 读一次，A 复用 n_tiles 次，C 需累加 (k_tiles-1) 次
        - mkn: N 在最内层，A 读一次，B 复用 m_tiles 次，C 需累加 (k_tiles-1) 次
        - nmk: K 在最内层（类似 mnk）
        - kmn: N 在最内层（类似 mkn）
        - knm: M 在最内层（类似 nkm）
        """
        m_blk = tile_m * m_tiles
        n_blk = tile_n * n_tiles
        k_blk = tile_k * k_tiles

        # 精确模型：根据 loop-order 区分数据复用
        if order == "mnk":
            # K 在最内层：A[m,k] 复用 n_tiles 次，B[k,n] 复用 m_tiles 次
            # C 在 K 维累加完成后写回，无需中间读写
            traffic = (
                (m_blk * k_blk) * a_bytes * n_tiles  # A 复用
                + (n_blk * k_blk) * b_bytes * m_tiles  # B 复用
                + (m_blk * n_blk) * c_bytes  # C 最终写回
            )
        elif order == "nkm":
            # M 在最内层：B[k,n] 读一次，A[m,k] 复用 n_tiles 次
            # C 需要 k_tiles-1 次累加读写（FP32 精度）
            traffic = (
                (n_blk * k_blk) * b_bytes  # B 读一次
                + (m_blk * k_blk) * a_bytes * n_tiles  # A 复用
                + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)  # C 累加读写
                + (m_blk * n_blk) * c_bytes  # C 最终写回
            )
        elif order == "mkn":
            # N 在最内层：A[m,k] 读一次，B[k,n] 复用 m_tiles 次
            # C 需要 k_tiles-1 次累加读写
            traffic = (
                (m_blk * k_blk) * a_bytes  # A 读一次
                + (n_blk * k_blk) * b_bytes * m_tiles  # B 复用
                + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)  # C 累加读写
                + (m_blk * n_blk) * c_bytes  # C 最终写回
            )
        elif order == "nmk":
            # K 在最内层（类似 mnk，交换 M/N）
            traffic = (
                (n_blk * k_blk) * b_bytes * m_tiles  # B 复用
                + (m_blk * k_blk) * a_bytes * n_tiles  # A 复用
                + (m_blk * n_blk) * c_bytes  # C 最终写回
            )
        elif order == "kmn":
            # N 在最内层（类似 mkn）
            traffic = (
                (m_blk * k_blk) * a_bytes  # A 读一次
                + (n_blk * k_blk) * b_bytes * m_tiles  # B 复用
                + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)  # C 累加
                + (m_blk * n_blk) * c_bytes  # C 写回
            )
        else:  # knm
            # M 在最内层（类似 nkm）
            traffic = (
                (n_blk * k_blk) * b_bytes  # B 读一次
                + (m_blk * k_blk) * a_bytes * n_tiles  # A 复用
                + (m_blk * n_blk) * accum_bytes * 2 * max(0, k_tiles - 1)  # C 累加
                + (m_blk * n_blk) * c_bytes  # C 写回
            )

        return int(traffic * g)

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
    支持 zigzag reorder 优化（prefill + causal attention 时 GEMM 减 40%）
    集成 Softmax 10步向量操作评估（P1.4/P1.5）
    """

    SUPPORTED_OPS = {"attention", "mha", "mla", "fa2", "flash_attention", "sdpa"}

    def __init__(self, is_prefill: bool = False, enable_zigzag: bool = False) -> None:
        self.is_prefill = is_prefill
        self.enable_zigzag = enable_zigzag

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
        """Attention 精评估（FlashAttention-2 style）

        P1.5 集成: GEMM (Cube) + Softmax 向量 (EU) 分开计算时间，合并 arch_urate
        """
        from math_model.L4_evaluation.evaluators.softmax_eval import (
            softmax_theoretical_and_real,
        )

        shape = op.local_shape

        # FA2 shape: {B, QS, KS, QD, VD}  (B = batch*heads)
        # Legacy shape: {B, S, H, D}
        if "QS" in shape or "KS" in shape:
            for key in ("B", "QS", "KS", "QD", "VD"):
                if key not in shape:
                    raise ValueError(
                        f"Missing '{key}' in FA2 op local_shape for op '{op.op_id}'. "
                        f"Available keys: {list(shape.keys())}"
                    )
            b = shape["B"]
            qs = shape["QS"]
            ks = shape["KS"]
            qd = shape["QD"]
            vd = shape["VD"]
            h = 1

            # B*QS redistribution across cores (CHIPMathica FA2 style)
            # Merge batch*heads and query sequence, redistribute evenly across
            # cores to eliminate alignment waste when QS is small (decode QS=1)
            if chip and chip.core_count > 0:
                total_works = b * qs
                b = chip.core_count
                qs = math.ceil(total_works / b)
        else:
            for key in ("B", "H", "D"):
                if key not in shape and key.lower() not in shape:
                    raise ValueError(
                        f"Missing '{key}' in attention op local_shape for op '{op.op_id}'. "
                        f"Available keys: {list(shape.keys())}"
                    )
            b = shape["B"]
            h = shape.get("H") or shape["num_heads"]
            qs = shape.get("S") or shape["seq_len"]
            ks = qs
            qd = shape.get("D") or shape["head_dim"]
            vd = qd

        qkv_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)
        out_bytes = self._get_dtype_bytes(op.attrs, "output_dtype_bytes", 2)

        tile_q = tile_config.tile_m or min(qs, 64)
        # FA2TilingEvaluator puts KS tile in tile_n, fallback to tile_k
        tile_k = tile_config.tile_n or tile_config.tile_k or min(ks, 64)

        q_tiles = math.ceil(qs / tile_q)
        k_tiles = math.ceil(ks / tile_k)

        # FLOPs: 2*B*H*QS*KS*QD (QK^T) + 2*B*H*QS*KS*VD (PV)
        flops = 2 * b * h * qs * ks * (qd + vd)

        # Traffic (tiled access pattern)
        # Q stays in SRAM while iterating K tiles, so loaded only q_tiles times
        q_loads = q_tiles
        k_loads = q_tiles * k_tiles
        v_loads = q_tiles * k_tiles
        o_writes = q_tiles

        traffic = (
            q_loads * b * h * tile_q * qd * qkv_bytes
            + k_loads * b * h * tile_k * qd * qkv_bytes
            + v_loads * b * h * tile_k * vd * qkv_bytes
            + o_writes * b * h * tile_q * vd * out_bytes
        )

        # LMEM: Q + K + V + P + O tiles
        lmem_bytes = (
            tile_q * qd * qkv_bytes
            + tile_k * qd * qkv_bytes
            + tile_k * vd * qkv_bytes
            + 2 * tile_q * tile_k * 4  # P (float32)
            + 4 * tile_q * vd * 4  # O (float32)
        )

        # ============ GEMM arch_urate ============
        cube_m = chip.cube_m if chip else 0
        cube_k = chip.cube_k if chip else 0
        cube_n = chip.cube_n if chip else 0
        if cube_m > 0 and cube_k > 0 and cube_n > 0:
            # QK^T: align(QS, cube_m) * align(QD, cube_k) * align(KS, cube_n)
            # PV:   align(QS, cube_m) * align(KS, cube_k) * align(VD, cube_n)
            gemm_real = qs * ks * (qd + vd)
            gemm_theo = (
                _align_up(qs, cube_m) * _align_up(qd, cube_k) * _align_up(ks, cube_n)
                + _align_up(qs, cube_m) * _align_up(ks, cube_k) * _align_up(vd, cube_n)
            )
            if self.enable_zigzag and self.is_prefill:
                gemm_real = int(gemm_real * 0.6)
                gemm_theo = int(gemm_theo * 0.6)
            aligned_flops = 2 * b * h * gemm_theo
        else:
            gemm_real = qs * ks * (qd + vd)
            gemm_theo = gemm_real
            aligned_flops = flops
            if self.enable_zigzag and self.is_prefill:
                gemm_real = int(gemm_real * 0.6)
                gemm_theo = int(gemm_theo * 0.6)
                aligned_flops = int(aligned_flops * 0.6)

        # ============ Softmax vector ops ============
        lane_num = chip.lane_per_core if chip else 0
        eu_num_total = chip.eu_num if chip else 0  # chip total EU (for throughput)
        # softmax alignment 需要每核 EU 数: eu_block = eu_per_core / lane_num / dtype
        # chip.eu_num 是芯片总量 (eu_per_lane * lane * cores), 需除以 core_count
        eu_per_core = eu_num_total // chip.core_count if chip and chip.core_count > 0 else 0
        vector_theo = 0
        vector_real = 0
        vector_t_ms = 0.0

        # P16+P17: softmax 始终用 BF16 精度计算 (数值稳定性), 对齐 CHIPMathica
        SOFTMAX_DTYPE_BYTES = 2

        if lane_num > 0 and eu_per_core > 0:
            # P16: 用 full (qs, ks) 而非 tile 维度, 对齐 CHIPMathica
            # P17: 用 BF16 做 eu_block 对齐, 对齐 CHIPMathica
            sv_theo, sv_real = softmax_theoretical_and_real(
                QS=qs, KS=ks,
                lane_num=lane_num, eu_num=eu_per_core,
                dtype_bytes=SOFTMAX_DTYPE_BYTES,
            )
            # arch_urate 用 per-head 值 (与 gemm_real/gemm_theo 一致)
            vector_theo = sv_theo
            vector_real = sv_real

            freq_ghz = chip.frequency_ghz if chip else 1.0
            # P17: throughput = eu * freq * dtype_bytes, 对齐 CHIPMathica
            vector_throughput = eu_num_total * freq_ghz * 1e9 * SOFTMAX_DTYPE_BYTES
            if vector_throughput > 0:
                # time 用 total (b * h) 计算
                vector_t_ms = sv_theo * b * h / vector_throughput * 1e3

        total_real = gemm_real + vector_real
        total_theo = gemm_theo + vector_theo
        arch_urate = total_real / total_theo if total_theo > 0 else 1.0

        # ============ Core utilization (B*H redistribute) ============
        active_cores = chip.core_count if chip else 1
        if chip and chip.core_count > 0:
            # For FA2: redistribute B*H*q_tiles across cores (CHIPMathica style)
            total_parallel_tiles = b * h * q_tiles
            active_cores = min(chip.core_count, max(1, total_parallel_tiles))
            effective_tflops = compute_tflops * active_cores / chip.core_count
        else:
            effective_tflops = compute_tflops

        t_gemm_ms = aligned_flops / (effective_tflops * 1e9) if effective_tflops > 0 else 0
        t_compute_ms = t_gemm_ms + vector_t_ms
        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        overlap_rate = chip.compute_dma_overlap_rate if chip else 0.0
        total_time = (
            min(t_compute_ms, t_memory_ms) * (1 - overlap_rate)
            + max(t_compute_ms, t_memory_ms)
        )

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
            extra={
                "q_tiles": q_tiles,
                "k_tiles": k_tiles,
                "arch_urate": arch_urate,
                "active_cores": active_cores,
                "overlap_rate": overlap_rate,
                "zigzag_applied": 1.0 if (self.enable_zigzag and self.is_prefill) else 0.0,
                "vector_theo": vector_theo,
                "vector_real": vector_real,
                "vector_t_ms": vector_t_ms,
                "t_gemm_ms": t_gemm_ms,
            },
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


class RMSNormPreciseEvaluator:
    """RMSNorm/LayerNorm 精评估器

    适用：rmsnorm/layernorm/rmsnorm_q/rmsnorm_kv
    特点：9步向量操作精确建模，考虑硬件对齐（lane_num/eu_num）
    关键慢操作：div_constant(31 cycles), rsqrt(30 cycles)
    """

    SUPPORTED_OPS = {"rmsnorm", "layernorm", "rmsnorm_q", "rmsnorm_kv"}

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
        """RMSNorm 精评估（9步向量操作）"""
        from math_model.L4_evaluation.evaluators.rmsnorm_eval import (
            rmsnorm_theoretical_and_real,
        )

        shape = op.local_shape
        # RMSNorm shape: (B*S, hidden) or (M, K)
        # 行维 = tokens, 列维 = hidden_dim
        batch_seq = shape.get("M", shape.get("B", 1))
        if "S" in shape:
            batch_seq = batch_seq * shape["S"]
        hidden_dim = shape.get("K", shape.get("N", shape.get("H", 1)))

        dtype_bytes = self._get_dtype_bytes(op.attrs, "input_dtype_bytes", 2)
        is_layernorm = "layernorm" in op.op_type.lower()

        lane_num = chip.lane_per_core if chip else 0
        eu_num = chip.eu_num if chip else 0

        # Traffic: read input + read gamma + write output
        elements = batch_seq * hidden_dim
        traffic = (2 * elements + hidden_dim) * dtype_bytes  # input + output + gamma

        vector_theo = 0
        vector_real = 0
        t_compute_ms = 0.0

        if lane_num > 0 and eu_num > 0:
            vector_theo, vector_real = rmsnorm_theoretical_and_real(
                batch_size=batch_seq,
                hidden_dim=hidden_dim,
                lane_num=lane_num,
                eu_num=eu_num,
                dtype_bytes=dtype_bytes,
                has_scale=True,
                has_bias=is_layernorm,
            )

            # 向量执行时间
            freq_ghz = chip.frequency_ghz if chip else 1.0
            ops_per_cycle = eu_num
            if ops_per_cycle > 0 and freq_ghz > 0:
                cycles = vector_theo / ops_per_cycle
                t_compute_ms = cycles / (freq_ghz * 1e6)  # GHz*1e6 = cycles/ms
        else:
            # fallback: 粗估
            flops_estimate = elements * 5
            t_compute_ms = flops_estimate / (compute_tflops * 1e9) if compute_tflops > 0 else 0

        t_memory_ms = traffic / (memory_bw_gbps * 1e6) if memory_bw_gbps > 0 else 0

        # compute-DMA overlap
        overlap_rate = chip.compute_dma_overlap_rate if chip else 0.0
        total_time = (
            min(t_compute_ms, t_memory_ms) * (1 - overlap_rate)
            + max(t_compute_ms, t_memory_ms)
        )

        arch_urate = vector_real / vector_theo if vector_theo > 0 else 1.0
        compute_urate = t_compute_ms / total_time if total_time > 0 else 0
        memory_urate = t_memory_ms / total_time if total_time > 0 else 0
        bottleneck = "compute" if t_compute_ms >= t_memory_ms else "memory"

        return PreciseMetrics(
            t_compute_ms=t_compute_ms,
            t_memory_ms=t_memory_ms,
            dram_traffic_bytes=int(traffic),
            lmem_bytes=0,
            flops=vector_real,
            compute_urate=compute_urate,
            memory_urate=memory_urate,
            bottleneck=bottleneck,
            best_loop_order="sequential",
            extra={
                "arch_urate": arch_urate,
                "vector_theo": vector_theo,
                "vector_real": vector_real,
                "overlap_rate": overlap_rate,
            },
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

    适用：relu/gelu/silu/add/mul 等简单逐元素操作
    特点：memory-bound，traffic = 2 * elements * dtype_bytes
    注意：rmsnorm/layernorm 已由 RMSNormPreciseEvaluator 专门处理
    """

    SUPPORTED_OPS = {
        "softmax",
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
        is_prefill: bool = False,
        enable_zigzag: bool = False,
    ) -> None:
        """初始化精评估器

        Args:
            compute_tflops: 峰值算力（TFLOPS）
            memory_bandwidth_gbps: 显存带宽（GB/s）
            registry: 评估器注册表（不指定则使用默认）
            is_prefill: 是否为 prefill 阶段
            enable_zigzag: 是否启用 zigzag reorder 优化
        """
        self.compute_tflops = compute_tflops
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self.is_prefill = is_prefill
        self.enable_zigzag = enable_zigzag

        if registry is not None:
            self.registry = registry
        else:
            self.registry = self._create_default_registry()

    def _create_default_registry(self) -> PreciseEvaluatorRegistry:
        """创建默认注册表"""
        registry = PreciseEvaluatorRegistry()
        registry.register(MatMulPreciseEvaluator())
        registry.register(AttentionPreciseEvaluator(
            is_prefill=self.is_prefill,
            enable_zigzag=self.enable_zigzag,
        ))
        registry.register(ConvPreciseEvaluator())
        registry.register(RMSNormPreciseEvaluator())
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
