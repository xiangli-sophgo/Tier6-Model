"""指令级 tiling (G5 指令级仿真模式)

MatMul Tiling 算法:
    1. 计算单核 LMEM 可用预算
    2. Double buffering 需要 2 份 A/B/C buffer
    3. 搜索最优 tile_m/n/k 使得 tile volume 最大
    4. 顺序分配 LMEM 地址: A0, A1, B0, B1, C0, C1

参考设计: docs/plans/G5-instruction-level-simulator.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L4_evaluation.g5.memory import lmem_budget_per_core


@dataclass
class LMEMLayout:
    """LMEM 地址布局 (double buffering)

    Attributes:
        a_addrs: A buffer 地址 [buf0, buf1]
        b_addrs: B buffer 地址 [buf0, buf1]
        c_addrs: C buffer 地址 [buf0, buf1]
        a_tile_bytes: 单个 A buffer 大小
        b_tile_bytes: 单个 B buffer 大小
        c_tile_bytes: 单个 C buffer 大小
        total_bytes: 总使用量
    """
    a_addrs: list[int]
    b_addrs: list[int]
    c_addrs: list[int]
    a_tile_bytes: int
    b_tile_bytes: int
    c_tile_bytes: int
    total_bytes: int


@dataclass
class TilingResult:
    """Tiling 结果

    Attributes:
        tile_m: M 维度 tile 大小
        tile_n: N 维度 tile 大小
        tile_k: K 维度 tile 大小
        layout: LMEM 地址布局
        m_tiles: M 方向 tile 数量
        n_tiles: N 方向 tile 数量
        k_tiles: K 方向 tile 数量
    """
    tile_m: int
    tile_n: int
    tile_k: int
    layout: LMEMLayout
    m_tiles: int
    n_tiles: int
    k_tiles: int


def _align_up(value: int, alignment: int) -> int:
    """向上对齐"""
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _calc_tile_bytes(
    tile_m: int, tile_n: int, tile_k: int,
    a_bytes: int, b_bytes: int, c_bytes: int,
    lane_num: int, align_bytes: int,
) -> tuple[int, int, int]:
    """计算单组 A/B/C tile 大小 (bytes)

    A_tile = align_up(tile_m, lane_num) * tile_k * a_bytes
    B_tile = tile_k * align_up(tile_n, align_bytes // b_bytes) * b_bytes
    C_tile = align_up(tile_m, lane_num) * tile_n * c_bytes
    """
    aligned_m = _align_up(tile_m, lane_num)
    b_align = align_bytes // b_bytes if b_bytes > 0 else 1
    aligned_n = _align_up(tile_n, b_align)

    a_tile = aligned_m * tile_k * a_bytes
    b_tile = tile_k * aligned_n * b_bytes
    c_tile = aligned_m * tile_n * c_bytes

    return a_tile, b_tile, c_tile


def tile_matmul(
    M: int, N: int, K: int,
    dtype_bytes: int,
    chip: ChipSpecImpl,
) -> TilingResult:
    """MatMul tiling

    搜索 tile_m/n/k 使 double buffering 下 LMEM 使用最优。

    Args:
        M: 矩阵 M 维度
        N: 矩阵 N 维度
        K: 矩阵 K 维度
        dtype_bytes: 输入数据类型字节数 (如 BF16=2)
        chip: 芯片规格

    Returns:
        TilingResult
    """
    budget = lmem_budget_per_core(chip)
    lane_num = chip.lane_per_core
    cube_m = chip.cube_m
    cube_n = chip.cube_n
    align_bytes = chip.align_bytes

    if cube_m <= 0 or cube_n <= 0:
        raise ValueError(
            f"Chip '{chip.name}' cube dimensions invalid: "
            f"cube_m={cube_m}, cube_n={cube_n}"
        )

    a_bytes = dtype_bytes
    b_bytes = dtype_bytes
    # 输出精度: BF16 输入 -> FP32 累加后截断回 BF16, 但 LMEM 中保存 FP32 中间结果
    # 最小化实现中假设输出与输入同精度
    c_bytes = dtype_bytes

    best: TilingResult | None = None
    best_volume = 0

    # tile_m 从 M 向下按 cube_m 步进
    tm = M
    while tm >= cube_m:
        # tile_n 从 N 向下按 cube_n 步进
        tn = N
        while tn >= cube_n:
            # 计算此 (tm, tn) 下的最大 tile_k
            _, _, c_one = _calc_tile_bytes(
                tm, tn, 1, a_bytes, b_bytes, c_bytes, lane_num, align_bytes,
            )
            # double buffering: 2 * (A + B + C) 需要 <= budget
            # A 和 B 随 tile_k 线性增长, C 不变
            # 2 * (a_per_k * tk + b_per_k * tk + c_one) <= budget
            a_per_k = _align_up(tm, lane_num) * a_bytes
            b_align = align_bytes // b_bytes if b_bytes > 0 else 1
            b_per_k = _align_up(tn, b_align) * b_bytes

            per_k = a_per_k + b_per_k
            if per_k <= 0:
                tn -= cube_n
                continue

            remaining = budget - 2 * c_one
            if remaining <= 0:
                tn -= cube_n
                continue

            max_tk = remaining // (2 * per_k)
            if max_tk <= 0:
                tn -= cube_n
                continue

            tk = min(max_tk, K)

            # 验证总量
            a_tile, b_tile, c_tile = _calc_tile_bytes(
                tm, tn, tk, a_bytes, b_bytes, c_bytes, lane_num, align_bytes,
            )
            total = 2 * (a_tile + b_tile + c_tile)
            if total > budget:
                tn -= cube_n
                continue

            volume = tm * tn * tk
            if volume > best_volume:
                best_volume = volume
                layout = _build_layout(a_tile, b_tile, c_tile)
                best = TilingResult(
                    tile_m=tm,
                    tile_n=tn,
                    tile_k=tk,
                    layout=layout,
                    m_tiles=math.ceil(M / tm),
                    n_tiles=math.ceil(N / tn),
                    k_tiles=math.ceil(K / tk),
                )

            tn -= cube_n
        tm -= cube_m

    if best is None:
        raise ValueError(
            f"Cannot tile MatMul({M}x{N}x{K}, dtype_bytes={dtype_bytes}) "
            f"into LMEM budget={budget} bytes for chip '{chip.name}'"
        )

    return best


def _build_layout(a_tile: int, b_tile: int, c_tile: int) -> LMEMLayout:
    """顺序分配 LMEM 地址: A0, A1, B0, B1, C0, C1"""
    offset = 0
    a0 = offset; offset += a_tile
    a1 = offset; offset += a_tile
    b0 = offset; offset += b_tile
    b1 = offset; offset += b_tile
    c0 = offset; offset += c_tile
    c1 = offset; offset += c_tile

    return LMEMLayout(
        a_addrs=[a0, a1],
        b_addrs=[b0, b1],
        c_addrs=[c0, c1],
        a_tile_bytes=a_tile,
        b_tile_bytes=b_tile,
        c_tile_bytes=c_tile,
        total_bytes=offset,
    )
