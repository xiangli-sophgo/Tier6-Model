"""
Flash Attention 2 精确评估器

移植自 DS_TPU_1209/performance/evaluate/compute/flash_attention/fa2_eval.py
核心功能：
1. Batch 维度分区策略搜索
2. Tile 大小搜索 (受 SRAM 约束)
3. DRAM 流量计算
4. 架构利用率计算 (含 Softmax 向量操作)
5. 计算-搬运重叠模型
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from .arch_config import AcceleratorMicroArch
from .utils import ceil_div, align_up
from .softmax_eval import softmax_theoretical_and_real

# 数据类型字节数
FP8_BYTES = 1
BF16_BYTES = 2
FP32_BYTES = 4


@dataclass
class FA2Result:
    """FA2 评估结果"""
    latency_us: float
    """总延迟 (微秒)"""

    compute_time_us: float
    """计算时间 (微秒)"""

    memory_time_us: float
    """访存时间 (微秒)"""

    flops: int
    """浮点运算数"""

    dram_traffic_bytes: int
    """DRAM 流量 (字节)"""

    arch_utilization: float
    """架构利用率 (0-1)"""

    effective_utilization: float
    """有效利用率 (0-1)"""

    best_tile: Tuple[int, int]
    """最佳 Tile 大小 (q_t, k_t)"""

    best_partition: int
    """最佳 Batch 分区数 P_B"""


class FA2Evaluator:
    """Flash Attention 2 精确评估器"""

    def __init__(self, arch: AcceleratorMicroArch):
        """
        初始化评估器

        Args:
            arch: 硬件微架构配置
        """
        self.arch = arch
        self._valid_partitions = self._compute_valid_partitions()
        self._cache: Dict[Tuple, FA2Result] = {}

    def _compute_valid_partitions(self) -> List[int]:
        """
        枚举所有合法的 Batch 分区方案

        FA2 只沿 Batch 维度分区: P_B ∈ divisors(num_cores)
        """
        partitions = []
        cores = self.arch.num_cores

        for p_b in range(1, cores + 1):
            if cores % p_b == 0:
                partitions.append(p_b)

        return partitions

    def _find_legal_tiles(
        self,
        QS: int,
        KS: int,
        QD: int,
        VD: int,
    ) -> List[Tuple[int, int]]:
        """
        搜索所有能放进 SRAM 的 Tile 大小

        SRAM 布局:
        - Q tile: [q_t, QD] × FP8
        - K tile: [k_t, QD] × FP8
        - V tile: [k_t, VD] × FP8
        - P tile: [q_t, k_t] × BF16 (attention scores)
        - O tile: [q_t, VD] × BF16 (output)
        """
        tiles = []
        cube_m = self.arch.cube_m
        cube_n = self.arch.cube_n
        cube_k = self.arch.cube_k
        max_cube_nk = max(cube_n, cube_k)
        sram_limit = self.arch.effective_sram_bytes
        lane_num = self.arch.lane_num
        align_bytes = self.arch.align_bytes

        def align_row(r: int) -> int:
            return align_up(r, lane_num)

        def align_col(c: int, elem_bytes: int) -> int:
            return align_up(c * elem_bytes, align_bytes)

        # 从大到小搜索 Tile
        q_start = align_up(QS, cube_m)
        k_start = align_up(KS, max_cube_nk)

        for q_t in range(q_start, 0, -cube_m):
            if q_t > QS * 2:
                continue
            align_row_q = align_row(q_t)

            for k_t in range(k_start, 0, -max_cube_nk):
                if k_t > KS * 2:
                    continue
                align_row_k = align_row(k_t)

                # 计算 SRAM 占用
                q_block = align_row_q * align_col(QD, FP8_BYTES)
                k_block = align_row_k * align_col(QD, FP8_BYTES)
                v_block = align_row_k * align_col(VD, FP8_BYTES)
                p_block = align_row_q * align_col(k_t, BF16_BYTES)
                o_block = align_row_q * align_col(VD, BF16_BYTES)

                # 总占用: Q + K + V + 2*P + 4*O
                occupied = q_block + k_block + v_block + 2 * p_block + 4 * o_block

                if occupied > sram_limit:
                    continue

                # Pareto 最优检查
                is_dominated = any(
                    q0 >= q_t and k0 >= k_t
                    for q0, k0 in tiles
                )
                if not is_dominated:
                    tiles.append((q_t, k_t))

        # 如果没找到合法 tile，使用保守默认值
        if not tiles:
            tiles.append((min(64, QS), min(64, KS)))

        return tiles

    def _calc_dram_traffic(
        self,
        QS: int,
        KS: int,
        QD: int,
        VD: int,
        q_t: int,
        k_t: int,
    ) -> int:
        """
        计算 DRAM 流量 (字节)

        流量模型:
        - Q: 加载一次
        - K: 每个 Q tile 重复加载一次
        - V: 每个 Q tile 重复加载一次
        - O: 写回一次
        """
        if QS <= 0 or KS <= 0 or q_t <= 0:
            return 0

        tile_num_q = ceil_div(QS, q_t)

        load_q = QS * QD * FP8_BYTES
        load_k = KS * QD * FP8_BYTES * tile_num_q
        load_v = KS * VD * FP8_BYTES * tile_num_q
        store_o = QS * VD * BF16_BYTES

        return load_q + load_k + load_v + store_o

    def _calc_arch_utilization(
        self,
        b_blk: int,
        QS: int,
        KS: int,
        QD: int,
        VD: int,
    ) -> Tuple[float, float]:
        """
        计算架构利用率和计算时间

        包含:
        - GEMM: QK^T 和 PV
        - Vector: Softmax 操作

        Returns:
            (arch_utilization, compute_time_us)
        """
        if b_blk <= 0:
            return 0.0, 0.0

        # GEMM 部分: QK^T + PV
        gemm_real = QS * KS * (QD + VD)
        gemm_theo = (
            align_up(QS, self.arch.cube_m) *
            align_up(QD, self.arch.cube_k) *
            align_up(KS, self.arch.cube_n) +
            align_up(QS, self.arch.cube_m) *
            align_up(KS, self.arch.cube_k) *
            align_up(VD, self.arch.cube_n)
        )

        # Softmax 向量操作
        vector_theo, vector_real = softmax_theoretical_and_real(
            QS, KS,
            self.arch.lane_num,
            self.arch.eu_num,
            BF16_BYTES,
        )

        # 架构利用率
        total_theo = gemm_theo + vector_theo
        total_real = gemm_real + vector_real
        arch_util = total_real / total_theo if total_theo > 0 else 0.0

        # 计算时间
        macs_per_cycle = self.arch.macs_per_cycle
        freq = self.arch.freq_ghz
        eu_num = self.arch.eu_num

        if macs_per_cycle <= 0 or freq <= 0:
            return arch_util, 0.0

        # GEMM 时间
        gemm_t_us = gemm_theo * b_blk / macs_per_cycle / freq / 1e3

        # Vector 时间
        vector_t_us = vector_theo * b_blk / eu_num / freq / BF16_BYTES / 1e3

        t_us = gemm_t_us + vector_t_us

        return arch_util, t_us

    def _evaluate_partition(
        self,
        P_B: int,
        B: int,
        QS: int,
        KS: int,
        QD: int,
        VD: int,
    ) -> Tuple[float, FA2Result]:
        """
        评估单个分区方案

        Returns:
            (total_time_us, FA2Result)
        """
        # 搜索最佳 tile
        tiles = self._find_legal_tiles(QS, KS, QD, VD)

        min_traffic = float('inf')
        best_tile = tiles[0] if tiles else (min(64, QS), min(64, KS))

        for q_t, k_t in tiles:
            traffic = self._calc_dram_traffic(QS, KS, QD, VD, q_t, k_t)
            if traffic < min_traffic:
                min_traffic = traffic
                best_tile = (q_t, k_t)

        q_t, k_t = best_tile
        b_nom = ceil_div(B, P_B)

        # 遍历所有核心
        total_flops = 0
        total_traffic = 0
        max_time = 0.0
        best_t_comp = 0.0
        best_t_dma = 0.0

        for i_b in range(P_B):
            b_start = i_b * b_nom
            b_blk = max(min(B - b_start, b_nom), 0)

            if b_blk <= 0:
                continue

            # FLOPs: 2 * B * QS * KS * (QD + VD)
            core_flops = 2 * b_blk * QS * KS * (QD + VD)

            # 架构利用率和计算时间
            arch_util, t_comp = self._calc_arch_utilization(b_blk, QS, KS, QD, VD)

            # DRAM 流量和访存时间
            traffic = b_blk * self._calc_dram_traffic(QS, KS, QD, VD, q_t, k_t)
            dma_bw = self.arch.dma_bandwidth_per_core
            t_dma = 1e6 * traffic / dma_bw if dma_bw > 0 else 1e6

            # 计算-搬运重叠
            overlap = self.arch.compute_dma_overlap_rate
            t_total = (min(t_comp, t_dma) * (1 - overlap) +
                       max(t_comp, t_dma))

            # 更新统计
            if t_total > max_time:
                max_time = t_total
                best_t_comp = t_comp
                best_t_dma = t_dma

            total_flops += core_flops
            total_traffic += traffic

        # 计算总体利用率
        if max_time > 0:
            overall_util = total_flops / (
                max_time * 1e3 *
                self.arch.num_cores *
                self.arch.macs_per_cycle *
                self.arch.freq_ghz * 2
            )
        else:
            overall_util = 0.0

        # 架构利用率
        if best_t_comp > 0 and max_time > 0:
            arch_util_avg = best_t_comp / max_time
        else:
            arch_util_avg = 0.0

        result = FA2Result(
            latency_us=max_time,
            compute_time_us=best_t_comp,
            memory_time_us=best_t_dma,
            flops=total_flops,
            dram_traffic_bytes=int(total_traffic),
            arch_utilization=arch_util_avg,
            effective_utilization=overall_util,
            best_tile=best_tile,
            best_partition=P_B,
        )

        return max_time, result

    def evaluate(
        self,
        B: int,
        QS: int,
        KS: int,
        QD: int,
        VD: int,
    ) -> FA2Result:
        """
        评估 Flash Attention 2

        Args:
            B: Batch size (通常是 num_heads)
            QS: Query 序列长度
            KS: Key/Value 序列长度
            QD: Query/Key 维度 (head_dim)
            VD: Value 维度 (通常等于 QD)

        Returns:
            FA2Result: 包含延迟、利用率、最佳配置等
        """
        # 检查缓存
        cache_key = (B, QS, KS, QD, VD)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 特殊情况处理
        if B <= 0 or QS <= 0 or KS <= 0 or QD <= 0 or VD <= 0:
            result = FA2Result(
                latency_us=0.0,
                compute_time_us=0.0,
                memory_time_us=0.0,
                flops=0,
                dram_traffic_bytes=0,
                arch_utilization=0.0,
                effective_utilization=0.0,
                best_tile=(0, 0),
                best_partition=1,
            )
            self._cache[cache_key] = result
            return result

        best_time = float('inf')
        best_result = None

        # 遍历所有分区方案
        for p_b in self._valid_partitions:
            time_us, result = self._evaluate_partition(p_b, B, QS, KS, QD, VD)

            if time_us < best_time and time_us > 0:
                best_time = time_us
                best_result = result

        # 如果没有找到有效结果，使用简单估算
        if best_result is None:
            total_flops = 2 * B * QS * KS * (QD + VD)
            peak_flops = self.arch.flops_per_second
            if peak_flops > 0:
                latency_us = total_flops / (peak_flops * 0.5) * 1e6
            else:
                latency_us = 0.0

            best_result = FA2Result(
                latency_us=latency_us,
                compute_time_us=latency_us,
                memory_time_us=0.0,
                flops=total_flops,
                dram_traffic_bytes=0,
                arch_utilization=0.5,
                effective_utilization=0.5,
                best_tile=(min(64, QS), min(64, KS)),
                best_partition=1,
            )

        self._cache[cache_key] = best_result
        return best_result

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# ==================== 便捷接口 ====================

_fa2_evaluator_cache: Dict[int, FA2Evaluator] = {}


def get_fa2_evaluator(arch: AcceleratorMicroArch) -> FA2Evaluator:
    """获取或创建 FA2 评估器 (缓存单例)"""
    key = id(arch)
    if key not in _fa2_evaluator_cache:
        _fa2_evaluator_cache[key] = FA2Evaluator(arch)
    return _fa2_evaluator_cache[key]


def eval_fa2(
    arch: AcceleratorMicroArch,
    B: int,
    QS: int,
    KS: int,
    QD: int,
    VD: int,
) -> FA2Result:
    """
    快速评估 Flash Attention 2

    Args:
        arch: 硬件微架构配置
        B: Batch size (num_heads)
        QS: Query 序列长度
        KS: Key/Value 序列长度
        QD: Query/Key 维度
        VD: Value 维度

    Returns:
        FA2Result
    """
    evaluator = get_fa2_evaluator(arch)
    return evaluator.evaluate(B, QS, KS, QD, VD)
