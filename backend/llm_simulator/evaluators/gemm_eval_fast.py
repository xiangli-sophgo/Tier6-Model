"""
GEMM 快速评估器 - 禁用 Tile 搜索
用于测试基准性能
"""

import os
from typing import Tuple, Optional
from .gemm_eval import GEMMEvaluator, GEMMResult
from .arch_config import AcceleratorMicroArch
from .utils import ceil_div

# 禁用tile搜索的环境变量
DISABLE_TILE_SEARCH = os.environ.get('GEMM_DISABLE_TILE_SEARCH', '0') == '1'


class FastGEMMEvaluator(GEMMEvaluator):
    """快速GEMM评估器 - 禁用tile和循环顺序搜索"""

    def _find_legal_tiles(
        self,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> list:
        """
        快速模式：直接返回固定tile大小，不搜索

        使用经验值：tile = (cube_m, cube_n, cube_k)
        """
        # 固定使用最小cube大小（最快，但可能不是最优）
        return [(self.arch.cube_m, self.arch.cube_n, self.arch.cube_k)]

    def _evaluate_partition(
        self,
        p_g: int,
        p_m: int,
        p_n: int,
        p_k: int,
        G: int,
        M: int,
        N: int,
        K: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> Tuple[float, GEMMResult]:
        """
        快速评估：使用固定tile和循环顺序
        """
        g_nom = ceil_div(G, p_g) if p_g > 0 else G
        m_nom = ceil_div(M, p_m) if p_m > 0 else M
        n_nom = ceil_div(N, p_n) if p_n > 0 else N
        k_nom = ceil_div(K, p_k) if p_k > 0 else K

        # 🚀 快速模式：固定tile和循环顺序
        best_tile = (self.arch.cube_m, self.arch.cube_n, self.arch.cube_k)
        best_order = 'mnk'  # 固定使用mnk顺序
        m_t, n_t, k_t = best_tile

        # 遍历所有核心，计算最长执行时间
        total_flops = 0
        total_traffic = 0
        max_time = 0.0
        best_t_comp = 0.0
        best_t_dma = 0.0

        for i_g in range(p_g):
            g_start = i_g * g_nom
            g_blk = max(min(G - g_start, g_nom), 0)

            for i_m in range(p_m):
                m_start = i_m * m_nom
                m_blk = max(min(M - m_start, m_nom), 0)

                for i_n in range(p_n):
                    n_start = i_n * n_nom
                    n_blk = max(min(N - n_start, n_nom), 0)

                    for i_k in range(p_k):
                        k_start = i_k * k_nom
                        k_blk = max(min(K - k_start, k_nom), 0)

                        if g_blk <= 0 or m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
                            continue

                        core_flops = 2 * g_blk * m_blk * n_blk * k_blk
                        arch_util, t_comp = self._calc_arch_utilization(g_blk, m_blk, n_blk, k_blk)

                        traffic = g_blk * self._calc_dram_traffic(
                            best_order, m_blk, n_blk, k_blk, m_t, n_t, k_t,
                            input_dtype_bytes, output_dtype_bytes
                        )

                        dma_bw = self.arch.dma_bandwidth_per_core
                        t_dma = 1e6 * traffic / dma_bw if dma_bw > 0 else 0.0

                        overlap = self.arch.compute_dma_overlap_rate
                        t_total = (min(t_comp, t_dma) * (1 - overlap) + max(t_comp, t_dma))

                        if t_total > max_time:
                            max_time = t_total
                            best_t_comp = t_comp
                            best_t_dma = t_dma

                        total_flops += core_flops
                        total_traffic += traffic

        if max_time > 0:
            overall_util = total_flops / (
                max_time * 1e3 *
                self.arch.num_cores *
                self.arch.macs_per_cycle *
                self.arch.freq_ghz * 2
            )
        else:
            overall_util = 0.0

        if best_t_comp > 0 and max_time > 0:
            arch_util_avg = best_t_comp / max_time
        else:
            arch_util_avg = 0.0

        result = GEMMResult(
            latency_us=max_time,
            compute_time_us=best_t_comp,
            memory_time_us=best_t_dma,
            flops=total_flops,
            dram_traffic_bytes=int(total_traffic),
            arch_utilization=arch_util_avg,
            effective_utilization=overall_util,
            best_tile=best_tile,
            best_loop_order=best_order,
            best_partition=(p_g, p_m, p_n, p_k),
        )

        return max_time, result


def create_gemm_evaluator(arch: AcceleratorMicroArch, fast_mode: bool = False, enable_partition_search: bool = True, max_gemm_processes: Optional[int] = None):
    """
    创建GEMM评估器

    Args:
        arch: 硬件架构
        fast_mode: 是否使用快速模式（禁用tile搜索）
        enable_partition_search: 是否启用分区搜索（禁用可极大提升速度）
        max_gemm_processes: GEMM 并行搜索的最大进程数（None 时自动设置）
    """
    if fast_mode or DISABLE_TILE_SEARCH:
        return FastGEMMEvaluator(arch, enable_partition_search=enable_partition_search, max_gemm_processes=max_gemm_processes)
    else:
        return GEMMEvaluator(arch, enable_partition_search=enable_partition_search, max_gemm_processes=max_gemm_processes)
