"""
GEMM 精确评估器

移植自 DS_TPU_1209/gemm.py，核心功能：
1. 多核分块策略搜索 (支持多进程并行)
2. Tile 大小搜索（受 SRAM 约束）
3. 循环顺序优化
4. 架构利用率计算
5. 计算-搬运重叠模型
"""

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count

from .arch_config import AcceleratorMicroArch
from .utils import ceil_div, align_up

# 是否启用多进程搜索 (可通过环境变量禁用)
ENABLE_MULTIPROCESS = os.environ.get('GEMM_DISABLE_MULTIPROCESS', '0') != '1'

# 数据类型字节数
DTYPE_BYTES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'fp8': 1,
    'int8': 1,
}


@dataclass
class GEMMResult:
    """GEMM 评估结果"""
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
    """架构利用率 (0-1)，考虑对齐损失"""

    effective_utilization: float
    """有效利用率 (0-1)，考虑访存瓶颈"""

    best_tile: Tuple[int, int, int]
    """最佳 Tile 大小 (m_t, n_t, k_t)"""

    best_loop_order: str
    """最佳循环顺序 ('mnk', 'nkm', 'mkn')"""

    best_partition: Tuple[int, int, int, int]
    """最佳多核分块 (P_G, P_M, P_N, P_K)"""


class GEMMEvaluator:
    """GEMM 精确评估器"""

    def __init__(self, arch: AcceleratorMicroArch):
        """
        初始化评估器

        Args:
            arch: 硬件微架构配置
        """
        self.arch = arch
        self._valid_partitions = self._compute_valid_partitions()
        self._cache: Dict[Tuple, GEMMResult] = {}

    def _compute_valid_partitions(self) -> List[Tuple[int, int, int, int]]:
        """
        枚举所有合法的多核分块方案

        约束: P_G × P_M × P_N × P_K = num_cores
        """
        partitions = []
        cores = self.arch.num_cores

        for p_g in range(1, cores + 1):
            if cores % p_g != 0:
                continue
            rem_m = cores // p_g

            for p_m in range(1, rem_m + 1):
                if rem_m % p_m != 0:
                    continue
                rem_n = rem_m // p_m

                for p_n in range(1, rem_n + 1):
                    if rem_n % p_n != 0:
                        continue
                    p_k = rem_n // p_n
                    partitions.append((p_g, p_m, p_n, p_k))

        return partitions

    def _find_legal_tiles(
        self,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> List[Tuple[int, int, int]]:
        """
        搜索所有能放进 SRAM 的 Tile 大小

        SRAM 布局:
        - A tile: [m_t, k_t] × input_dtype_bytes
        - B tile: [k_t, n_t] × input_dtype_bytes
        - C tile: [m_t, n_t] × output_dtype_bytes
        """
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return [(0, 0, 0)]

        tiles = []
        cube_m = self.arch.cube_m
        cube_n = self.arch.cube_n
        cube_k = self.arch.cube_k
        sram_limit = self.arch.effective_sram_bytes
        lane_num = self.arch.lane_num
        align_bytes = self.arch.align_bytes

        # 对齐函数
        def align_row(r: int) -> int:
            return align_up(r, lane_num)

        def align_col(c: int, elem_bytes: int) -> int:
            return align_up(c * elem_bytes, align_bytes)

        # 从大到小搜索 Tile (越大越好，数据复用越多)
        m_start = align_up(m_blk, cube_m)
        n_start = align_up(n_blk, cube_n)

        for m_t in range(m_start, 0, -cube_m):
            if m_t > m_blk * 2:  # 跳过过大的 tile
                continue
            align_row_m = align_row(m_t)

            for n_t in range(n_start, 0, -cube_n):
                if n_t > n_blk * 2:
                    continue
                align_col_n = align_col(n_t, output_dtype_bytes)
                align_row_n = align_row(n_t)

                # C tile 必须放得下
                c_tile_bytes = align_row_n * align_col_n
                avail = sram_limit - c_tile_bytes

                if avail <= 0:
                    continue

                # 计算最大 k_t
                bytes_per_k = (align_row_m + align_row_n) * input_dtype_bytes
                if bytes_per_k <= 0:
                    max_k = k_blk
                else:
                    max_k = int(avail / bytes_per_k)

                if max_k <= 0:
                    continue

                # 对齐到 cube_k
                k_t = min(k_blk, max_k)
                if k_t >= cube_k:
                    k_t = (k_t // cube_k) * cube_k

                if k_t <= 0:
                    continue

                # Pareto 最优检查 (去除被支配的 tile)
                is_dominated = any(
                    m0 >= m_t and n0 >= n_t and k0 >= k_t
                    for m0, n0, k0 in tiles
                )
                if not is_dominated:
                    tiles.append((m_t, n_t, k_t))

        # 如果没找到合法 tile，使用最小的 cube 大小
        if not tiles:
            tiles.append((cube_m, cube_n, min(k_blk, cube_k)))

        return tiles

    def _calc_dram_traffic(
        self,
        loop_order: str,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        m_t: int,
        n_t: int,
        k_t: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> int:
        """
        计算 DRAM 流量 (字节)

        不同循环顺序的流量差异:
        - mnk: K 在最内层，A/B 各重复加载 tile_num_n/tile_num_m 次
        - nkm: M 在最内层，B 只加载一次，但 C 需要多次累加
        - mkn: N 在最内层，A 只加载一次，但 C 需要多次累加
        """
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return 0

        if m_t <= 0 or n_t <= 0 or k_t <= 0:
            return 0

        tile_num_m = ceil_div(m_blk, m_t)
        tile_num_n = ceil_div(n_blk, n_t)
        tile_num_k = ceil_div(k_blk, k_t)

        a_size = m_blk * k_blk * input_dtype_bytes
        b_size = n_blk * k_blk * input_dtype_bytes
        c_size = m_blk * n_blk * output_dtype_bytes

        if loop_order == 'mnk':
            # A 重复 tile_num_n 次, B 重复 tile_num_m 次
            return a_size * tile_num_n + b_size * tile_num_m + c_size

        elif loop_order == 'nkm':
            # B 只加载一次, A 重复 tile_num_n 次
            # C 需要 tile_num_k - 1 次累加 (读+写 FP32)
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return b_size + a_size * tile_num_n + partial_sum_traffic + c_size

        else:  # mkn
            # A 只加载一次, B 重复 tile_num_m 次
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return a_size + b_size * tile_num_m + partial_sum_traffic + c_size

    def _calc_arch_utilization(
        self,
        g_blk: int,
        m_blk: int,
        n_blk: int,
        k_blk: int,
    ) -> Tuple[float, float]:
        """
        计算架构利用率和计算时间

        架构利用率 = 实际 MACs / 对齐后理论 MACs

        Returns:
            (arch_utilization, compute_time_us)
        """
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return 0.0, 0.0

        # 实际 MAC 数
        real_macs = m_blk * n_blk * k_blk

        # 对齐后的理论 MAC 数
        theo_macs = (
            align_up(m_blk, self.arch.cube_m) *
            align_up(k_blk, self.arch.cube_k) *
            align_up(n_blk, self.arch.cube_n)
        )

        arch_util = real_macs / theo_macs if theo_macs > 0 else 0.0

        # 计算时间 (微秒)
        # t = theo_macs × g_blk / macs_per_cycle / freq_ghz / 1e3
        macs_per_cycle = self.arch.macs_per_cycle
        freq = self.arch.freq_ghz
        if macs_per_cycle <= 0 or freq <= 0:
            t_us = 0.0
        else:
            t_us = theo_macs * g_blk / macs_per_cycle / freq / 1e3

        return arch_util, t_us

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
        评估单个分块方案

        Returns:
            (total_time_us, GEMMResult)
        """
        # 每核分配的维度
        g_nom = ceil_div(G, p_g) if p_g > 0 else G
        m_nom = ceil_div(M, p_m) if p_m > 0 else M
        n_nom = ceil_div(N, p_n) if p_n > 0 else N
        k_nom = ceil_div(K, p_k) if p_k > 0 else K

        # 搜索最佳 tile 和循环顺序
        tiles = self._find_legal_tiles(m_nom, n_nom, k_nom, input_dtype_bytes, output_dtype_bytes)

        min_traffic = float('inf')
        best_tile = tiles[0] if tiles else (self.arch.cube_m, self.arch.cube_n, self.arch.cube_k)
        best_order = 'mnk'

        for m_t, n_t, k_t in tiles:
            if m_t <= 0:
                continue
            for order in ('mnk', 'nkm', 'mkn'):
                traffic = self._calc_dram_traffic(
                    order, m_nom, n_nom, k_nom, m_t, n_t, k_t,
                    input_dtype_bytes, output_dtype_bytes
                )
                if traffic < min_traffic:
                    min_traffic = traffic
                    best_tile = (m_t, n_t, k_t)
                    best_order = order

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

                        # 计算当前核心的 FLOPs
                        core_flops = 2 * g_blk * m_blk * n_blk * k_blk

                        # 计算架构利用率和计算时间
                        arch_util, t_comp = self._calc_arch_utilization(g_blk, m_blk, n_blk, k_blk)

                        # 计算 DRAM 流量和访存时间
                        traffic = g_blk * self._calc_dram_traffic(
                            best_order, m_blk, n_blk, k_blk, m_t, n_t, k_t,
                            input_dtype_bytes, output_dtype_bytes
                        )

                        dma_bw = self.arch.dma_bandwidth_per_core
                        t_dma = 1e6 * traffic / dma_bw if dma_bw > 0 else 0.0

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
        # 公式: total_flops / (max_time_us * 1e3 * cores * macs_per_cycle * freq * 2)
        # 与 DS_TPU 公式完全一致
        if max_time > 0:
            overall_util = total_flops / (
                max_time * 1e3 *
                self.arch.num_cores *
                self.arch.macs_per_cycle *
                self.arch.freq_ghz * 2
            )
        else:
            overall_util = 0.0

        # 计算架构利用率（基于最慢核心）
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

    def evaluate(
        self,
        G: int,
        M: int,
        K: int,
        N: int,
        input_dtype: str = "bf16",
        output_dtype: str = "bf16",
        use_multiprocess: bool = True,
    ) -> GEMMResult:
        """
        评估 GEMM: C[G, M, N] = A[G, M, K] × B[G, K, N]

        Args:
            G: Batch/Group 维度 (可以是 1)
            M: 输出行数
            K: 累加维度
            N: 输出列数
            input_dtype: 输入数据类型 ('fp8', 'bf16', 'fp16')
            output_dtype: 输出数据类型 ('bf16', 'fp32')
            use_multiprocess: 是否使用多进程并行搜索 (默认 True)

        Returns:
            GEMMResult: 包含延迟、利用率、最佳配置等
        """
        # 检查缓存
        cache_key = (G, M, K, N, input_dtype, output_dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]

        input_bytes = DTYPE_BYTES.get(input_dtype, 2)
        output_bytes = DTYPE_BYTES.get(output_dtype, 2)

        # 特殊情况处理
        if G <= 0 or M <= 0 or K <= 0 or N <= 0:
            result = GEMMResult(
                latency_us=0.0,
                compute_time_us=0.0,
                memory_time_us=0.0,
                flops=0,
                dram_traffic_bytes=0,
                arch_utilization=0.0,
                effective_utilization=0.0,
                best_tile=(0, 0, 0),
                best_loop_order='mnk',
                best_partition=(1, 1, 1, 1),
            )
            self._cache[cache_key] = result
            return result

        best_time = float('inf')
        best_result_dict = None

        # 判断是否使用多进程
        use_mp = use_multiprocess and ENABLE_MULTIPROCESS and len(self._valid_partitions) > 1

        if use_mp:
            # 多进程并行搜索
            best_result_dict = self._evaluate_parallel(G, M, N, K, input_bytes, output_bytes)
        else:
            # 串行搜索
            for partition in self._valid_partitions:
                p_g, p_m, p_n, p_k = partition

                time_us, result = self._evaluate_partition(
                    p_g, p_m, p_n, p_k,
                    G, M, N, K,
                    input_bytes, output_bytes,
                )

                if time_us < best_time and time_us > 0:
                    best_time = time_us
                    best_result_dict = {
                        'latency_us': result.latency_us,
                        'compute_time_us': result.compute_time_us,
                        'memory_time_us': result.memory_time_us,
                        'flops': result.flops,
                        'dram_traffic_bytes': result.dram_traffic_bytes,
                        'arch_utilization': result.arch_utilization,
                        'effective_utilization': result.effective_utilization,
                        'best_tile': result.best_tile,
                        'best_loop_order': result.best_loop_order,
                        'best_partition': result.best_partition,
                    }

        # 如果没有找到有效结果，返回一个基于简单估算的结果
        if best_result_dict is None:
            total_flops = 2 * G * M * N * K
            # 简单估算：假设 50% 利用率
            peak_flops = self.arch.flops_per_second
            if peak_flops > 0:
                latency_us = total_flops / (peak_flops * 0.5) * 1e6
            else:
                latency_us = 0.0

            best_result = GEMMResult(
                latency_us=latency_us,
                compute_time_us=latency_us,
                memory_time_us=0.0,
                flops=total_flops,
                dram_traffic_bytes=0,
                arch_utilization=0.5,
                effective_utilization=0.5,
                best_tile=(self.arch.cube_m, self.arch.cube_n, self.arch.cube_k),
                best_loop_order='mnk',
                best_partition=(1, 1, 1, self.arch.num_cores),
            )
        else:
            best_result = GEMMResult(
                latency_us=best_result_dict['latency_us'],
                compute_time_us=best_result_dict['compute_time_us'],
                memory_time_us=best_result_dict['memory_time_us'],
                flops=best_result_dict['flops'],
                dram_traffic_bytes=best_result_dict['dram_traffic_bytes'],
                arch_utilization=best_result_dict['arch_utilization'],
                effective_utilization=best_result_dict['effective_utilization'],
                best_tile=best_result_dict['best_tile'],
                best_loop_order=best_result_dict['best_loop_order'],
                best_partition=best_result_dict['best_partition'],
            )

        self._cache[cache_key] = best_result
        return best_result

    def _evaluate_parallel(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        input_bytes: int,
        output_bytes: int,
    ) -> Optional[dict]:
        """
        使用多进程并行评估所有分块方案

        Returns:
            最佳结果的字典，如果无有效结果则返回 None
        """
        # 准备架构参数 (用于 pickle)
        arch_params = {
            'cube_m': self.arch.cube_m,
            'cube_n': self.arch.cube_n,
            'cube_k': self.arch.cube_k,
            'effective_sram_bytes': self.arch.effective_sram_bytes,
            'lane_num': self.arch.lane_num,
            'align_bytes': self.arch.align_bytes,
            'macs_per_cycle': self.arch.macs_per_cycle,
            'freq_ghz': self.arch.freq_ghz,
            'dma_bandwidth_per_core': self.arch.dma_bandwidth_per_core,
            'compute_dma_overlap_rate': self.arch.compute_dma_overlap_rate,
            'num_cores': self.arch.num_cores,
        }

        # 准备任务列表
        tasks = [
            (p_g, p_m, p_n, p_k, G, M, N, K, input_bytes, output_bytes, arch_params)
            for p_g, p_m, p_n, p_k in self._valid_partitions
        ]

        # 使用多进程池
        num_processes = min(len(tasks), cpu_count())

        try:
            with Pool(processes=num_processes) as pool:
                results = pool.map(_evaluate_partition_worker, tasks)
        except Exception:
            # 如果多进程失败，回退到串行
            results = [_evaluate_partition_worker(task) for task in tasks]

        # 找最优结果
        best_time = float('inf')
        best_result_dict = None

        for time_us, result_dict in results:
            if time_us < best_time and time_us > 0:
                best_time = time_us
                best_result_dict = result_dict

        return best_result_dict

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# ==================== 多进程辅助函数 ====================

def _evaluate_partition_worker(args):
    """
    多进程 worker 函数

    必须是模块级函数才能被 pickle
    """
    (p_g, p_m, p_n, p_k, G, M, N, K, input_bytes, output_bytes, arch_params) = args

    # 重建简化的架构对象用于计算
    arch = SimpleNamespace(**arch_params)

    # 创建临时评估器
    evaluator = _PartitionEvaluator(arch)
    return evaluator.evaluate_partition(
        p_g, p_m, p_n, p_k,
        G, M, N, K,
        input_bytes, output_bytes,
    )


class _PartitionEvaluator:
    """
    轻量级分区评估器，用于多进程
    """

    def __init__(self, arch):
        self.arch = arch

    def _find_legal_tiles(
        self,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> List[Tuple[int, int, int]]:
        """搜索所有能放进 SRAM 的 Tile 大小"""
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return [(0, 0, 0)]

        tiles = []
        cube_m = self.arch.cube_m
        cube_n = self.arch.cube_n
        cube_k = self.arch.cube_k
        sram_limit = self.arch.effective_sram_bytes
        lane_num = self.arch.lane_num
        align_bytes = self.arch.align_bytes

        def align_row(r: int) -> int:
            return align_up(r, lane_num)

        def align_col(c: int, elem_bytes: int) -> int:
            return align_up(c * elem_bytes, align_bytes)

        m_start = align_up(m_blk, cube_m)
        n_start = align_up(n_blk, cube_n)

        for m_t in range(m_start, 0, -cube_m):
            if m_t > m_blk * 2:
                continue
            align_row_m = align_row(m_t)

            for n_t in range(n_start, 0, -cube_n):
                if n_t > n_blk * 2:
                    continue
                align_col_n = align_col(n_t, output_dtype_bytes)
                align_row_n = align_row(n_t)

                c_tile_bytes = align_row_n * align_col_n
                avail = sram_limit - c_tile_bytes

                if avail <= 0:
                    continue

                bytes_per_k = (align_row_m + align_row_n) * input_dtype_bytes
                if bytes_per_k <= 0:
                    max_k = k_blk
                else:
                    max_k = int(avail / bytes_per_k)

                if max_k <= 0:
                    continue

                k_t = min(k_blk, max_k)
                if k_t >= cube_k:
                    k_t = (k_t // cube_k) * cube_k

                if k_t <= 0:
                    continue

                is_dominated = any(
                    m0 >= m_t and n0 >= n_t and k0 >= k_t
                    for m0, n0, k0 in tiles
                )
                if not is_dominated:
                    tiles.append((m_t, n_t, k_t))

        if not tiles:
            tiles.append((cube_m, cube_n, min(k_blk, cube_k)))

        return tiles

    def _calc_dram_traffic(
        self,
        loop_order: str,
        m_blk: int,
        n_blk: int,
        k_blk: int,
        m_t: int,
        n_t: int,
        k_t: int,
        input_dtype_bytes: int,
        output_dtype_bytes: int,
    ) -> int:
        """计算 DRAM 流量"""
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return 0
        if m_t <= 0 or n_t <= 0 or k_t <= 0:
            return 0

        tile_num_m = ceil_div(m_blk, m_t)
        tile_num_n = ceil_div(n_blk, n_t)
        tile_num_k = ceil_div(k_blk, k_t)

        a_size = m_blk * k_blk * input_dtype_bytes
        b_size = n_blk * k_blk * input_dtype_bytes
        c_size = m_blk * n_blk * output_dtype_bytes

        if loop_order == 'mnk':
            return a_size * tile_num_n + b_size * tile_num_m + c_size
        elif loop_order == 'nkm':
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return b_size + a_size * tile_num_n + partial_sum_traffic + c_size
        else:  # mkn
            fp32_bytes = 4
            partial_sum_traffic = m_blk * n_blk * fp32_bytes * 2 * max(0, tile_num_k - 1)
            return a_size + b_size * tile_num_m + partial_sum_traffic + c_size

    def _calc_arch_utilization(
        self,
        g_blk: int,
        m_blk: int,
        n_blk: int,
        k_blk: int,
    ) -> Tuple[float, float]:
        """计算架构利用率和计算时间"""
        if m_blk <= 0 or n_blk <= 0 or k_blk <= 0:
            return 0.0, 0.0

        real_macs = m_blk * n_blk * k_blk
        theo_macs = (
            align_up(m_blk, self.arch.cube_m) *
            align_up(k_blk, self.arch.cube_k) *
            align_up(n_blk, self.arch.cube_n)
        )

        arch_util = real_macs / theo_macs if theo_macs > 0 else 0.0

        macs_per_cycle = self.arch.macs_per_cycle
        freq = self.arch.freq_ghz
        if macs_per_cycle <= 0 or freq <= 0:
            t_us = 0.0
        else:
            t_us = theo_macs * g_blk / macs_per_cycle / freq / 1e3

        return arch_util, t_us

    def evaluate_partition(
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
    ) -> Tuple[float, dict]:
        """评估单个分块方案，返回 (time, result_dict)"""
        g_nom = ceil_div(G, p_g) if p_g > 0 else G
        m_nom = ceil_div(M, p_m) if p_m > 0 else M
        n_nom = ceil_div(N, p_n) if p_n > 0 else N
        k_nom = ceil_div(K, p_k) if p_k > 0 else K

        tiles = self._find_legal_tiles(m_nom, n_nom, k_nom, input_dtype_bytes, output_dtype_bytes)

        min_traffic = float('inf')
        best_tile = tiles[0] if tiles else (self.arch.cube_m, self.arch.cube_n, self.arch.cube_k)
        best_order = 'mnk'

        for m_t, n_t, k_t in tiles:
            if m_t <= 0:
                continue
            for order in ('mnk', 'nkm', 'mkn'):
                traffic = self._calc_dram_traffic(
                    order, m_nom, n_nom, k_nom, m_t, n_t, k_t,
                    input_dtype_bytes, output_dtype_bytes
                )
                if traffic < min_traffic:
                    min_traffic = traffic
                    best_tile = (m_t, n_t, k_t)
                    best_order = order

        m_t, n_t, k_t = best_tile

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
                        t_total = (min(t_comp, t_dma) * (1 - overlap) +
                                   max(t_comp, t_dma))

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

        result_dict = {
            'latency_us': max_time,
            'compute_time_us': best_t_comp,
            'memory_time_us': best_t_dma,
            'flops': total_flops,
            'dram_traffic_bytes': int(total_traffic),
            'arch_utilization': arch_util_avg,
            'effective_utilization': overall_util,
            'best_tile': best_tile,
            'best_loop_order': best_order,
            'best_partition': (p_g, p_m, p_n, p_k),
        }

        return max_time, result_dict


# ==================== 便捷接口 ====================

_evaluator_cache: Dict[int, GEMMEvaluator] = {}


def get_gemm_evaluator(arch: AcceleratorMicroArch) -> GEMMEvaluator:
    """获取或创建 GEMM 评估器 (缓存单例)"""
    key = id(arch)
    if key not in _evaluator_cache:
        _evaluator_cache[key] = GEMMEvaluator(arch)
    return _evaluator_cache[key]


def eval_gemm(
    arch: AcceleratorMicroArch,
    G: int,
    M: int,
    K: int,
    N: int,
    input_dtype: str = "bf16",
    output_dtype: str = "bf16",
) -> GEMMResult:
    """
    快速评估 GEMM

    Args:
        arch: 硬件微架构配置
        G, M, K, N: GEMM 维度
        input_dtype: 输入类型
        output_dtype: 输出类型

    Returns:
        GEMMResult
    """
    evaluator = get_gemm_evaluator(arch)
    return evaluator.evaluate(G, M, K, N, input_dtype, output_dtype)
