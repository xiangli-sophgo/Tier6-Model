"""
GEMM 评估器对比测试

对比 Tier6+Model/evaluators/gemm_eval.py 与 DS_TPU_1209/performance/evaluate/compute/matmul/matmul_eval.py
确保两者功能一致、结果一致
"""

import sys

# 添加路径
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

# 直接导入 DS_TPU 的模块 (避免包导入问题)
exec(open('/Users/lixiang/Documents/工作/code/DS_TPU_1209/model/dtypes.py').read())

# 手动定义 DS_TPU 需要的类
class TPUConfigBase:
    """TPU配置基类 (从 DS_TPU 复制)"""
    def __init__(self, **kwargs):
        self.core = kwargs.get('core', 32)
        self.flops = kwargs.get('flops', 256.0 * 1024 * 1e9)
        self.dram = kwargs.get('dram', 80.0e9)
        self.dram_bw = kwargs.get('dram_bw', 4000e9 * 0.80)
        self.intra_bw = kwargs.get('intra_bw', 500.0e9)
        self.inter_bw = kwargs.get('inter_bw', 40.0e9)
        self.discount_rate = kwargs.get('discount_rate', 1.0)

        self.tpu_cores = self.core
        self.cube_m = kwargs.get('cube_m', 16)
        self.cube_k = kwargs.get('cube_k', 32)
        self.cube_n = kwargs.get('cube_n', 8)
        self.sram_size = kwargs.get('sram_size', 2 * (1 << 20))
        self.lane_num = kwargs.get('lane_num', 16)
        self.eu_num = kwargs.get('eu_num', 512)
        self.align_bytes = kwargs.get('align_bytes', 32)

        self.dma_bw = self.dram_bw / self.tpu_cores
        self.macs_per_cycle = self.cube_m * self.cube_k * self.cube_n
        self.freq = self.flops / (2.0 * self.core * self.macs_per_cycle * 1e9)
        self.tpu_gdma_overlap_rate = kwargs.get('tpu_gdma_overlap_rate', 0.8)


# DS_TPU MatmulEval 需要的工具函数
import math
from functools import partial
from typing import List, Tuple, Dict, Optional

FP8_BYTES = 1
BF16_BYTES = 2
FP32_BYTES = 4

def ceil_div(x, y):
    return (x + y - 1) // y

def align_up(x, y):
    return ((x + y - 1) // y) * y


class Tile_Info:
    """Tile信息类"""
    def __init__(self, tpu_idx):
        self.tpu_idx = tpu_idx
        self.block_dims = {}
        self.tile_dims = {}

    def add_block(self, **block_dims):
        self.block_dims.update(block_dims)

    def add_tile(self, **tile_dims):
        self.tile_dims.update(tile_dims)

    def add_perf(self, flops, traffic, t_total, comp_elapse, dma_elapse, arch_urate, urate):
        self.flops = flops
        self.traffic = traffic
        self.t_total = t_total
        self.comp_elapse = comp_elapse
        self.dma_elapse = dma_elapse
        self.arch_urate = arch_urate
        self.urate = urate


class Block_Info:
    """Block信息类"""
    def __init__(self, dims, procs):
        self.dims = dims
        self.procs = procs
        self.tiles = []
        self.elapse = 0
        self.urate = 0
        self.traffic = 0
        self.flops = 0
        self.active_cores = 0
        self.order = None
        self.comp_elapse = 0
        self.dma_elapse = 0

    def add_perf(self, flops, traffic, t_total, comp_elapse, dma_elapse, urate, active_cores, order=None):
        self.flops = flops
        self.traffic = traffic
        self.elapse = t_total
        self.comp_elapse = float(comp_elapse)
        self.dma_elapse = float(dma_elapse)
        self.urate = urate
        self.active_cores = active_cores
        if order is not None:
            self.order = order
        if len(self.tiles) > 0:
            self.best_tile = self.tiles[0].tile_dims
        else:
            self.best_tile = None
        try:
            self.best_partition = {'dims': self.dims, 'procs': self.procs}
        except Exception:
            self.best_partition = None

    def add_tile_info(self, info):
        self.tiles.append(info)


class MatmulEval:
    """DS_TPU MatmulEval (简化版，单线程)"""

    def __init__(self, arch):
        self.arch = arch
        self.all_valid_partitions = self._valid_partition()

    def _valid_partition(self):
        blocks = []
        for P_G in range(1, self.arch.tpu_cores + 1):
            if self.arch.tpu_cores % P_G:
                continue
            rem_m = self.arch.tpu_cores // P_G
            for P_M in range(1, rem_m + 1):
                if rem_m % P_M:
                    continue
                rem_n = rem_m // P_M
                for P_N in range(1, rem_n + 1):
                    if rem_n % P_N:
                        continue
                    blocks.append((P_G, P_M, P_N, rem_n // P_N))
        return blocks

    def legal_tiles(self, m_blk, n_blk, k_blk):
        if m_blk * n_blk * k_blk == 0:
            return [(0, 0, 0)]

        all_tiles = []
        cube_m, cube_n, cube_k = self.arch.cube_m, self.arch.cube_n, self.arch.cube_k
        sram_limit = self.arch.sram_size * 0.45
        align_row = lambda r: align_up(r, self.arch.lane_num)
        align_col = lambda c, elem_bytes: align_up(c * elem_bytes, self.arch.align_bytes)

        for m_t in range(align_up(m_blk, cube_m), 0, -cube_m):
            align_row_m = align_row(m_t)
            for n_t in range(align_up(n_blk, cube_n), 0, -cube_n):
                align_col_n = align_col(n_t, BF16_BYTES)
                align_row_n = align_row(n_t)

                avail = sram_limit - align_row_n * align_col_n
                if avail <= 0:
                    continue
                max_k = int(avail / ((align_row_m + align_row_n) * FP8_BYTES))
                if max_k == 0:
                    continue
                align_k = align_up(min(k_blk, max_k), cube_k)
                if align_k < cube_k:
                    k_t = max_k
                else:
                    k_t = (align_k - cube_k) if align_k > max_k else align_k
                    if k_t == 0:
                        continue
                if self._is_pareto_max_tiles(all_tiles, m_t, n_t, k_t):
                    all_tiles.append((m_t, n_t, k_t))
        return all_tiles

    def _is_pareto_max_tiles(self, conds, m_t, n_t, k_t):
        if len(conds) == 0:
            return True
        for m0, n0, k0 in conds:
            if m0 >= m_t and n0 >= n_t and k0 >= k_t:
                return False
        return True

    def dram_traffic(self, loop_order, m_blk, n_blk, k_blk, m_t, n_t, k_t):
        if m_blk * n_blk * k_blk == 0:
            return 0
        tile_num_m = ceil_div(m_blk, m_t)
        tile_num_n = ceil_div(n_blk, n_t)
        tile_num_k = ceil_div(k_blk, k_t)
        if loop_order == 'mnk':
            return (m_blk * k_blk) * FP8_BYTES * tile_num_n + \
                (n_blk * k_blk) * FP8_BYTES * tile_num_m + \
                (m_blk * n_blk) * BF16_BYTES
        elif loop_order == 'nkm':
            return (n_blk * k_blk) * FP8_BYTES + \
                (m_blk * k_blk) * FP8_BYTES * tile_num_n + \
                (m_blk * n_blk) * FP32_BYTES * 2 * (tile_num_k - 1) + \
                (m_blk * n_blk) * BF16_BYTES
        else:
            return (m_blk * k_blk) * FP8_BYTES + \
                (n_blk * k_blk) * FP8_BYTES * tile_num_m + \
                (m_blk * n_blk) * FP32_BYTES * 2 * (tile_num_k - 1) + \
                (m_blk * n_blk) * BF16_BYTES

    def calc_arch_urate(self, g_blk, m_blk, n_blk, k_blk):
        if m_blk * n_blk * k_blk == 0:
            return 0, 0
        real = m_blk * n_blk * k_blk
        theo = align_up(m_blk, self.arch.cube_m) * \
               align_up(k_blk, self.arch.cube_k) * \
               align_up(n_blk, self.arch.cube_n)
        arch_urate = real / theo
        t_us = theo * g_blk / self.arch.macs_per_cycle / self.arch.freq / 1e3
        return arch_urate, t_us

    def evaluate_partition(self, P_G, P_M, P_N, P_K, G, M, N, K):
        g_nom = ceil_div(G, P_G)
        m_nom = ceil_div(M, P_M)
        n_nom = ceil_div(N, P_N)
        k_nom = ceil_div(K, P_K)

        min_traffic = math.inf
        best_tile = None
        best_order = None
        for (m_t, n_t, k_t) in self.legal_tiles(m_nom, n_nom, k_nom):
            for order in ('mnk', 'nkm', 'mkn'):
                traffic = self.dram_traffic(order, m_nom, n_nom, k_nom, m_t, n_t, k_t)
                if traffic < min_traffic:
                    min_traffic = traffic
                    best_tile = (m_t, n_t, k_t)
                    best_order = order

        if best_tile is None:
            return None, math.inf

        m_t, n_t, k_t = best_tile
        total_flops = 0
        total_traffic = 0.0
        max_time = 0
        best_comp_elapse = 0
        best_dma_elapse = 0
        npu = 0
        active_cores = 0
        partition = Block_Info(
            dims={'G': G, 'M': M, 'N': N, 'K': K},
            procs={'P_G': P_G, 'P_M': P_M, 'P_N': P_N, 'P_K': P_K}
        )

        for i_g in range(P_G):
            g_start = i_g * g_nom
            g_blk = max(min(G - g_start, g_nom), 0)
            for i_m in range(P_M):
                m_start = i_m * m_nom
                m_blk = max(min(M - m_start, m_nom), 0)
                for i_n in range(P_N):
                    n_start = i_n * n_nom
                    n_blk = max(min(N - n_start, n_nom), 0)
                    for i_k in range(P_K):
                        k_start = i_k * k_nom
                        k_blk = max(min(K - k_start, k_nom), 0)

                        flops_val = 2 * g_blk * m_blk * n_blk * k_blk
                        arch_urate, comp_elapse = self.calc_arch_urate(g_blk, m_blk, n_blk, k_blk)
                        traffic = g_blk * self.dram_traffic(best_order, m_blk, n_blk, k_blk, m_t, n_t, k_t)
                        dma_elapse = 1e6 * traffic / self.arch.dma_bw
                        t_total = min(comp_elapse, dma_elapse) * (1 - self.arch.tpu_gdma_overlap_rate) + \
                                  max(comp_elapse, dma_elapse)
                        real_util = 0 if t_total == 0 else comp_elapse / t_total * arch_urate
                        if t_total > max_time:
                            best_comp_elapse = comp_elapse
                            best_dma_elapse = dma_elapse
                        max_time = max(max_time, t_total)

                        info = Tile_Info(npu)
                        info.add_block(G=g_blk, M=m_blk, N=n_blk, K=k_blk)
                        info.add_tile(order=best_order, M=m_t, N=n_t, K=k_t)
                        info.add_perf(flops_val, traffic, t_total, comp_elapse, dma_elapse, arch_urate, real_util)
                        partition.add_tile_info(info)
                        if real_util != 0:
                            active_cores += 1
                        total_flops += flops_val
                        total_traffic += traffic
                        npu += 1

        if max_time == 0:
            urate = 0
        else:
            urate = total_flops / (max_time * 1e3 * self.arch.tpu_cores * self.arch.macs_per_cycle * self.arch.freq * 2)

        partition.add_perf(total_flops, total_traffic, max_time, best_comp_elapse,
                          best_dma_elapse, urate, active_cores, best_order)

        return partition, max_time

    def eval_p(self, name, G, M, N, K):
        """评估所有分区并返回最佳分区 (单线程版本)"""
        min_time = math.inf
        best_partition = None

        for partition_config in self.all_valid_partitions:
            P_G, P_M, P_N, P_K = partition_config
            partition, t_total = self.evaluate_partition(P_G, P_M, P_N, P_K, G, M, N, K)
            if partition is not None and t_total < min_time:
                min_time = t_total
                best_partition = partition

        return best_partition


# 导入 Tier6+ 的评估器
from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
from llm_simulator.evaluators.gemm_eval import GEMMEvaluator


def create_matching_configs():
    """创建匹配的配置"""

    # DS_TPU 配置
    ds_config = TPUConfigBase(
        core=64,
        flops=64.0 * 1e12,
        dram_bw=500e9,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        sram_size=16 * (1 << 20),
        lane_num=64,
        align_bytes=128,
        tpu_gdma_overlap_rate=0.8,
    )

    # 从 ds_config 推导频率
    freq_ghz = ds_config.freq

    # Tier6+ 配置 (完全匹配)
    tier6_config = AcceleratorMicroArch(
        num_cores=64,
        cube_m=16,
        cube_k=32,
        cube_n=8,
        freq_ghz=freq_ghz,
        sram_size_bytes=16 * (1 << 20),
        sram_utilization=0.45,
        dram_bandwidth_bytes=500e9,
        lane_num=64,
        align_bytes=128,
        compute_dma_overlap_rate=0.8,
    )

    return ds_config, tier6_config


def compare_valid_partitions(ds_eval, tier6_eval):
    """对比有效分区枚举"""
    print("=" * 60)
    print("1. 对比有效分区枚举")
    print("=" * 60)

    ds_parts = set(ds_eval.all_valid_partitions)
    tier6_parts = set(tier6_eval._valid_partitions)

    print(f"DS_TPU 分区数: {len(ds_parts)}")
    print(f"Tier6+ 分区数: {len(tier6_parts)}")

    if ds_parts == tier6_parts:
        print("✓ 分区枚举完全一致!")
        return True
    else:
        print("✗ 分区枚举不一致!")
        only_in_ds = ds_parts - tier6_parts
        only_in_tier6 = tier6_parts - ds_parts
        if only_in_ds:
            print(f"  仅在 DS_TPU: {list(only_in_ds)[:5]}...")
        if only_in_tier6:
            print(f"  仅在 Tier6+: {list(only_in_tier6)[:5]}...")
        return False


def compare_full_evaluation(ds_eval, tier6_eval, G, M, N, K):
    """对比完整评估"""
    print(f"\n完整评估 GEMM[{G}, {M}, {N}, {K}]:")

    # DS_TPU
    ds_result = ds_eval.eval_p(f"test_{G}_{M}_{N}_{K}", G, M, N, K)

    # Tier6+
    tier6_result = tier6_eval.evaluate(G, M, K, N, input_dtype="fp8", output_dtype="bf16")

    if ds_result:
        print(f"\n  DS_TPU 结果:")
        print(f"    延迟: {ds_result.elapse:.3f} μs")
        print(f"    利用率: {ds_result.urate:.4f}")
        print(f"    计算时间: {ds_result.comp_elapse:.3f} μs")
        print(f"    DMA时间: {ds_result.dma_elapse:.3f} μs")
        print(f"    FLOPS: {ds_result.flops}")
        print(f"    流量: {ds_result.traffic:.0f} bytes")
        print(f"    循环顺序: {ds_result.order}")
    else:
        print("  DS_TPU: 无结果")

    print(f"\n  Tier6+ 结果:")
    print(f"    延迟: {tier6_result.latency_us:.3f} μs")
    print(f"    利用率: {tier6_result.effective_utilization:.4f}")
    print(f"    计算时间: {tier6_result.compute_time_us:.3f} μs")
    print(f"    DMA时间: {tier6_result.memory_time_us:.3f} μs")
    print(f"    FLOPS: {tier6_result.flops}")
    print(f"    流量: {tier6_result.dram_traffic_bytes} bytes")
    print(f"    循环顺序: {tier6_result.best_loop_order}")

    # 对比
    if ds_result:
        latency_diff = abs(ds_result.elapse - tier6_result.latency_us)
        latency_rel_diff = latency_diff / ds_result.elapse * 100 if ds_result.elapse > 0 else 0

        util_diff = abs(ds_result.urate - tier6_result.effective_utilization)

        print(f"\n  对比:")
        print(f"    延迟差异: {latency_diff:.3f} μs ({latency_rel_diff:.2f}%)")
        print(f"    利用率差异: {util_diff:.4f}")

        is_match = latency_rel_diff < 10.0  # 允许 10% 误差
        print(f"    结果: {'✓ 匹配' if is_match else '✗ 不匹配'}")
        return is_match, ds_result, tier6_result

    return False, None, tier6_result


def run_comprehensive_comparison():
    """运行全面的对比测试"""
    print("=" * 60)
    print("GEMM 评估器对比测试")
    print("DS_TPU_1209 vs Tier6+Model")
    print("=" * 60)

    # 创建匹配的配置
    ds_config, tier6_config = create_matching_configs()

    print("\n配置信息:")
    print(f"  核心数: {ds_config.tpu_cores}")
    print(f"  FLOPS: {ds_config.flops / 1e12:.1f} TFLOPS")
    print(f"  频率: {ds_config.freq:.3f} GHz")
    print(f"  DRAM带宽: {ds_config.dram_bw / 1e9:.1f} GB/s")
    print(f"  Cube大小: ({ds_config.cube_m}, {ds_config.cube_k}, {ds_config.cube_n})")
    print(f"  MACs/cycle: {ds_config.macs_per_cycle}")
    print(f"  SRAM: {ds_config.sram_size / (1 << 20):.1f} MB")
    print(f"  重叠率: {ds_config.tpu_gdma_overlap_rate}")

    print(f"\nTier6+ 配置:")
    print(f"  核心数: {tier6_config.num_cores}")
    print(f"  FLOPS: {tier6_config.flops_per_second / 1e12:.1f} TFLOPS")
    print(f"  频率: {tier6_config.freq_ghz:.3f} GHz")
    print(f"  DRAM带宽: {tier6_config.dram_bandwidth_bytes / 1e9:.1f} GB/s")
    print(f"  Cube大小: ({tier6_config.cube_m}, {tier6_config.cube_k}, {tier6_config.cube_n})")
    print(f"  MACs/cycle: {tier6_config.macs_per_cycle}")
    print(f"  SRAM: {tier6_config.sram_size_bytes / (1 << 20):.1f} MB (可用: {tier6_config.effective_sram_bytes / (1 << 20):.1f} MB)")

    # 创建评估器
    ds_eval = MatmulEval(ds_config)
    tier6_eval = GEMMEvaluator(tier6_config)

    # 1. 对比分区枚举
    partition_match = compare_valid_partitions(ds_eval, tier6_eval)

    # 2. 对比 DRAM 流量计算
    print("\n" + "=" * 60)
    print("2. 对比 DRAM 流量计算")
    print("=" * 60)

    test_cases = [
        ('mnk', 256, 256, 256, 64, 64, 64),
        ('nkm', 256, 256, 256, 64, 64, 64),
        ('mkn', 256, 256, 256, 64, 64, 64),
    ]

    traffic_match = True
    for loop_order, m_blk, n_blk, k_blk, m_t, n_t, k_t in test_cases:
        ds_traffic = ds_eval.dram_traffic(loop_order, m_blk, n_blk, k_blk, m_t, n_t, k_t)
        tier6_traffic = tier6_eval._calc_dram_traffic(loop_order, m_blk, n_blk, k_blk, m_t, n_t, k_t, 1, 2)
        match = "✓" if ds_traffic == tier6_traffic else "✗"
        if ds_traffic != tier6_traffic:
            traffic_match = False
        print(f"  {loop_order}: DS={ds_traffic}, Tier6+={tier6_traffic} {match}")

    # 3. 对比架构利用率
    print("\n" + "=" * 60)
    print("3. 对比架构利用率计算")
    print("=" * 60)

    test_dims = [
        (1, 64, 64, 64),
        (1, 256, 256, 256),
        (1, 100, 100, 100),
    ]

    urate_match = True
    for g, m, n, k in test_dims:
        ds_urate, ds_time = ds_eval.calc_arch_urate(g, m, n, k)
        tier6_urate, tier6_time = tier6_eval._calc_arch_utilization(g, m, n, k)
        urate_ok = abs(ds_urate - tier6_urate) < 0.001
        time_ok = abs(ds_time - tier6_time) < 0.1
        if not urate_ok or not time_ok:
            urate_match = False
        print(f"  [{g},{m},{n},{k}]: 利用率 DS={ds_urate:.4f} Tier6+={tier6_urate:.4f} {'✓' if urate_ok else '✗'}")
        print(f"              时间 DS={ds_time:.3f}μs Tier6+={tier6_time:.3f}μs {'✓' if time_ok else '✗'}")

    # 4. 完整评估对比
    print("\n" + "=" * 60)
    print("4. 完整评估对比")
    print("=" * 60)

    test_gemms = [
        (1, 1024, 1024, 1024),
        (1, 512, 4096, 4096),
        (1, 48, 7168, 2048),
        (1, 256, 256, 11008),
        (32, 128, 128, 128),
    ]

    results = []
    for G, M, N, K in test_gemms:
        is_match, _, _ = compare_full_evaluation(ds_eval, tier6_eval, G, M, N, K)
        results.append((G, M, N, K, is_match))

    # 5. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for r in results if r[4])
    total = len(results)

    print(f"\n分区枚举: {'✓' if partition_match else '✗'}")
    print(f"DRAM流量: {'✓' if traffic_match else '✗'}")
    print(f"架构利用率: {'✓' if urate_match else '✗'}")
    print(f"完整评估: {passed}/{total} 通过")

    print("\n详细结果:")
    for G, M, N, K, is_match in results:
        status = "✓" if is_match else "✗"
        print(f"  GEMM[{G}, {M}, {N}, {K}]: {status}")

    if partition_match and traffic_match and urate_match and passed == total:
        print("\n✓ 所有测试通过! GEMM 评估器实现一致。")
        return True
    else:
        print("\n✗ 存在不一致，需要检查。")
        return False


if __name__ == "__main__":
    run_comprehensive_comparison()
