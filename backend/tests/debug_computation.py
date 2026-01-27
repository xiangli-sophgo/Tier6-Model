#!/usr/bin/env python3
"""调试计算时间公式"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接调用 _calc_arch_utilization
def align_up(x, a):
    return ((x + a - 1) // a) * a

def ceil_div(a, b):
    return (a + b - 1) // b

# partition (1,1,8,8)
M, K, N = 384, 7168, 2048
P_M, P_N, P_K = 1, 8, 8

m_blk = ceil_div(M, P_M)  # 384
n_blk = ceil_div(N, P_N)  # 256
k_blk = ceil_div(K, P_K)  # 896
g_blk = 1

cube_m, cube_n, cube_k = 32, 32, 32

print("单核维度:")
print(f"  g_blk={g_blk}, m_blk={m_blk}, n_blk={n_blk}, k_blk={k_blk}")

# 计算 theo_macs
theo_macs = align_up(m_blk, cube_m) * align_up(k_blk, cube_k) * align_up(n_blk, cube_n)
print(f"\ntheo_macs 计算:")
print(f"  align_up({m_blk}, {cube_m}) = {align_up(m_blk, cube_m)}")
print(f"  align_up({k_blk}, {cube_k}) = {align_up(k_blk, cube_k)}")
print(f"  align_up({n_blk}, {cube_n}) = {align_up(n_blk, cube_n)}")
print(f"  theo_macs = {align_up(m_blk, cube_m)} × {align_up(k_blk, cube_k)} × {align_up(n_blk, cube_n)}")
print(f"            = {theo_macs:,}")

# 计算时间
macs_per_cycle = cube_m * cube_n
freq_ghz = 1.2

print(f"\n计算时间公式:")
print(f"  macs_per_cycle = {cube_m} × {cube_n} = {macs_per_cycle}")
print(f"  freq_ghz = {freq_ghz}")

t_us = theo_macs * g_blk / macs_per_cycle / freq_ghz / 1e3

print(f"\n  t_us = theo_macs × g_blk / macs_per_cycle / freq_ghz / 1e3")
print(f"       = {theo_macs} × {g_blk} / {macs_per_cycle} / {freq_ghz} / 1000")
print(f"       = {theo_macs * g_blk} / {macs_per_cycle} / {freq_ghz} / 1000")
print(f"       = {theo_macs * g_blk / macs_per_cycle} / {freq_ghz} / 1000")
print(f"       = {theo_macs * g_blk / macs_per_cycle / freq_ghz} / 1000")
print(f"       = {t_us:.2f} μs")

print(f"\n预期: 71.68 μs")
print(f"实际: {t_us:.2f} μs")
print(f"{'✅ 正确' if abs(t_us - 71.68) < 1 else '❌ 错误'}")

# 现在调用实际的评估器
print(f"\n{'='*80}")
print("调用实际评估器:")
print("="*80)

from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
from llm_simulator.evaluators.gemm_eval import GEMMEvaluator

arch = AcceleratorMicroArch(
    name="Test",
    num_cores=64,
    cube_m=32, cube_n=32, cube_k=32,
    sram_size_bytes=8*1024*1024,
    sram_utilization=0.45,
    lane_num=32, align_bytes=64,
    freq_ghz=1.2,
    dram_bandwidth_bytes=273e9,
    compute_dma_overlap_rate=0.5,
)

print(f"\n架构参数:")
print(f"  cube_m={arch.cube_m}, cube_n={arch.cube_n}, cube_k={arch.cube_k}")
print(f"  macs_per_cycle={arch.macs_per_cycle}")
print(f"  freq_ghz={arch.freq_ghz}")

# 直接调用 _calc_arch_utilization
evaluator = GEMMEvaluator(arch, enable_partition_search=False)
arch_util, t_comp = evaluator._calc_arch_utilization(g_blk, m_blk, n_blk, k_blk)

print(f"\n_calc_arch_utilization 返回:")
print(f"  arch_util: {arch_util:.4f}")
print(f"  t_comp: {t_comp:.2f} μs")
print(f"\n预期 t_comp: 71.68 μs")
print(f"{'✅ 正确' if abs(t_comp - 71.68) < 1 else '❌ 错误'}")
