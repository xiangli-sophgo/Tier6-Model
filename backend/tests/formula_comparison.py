#!/usr/bin/env python3
"""
对比两个系统的计算公式细节
"""

import sys
from pathlib import Path

tier6_backend = Path(__file__).parent.parent
sys.path.insert(0, str(tier6_backend))

from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
from llm_simulator.evaluators.gemm_eval import GEMMEvaluator

def ceil_div(a, b):
    return (a + b - 1) // b

def align_up(x, a):
    return ceil_div(x, a) * a


# 统一架构参数
unified_config = {
    "num_cores": 64,
    "cube_m": 32,
    "cube_n": 32,
    "cube_k": 32,
    "sram_size": 8 * 1024 * 1024,
    "lane_num": 32,
    "align_bytes": 64,
    "freq_ghz": 1.2,  # GHz
    "dma_bw": 273e9,  # bytes/s
    "overlap_rate": 0.5,
}

# 测试算子
G, M, K, N = 1, 384, 7168, 2048

print("="*80)
print("手动计算对比")
print("="*80)

# 创建 Tier6 架构
tier6_arch = AcceleratorMicroArch(
    name="Test",
    num_cores=unified_config["num_cores"],
    cube_m=unified_config["cube_m"],
    cube_n=unified_config["cube_n"],
    cube_k=unified_config["cube_k"],
    sram_size_bytes=unified_config["sram_size"],
    sram_utilization=0.45,
    lane_num=unified_config["lane_num"],
    align_bytes=unified_config["align_bytes"],
    freq_ghz=unified_config["freq_ghz"],
    dram_bandwidth_bytes=unified_config["dma_bw"],
    compute_dma_overlap_rate=unified_config["overlap_rate"],
)

print(f"\n架构参数:")
print(f"  cube: {tier6_arch.cube_m} × {tier6_arch.cube_n} × {tier6_arch.cube_k}")
print(f"  macs_per_cycle: {tier6_arch.macs_per_cycle}")
print(f"  freq_ghz: {tier6_arch.freq_ghz}")
print(f"  dram_bw: {tier6_arch.dram_bandwidth_bytes/1e9:.2f} GB/s")
print(f"  overlap_rate: {tier6_arch.compute_dma_overlap_rate}")

# 手动计算（假设 partition = (1,1,16,2), tile = (384,448,1024)）
P_M, P_N, P_K = 1, 16, 2
m_nom = ceil_div(M, P_M)
n_nom = ceil_div(N, P_N)
k_nom = ceil_div(K, P_K)

print(f"\n假设分区: P_M={P_M}, P_N={P_N}, P_K={P_K}")
print(f"每核维度: M={m_nom}, N={n_nom}, K={k_nom}")

# 理论 MACs
theo_macs = align_up(m_nom, tier6_arch.cube_m) * \
            align_up(k_nom, tier6_arch.cube_k) * \
            align_up(n_nom, tier6_arch.cube_n)

real_macs = m_nom * n_nom * k_nom
arch_util = real_macs / theo_macs

print(f"\nMACs:")
print(f"  real_macs: {real_macs:,}")
print(f"  theo_macs: {theo_macs:,}")
print(f"  arch_util: {arch_util*100:.1f}%")

# Tier6 公式
t_us_tier6 = theo_macs * G / tier6_arch.macs_per_cycle / tier6_arch.freq_ghz / 1e3
print(f"\nTier6 公式 (theo / macs / freq_ghz / 1e3):")
print(f"  = {theo_macs} / {tier6_arch.macs_per_cycle} / {tier6_arch.freq_ghz} / 1000")
print(f"  = {t_us_tier6:.2f} μs")

# DS_TPU 公式 (freq 单位是 Hz)
freq_hz = tier6_arch.freq_ghz * 1e9
t_us_ds = theo_macs * G / tier6_arch.macs_per_cycle / freq_hz / 1e3
print(f"\nDS_TPU 公式 (theo / macs / freq_hz / 1e3):")
print(f"  = {theo_macs} / {tier6_arch.macs_per_cycle} / {freq_hz} / 1000")
print(f"  = {t_us_ds:.2f} μs")

# 正确公式
t_us_correct = theo_macs * G / tier6_arch.macs_per_cycle / freq_hz * 1e6
print(f"\n正确公式 (theo / macs / freq_hz * 1e6):")
print(f"  = {theo_macs} / {tier6_arch.macs_per_cycle} / {freq_hz} * 1e6")
print(f"  = {t_us_correct:.2f} μs")

print(f"\nDS_TPU 实际结果: 43.01 μs (计算时间)")
print(f"差异分析:")
print(f"  Tier6 公式结果: {t_us_tier6:.2f} μs (相差 {t_us_tier6/43.01:.2f}x)")
print(f"  正确公式结果: {t_us_correct:.2f} μs (相差 {t_us_correct/43.01:.2f}x)")

# 搬运时间
input_bytes = M * K + K * N  # FP8
output_bytes = M * N * 2      # BF16
dram_traffic = input_bytes + output_bytes

t_dma_tier6 = dram_traffic / tier6_arch.dram_bandwidth_bytes * 1e6
print(f"\n搬运时间:")
print(f"  DRAM 流量: {dram_traffic:,} bytes")
print(f"  搬运时间: {t_dma_tier6:.2f} μs")
print(f"  DS_TPU 实际: 11.96 μs")
