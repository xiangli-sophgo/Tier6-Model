#!/usr/bin/env python3
"""验证 DS_TPU 的架构参数是否正确设置"""

import sys
from pathlib import Path

ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")
sys.path.insert(0, str(ds_tpu_root))

from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
from performance.evaluate.compute.comp_eval_base import TPUArch

# 按照 aligned_comparison.py 的方式创建架构
unified_config = {
    "num_cores": 64,
    "cube_m": 32,
    "cube_n": 32,
    "cube_k": 32,
    "freq_ghz": 1.2,
}

macs_per_cycle = unified_config["cube_m"] * unified_config["cube_n"]
unified_config["flops"] = 2.0 * unified_config["num_cores"] * macs_per_cycle * unified_config["freq_ghz"] * 1e9

print("配置参数:")
print(f"  flops: {unified_config['flops']/1e12:.2f} TFLOPS")

ds_arch = TPUArch(
    core=unified_config["num_cores"],
    cube_m=unified_config["cube_m"],
    cube_n=unified_config["cube_n"],
    cube_k=unified_config["cube_k"],
    flops=unified_config["flops"],
    dram_bw=273e9,
    sram_size=8*1024*1024,
    lane_num=32, align_bytes=64,
    tpu_gdma_overlap_rate=0.5,
)

print(f"\nDS_TPU 架构反推结果:")
print(f"  freq: {ds_arch.freq}")
print(f"  macs_per_cycle: {ds_arch.macs_per_cycle}")

# 手动计算 partition (1,1,16,4)
def align_up(x, a):
    return ((x + a - 1) // a) * a

M, K, N = 384, 7168, 2048
P_M, P_N, P_K = 1, 16, 4

m_nom = (M + P_M - 1) // P_M  # 384
n_nom = (N + P_N - 1) // P_N  # 128
k_nom = (K + P_K - 1) // P_K  # 1792

theo_macs = align_up(m_nom, 32) * align_up(k_nom, 32) * align_up(n_nom, 32)

print(f"\npartition (1, 1, 16, 4):")
print(f"  单核: M={m_nom}, N={n_nom}, K={k_nom}")
print(f"  theo_macs: {theo_macs:,}")

t_comp = theo_macs / ds_arch.macs_per_cycle / ds_arch.freq / 1e3
print(f"  计算时间: {t_comp:.2f} μs")

# 调用 DS_TPU 的 calc_arch_urate
evaluator = MatmulEval(ds_arch, input_dtype='fp8', output_dtype='bf16')
arch_util, ds_t_comp = evaluator.calc_arch_urate(1, m_nom, n_nom, k_nom)

print(f"\nDS_TPU calc_arch_urate:")
print(f"  返回的计算时间: {ds_t_comp:.2f} μs")

print(f"\n预期: 2.24 μs")
print(f"实际: {ds_t_comp:.2f} μs")
