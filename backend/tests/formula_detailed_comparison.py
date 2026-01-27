#!/usr/bin/env python3
"""
详细对比 Tier6 和 DS_TPU 的计算公式

逐步验证：
1. 搬运量计算
2. 计算时间公式
3. 重叠模型
4. 精度（数据类型字节数）
"""

import sys
from pathlib import Path

tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def align_up(x, a):
    return ((x + a - 1) // a) * a


def ceil_div(a, b):
    return (a + b - 1) // b


def main():
    print("="*80)
    print("详细公式对比")
    print("="*80)

    # 统一参数
    G, M, K, N = 1, 384, 7168, 2048
    cube_m, cube_n, cube_k = 32, 32, 32
    freq_ghz = 1.2
    dma_bw = 273e9
    num_cores = 64

    # DS_TPU 选择的 partition 和 tile
    P_M, P_N, P_K = 1, 16, 4
    m_t, n_t, k_t = 384, 448, 512

    # 单核维度
    m_nom = ceil_div(M, P_M)
    n_nom = ceil_div(N, P_N)
    k_nom = ceil_div(K, P_K)

    print(f"\n【配置】")
    print(f"形状: G={G}, M={M}, K={K}, N={N}")
    print(f"Partition: P_M={P_M}, P_N={P_N}, P_K={P_K}")
    print(f"单核维度: m={m_nom}, n={n_nom}, k={k_nom}")
    print(f"Tile: m_t={m_t}, n_t={n_t}, k_t={k_t}")

    # ============================================================
    # 1. 精度（字节数）
    # ============================================================
    print(f"\n{'='*80}")
    print("1. 数据类型字节数")
    print("="*80)

    # Tier6
    from llm_simulator.evaluators.gemm_eval import DTYPE_BYTES
    tier6_fp8 = DTYPE_BYTES.get('fp8', 1)
    tier6_bf16 = DTYPE_BYTES.get('bf16', 2)

    # DS_TPU
    from performance.evaluate.compute.comp_eval_base import bytes_per_element
    ds_fp8 = bytes_per_element('fp8')
    ds_bf16 = bytes_per_element('bfloat16')

    print(f"Tier6:  fp8={tier6_fp8}, bf16={tier6_bf16}")
    print(f"DS_TPU: fp8={ds_fp8}, bf16={ds_bf16}")
    print(f"✅ 一致" if tier6_fp8 == ds_fp8 and tier6_bf16 == ds_bf16 else "❌ 不一致")

    input_bytes = tier6_fp8
    output_bytes = tier6_bf16

    # ============================================================
    # 2. 搬运量计算
    # ============================================================
    print(f"\n{'='*80}")
    print("2. 搬运量计算 (mnk 循环顺序)")
    print("="*80)

    # Tier6 公式
    tile_num_m = ceil_div(m_nom, m_t)
    tile_num_n = ceil_div(n_nom, n_t)
    tile_num_k = ceil_div(k_nom, k_t)

    a_size = m_nom * k_nom * input_bytes
    b_size = n_nom * k_nom * input_bytes
    c_size = m_nom * n_nom * output_bytes

    tier6_traffic = a_size * tile_num_n + b_size * tile_num_m + c_size
    tier6_traffic_total = tier6_traffic * G

    print(f"\nTier6 (单核):")
    print(f"  tile_num: m={tile_num_m}, n={tile_num_n}, k={tile_num_k}")
    print(f"  A: {m_nom}×{k_nom}×{input_bytes} = {a_size:,} bytes")
    print(f"  B: {n_nom}×{k_nom}×{input_bytes} = {b_size:,} bytes")
    print(f"  C: {m_nom}×{n_nom}×{output_bytes} = {c_size:,} bytes")
    print(f"  Traffic (mnk): A×{tile_num_n} + B×{tile_num_m} + C")
    print(f"              = {a_size}×{tile_num_n} + {b_size}×{tile_num_m} + {c_size}")
    print(f"              = {tier6_traffic_total:,} bytes")

    # DS_TPU 公式
    from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
    from performance.evaluate.compute.comp_eval_base import TPUArch

    ds_arch = TPUArch(
        core=num_cores,
        cube_m=cube_m, cube_n=cube_n, cube_k=cube_k,
        flops=2.0 * num_cores * cube_m * cube_n * freq_ghz * 1e9,
        dram_bw=dma_bw,
        sram_size=8*1024*1024,
        lane_num=32, align_bytes=64,
        tpu_gdma_overlap_rate=0.5,
    )
    ds_evaluator = MatmulEval(ds_arch, input_dtype='fp8', output_dtype='bf16')
    ds_traffic = ds_evaluator.dram_traffic('mnk', m_nom, n_nom, k_nom, m_t, n_t, k_t)
    ds_traffic_total = ds_traffic * G

    print(f"\nDS_TPU (单核):")
    print(f"  Traffic: {ds_traffic_total:,} bytes")

    traffic_diff = abs(tier6_traffic_total - ds_traffic_total) / ds_traffic_total * 100
    print(f"\n对比:")
    print(f"  Tier6:  {tier6_traffic_total:,} bytes")
    print(f"  DS_TPU: {ds_traffic_total:,} bytes")
    print(f"  差异: {traffic_diff:.2f}%")
    print(f"  {'✅ 一致' if traffic_diff < 1 else '❌ 不一致'}")

    # ============================================================
    # 3. 计算时间公式
    # ============================================================
    print(f"\n{'='*80}")
    print("3. 计算时间公式")
    print("="*80)

    # 对齐后的 MACs
    theo_macs = align_up(m_nom, cube_m) * align_up(k_nom, cube_k) * align_up(n_nom, cube_n)
    real_macs = m_nom * k_nom * n_nom
    arch_util = real_macs / theo_macs

    print(f"\n单核 MACs:")
    print(f"  real_macs: {real_macs:,}")
    print(f"  theo_macs: {theo_macs:,}")
    print(f"  arch_util: {arch_util*100:.2f}%")

    # Tier6 公式
    macs_per_cycle = cube_m * cube_n
    tier6_t_comp = theo_macs * G / macs_per_cycle / freq_ghz / 1e3

    print(f"\nTier6 公式:")
    print(f"  t_comp = theo_macs × G / macs_per_cycle / freq_ghz / 1e3")
    print(f"         = {theo_macs} × {G} / {macs_per_cycle} / {freq_ghz} / 1000")
    print(f"         = {tier6_t_comp:.2f} μs")

    # DS_TPU 公式
    ds_arch_util, ds_t_comp = ds_evaluator.calc_arch_urate(G, m_nom, n_nom, k_nom)

    print(f"\nDS_TPU 公式:")
    print(f"  arch_util: {ds_arch_util*100:.2f}%")
    print(f"  t_comp: {ds_t_comp:.2f} μs")

    comp_diff = abs(tier6_t_comp - ds_t_comp) / ds_t_comp * 100
    print(f"\n对比:")
    print(f"  Tier6:  {tier6_t_comp:.2f} μs")
    print(f"  DS_TPU: {ds_t_comp:.2f} μs")
    print(f"  差异: {comp_diff:.2f}%")
    print(f"  {'✅ 一致' if comp_diff < 1 else '❌ 不一致'}")

    # ============================================================
    # 4. 搬运时间
    # ============================================================
    print(f"\n{'='*80}")
    print("4. 搬运时间")
    print("="*80)

    # Tier6: 使用每核带宽
    dma_bw_per_core = dma_bw / num_cores
    tier6_t_dma = tier6_traffic_total * 1e6 / dma_bw_per_core

    print(f"\nTier6:")
    print(f"  dma_bw_per_core: {dma_bw_per_core/1e9:.2f} GB/s")
    print(f"  t_dma = traffic × 1e6 / dma_bw_per_core")
    print(f"        = {tier6_traffic_total:,} × 1e6 / {dma_bw_per_core}")
    print(f"        = {tier6_t_dma:.2f} μs")

    # DS_TPU: 也是使用每核带宽（dma_bw）
    ds_t_dma = ds_traffic_total * 1e6 / ds_arch.dma_bw

    print(f"\nDS_TPU:")
    print(f"  dma_bw (per core): {ds_arch.dma_bw/1e9:.2f} GB/s")
    print(f"  t_dma: {ds_t_dma:.2f} μs")

    dma_diff = abs(tier6_t_dma - ds_t_dma) / ds_t_dma * 100
    print(f"\n对比:")
    print(f"  Tier6:  {tier6_t_dma:.2f} μs")
    print(f"  DS_TPU: {ds_t_dma:.2f} μs")
    print(f"  差异: {dma_diff:.2f}%")
    print(f"  {'✅ 一致' if dma_diff < 1 else '❌ 不一致'}")

    # ============================================================
    # 5. 重叠模型
    # ============================================================
    print(f"\n{'='*80}")
    print("5. 重叠模型")
    print("="*80)

    overlap_rate = 0.5

    # Tier6
    tier6_overlap = min(tier6_t_comp, tier6_t_dma) * (1 - overlap_rate)
    tier6_t_total = tier6_overlap + max(tier6_t_comp, tier6_t_dma)

    print(f"\nTier6:")
    print(f"  overlap = min(t_comp, t_dma) × (1 - overlap_rate)")
    print(f"          = min({tier6_t_comp:.2f}, {tier6_t_dma:.2f}) × (1 - {overlap_rate})")
    print(f"          = {tier6_overlap:.2f} μs")
    print(f"  t_total = overlap + max(t_comp, t_dma)")
    print(f"          = {tier6_overlap:.2f} + {max(tier6_t_comp, tier6_t_dma):.2f}")
    print(f"          = {tier6_t_total:.2f} μs")

    # DS_TPU
    ds_overlap = min(ds_t_comp, ds_t_dma) * (1 - overlap_rate)
    ds_t_total = ds_overlap + max(ds_t_comp, ds_t_dma)

    print(f"\nDS_TPU:")
    print(f"  overlap: {ds_overlap:.2f} μs")
    print(f"  t_total: {ds_t_total:.2f} μs")

    total_diff = abs(tier6_t_total - ds_t_total) / ds_t_total * 100
    print(f"\n对比:")
    print(f"  Tier6:  {tier6_t_total:.2f} μs")
    print(f"  DS_TPU: {ds_t_total:.2f} μs")
    print(f"  差异: {total_diff:.2f}%")
    print(f"  {'✅ 一致' if total_diff < 1 else '❌ 不一致'}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n{'='*80}")
    print("总结")
    print("="*80)

    checks = [
        ("数据类型字节数", tier6_fp8 == ds_fp8 and tier6_bf16 == ds_bf16),
        ("搬运量计算", traffic_diff < 1),
        ("计算时间公式", comp_diff < 1),
        ("搬运时间", dma_diff < 1),
        ("总延迟", total_diff < 1),
    ]

    for name, passed in checks:
        print(f"  {name}: {'✅' if passed else '❌'}")

    if all(p for _, p in checks):
        print(f"\n✅ 所有公式一致！")
    else:
        print(f"\n❌ 存在公式差异！")


if __name__ == "__main__":
    main()
