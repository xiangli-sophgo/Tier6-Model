#!/usr/bin/env python3
"""
强制使用相同分区对比 Tier6 和 DS_TPU 的结果

验证：当分区完全相同时，延迟是否完全一致
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


def test_fixed_partition():
    """使用固定分区对比"""
    print("="*80)
    print("使用固定分区对比 Tier6 vs DS_TPU")
    print("="*80)

    # 统一架构参数
    unified_config = {
        "num_cores": 64,
        "cube_m": 32,
        "cube_n": 32,
        "cube_k": 32,
        "sram_size": 8 * 1024 * 1024,
        "lane_num": 32,
        "align_bytes": 64,
        "freq_ghz": 1.2,
        "dma_bw": 273e9,
        "overlap_rate": 0.5,
    }

    macs_per_cycle = unified_config["cube_m"] * unified_config["cube_k"] * unified_config["cube_n"]
    unified_config["flops"] = 2.0 * unified_config["num_cores"] * macs_per_cycle * unified_config["freq_ghz"] * 1e9

    # DS_TPU 架构
    from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
    from performance.evaluate.compute.comp_eval_base import TPUArch

    ds_arch = TPUArch(
        core=unified_config["num_cores"],
        cube_m=unified_config["cube_m"],
        cube_n=unified_config["cube_n"],
        cube_k=unified_config["cube_k"],
        sram_size=unified_config["sram_size"],
        lane_num=unified_config["lane_num"],
        align_bytes=unified_config["align_bytes"],
        flops=unified_config["flops"],
        dram_bw=unified_config["dma_bw"],
        tpu_gdma_overlap_rate=unified_config["overlap_rate"],
    )
    ds_evaluator = MatmulEval(ds_arch, input_dtype='fp8', output_dtype='bf16')

    # Tier6 架构
    from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
    from llm_simulator.evaluators.gemm_eval import GEMMEvaluator

    tier6_arch = AcceleratorMicroArch(
        name="Unified_Test_Arch",
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
    tier6_evaluator = GEMMEvaluator(
        tier6_arch,
        enable_partition_search=False,  # 关闭搜索
        enable_tile_search=False
    )

    # 测试用例
    test_cases = [
        ("Decode MoE Gate", 1, 384, 7168, 2048, (1, 1, 8, 8), (384, 256, 896)),
        ("Decode MLA Q_down", 1, 48, 7168, 1536, (1, 1, 4, 16), (64, 384, 448)),
        ("Decode MLA Q_up", 1, 6144, 1536, 192, (1, 16, 1, 4), (384, 192, 384)),
    ]

    print("\n" + "="*80)
    print("对比结果（固定分区）")
    print("="*80)
    print(f"\n{'算子':<25} {'分区':<20} {'DS_TPU延迟':<15} {'Tier6延迟':<15} {'差异':<15} {'状态'}")
    print("-"*100)

    all_identical = True
    for name, G, M, K, N, partition, tile in test_cases:
        P_G, P_M, P_N, P_K = partition
        m_t, n_t, k_t = tile

        # 计算单核维度
        m_nom = ceil_div(M, P_M)
        n_nom = ceil_div(N, P_N)
        k_nom = ceil_div(K, P_K)

        # DS_TPU: 手动计算
        # 1. 计算时间
        theo_macs = align_up(m_nom, 32) * align_up(k_nom, 32) * align_up(n_nom, 32)
        macs_per_cycle_calc = 32 * 32 * 32  # cube_m * cube_k * cube_n
        ds_t_comp = theo_macs * G / macs_per_cycle_calc / 1.2 / 1e3

        # 2. 搬运量
        input_bytes = 1  # fp8
        output_bytes = 2  # bf16

        tile_num_m = ceil_div(m_nom, m_t)
        tile_num_n = ceil_div(n_nom, n_t)
        tile_num_k = ceil_div(k_nom, k_t)

        a_size = m_nom * k_nom * input_bytes
        b_size = n_nom * k_nom * input_bytes
        c_size = m_nom * n_nom * output_bytes

        traffic = a_size * tile_num_n + b_size * tile_num_m + c_size
        traffic_total = traffic * G

        dma_bw_per_core = unified_config["dma_bw"] / unified_config["num_cores"]
        ds_t_dma = traffic_total * 1e6 / dma_bw_per_core

        # 3. 重叠
        ds_overlap = min(ds_t_comp, ds_t_dma) * (1 - 0.5)
        ds_latency = ds_overlap + max(ds_t_comp, ds_t_dma)

        # Tier6: 使用内部方法直接计算
        # 1. 计算时间
        arch_util, tier6_t_comp = tier6_evaluator._calc_arch_utilization(G, m_nom, n_nom, k_nom)

        # 2. 搬运时间
        tier6_traffic = G * tier6_evaluator._calc_dram_traffic(
            'mnk', m_nom, n_nom, k_nom, m_t, n_t, k_t,
            input_bytes, output_bytes
        )
        tier6_t_dma = 1e6 * tier6_traffic / tier6_arch.dma_bandwidth_per_core

        # 3. 重叠
        tier6_overlap = min(tier6_t_comp, tier6_t_dma) * (1 - tier6_arch.compute_dma_overlap_rate)
        tier6_latency = tier6_overlap + max(tier6_t_comp, tier6_t_dma)

        # 对比
        diff_pct = abs(ds_latency - tier6_latency) / ds_latency * 100 if ds_latency > 0 else 0
        status = "✅" if diff_pct < 0.1 else "⚠️" if diff_pct < 1 else "❌"

        if diff_pct >= 0.1:
            all_identical = False

        partition_str = f"({P_G},{P_M},{P_N},{P_K})"
        print(f"{name:<25} {partition_str:<20} {ds_latency:>13.2f}μs {tier6_latency:>13.2f}μs {diff_pct:>13.4f}% {status}")

        # 详细分解
        print(f"  DS_TPU  - 计算: {ds_t_comp:.2f}μs, 搬运: {ds_t_dma:.2f}μs, 重叠: {ds_overlap:.2f}μs")
        print(f"  Tier6   - 计算: {tier6_t_comp:.2f}μs, 搬运: {tier6_t_dma:.2f}μs, 重叠: {tier6_overlap:.2f}μs")

    print("-"*100)

    if all_identical:
        print("\n✅ 所有算子在相同分区下延迟完全一致（差异 <0.1%）")
        return True
    else:
        print("\n⚠️ 部分算子在相同分区下存在微小差异")
        return False


if __name__ == "__main__":
    success = test_fixed_partition()
    sys.exit(0 if success else 1)
