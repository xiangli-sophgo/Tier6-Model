#!/usr/bin/env python3
"""
使用相同架构参数对比 Tier6 和 DS_TPU 的评估结果

验证：
1. 使用完全相同的架构配置
2. 评估相同的 GEMM 算子
3. 对比延迟是否一致
"""

import sys
from pathlib import Path

# 添加路径
tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def test_aligned_comparison():
    """使用相同架构参数对比"""
    print("="*80)
    print("使用相同架构参数对比 Tier6 vs DS_TPU")
    print("="*80)

    # ====================
    # 统一架构参数
    # ====================
    unified_config = {
        "num_cores": 64,
        "cube_m": 32,
        "cube_n": 32,
        "cube_k": 32,
        "sram_size": 8 * 1024 * 1024,  # 8 MB
        "lane_num": 32,
        "align_bytes": 64,
        "freq_ghz": 1.2,  # 1.2 GHz (for Tier6)
        "dma_bw": 273e9,  # 273 GB/s
        "overlap_rate": 0.5,
    }

    # DS_TPU 需要 flops 参数来反推 freq
    # flops = 2 * cores * macs_per_cycle * freq
    # macs_per_cycle = cube_m * cube_k * cube_n (不是 cube_m * cube_n!)
    macs_per_cycle = unified_config["cube_m"] * unified_config["cube_k"] * unified_config["cube_n"]
    unified_config["flops"] = 2.0 * unified_config["num_cores"] * macs_per_cycle * unified_config["freq_ghz"] * 1e9

    print("\n【统一架构参数】")
    print(f"  num_cores: {unified_config['num_cores']}")
    print(f"  cube: {unified_config['cube_m']} × {unified_config['cube_n']} × {unified_config['cube_k']}")
    print(f"  sram_size: {unified_config['sram_size']/1024/1024:.0f} MB")
    print(f"  lane_num: {unified_config['lane_num']}")
    print(f"  align_bytes: {unified_config['align_bytes']}")
    print(f"  freq: {unified_config['freq_ghz']:.2f} GHz")
    print(f"  dma_bw: {unified_config['dma_bw']/1e9:.2f} GB/s")
    print(f"  overlap_rate: {unified_config['overlap_rate']}")
    print(f"  flops: {unified_config['flops']/1e12:.2f} TFLOPS")

    # ====================
    # DS_TPU 评估
    # ====================
    print("\n" + "="*80)
    print("【DS_TPU】")
    print("="*80)

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
        flops=unified_config["flops"],  # DS_TPU 从 flops 反推 freq
        dram_bw=unified_config["dma_bw"],
        tpu_gdma_overlap_rate=unified_config["overlap_rate"],
    )
    ds_evaluator = MatmulEval(ds_arch, input_dtype='fp8', output_dtype='bf16')

    # ====================
    # Tier6 评估
    # ====================
    print("\n" + "="*80)
    print("【Tier6】")
    print("="*80)

    from llm_simulator.evaluators.arch_config import AcceleratorMicroArch
    from llm_simulator.evaluators.gemm_eval import GEMMEvaluator

    tier6_arch = AcceleratorMicroArch(
        name="Unified_Test_Arch",
        num_cores=unified_config["num_cores"],
        cube_m=unified_config["cube_m"],
        cube_n=unified_config["cube_n"],
        cube_k=unified_config["cube_k"],
        sram_size_bytes=unified_config["sram_size"],
        sram_utilization=0.45,  # Tier6 默认值
        lane_num=unified_config["lane_num"],
        align_bytes=unified_config["align_bytes"],
        freq_ghz=unified_config["freq_ghz"],
        dram_bandwidth_bytes=unified_config["dma_bw"],
        compute_dma_overlap_rate=unified_config["overlap_rate"],
    )
    tier6_evaluator = GEMMEvaluator(
        tier6_arch,
        enable_partition_search=True,
        enable_tile_search=True
    )

    # ====================
    # 测试用例
    # ====================
    test_cases = [
        ("Decode MoE Gate", 1, 384, 7168, 2048),
        ("Decode MLA Q_down", 1, 48, 7168, 1536),
        ("Decode MLA Q_up", 1, 6144, 1536, 192),
    ]

    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    print(f"\n{'算子':<25} {'DS_TPU延迟':<15} {'Tier6延迟':<15} {'差异':<15} {'状态'}")
    print("-"*80)

    all_close = True
    for name, G, M, K, N in test_cases:
        # DS_TPU
        ds_result = ds_evaluator.eval_p(name, G, M, K, N)
        ds_latency = ds_result.elapse if ds_result else 0

        # Tier6
        tier6_result = tier6_evaluator.evaluate(
            G, M, K, N,
            input_dtype='fp8',
            output_dtype='bf16',
            use_multiprocess=True
        )
        tier6_latency = tier6_result.latency_us

        # 对比
        diff_pct = abs(ds_latency - tier6_latency) / ds_latency * 100 if ds_latency > 0 else 0
        status = "✅" if diff_pct < 10 else "⚠️" if diff_pct < 30 else "❌"

        if diff_pct >= 30:
            all_close = False

        print(f"{name:<25} {ds_latency:>13.2f}μs {tier6_latency:>13.2f}μs {diff_pct:>13.1f}% {status}")

        # 详细分解
        overlap_time_ds = min(ds_result.comp_elapse, ds_result.dma_elapse) * ds_arch.tpu_gdma_overlap_rate
        overlap_time_tier6 = min(tier6_result.compute_time_us, tier6_result.memory_time_us) * tier6_arch.compute_dma_overlap_rate

        print(f"  DS_TPU  - 计算: {ds_result.comp_elapse:.2f}μs, "
              f"搬运: {ds_result.dma_elapse:.2f}μs, "
              f"重叠: {overlap_time_ds:.2f}μs, "
              f"利用率: {ds_result.urate*100:.1f}%")
        print(f"            partition: {ds_result.best_partition['procs']}, tile: {ds_result.best_tile}")

        print(f"  Tier6   - 计算: {tier6_result.compute_time_us:.2f}μs, "
              f"搬运: {tier6_result.memory_time_us:.2f}μs, "
              f"重叠: {overlap_time_tier6:.2f}μs, "
              f"利用率: {tier6_result.arch_utilization*100:.1f}%")
        print(f"            partition: {tier6_result.best_partition}, tile: {tier6_result.best_tile}")

    print("-"*80)

    if all_close:
        print("\n✅ 所有算子延迟差异 <30%")
        return True
    else:
        print("\n❌ 部分算子延迟差异 >=30%")
        return False


if __name__ == "__main__":
    success = test_aligned_comparison()
    sys.exit(0 if success else 1)
