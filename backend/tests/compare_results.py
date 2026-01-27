#!/usr/bin/env python3
"""
对比 Tier6+Model 和 DS_TPU 的评估结果

验证：
1. 算子延迟是否一致
2. 模型总延迟是否一致
3. MFU、吞吐量等指标是否一致
"""

import sys
import json
from pathlib import Path

# 添加路径
tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def test_single_operator_comparison():
    """对比单个算子的评估结果"""
    print("="*80)
    print("单个 GEMM 算子对比")
    print("="*80)

    # DS_TPU 评估
    from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
    from performance.evaluate.compute.comp_eval_base import TPUArch

    ds_arch = TPUArch(
        tpu_cores=64,
        cube_m=32, cube_n=32, cube_k=32,
        sram_size=8*1024*1024,
        lane_num=32, align_bytes=64,
        macs_per_cycle=32*32, freq=1.2e9,
        dma_bw=273*1e9,
        tpu_gdma_overlap_rate=0.5,
    )
    ds_evaluator = MatmulEval(ds_arch, input_dtype='fp8', output_dtype='bf16')

    # Tier6 评估
    from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator

    tier6_arch = get_arch_preset("SG2260E")
    tier6_evaluator = create_gemm_evaluator(
        tier6_arch,
        fast_mode=False,
        enable_partition_search=True
    )

    # 测试用例（相同的 GEMM 形状）
    test_cases = [
        ("Decode MoE Gate", 1, 384, 7168, 2048),
        ("Decode MLA Q_down", 1, 48, 7168, 1536),
        ("Decode MLA Q_up", 1, 6144, 1536, 192),
    ]

    print("\n" + "-"*80)
    print(f"{'算子':<25} {'DS_TPU延迟':<15} {'Tier6延迟':<15} {'差异':<15} {'状态'}")
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

    print("-"*80)

    if all_close:
        print("\n✅ 所有算子延迟差异 <30%")
    else:
        print("\n❌ 部分算子延迟差异 >=30%")

    return all_close


def test_model_level_comparison():
    """对比完整模型的评估结果"""
    print("\n" + "="*80)
    print("完整模型评估对比 (Decode 模式)")
    print("="*80)

    # DS_TPU 评估
    from model.model_factories import model_factory
    from tpu.tpu_factories import tpu_factory
    from performance.analyzer import PerformanceAnalyzer
    from config.deployment_config import DeploymentConfig
    from config.config_loader import load_model_config
    import time

    model_cfg = load_model_config('deepseek-v3.2')
    deploy_cfg = DeploymentConfig(
        batch_size=1536,
        q_seq_len=1,  # Decode
        kv_seq_len=8192,
        tp=1, dp=32, moe_tp=1, ep=32,
        is_prefill=False,
    )

    print("\n【DS_TPU】")
    ds_start = time.time()
    model = model_factory.create_model(model_cfg, deploy_cfg.__dict__, 'DeepSeek-V3.2')
    tpu = tpu_factory.create_tpu('v1', {'core': 64})
    analyzer = PerformanceAnalyzer(model, tpu, deploy_cfg, {})
    ds_time = time.time() - ds_start

    ds_perf = analyzer.analysis_summary.get('performance', {})
    ds_latency = ds_perf.get('total_elapse_us', 0) / 1000  # us -> ms
    ds_tps = ds_perf.get('tps', 0)
    ds_mfu = ds_perf.get('mfu', 0) * 100

    print(f"  评估耗时: {ds_time:.2f}s")
    print(f"  模拟延迟: {ds_latency:.2f}ms")
    print(f"  吞吐量: {ds_tps:.2f} tokens/s")
    print(f"  MFU: {ds_mfu:.2f}%")

    # Tier6 评估（只评估算子，不运行完整模拟）
    print("\n【Tier6+Model - 算子级别评估】")
    from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator

    tier6_arch = get_arch_preset("SG2260E")
    tier6_gemm = create_gemm_evaluator(tier6_arch, fast_mode=False, enable_partition_search=True)

    # 手动评估一个代表性层（MoE层）
    print("\n  评估代表性 MoE 层:")

    # MLA Attention 部分
    mla_ops = [
        ("Q_down", 1, 48, 7168, 1536),
        ("Q_up", 1, 6144, 1536, 192),
        ("KV_down", 1, 48, 7168, 512),
        ("KV_nope_up", 1, 6144, 512, 128),
        ("KV_v_up", 1, 6144, 512, 128),
        ("O_proj", 1, 48, 16384, 7168),
    ]

    mla_total_latency = 0
    for name, G, M, K, N in mla_ops:
        result = tier6_gemm.evaluate(G, M, K, N, 'fp8', 'bf16', use_multiprocess=True)
        mla_total_latency += result.latency_us
        print(f"    {name}: {result.latency_us:.2f}μs")

    # MoE FFN 部分
    moe_ops = [
        ("Routed Gate", 1, 384, 7168, 2048),   # 48*8 experts
        ("Routed Up", 1, 384, 7168, 2048),
        ("Routed Down", 1, 384, 2048, 7168),
        ("Shared Gate", 1, 48, 7168, 2048),
        ("Shared Up", 1, 48, 7168, 2048),
        ("Shared Down", 1, 48, 2048, 7168),
    ]

    moe_total_latency = 0
    for name, G, M, K, N in moe_ops:
        result = tier6_gemm.evaluate(G, M, K, N, 'fp8', 'bf16', use_multiprocess=True)
        moe_total_latency += result.latency_us
        print(f"    {name}: {result.latency_us:.2f}μs")

    # 单层延迟
    layer_latency = (mla_total_latency + moe_total_latency) / 1000  # us -> ms
    print(f"\n  单层总延迟: {layer_latency:.2f}ms")

    # 估算 61 层的总延迟（简化）
    # 实际上前3层是 Dense，后58层是 MoE，这里简化为全部 MoE
    total_layers = 61
    tier6_estimated_latency = layer_latency * total_layers
    tier6_estimated_tps = 1000 / tier6_estimated_latency * 1536  # batch_size

    print(f"  估算 61 层延迟: {tier6_estimated_latency:.2f}ms")
    print(f"  估算吞吐量: {tier6_estimated_tps:.2f} tokens/s")

    # 对比
    print("\n" + "-"*80)
    print("对比结果")
    print("-"*80)

    latency_diff = abs(ds_latency - tier6_estimated_latency) / ds_latency * 100
    tps_diff = abs(ds_tps - tier6_estimated_tps) / ds_tps * 100

    print(f"\n延迟对比:")
    print(f"  DS_TPU:  {ds_latency:.2f}ms")
    print(f"  Tier6:   {tier6_estimated_latency:.2f}ms")
    print(f"  差异:    {latency_diff:.1f}%")

    print(f"\n吞吐量对比:")
    print(f"  DS_TPU:  {ds_tps:.2f} tokens/s")
    print(f"  Tier6:   {tier6_estimated_tps:.2f} tokens/s")
    print(f"  差异:    {tps_diff:.1f}%")

    if latency_diff < 20 and tps_diff < 20:
        print("\n✅ 模型级别评估结果基本一致（差异 <20%）")
        return True
    else:
        print("\n⚠️  模型级别评估结果存在差异（差异 >=20%）")
        return False


def main():
    print("="*80)
    print("🔬 Tier6+Model vs DS_TPU 结果对比")
    print("="*80)

    success1 = test_single_operator_comparison()
    success2 = test_model_level_comparison()

    print("\n" + "="*80)
    if success1 and success2:
        print("✅ 所有对比测试通过！两个系统的评估结果一致")
    else:
        print("⚠️  对比测试发现差异，需要进一步分析")
    print("="*80)


if __name__ == "__main__":
    main()
