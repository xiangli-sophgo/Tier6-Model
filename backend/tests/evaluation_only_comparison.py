#!/usr/bin/env python3
"""
纯算子评估对比（不运行完整模拟）

只对比：创建模型 + 评估所有算子的时间
"""

import sys
import time
from pathlib import Path

tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def test_ds_tpu_evaluation_only():
    """DS_TPU: 纯算子评估"""
    from model.model_factories import model_factory
    from tpu.tpu_factories import tpu_factory
    from performance.analyzer import PerformanceAnalyzer
    from config.deployment_config import DeploymentConfig
    from config.config_loader import load_model_config

    print("\n" + "="*80)
    print("DS_TPU - 纯算子评估")
    print("="*80)

    model_cfg = load_model_config("deepseek-v3.2")
    deploy_cfg = DeploymentConfig(
        batch_size=1536, q_seq_len=1, kv_seq_len=8192,
        tp=1, dp=32, moe_tp=1, ep=32, is_prefill=False,
    )

    # 计时开始
    total_start = time.time()

    # 创建模型
    model_start = time.time()
    model = model_factory.create_model(model_cfg, deploy_cfg.__dict__, "DeepSeek-V3.2")
    model_time = time.time() - model_start

    # 创建 TPU
    tpu = tpu_factory.create_tpu('v1', {'core': 64})

    # 评估（PerformanceAnalyzer 会自动评估所有算子）
    eval_start = time.time()
    analyzer = PerformanceAnalyzer(model, tpu, deploy_cfg, {})
    eval_time = time.time() - eval_start

    total_time = time.time() - total_start

    print(f"\n⏱️  时间分解:")
    print(f"   模型创建: {model_time*1000:.2f}ms")
    print(f"   算子评估: {eval_time*1000:.2f}ms")
    print(f"   总耗时: {total_time*1000:.2f}ms ({total_time:.2f}s)")

    return {
        "model_time_s": model_time,
        "eval_time_s": eval_time,
        "total_time_s": total_time,
        "num_layers": len(model.layers),
    }


def test_tier6_evaluation_only():
    """Tier6: 纯算子评估（61层）"""
    from llm_simulator.layers import MLAAbsorbv32Layer, MoELayer, MLPLayer
    from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator

    print("\n" + "="*80)
    print("Tier6+Model - 纯算子评估 (61层)")
    print("="*80)

    # 配置
    mla_cfg = {
        'hidden_dim': 7168, 'num_heads': 128,
        'qk_nope_dim': 128, 'qk_rope_dim': 64, 'v_head_dim': 128,
        'kv_lora_rank': 512, 'q_lora_rank': 1536,
        'batch_size': 1536, 'seq_len': 1, 'kv_seq_len': 8192,
        'tp': 1, 'comm_protocol': 1,
    }

    dense_mlp_cfg = {
        'hidden_dim': 7168,
        'inter_dim': 18432,
        'batch_size': 1536,
        'seq_len': 1,
        'tp': 1,
        'comm_protocol': 1,
    }

    moe_cfg = {
        'hidden_dim': 7168,
        'inter_dim': 18432,
        'num_experts': 256,
        'num_experts_per_tok': 8,
        'expert_intermediate_size': 2048,
        'batch_size': 1536,
        'seq_len': 1,
        'tp': 1,
        'comm_protocol': 1,
    }

    # 创建评估器
    arch = get_arch_preset("SG2260E")
    gemm_eval = create_gemm_evaluator(arch, fast_mode=False, enable_partition_search=True)

    total_start = time.time()

    # 创建层（61层：3 Dense + 58 MoE，都有 MLA）
    create_start = time.time()
    layers = []

    # 前3层：Dense MLP + MLA
    for i in range(3):
        mla = MLAAbsorbv32Layer(f"mla_{i}", mla_cfg)
        mlp = MLPLayer(f"dense_mlp_{i}", dense_mlp_cfg)
        layers.append((mla, mlp))

    # 后58层：MoE + MLA
    for i in range(3, 61):
        mla = MLAAbsorbv32Layer(f"mla_{i}", mla_cfg)
        moe = MoELayer(f"moe_{i}", moe_cfg)
        layers.append((mla, moe))

    create_time = time.time() - create_start

    # 评估所有层
    eval_start = time.time()
    for i, (attn, ffn) in enumerate(layers):
        # 评估 Attention
        for op in attn.comp_ops:
            if op.operator_type == "MatMulOperator":
                result = gemm_eval.evaluate(
                    G=op.parallel_params.get("G", 1),
                    M=op.parallel_params.get("M", 1),
                    K=op.parallel_params.get("K", 1),
                    N=op.parallel_params.get("N", 1),
                    input_dtype='fp8', output_dtype='bf16',
                    use_multiprocess=True,
                )
                op.elapse = result.latency_us

        # 评估 FFN
        for op in ffn.comp_ops:
            if op.operator_type == "MatMulOperator":
                result = gemm_eval.evaluate(
                    G=op.parallel_params.get("G", 1),
                    M=op.parallel_params.get("M", 1),
                    K=op.parallel_params.get("K", 1),
                    N=op.parallel_params.get("N", 1),
                    input_dtype='fp8', output_dtype='bf16',
                    use_multiprocess=True,
                )
                op.elapse = result.latency_us

        # 每10层报告一次进度
        if (i + 1) % 10 == 0:
            print(f"   已评估 {i+1}/61 层...")

    eval_time = time.time() - eval_start
    total_time = time.time() - total_start

    print(f"\n⏱️  时间分解:")
    print(f"   创建61层: {create_time*1000:.2f}ms")
    print(f"   评估61层: {eval_time*1000:.2f}ms")
    print(f"   总耗时: {total_time*1000:.2f}ms ({total_time:.2f}s)")

    # 打印缓存统计
    gemm_eval.print_cache_stats()

    return {
        "create_time_s": create_time,
        "eval_time_s": eval_time,
        "total_time_s": total_time,
        "num_layers": 61,
    }


def main():
    print("="*80)
    print("🔬 纯算子评估性能对比（不含完整模拟）")
    print("="*80)

    # 测试 DS_TPU
    try:
        ds_result = test_ds_tpu_evaluation_only()
    except Exception as e:
        print(f"\n❌ DS_TPU 失败: {e}")
        import traceback
        traceback.print_exc()
        ds_result = None

    # 测试 Tier6
    try:
        tier6_result = test_tier6_evaluation_only()
    except Exception as e:
        print(f"\n❌ Tier6 失败: {e}")
        import traceback
        traceback.print_exc()
        tier6_result = None

    # 对比
    if ds_result and tier6_result:
        print("\n" + "="*80)
        print("📊 对比总结")
        print("="*80)

        print(f"\n【DS_TPU】")
        print(f"  模型创建: {ds_result['model_time_s']:.2f}s")
        print(f"  算子评估: {ds_result['eval_time_s']:.2f}s")
        print(f"  总耗时: {ds_result['total_time_s']:.2f}s")
        print(f"  层数: {ds_result['num_layers']}")

        print(f"\n【Tier6】")
        print(f"  创建61层: {tier6_result['create_time_s']:.2f}s")
        print(f"  评估61层: {tier6_result['eval_time_s']:.2f}s")
        print(f"  总耗时: {tier6_result['total_time_s']:.2f}s")
        print(f"  层数: {tier6_result['num_layers']}")

        print(f"\n【速度比】")
        speedup = tier6_result['total_time_s'] / ds_result['total_time_s']
        print(f"  {speedup:.2f}x {'(Tier6慢)' if speedup > 1 else '(Tier6快)'}")

        print(f"\n【结论】")
        if speedup > 10:
            print(f"  ⚠️  Tier6 比 DS_TPU 慢 {speedup:.1f}倍")
            print(f"  主要瓶颈在算子评估阶段")
        elif speedup > 1:
            print(f"  ⚠️  Tier6 比 DS_TPU 慢 {(speedup-1)*100:.0f}%")
            print(f"  瓶颈在算子评估阶段")
        else:
            print(f"  ✅ Tier6 比 DS_TPU 快")
            print(f"  瓶颈不在算子评估，而在模拟逻辑其他部分")


if __name__ == "__main__":
    main()
