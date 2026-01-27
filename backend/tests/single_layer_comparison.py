#!/usr/bin/env python3
"""
单层评估性能对比

对比 DS_TPU 和 Tier6+Model 评估单个 Transformer 层的时间
"""

import sys
import time
from pathlib import Path

tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def test_ds_tpu_single_layer():
    """测试 DS_TPU 评估单层的时间"""
    from model.model_factories import model_factory
    from tpu.tpu_factories import tpu_factory
    from performance.analyzer import PerformanceAnalyzer
    from config.deployment_config import DeploymentConfig
    from config.config_loader import load_model_config

    print("\n" + "="*80)
    print("DS_TPU - 评估单层 (Decode 模式)")
    print("="*80)

    # 加载模型配置
    model_cfg = load_model_config("deepseek-v3.2")

    # 部署配置（Decode 模式）
    deploy_cfg = DeploymentConfig(
        batch_size=1536,
        q_seq_len=1,  # Decode: 只处理 1 个新 token
        kv_seq_len=8192,  # KV cache 长度
        tp=1, dp=32, moe_tp=1, ep=32,
        is_prefill=False,
        enable_tp_sp=True,
        comm_protocol=1,
    )

    # 创建模型和 TPU
    print("\n⏱️  创建模型...")
    model_start = time.time()
    model = model_factory.create_model(model_cfg, deploy_cfg.__dict__, "DeepSeek-V3.2")
    model_time = (time.time() - model_start) * 1000

    print(f"   模型创建耗时: {model_time:.2f}ms")
    print(f"   模型层数: {len(model.layers)}")

    tpu = tpu_factory.create_tpu('v1', {'core': 64})

    # 分析性能（会评估所有层）
    print("\n⏱️  性能分析...")
    analysis_start = time.time()
    analyzer = PerformanceAnalyzer(model, tpu, deploy_cfg, {})
    analysis_time = (time.time() - analysis_start) * 1000

    print(f"   分析耗时: {analysis_time:.2f}ms")
    print(f"   平均每层: {analysis_time/len(model.layers):.2f}ms/层")

    perf = analyzer.analysis_summary.get('performance', {})
    print(f"\n📊 性能结果:")
    print(f"   延迟: {perf.get('total_elapse_us', 0):.2f}μs")
    print(f"   吞吐: {perf.get('tps', 0):.2f} tokens/s")
    print(f"   MFU: {perf.get('mfu', 0)*100:.2f}%")

    return {
        "model_time_ms": model_time,
        "analysis_time_ms": analysis_time,
        "total_time_ms": model_time + analysis_time,
        "num_layers": len(model.layers),
        "time_per_layer_ms": analysis_time / len(model.layers),
    }


def test_tier6_single_layer():
    """测试 Tier6+Model 评估单层的时间"""
    from llm_simulator.layers import MLAAbsorbv32Layer, MoELayer
    from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator, FA2Evaluator, AllReduceEval

    print("\n" + "="*80)
    print("Tier6+Model - 评估单层 (Decode 模式)")
    print("="*80)

    # 层配置（DeepSeek V3.2 MoE 层 + MLA Attention）
    layer_cfg = {
        "hidden_dim": 7168,
        "num_heads": 128,
        "qk_nope_dim": 128,
        "qk_rope_dim": 64,
        "v_head_dim": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "batch_size": 1536,
        "seq_len": 1,  # Decode: 只处理 1 个新 token
        "kv_seq_len": 8192,  # KV cache 长度
        "tp": 1,
        "comm_protocol": 1,
    }

    moe_cfg = {
        "hidden_dim": 7168,
        "inter_dim": 18432,  # Dense层的 inter_dim
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "expert_intermediate_size": 2048,  # 专家的 inter_dim
        "batch_size": 1536,
        "seq_len": 1,
        "tp": 1,
        "comm_protocol": 1,
    }

    # 创建层
    print("\n⏱️  创建层...")
    layer_start = time.time()
    mla_layer = MLAAbsorbv32Layer("mla", layer_cfg)
    moe_layer = MoELayer("moe", moe_cfg)
    layer_time = (time.time() - layer_start) * 1000

    print(f"   层创建耗时: {layer_time:.2f}ms")
    print(f"   MLA 算子: {len(mla_layer.comp_ops)} compute + {len(mla_layer.comm_ops)} comm")
    print(f"   MoE 算子: {len(moe_layer.comp_ops)} compute + {len(moe_layer.comm_ops)} comm")

    # 创建评估器
    arch = get_arch_preset("SG2260E")
    gemm_eval = create_gemm_evaluator(arch, fast_mode=False, enable_partition_search=True)
    fa2_eval = FA2Evaluator(arch)
    allreduce_eval = AllReduceEval(arch)

    # 评估 MLA 层
    print("\n⏱️  评估 MLA 层...")
    mla_eval_start = time.time()
    for op in mla_layer.comp_ops:
        if op.operator_type == "MatMulOperator":
            result = gemm_eval.evaluate(
                G=op.parallel_params.get("G", 1),
                M=op.parallel_params.get("M", 1),
                K=op.parallel_params.get("K", 1),
                N=op.parallel_params.get("N", 1),
                input_dtype='fp8',
                output_dtype='bf16',
                use_multiprocess=True,
            )
            op.elapse = result.latency_us
    mla_eval_time = (time.time() - mla_eval_start) * 1000

    print(f"   MLA 评估耗时: {mla_eval_time:.2f}ms")

    # 评估 MoE 层
    print("\n⏱️  评估 MoE 层...")
    moe_eval_start = time.time()
    for op in moe_layer.comp_ops:
        if op.operator_type == "MatMulOperator":
            result = gemm_eval.evaluate(
                G=op.parallel_params.get("G", 1),
                M=op.parallel_params.get("M", 1),
                K=op.parallel_params.get("K", 1),
                N=op.parallel_params.get("N", 1),
                input_dtype='fp8',
                output_dtype='bf16',
                use_multiprocess=True,
            )
            op.elapse = result.latency_us
    moe_eval_time = (time.time() - moe_eval_start) * 1000

    print(f"   MoE 评估耗时: {moe_eval_time:.2f}ms")

    # 打印缓存统计
    gemm_eval.print_cache_stats()

    single_layer_time = mla_eval_time + moe_eval_time
    print(f"\n📊 单层总耗时: {single_layer_time:.2f}ms")

    return {
        "layer_time_ms": layer_time,
        "mla_eval_time_ms": mla_eval_time,
        "moe_eval_time_ms": moe_eval_time,
        "total_time_ms": layer_time + single_layer_time,
        "single_layer_time_ms": single_layer_time,
    }


def main():
    print("="*80)
    print("🔬 单层评估性能对比")
    print("="*80)

    # 测试 DS_TPU
    try:
        ds_result = test_ds_tpu_single_layer()
    except Exception as e:
        print(f"\n❌ DS_TPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        ds_result = None

    # 测试 Tier6
    try:
        tier6_result = test_tier6_single_layer()
    except Exception as e:
        print(f"\n❌ Tier6+Model 测试失败: {e}")
        import traceback
        traceback.print_exc()
        tier6_result = None

    # 对比
    if ds_result and tier6_result:
        print("\n" + "="*80)
        print("📊 性能对比总结")
        print("="*80)

        print(f"\n【DS_TPU】")
        print(f"  模型创建: {ds_result['model_time_ms']:.2f}ms")
        print(f"  性能分析: {ds_result['analysis_time_ms']:.2f}ms ({ds_result['num_layers']}层)")
        print(f"  平均每层: {ds_result['time_per_layer_ms']:.2f}ms/层")
        print(f"  总耗时: {ds_result['total_time_ms']:.2f}ms")

        print(f"\n【Tier6+Model】")
        print(f"  层创建: {tier6_result['layer_time_ms']:.2f}ms")
        print(f"  MLA 评估: {tier6_result['mla_eval_time_ms']:.2f}ms")
        print(f"  MoE 评估: {tier6_result['moe_eval_time_ms']:.2f}ms")
        print(f"  单层总耗时: {tier6_result['single_layer_time_ms']:.2f}ms")

        print(f"\n【对比】")
        speedup = tier6_result['single_layer_time_ms'] / ds_result['time_per_layer_ms']
        print(f"  DS_TPU 每层: {ds_result['time_per_layer_ms']:.2f}ms")
        print(f"  Tier6 单层: {tier6_result['single_layer_time_ms']:.2f}ms")
        print(f"  速度比: {speedup:.2f}x {'(Tier6慢)' if speedup > 1 else '(Tier6快)'}")

        # 如果按 61 层计算
        tier6_61_layers = tier6_result['single_layer_time_ms'] * 61
        print(f"\n  推算 61 层总耗时:")
        print(f"    DS_TPU: {ds_result['analysis_time_ms']:.2f}ms")
        print(f"    Tier6: {tier6_61_layers:.2f}ms")


if __name__ == "__main__":
    main()
