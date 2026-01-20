#!/usr/bin/env python3
"""
Tier6+ 新架构 vs DS_TPU 对比测试

验证相同配置下两者的分析结果是否一致
"""

import sys
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/DS_TPU_1209')

from llm_simulator.models.deepseek import DeepSeekModel
from llm_simulator.analyzer import PerformanceAnalyzer
from llm_simulator.evaluators import get_arch_preset


def test_single_matmul_comparison():
    """对比单个 GEMM 算子的评估结果"""
    print("=" * 80)
    print("单 GEMM 算子对比测试")
    print("=" * 80)

    # Tier6+ 评估
    from llm_simulator.evaluators import GEMMEvaluator, get_arch_preset
    arch = get_arch_preset('SG2260E')
    tier6_eval = GEMMEvaluator(arch)

    # 测试用例: Dense MLP Gate Projection
    # DeepSeek V3: hidden_dim=7168, inter_dim=18432, batch*seq=1536
    G, M, K, N = 1, 1536, 7168, 18432
    tier6_result = tier6_eval.evaluate(G, M, K, N, 'bf16', 'bf16')

    print(f"\nTier6+ GEMM 结果 (G={G}, M={M}, K={K}, N={N}):")
    print(f"  延迟: {tier6_result.latency_us:.3f} us")
    print(f"  计算时间: {tier6_result.compute_time_us:.3f} us")
    print(f"  访存时间: {tier6_result.memory_time_us:.3f} us")
    print(f"  利用率: {tier6_result.effective_utilization:.4f}")
    print(f"  Best Tile: {tier6_result.best_tile}")
    print(f"  Best Partition: {tier6_result.best_partition}")

    # DS_TPU 评估
    try:
        sys.path.insert(0, '/Users/lixiang/Documents/工作/code/DS_TPU_1209')
        from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
        from config.tpu_config import TPUV1

        tpu_config = TPUV1(core=64)
        ds_eval = MatmulEval(tpu_config)

        # DS_TPU 使用 Operator 对象
        class MockOperator:
            def __init__(self):
                self.parallel_params = {'G': G, 'M': M, 'K': K, 'N': N}
                self.elapse = 0
                self.comp_elapse = 0
                self.dma_elapse = 0
                self.dram_traffic = 0
                self.urate = 0
                self.best_tile = None
                self.best_partition = None

        op = MockOperator()
        ds_eval.evaluate(op)

        print(f"\nDS_TPU GEMM 结果:")
        print(f"  延迟: {op.elapse:.3f} us")
        print(f"  计算时间: {op.comp_elapse:.3f} us")
        print(f"  访存时间: {op.dma_elapse:.3f} us")
        print(f"  利用率: {op.urate:.4f}")
        print(f"  Best Tile: {op.best_tile}")
        print(f"  Best Partition: {op.best_partition}")

        # 对比
        print(f"\n对比:")
        latency_diff = abs(tier6_result.latency_us - op.elapse) / op.elapse * 100
        urate_diff = abs(tier6_result.effective_utilization - op.urate)
        print(f"  延迟差异: {latency_diff:.2f}%")
        print(f"  利用率差异: {urate_diff:.4f}")

        return latency_diff < 1.0  # 误差 < 1%

    except ImportError as e:
        print(f"\n无法导入 DS_TPU: {e}")
        print("跳过 DS_TPU 对比")
        return True


def test_single_allreduce_comparison():
    """对比单个 AllReduce 的评估结果"""
    print("\n" + "=" * 80)
    print("单 AllReduce 对比测试")
    print("=" * 80)

    # 测试用例: TP=8, 通信 2MB 数据
    tp = 8
    comm_bytes = 2 * 1024 * 1024  # 2MB
    comm_protocol = 1

    # Tier6+ 评估
    from llm_simulator.evaluators import AllReduceEval, get_arch_preset
    arch = get_arch_preset('SG2260E')
    tier6_eval = AllReduceEval(arch)
    tier6_result = tier6_eval.evaluate(tp, comm_bytes, comm_protocol)

    print(f"\nTier6+ AllReduce 结果 (tp={tp}, bytes={comm_bytes}):")
    print(f"  延迟: {tier6_result.latency_us:.3f} us")

    # DS_TPU 评估
    try:
        sys.path.insert(0, '/Users/lixiang/Documents/工作/code/DS_TPU_1209')
        from performance.evaluate.communication.allreduce_eval import AllReduceEval as DS_AllReduceEval
        from config.tpu_config import TPUV1

        tpu_config = TPUV1(core=64)
        ds_eval = DS_AllReduceEval(tpu_config)
        ds_latency, _ = ds_eval.evaluate_raw(tp, comm_bytes, comm_protocol)

        print(f"\nDS_TPU AllReduce 结果:")
        print(f"  延迟: {ds_latency:.3f} us")

        # 对比
        print(f"\n对比:")
        latency_diff = abs(tier6_result.latency_us - ds_latency) / ds_latency * 100
        print(f"  延迟差异: {latency_diff:.2f}%")

        return latency_diff < 1.0

    except ImportError as e:
        print(f"\n无法导入 DS_TPU: {e}")
        return True


def test_model_structure_comparison():
    """对比模型结构"""
    print("\n" + "=" * 80)
    print("模型结构对比")
    print("=" * 80)

    # Tier6+ 模型
    tier6_model = DeepSeekModel(
        name="deepseek-v3",
        config={
            'hidden_dim': 7168,
            'inter_dim': 18432,
            'vocab_size': 151936,
            'n_layers': 61,
            'n_dense_layers': 3,
            'n_moe_layers': 58,
            'num_heads': 128,
            'head_dim': 128,
            'kv_lora_rank': 512,
            'q_lora_rank': 1536,
            'num_experts': 256,
            'num_activated_experts': 8,
            'num_shared_experts': 1,
            'expert_inter_dim': 2048,
            'batch_size': 1536,
            'seq_len': 1,
            'tp': 1,
            'moe_tp': 1,
            'ep': 32,
            'comm_protocol': 1,
            'is_prefill': False,
        }
    )

    print(f"\nTier6+ 模型结构:")
    print(f"  层数: {len(tier6_model.layers)}")
    print(f"  层数映射: {tier6_model.layer_counts}")
    print(f"  算子类型: {tier6_model.operator_types}")

    # 统计算子
    tier6_ops = {}
    for op_type, ops in tier6_model.operator_map.items():
        tier6_ops[op_type] = len(ops)
    print(f"  算子数量: {tier6_ops}")

    # DS_TPU 模型
    try:
        sys.path.insert(0, '/Users/lixiang/Documents/工作/code/DS_TPU_1209')
        from model.model_factories import model_factory
        from config.deployment_config import DeploymentConfig

        deploy_cfg = DeploymentConfig(
            batch_size=1536,
            q_seq_len=1,
            kv_seq_len=1,
            tp=1,
            dp=32,
            moe_tp=1,
            ep=32,
            is_prefill=False,
            comm_protocol=1,
        )

        ds_model_cfg = {
            'name': 'DeepSeek-V3',
            'version': 'v3.2',
            'hidden_dim': 7168,
            'inter_dim': 18432,
            'vocab_size': 151936,
            'n_layers': 61,
            'n_dense_layers': 3,
            'n_moe_layers': 58,
            'n_heads': 128,
            'head_dim': 128,
            'kv_lora_rank': 512,
            'q_lora_rank': 1536,
            'n_routed_experts': 256,
            'n_activated_experts': 8,
            'n_shared_experts': 1,
        }

        ds_model = model_factory.create_model(ds_model_cfg, deploy_cfg.__dict__, "deepseek-v3")

        print(f"\nDS_TPU 模型结构:")
        print(f"  层数: {len(ds_model.layers)}")

        # 统计算子
        ds_ops = {}
        for op_type, ops in ds_model.operator_map.items():
            ds_ops[op_type] = len(ops)
        print(f"  算子数量: {ds_ops}")

        return True

    except ImportError as e:
        print(f"\n无法导入 DS_TPU: {e}")
        return True


def test_full_analysis_comparison():
    """对比完整分析结果"""
    print("\n" + "=" * 80)
    print("完整分析结果对比")
    print("=" * 80)

    # 使用简化配置进行测试
    batch_size = 64
    seq_len = 1

    # Tier6+ 分析
    tier6_model = DeepSeekModel(
        name="deepseek-test",
        config={
            'hidden_dim': 7168,
            'inter_dim': 18432,
            'vocab_size': 151936,
            'n_layers': 4,  # 简化
            'n_dense_layers': 1,
            'n_moe_layers': 3,
            'num_heads': 128,
            'head_dim': 128,
            'kv_lora_rank': 512,
            'q_lora_rank': 1536,
            'num_experts': 256,
            'num_activated_experts': 8,
            'num_shared_experts': 1,
            'expert_inter_dim': 2048,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'tp': 1,
            'moe_tp': 1,
            'ep': 1,
            'comm_protocol': 1,
            'is_prefill': False,
        }
    )

    arch = get_arch_preset('SG2260E')
    analyzer = PerformanceAnalyzer(tier6_model, arch)
    tier6_summary = analyzer.get_summary(batch_size=batch_size, seq_len=seq_len)

    print(f"\nTier6+ 分析结果:")
    print(f"  总延迟: {tier6_summary['performance']['total_elapse_ms']:.3f} ms")
    print(f"  MFU: {tier6_summary['performance']['mfu']:.4f}" if tier6_summary['performance']['mfu'] else "  MFU: N/A")
    print(f"  TPS: {tier6_summary['performance']['tps']:.2f}")

    # 打印各层延迟
    print(f"\n  各层延迟:")
    for layer_name, layer_info in tier6_summary['performance']['layers'].items():
        count = layer_info.get('count', 1)
        elapse = layer_info['perf']['elapse']
        print(f"    {layer_name} (x{count}): {elapse:.3f} us")

    return True


def main():
    results = []

    try:
        results.append(("单 GEMM 对比", test_single_matmul_comparison()))
    except Exception as e:
        print(f"单 GEMM 对比失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("单 GEMM 对比", False))

    try:
        results.append(("单 AllReduce 对比", test_single_allreduce_comparison()))
    except Exception as e:
        print(f"单 AllReduce 对比失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("单 AllReduce 对比", False))

    try:
        results.append(("模型结构对比", test_model_structure_comparison()))
    except Exception as e:
        print(f"模型结构对比失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("模型结构对比", False))

    try:
        results.append(("完整分析对比", test_full_analysis_comparison()))
    except Exception as e:
        print(f"完整分析对比失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("完整分析对比", False))

    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == '__main__':
    main()
