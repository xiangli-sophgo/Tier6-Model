#!/usr/bin/env python3
"""
DS_TPU vs Tier6+ 对比测试 (独立运行版)

使用动态模块加载避免导入问题
"""

import sys
import os
import importlib.util

# 设置路径
DS_TPU_PATH = '/Users/lixiang/Documents/工作/code/DS_TPU_1209'
TIER6_PATH = '/Users/lixiang/Documents/工作/code/Tier6+Model/backend'

sys.path.insert(0, TIER6_PATH)
os.chdir(DS_TPU_PATH)
sys.path.insert(0, DS_TPU_PATH)


def load_module(name, filepath):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_dstpu_modules():
    """加载 DS_TPU 所需模块"""
    # 按依赖顺序加载
    load_module('config', os.path.join(DS_TPU_PATH, 'config/__init__.py')
                if os.path.exists(os.path.join(DS_TPU_PATH, 'config/__init__.py'))
                else os.path.join(DS_TPU_PATH, 'config/config_loader.py'))

    # 加载 tpu 模块
    load_module('tpu', os.path.join(DS_TPU_PATH, 'tpu/__init__.py')
                if os.path.exists(os.path.join(DS_TPU_PATH, 'tpu/__init__.py'))
                else os.path.join(DS_TPU_PATH, 'tpu/tpu_v1.py'))

    tpu_v1 = load_module('tpu.tpu_v1', os.path.join(DS_TPU_PATH, 'tpu/tpu_v1.py'))
    return tpu_v1


def compare_gemm():
    """对比 GEMM 评估"""
    print("=" * 80)
    print("GEMM 算子对比测试")
    print("=" * 80)

    # 测试用例
    test_cases = [
        # (G, M, K, N, description)
        (1, 1, 7168, 1536, "MLA q_a_proj (small batch)"),
        (1, 64, 7168, 18432, "Dense MLP Gate (medium batch)"),
        (1, 1536, 7168, 18432, "Dense MLP Gate (large batch)"),
        (256, 8, 7168, 2048, "MoE Routed Expert"),
    ]

    # Tier6+ 评估器
    from llm_simulator.evaluators import GEMMEvaluator, get_arch_preset
    arch = get_arch_preset('sg2262')
    tier6_eval = GEMMEvaluator(arch)

    # DS_TPU 评估器
    try:
        tpu_v1 = setup_dstpu_modules()
        matmul_eval_mod = load_module(
            'matmul_eval',
            os.path.join(DS_TPU_PATH, 'performance/evaluate/compute/matmul/matmul_eval.py')
        )

        tpu_config = tpu_v1.SG2262(core=64)
        ds_eval = matmul_eval_mod.MatmulEval(tpu_config)
        ds_available = True
    except Exception as e:
        print(f"DS_TPU 加载失败: {e}")
        ds_available = False

    results = []

    for G, M, K, N, desc in test_cases:
        print(f"\n{desc}: G={G}, M={M}, K={K}, N={N}")

        # Tier6+ 评估
        tier6_result = tier6_eval.evaluate(G, M, K, N, 'bf16', 'bf16')
        tier6_latency = tier6_result.latency_us
        tier6_urate = tier6_result.effective_utilization

        if ds_available:
            # DS_TPU 评估
            class MockOperator:
                def __init__(self):
                    self.name = desc
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
            ds_latency = op.elapse
            ds_urate = op.urate

            # 对比
            latency_diff = abs(tier6_latency - ds_latency) / ds_latency * 100 if ds_latency > 0 else 0
            urate_diff = abs(tier6_urate - ds_urate)

            print(f"  DS_TPU:  延迟={ds_latency:.3f}us, 利用率={ds_urate:.4f}")
            print(f"  Tier6+:  延迟={tier6_latency:.3f}us, 利用率={tier6_urate:.4f}")
            print(f"  差异:    延迟={latency_diff:.2f}%, 利用率差={urate_diff:.4f}")

            results.append({
                'desc': desc,
                'ds_latency': ds_latency,
                'tier6_latency': tier6_latency,
                'latency_diff': latency_diff,
                'urate_diff': urate_diff,
                'match': latency_diff < 5.0  # 允许 5% 误差
            })
        else:
            print(f"  Tier6+:  延迟={tier6_latency:.3f}us, 利用率={tier6_urate:.4f}")
            print(f"  DS_TPU:  不可用")

    return results


def compare_allreduce():
    """对比 AllReduce 评估"""
    print("\n" + "=" * 80)
    print("AllReduce 通信对比测试")
    print("=" * 80)

    # 测试用例
    test_cases = [
        (2, 1024*1024, 1, "TP=2, 1MB"),
        (4, 2*1024*1024, 1, "TP=4, 2MB"),
        (8, 4*1024*1024, 1, "TP=8, 4MB"),
    ]

    # Tier6+ 评估器
    from llm_simulator.evaluators import AllReduceEval, get_arch_preset
    arch = get_arch_preset('sg2262')
    tier6_eval = AllReduceEval(arch)

    # DS_TPU 评估器
    try:
        tpu_v1 = setup_dstpu_modules()
        allreduce_eval_mod = load_module(
            'allreduce_eval',
            os.path.join(DS_TPU_PATH, 'performance/evaluate/communication/allreduce_eval.py')
        )

        tpu_config = tpu_v1.SG2262(core=64)
        ds_eval = allreduce_eval_mod.AllReduceEval(tpu_config)
        ds_available = True
    except Exception as e:
        print(f"DS_TPU AllReduce 加载失败: {e}")
        ds_available = False

    results = []

    for tp, comm_bytes, protocol, desc in test_cases:
        print(f"\n{desc}")

        # Tier6+ 评估
        tier6_result = tier6_eval.evaluate(tp, comm_bytes, protocol)
        tier6_latency = tier6_result.latency_us

        if ds_available:
            # DS_TPU 评估
            ds_latency, _ = ds_eval.evaluate_raw(tp, comm_bytes, protocol)

            # 对比
            latency_diff = abs(tier6_latency - ds_latency) / ds_latency * 100 if ds_latency > 0 else 0

            print(f"  DS_TPU:  延迟={ds_latency:.3f}us")
            print(f"  Tier6+:  延迟={tier6_latency:.3f}us")
            print(f"  差异:    {latency_diff:.2f}%")

            results.append({
                'desc': desc,
                'ds_latency': ds_latency,
                'tier6_latency': tier6_latency,
                'latency_diff': latency_diff,
                'match': latency_diff < 5.0
            })
        else:
            print(f"  Tier6+:  延迟={tier6_latency:.3f}us")
            print(f"  DS_TPU:  不可用")

    return results


def compare_mla():
    """对比 MLA 层评估"""
    print("\n" + "=" * 80)
    print("MLA 层对比测试")
    print("=" * 80)

    # Tier6+ MLA
    from llm_simulator.models import create_deepseek_v3
    from llm_simulator.evaluators import get_arch_preset

    arch = get_arch_preset('sg2262')

    for mla_type in ['mla', 'mla_absorb']:
        print(f"\n{mla_type.upper()} 变体:")

        model = create_deepseek_v3(
            batch_size=1,
            seq_len=1,
            kv_seq_len=4096,
            tp=1,
            mla_type=mla_type
        )

        # 获取 MLA 层的算子
        for layer in model.layers:
            if 'mla' in layer.name.lower():
                print(f"  计算算子: {[op.op_type.name for op in layer.comp_ops]}")
                print(f"  通信算子: {[op.comm_kind for op in layer.comm_ops] if layer.comm_ops else 'None'}")

                # 计算 FLOPs
                total_flops = sum(op.flops for op in layer.comp_ops)
                total_param = sum(op.param for op in layer.comp_ops)
                print(f"  总 FLOPs: {total_flops / 1e9:.4f} GFLOPs")
                print(f"  总参数: {total_param / 1e6:.2f} M")
                break


def main():
    print("DS_TPU vs Tier6+ 对比测试")
    print("=" * 80)

    gemm_results = compare_gemm()
    comm_results = compare_allreduce()
    compare_mla()

    # 汇总
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    if gemm_results:
        print("\nGEMM 对比:")
        for r in gemm_results:
            status = "✓ MATCH" if r['match'] else "✗ DIFF"
            print(f"  {r['desc']}: {status} (延迟差异 {r['latency_diff']:.2f}%)")

    if comm_results:
        print("\nAllReduce 对比:")
        for r in comm_results:
            status = "✓ MATCH" if r['match'] else "✗ DIFF"
            print(f"  {r['desc']}: {status} (延迟差异 {r['latency_diff']:.2f}%)")

    # 总结
    if gemm_results and comm_results:
        all_match = all(r['match'] for r in gemm_results + comm_results)
        print(f"\n总结: {'所有测试通过!' if all_match else '存在差异，请检查'}")


if __name__ == '__main__':
    main()
