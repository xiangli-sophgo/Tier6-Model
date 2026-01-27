#!/usr/bin/env python3
"""
算子级别性能对比

直接对比单个算子的评估时间（GEMM, FA2, AllReduce等）
"""

import sys
import time
from pathlib import Path

# 添加路径
tier6_backend = Path(__file__).parent.parent
ds_tpu_root = Path("/Users/lixiang/Documents/工作/code/DS_TPU_1209")

sys.path.insert(0, str(tier6_backend))
sys.path.insert(0, str(ds_tpu_root))


def benchmark_ds_tpu_matmul():
    """测试 DS_TPU 的 GEMM 算子评估"""
    from performance.evaluate.compute.matmul.matmul_eval import MatmulEval
    from performance.evaluate.compute.comp_eval_base import TPUArch

    # 创建 TPU 架构配置
    arch = TPUArch(
        tpu_cores=64,
        cube_m=32,
        cube_n=32,
        cube_k=32,
        sram_size=8*1024*1024,  # 8MB
        lane_num=32,
        align_bytes=64,
        macs_per_cycle=32*32,
        freq=1.2e9,  # 1.2GHz
        dma_bw=273*1e9,  # 273GB/s
        tpu_gdma_overlap_rate=0.5,
    )

    evaluator = MatmulEval(arch, input_dtype='fp8', output_dtype='bf16')

    # 测试用例：DeepSeek V3.2 MoE 层的 gate 投影
    # [batch, hidden] @ [hidden, expert_inter] = [48, 7168] @ [7168, 2048]
    test_cases = [
        ("MoE Gate", 1, 48, 2048, 7168),
        ("MoE Up", 1, 48, 2048, 7168),
        ("MoE Down", 1, 48, 7168, 2048),
        ("MLA Q_down", 1, 48, 1536, 7168),
        ("MLA Q_up", 1, 48*128, 192, 1536),
    ]

    results = []
    print("\n" + "="*80)
    print("DS_TPU GEMM 算子评估")
    print("="*80)

    for name, G, M, N, K in test_cases:
        print(f"\n📊 测试: {name} ({G}, {M}, {N}, {K})")

        start = time.time()
        result = evaluator.eval_p(name, G, M, N, K)
        elapsed = time.time() - start

        if result:
            latency = result.perf['elapse']
            print(f"   ⏱️  评估耗时: {elapsed*1000:.2f}ms")
            print(f"   🔢 模拟延迟: {latency:.2f}μs")
            print(f"   💾 DRAM流量: {result.perf['dram_traffic']/1e6:.2f}MB")
            print(f"   📈 利用率: {result.perf['urate']*100:.2f}%")

        results.append({
            "name": name,
            "dims": (G, M, N, K),
            "eval_time_ms": elapsed * 1000,
            "sim_latency_us": result.perf['elapse'] if result else 0,
        })

    return results


def benchmark_tier6_gemm():
    """测试 Tier6+Model 的 GEMM 算子评估"""
    from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator

    # 获取架构配置
    arch = get_arch_preset("SG2260E")

    # 测试用例（与 DS_TPU 相同）
    test_cases = [
        ("MoE Gate", 1, 48, 2048, 7168),
        ("MoE Up", 1, 48, 2048, 7168),
        ("MoE Down", 1, 48, 7168, 2048),
        ("MLA Q_down", 1, 48, 1536, 7168),
        ("MLA Q_up", 1, 48*128, 192, 1536),
    ]

    results = []

    # 测试 1: 不搜索（fast_mode=True）
    print("\n" + "="*80)
    print("Tier6+Model GEMM 算子评估 (fast_mode=True, 不搜索)")
    print("="*80)

    evaluator_fast = create_gemm_evaluator(arch, fast_mode=True, enable_partition_search=False)

    for name, G, M, N, K in test_cases:
        print(f"\n📊 测试: {name} ({G}, {M}, {N}, {K})")

        start = time.time()
        result = evaluator_fast.evaluate(G, M, K, N, input_dtype='fp8', output_dtype='bf16')
        elapsed = time.time() - start

        print(f"   ⏱️  评估耗时: {elapsed*1000:.2f}ms")
        print(f"   🔢 模拟延迟: {result.latency_us:.2f}μs")
        print(f"   💾 DRAM流量: {result.dram_traffic_bytes/1e6:.2f}MB")
        print(f"   📈 利用率: {result.effective_utilization*100:.2f}%")

        results.append({
            "name": name,
            "mode": "fast",
            "dims": (G, M, N, K),
            "eval_time_ms": elapsed * 1000,
            "sim_latency_us": result.latency_us,
        })

    # 测试 2: 完整搜索（fast_mode=False, enable_partition_search=True）
    print("\n" + "="*80)
    print("Tier6+Model GEMM 算子评估 (fast_mode=False, 完整搜索)")
    print("="*80)

    evaluator_search = create_gemm_evaluator(arch, fast_mode=False, enable_partition_search=True)

    for name, G, M, N, K in test_cases:
        print(f"\n📊 测试: {name} ({G}, {M}, {N}, {K})")

        start = time.time()
        result = evaluator_search.evaluate(G, M, K, N, input_dtype='fp8', output_dtype='bf16', use_multiprocess=True)
        elapsed = time.time() - start

        print(f"   ⏱️  评估耗时: {elapsed*1000:.2f}ms")
        print(f"   🔢 模拟延迟: {result.latency_us:.2f}μs")
        print(f"   💾 DRAM流量: {result.dram_traffic_bytes/1e6:.2f}MB")
        print(f"   📈 利用率: {result.effective_utilization*100:.2f}%")

        results.append({
            "name": name,
            "mode": "search",
            "dims": (G, M, N, K),
            "eval_time_ms": elapsed * 1000,
            "sim_latency_us": result.latency_us,
        })

    return results


def main():
    print("="*80)
    print("🔬 算子级别性能对比测试")
    print("="*80)

    # 测试 DS_TPU
    try:
        ds_results = benchmark_ds_tpu_matmul()
    except Exception as e:
        print(f"\n❌ DS_TPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        ds_results = []

    # 测试 Tier6+Model
    try:
        tier6_results = benchmark_tier6_gemm()
    except Exception as e:
        print(f"\n❌ Tier6+Model 测试失败: {e}")
        import traceback
        traceback.print_exc()
        tier6_results = []

    # 对比总结
    print("\n" + "="*80)
    print("📊 性能对比总结")
    print("="*80)

    if ds_results and tier6_results:
        tier6_fast = [r for r in tier6_results if r.get("mode") == "fast"]
        tier6_search = [r for r in tier6_results if r.get("mode") == "search"]

        print("\n【评估时间对比】")
        print(f"{'算子':<15} {'DS_TPU':<12} {'Tier6(Fast)':<15} {'Tier6(Search)':<15} {'加速比':<10}")
        print("-" * 80)

        for i, ds in enumerate(ds_results):
            name = ds["name"]
            ds_time = ds["eval_time_ms"]
            t6_fast_time = tier6_fast[i]["eval_time_ms"] if i < len(tier6_fast) else 0
            t6_search_time = tier6_search[i]["eval_time_ms"] if i < len(tier6_search) else 0

            speedup_fast = ds_time / t6_fast_time if t6_fast_time > 0 else 0
            speedup_search = ds_time / t6_search_time if t6_search_time > 0 else 0

            print(f"{name:<15} {ds_time:>10.2f}ms {t6_fast_time:>13.2f}ms {t6_search_time:>13.2f}ms {speedup_search:>8.2f}x")

        # 平均加速比
        avg_ds = sum(r["eval_time_ms"] for r in ds_results) / len(ds_results)
        avg_t6_fast = sum(r["eval_time_ms"] for r in tier6_fast) / len(tier6_fast)
        avg_t6_search = sum(r["eval_time_ms"] for r in tier6_search) / len(tier6_search)

        print("-" * 80)
        print(f"{'平均':<15} {avg_ds:>10.2f}ms {avg_t6_fast:>13.2f}ms {avg_t6_search:>13.2f}ms {avg_ds/avg_t6_search:>8.2f}x")

        print("\n【模拟精度对比（延迟）】")
        print(f"{'算子':<15} {'DS_TPU':<12} {'Tier6(Fast)':<15} {'Tier6(Search)':<15} {'误差':<10}")
        print("-" * 80)

        for i, ds in enumerate(ds_results):
            name = ds["name"]
            ds_lat = ds["sim_latency_us"]
            t6_fast_lat = tier6_fast[i]["sim_latency_us"] if i < len(tier6_fast) else 0
            t6_search_lat = tier6_search[i]["sim_latency_us"] if i < len(tier6_search) else 0

            error = abs(ds_lat - t6_search_lat) / ds_lat * 100 if ds_lat > 0 else 0

            print(f"{name:<15} {ds_lat:>10.2f}μs {t6_fast_lat:>13.2f}μs {t6_search_lat:>13.2f}μs {error:>8.1f}%")


if __name__ == "__main__":
    main()
