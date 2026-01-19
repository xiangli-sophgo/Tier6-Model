#!/usr/bin/env python3
"""
GEMM 多进程搜索测试

验证多进程和串行搜索产生相同结果
"""

import sys
import time
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

from llm_simulator.evaluators import (
    get_arch_preset,
    GEMMEvaluator,
    eval_gemm,
)


def test_multiprocess_consistency():
    """测试多进程和串行搜索结果一致性"""
    print("=" * 80)
    print("GEMM 多进程搜索一致性测试")
    print("=" * 80)

    arch = get_arch_preset('SG2260E')

    # 测试用例
    test_cases = [
        # (G, M, K, N, input_dtype, output_dtype)
        (1, 128, 512, 256, 'bf16', 'bf16'),
        (1, 1024, 4096, 4096, 'bf16', 'bf16'),
        (8, 256, 1024, 512, 'fp8', 'bf16'),
        (16, 64, 256, 128, 'bf16', 'bf16'),
        (1, 4096, 7168, 2048, 'bf16', 'bf16'),  # 大规模 GEMM
    ]

    all_passed = True
    results = []

    for G, M, K, N, in_dtype, out_dtype in test_cases:
        print(f"\n测试 GEMM: G={G}, M={M}, K={K}, N={N}, dtype={in_dtype}->{out_dtype}")

        # 创建新评估器（清空缓存）
        evaluator = GEMMEvaluator(arch)

        # 串行评估
        t0 = time.time()
        result_serial = evaluator.evaluate(G, M, K, N, in_dtype, out_dtype, use_multiprocess=False)
        t_serial = time.time() - t0

        # 清空缓存
        evaluator.clear_cache()

        # 多进程评估
        t0 = time.time()
        result_mp = evaluator.evaluate(G, M, K, N, in_dtype, out_dtype, use_multiprocess=True)
        t_mp = time.time() - t0

        # 比较结果
        latency_match = abs(result_serial.latency_us - result_mp.latency_us) < 1e-6
        tile_match = result_serial.best_tile == result_mp.best_tile
        order_match = result_serial.best_loop_order == result_mp.best_loop_order
        partition_match = result_serial.best_partition == result_mp.best_partition

        passed = latency_match and tile_match and order_match and partition_match

        status = "PASS" if passed else "FAIL"
        speedup = t_serial / t_mp if t_mp > 0 else 0

        print(f"  串行: {t_serial*1000:.1f}ms, 多进程: {t_mp*1000:.1f}ms, 加速比: {speedup:.2f}x")
        print(f"  延迟: {result_serial.latency_us:.3f}us vs {result_mp.latency_us:.3f}us")
        print(f"  Tile: {result_serial.best_tile} vs {result_mp.best_tile}")
        print(f"  分块: {result_serial.best_partition} vs {result_mp.best_partition}")
        print(f"  状态: {status}")

        if not passed:
            all_passed = False

        results.append({
            'case': (G, M, K, N),
            'passed': passed,
            't_serial': t_serial,
            't_mp': t_mp,
            'speedup': speedup,
        })

    # 汇总
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    total = len(results)
    passed_count = sum(1 for r in results if r['passed'])
    avg_speedup = sum(r['speedup'] for r in results) / total if total > 0 else 0

    print(f"通过: {passed_count}/{total}")
    print(f"平均加速比: {avg_speedup:.2f}x")

    if all_passed:
        print("\n所有测试通过! 多进程搜索与串行搜索结果一致")
    else:
        print("\n存在不一致的结果!")

    return all_passed


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 80)
    print("GEMM 基本功能测试")
    print("=" * 80)

    arch = get_arch_preset('SG2260E')

    # 简单测试
    result = eval_gemm(arch, G=1, M=1024, K=4096, N=4096)

    print(f"\n测试 GEMM [1, 1024, 4096] x [4096, 4096]:")
    print(f"  延迟: {result.latency_us:.3f} us")
    print(f"  计算时间: {result.compute_time_us:.3f} us")
    print(f"  访存时间: {result.memory_time_us:.3f} us")
    print(f"  FLOPs: {result.flops:,}")
    print(f"  DRAM 流量: {result.dram_traffic_bytes:,} bytes")
    print(f"  架构利用率: {result.arch_utilization:.2%}")
    print(f"  有效利用率: {result.effective_utilization:.2%}")
    print(f"  最佳 Tile: {result.best_tile}")
    print(f"  最佳循环顺序: {result.best_loop_order}")
    print(f"  最佳分块: {result.best_partition}")

    # 验证结果合理性
    passed = (
        result.latency_us > 0 and
        result.flops == 2 * 1024 * 4096 * 4096 and
        result.effective_utilization > 0 and result.effective_utilization <= 1
    )

    print(f"\n基本功能测试: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    """主测试函数"""
    results = []
    results.append(("基本功能", test_basic_functionality()))
    results.append(("多进程一致性", test_multiprocess_consistency()))

    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
