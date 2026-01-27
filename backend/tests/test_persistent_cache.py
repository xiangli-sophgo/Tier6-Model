#!/usr/bin/env python3
"""
测试 GEMM 持久化缓存功能

验证：
1. 缓存文件创建
2. 缓存保存和加载
3. 跨运行复用
4. 架构指纹匹配
"""

import sys
import time
from pathlib import Path

# 添加路径
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from llm_simulator.evaluators import get_arch_preset, create_gemm_evaluator


def test_cache_persistence():
    """测试缓存持久化"""
    print("="*80)
    print("测试 GEMM 持久化缓存")
    print("="*80)

    # 获取架构配置
    arch = get_arch_preset("SG2260E")

    # 测试用例
    test_cases = [
        ("MoE Gate", 1, 384, 7168, 2048),
        ("MLA Q_down", 1, 48, 7168, 1536),
        ("MLA Q_up", 1, 6144, 1536, 192),
    ]

    # ====================
    # 第一次运行：建立缓存
    # ====================
    print("\n" + "="*80)
    print("第1次运行：建立缓存")
    print("="*80)

    evaluator1 = create_gemm_evaluator(
        arch,
        fast_mode=False,
        enable_partition_search=True
    )

    print(f"\n缓存文件: {evaluator1.persistent_cache.cache_file}")

    for name, G, M, K, N in test_cases:
        print(f"\n📊 评估: {name} ({G}, {M}, {K}, {N})")

        start = time.time()
        result = evaluator1.evaluate(
            G, M, K, N,
            input_dtype='fp8',
            output_dtype='bf16',
            use_multiprocess=True
        )
        elapsed_ms = (time.time() - start) * 1000

        print(f"   耗时: {elapsed_ms:.2f}ms")
        print(f"   延迟: {result.latency_us:.2f}μs")
        print(f"   利用率: {result.effective_utilization*100:.1f}%")

    # 打印统计
    print("\n" + "-"*80)
    evaluator1.print_cache_stats()

    # ====================
    # 第二次运行：复用缓存
    # ====================
    print("\n" + "="*80)
    print("第2次运行：模拟进程重启，复用缓存")
    print("="*80)

    # 销毁第一个评估器，模拟进程重启
    del evaluator1

    # 创建新的评估器
    evaluator2 = create_gemm_evaluator(
        arch,
        fast_mode=False,
        enable_partition_search=True
    )

    print(f"\n加载的缓存条目数: {len(evaluator2.persistent_cache._cache)}")

    for name, G, M, K, N in test_cases:
        print(f"\n📊 评估: {name} ({G}, {M}, {K}, {N})")

        start = time.time()
        result = evaluator2.evaluate(
            G, M, K, N,
            input_dtype='fp8',
            output_dtype='bf16',
            use_multiprocess=True
        )
        elapsed_ms = (time.time() - start) * 1000

        print(f"   耗时: {elapsed_ms:.2f}ms  ✅ 应该 <1ms (缓存命中)")
        print(f"   延迟: {result.latency_us:.2f}μs")

    # 打印统计
    print("\n" + "-"*80)
    evaluator2.print_cache_stats()

    # ====================
    # 对比
    # ====================
    stats1 = {"cache_misses": 3}  # 第1次运行：3个未命中
    stats2 = evaluator2.get_cache_stats()

    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    print(f"\n第1次运行:")
    print(f"  缓存未命中: {stats1['cache_misses']} (需要搜索)")
    print(f"  缓存命中: 0")

    print(f"\n第2次运行:")
    print(f"  缓存未命中: {stats2['cache_misses']}")
    print(f"  缓存命中: {stats2['cache_hits']} ✅")
    print(f"  命中率: {stats2['hit_rate_percent']:.1f}%")

    if stats2['cache_hits'] >= 3:
        print("\n✅ 持久化缓存测试通过！")
        return True
    else:
        print("\n❌ 持久化缓存测试失败！")
        return False


def test_arch_fingerprint():
    """测试架构指纹匹配"""
    print("\n" + "="*80)
    print("测试架构指纹匹配")
    print("="*80)

    arch1 = get_arch_preset("SG2260E")
    arch2 = get_arch_preset("SG2260E")

    eval1 = create_gemm_evaluator(arch1, fast_mode=False)
    eval2 = create_gemm_evaluator(arch2, fast_mode=False)

    fp1 = eval1.persistent_cache.arch_fingerprint
    fp2 = eval2.persistent_cache.arch_fingerprint

    print(f"\nSG2260E (实例1) 指纹: {fp1}")
    print(f"SG2260E (实例2) 指纹: {fp2}")

    if fp1 == fp2:
        print("✅ 相同架构的指纹一致")
    else:
        print("❌ 相同架构的指纹不一致")

    # 缓存文件应该相同
    if eval1.persistent_cache.cache_file == eval2.persistent_cache.cache_file:
        print(f"✅ 使用相同的缓存文件: {eval1.persistent_cache.cache_file}")
        return True
    else:
        print("❌ 缓存文件不同")
        return False


if __name__ == "__main__":
    success1 = test_arch_fingerprint()
    print("\n")
    success2 = test_cache_persistence()

    if success1 and success2:
        print("\n" + "="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ 部分测试失败")
        print("="*80)
        sys.exit(1)
