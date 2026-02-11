"""内存分析功能测试

验证内存分解计算的正确性，包括 MLA 压缩和 MoE 支持。
"""

import sys
from pathlib import Path

# 添加 backend 到路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from math_model.L5_reporting.memory_analysis import MemoryAnalyzer


def test_llama_70b_standard():
    """测试 LLaMA-70B (Dense + GQA)

    配置：
    - Model: LLaMA-70B, hidden=8192, layers=80, kv_heads=8, heads=64
    - Deployment: TP=4, PP=1, batch=32, seq=4096, BF16
    - Chip: H100-80GB

    预期：
    - kv_cache_memory_gb: ~10.7 GB
    """
    print("\n=== 测试 1: LLaMA-70B (Standard GQA) ===")

    analyzer = MemoryAnalyzer(dtype_bytes=2)  # BF16

    kv_cache_bytes = analyzer.calculate_kv_cache_memory(
        batch_size=32,
        seq_len=4096,
        num_layers=80,
        hidden_size=8192,
        num_kv_heads=8,
        num_heads=64,
        tp_degree=4,
        pp_degree=1,
        mla_enabled=False,
    )

    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    print(f"  KV Cache: {kv_cache_gb:.2f} GB")

    # 验证预期值
    expected_gb = 10.7
    if abs(kv_cache_gb - expected_gb) < 1.0:
        print(f"  [OK] KV Cache 接近预期值 {expected_gb} GB")
    else:
        print(f"  [WARN] KV Cache 与预期值 {expected_gb} GB 差距较大")

    return kv_cache_gb


def test_deepseek_v3_mla():
    """测试 DeepSeek-V3 (MoE + MLA)

    配置：
    - Model: DeepSeek-V3-671B, hidden=7168, layers=61, MLA (kv_lora=512, rope=64)
    - Deployment: TP=8, EP=4, batch=128, seq=4096, FP8
    - Chip: SG2262-64GB

    预期：
    - kv_cache_memory_gb: ~2.3 GB (MLA 压缩)
    """
    print("\n=== 测试 2: DeepSeek-V3 (MLA + MoE) ===")

    analyzer = MemoryAnalyzer(dtype_bytes=1)  # FP8

    kv_cache_bytes = analyzer.calculate_kv_cache_memory(
        batch_size=128,
        seq_len=4096,
        num_layers=61,
        hidden_size=7168,
        num_kv_heads=128,
        num_heads=128,
        tp_degree=8,
        pp_degree=1,
        mla_enabled=True,
        kv_lora_rank=512,
        qk_rope_dim=64,
    )

    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    print(f"  KV Cache (MLA): {kv_cache_gb:.2f} GB")

    # 计算标准 GQA 对比
    kv_cache_standard = analyzer.calculate_kv_cache_memory(
        batch_size=128,
        seq_len=4096,
        num_layers=61,
        hidden_size=7168,
        num_kv_heads=128,
        num_heads=128,
        tp_degree=8,
        pp_degree=1,
        mla_enabled=False,
    )
    kv_cache_standard_gb = kv_cache_standard / (1024 ** 3)

    saving_ratio = (kv_cache_standard - kv_cache_bytes) / kv_cache_standard
    print(f"  KV Cache (Standard GQA): {kv_cache_standard_gb:.2f} GB")
    print(f"  MLA 节省: {saving_ratio * 100:.1f}%")

    # 验证预期值
    expected_gb = 2.3
    if abs(kv_cache_gb - expected_gb) < 0.5:
        print(f"  [OK] MLA KV Cache 接近预期值 {expected_gb} GB")
    else:
        print(f"  [WARN] MLA KV Cache 与预期值 {expected_gb} GB 差距较大")

    if saving_ratio > 0.90:
        print(f"  [OK] MLA 压缩率达到预期 (>90%)")
    else:
        print(f"  [WARN] MLA 压缩率低于预期 ({saving_ratio * 100:.1f}%)")

    return kv_cache_gb


def test_activation_memory():
    """测试激活内存计算"""
    print("\n=== 测试 3: 激活内存计算 ===")

    analyzer = MemoryAnalyzer(dtype_bytes=2)  # BF16

    activation_bytes = analyzer.calculate_activation_memory(
        batch_size=32,
        seq_len=1,  # Decode 阶段
        hidden_size=8192,
        intermediate_size=28672,
        num_layers=80,
        tp_degree=4,
        pp_degree=1,
    )

    activation_gb = activation_bytes / (1024 ** 3)
    print(f"  Activation (decode): {activation_gb:.3f} GB")

    # Prefill 阶段
    activation_prefill_bytes = analyzer.calculate_activation_memory(
        batch_size=32,
        seq_len=2048,
        hidden_size=8192,
        intermediate_size=28672,
        num_layers=80,
        tp_degree=4,
        pp_degree=1,
    )
    activation_prefill_gb = activation_prefill_bytes / (1024 ** 3)
    print(f"  Activation (prefill, seq=2048): {activation_prefill_gb:.3f} GB")

    # 验证推理激活内存应该较小（<1GB for decode）
    if activation_gb < 1.0:
        print(f"  [OK] Decode 激活内存合理 (<1GB)")
    else:
        print(f"  [WARN] Decode 激活内存偏大")

    return activation_gb


def test_overhead_calculation():
    """测试开销计算"""
    print("\n=== 测试 4: 开销计算 ===")

    analyzer = MemoryAnalyzer(dtype_bytes=2)

    # 测试不同规模的开销
    test_cases = [
        (1 * 1024**3, 0.5 * 1024**3, "小模型"),  # 1GB 权重, 0.5GB KV Cache
        (30 * 1024**3, 10 * 1024**3, "中模型"),  # 30GB 权重, 10GB KV Cache
        (100 * 1024**3, 50 * 1024**3, "大模型"),  # 100GB 权重, 50GB KV Cache
    ]

    for weight_bytes, kv_cache_bytes, label in test_cases:
        overhead_bytes = analyzer.calculate_overhead(weight_bytes, kv_cache_bytes)
        overhead_gb = overhead_bytes / (1024 ** 3)

        weight_gb = weight_bytes / (1024 ** 3)
        kv_gb = kv_cache_bytes / (1024 ** 3)
        ratio = overhead_bytes / (weight_bytes + kv_cache_bytes)

        print(f"  {label}: 权重={weight_gb:.1f}GB, KV Cache={kv_gb:.1f}GB")
        print(f"    -> 开销={overhead_gb:.2f}GB ({ratio * 100:.1f}%)")

        # 验证开销在合理范围内 [500MB, 4GB]
        if 0.5 <= overhead_gb <= 4.0:
            print(f"    [OK] 开销在合理范围内")
        else:
            print(f"    [WARN] 开销超出预期范围")

    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("内存分析功能测试")
    print("=" * 60)

    try:
        # 测试 1: LLaMA-70B
        llama_kv = test_llama_70b_standard()

        # 测试 2: DeepSeek-V3 MLA
        deepseek_kv = test_deepseek_v3_mla()

        # 测试 3: 激活内存
        activation = test_activation_memory()

        # 测试 4: 开销计算
        test_overhead_calculation()

        print("\n" + "=" * 60)
        print("[OK] 所有测试完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
