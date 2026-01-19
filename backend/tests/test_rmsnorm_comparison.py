#!/usr/bin/env python3
"""
RMSNorm 评估器对比测试

对比 Tier6+ 和 DS_TPU 的 RMSNorm 实现结果
"""

import sys
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

# Tier6+ 实现
from llm_simulator.evaluators.rmsnorm_eval import rmsnorm_theoretical_and_real as tier6_rmsnorm
from llm_simulator.evaluators.utils import align_up


# DS_TPU 实现 (直接内联，避免复杂依赖)
def ds_rmsnorm(QS, KS, lane_num, eu_num, BF16_BYTES, align_up, has_scale=False, has_bias=False):
    """
    DS_TPU 原版 RMSNorm 评估器
    直接从 DS_TPU_1209/performance/evaluate/compute/rmsnorm_eval.py 复制
    """
    rmsnorm_steps = [
        ('square', 0, 1),
        ('reduce_sum', 1, 1),
        ('div_constant', 1, 31),
        ('add_constant', 1, 1),
        ('rsqrt', 1, 30),
        ('mul', 0, 1),
        ('mul_scale', 0, 1),
        ('add_bias', 0, 1),
        ('data_convert', 0, 1),
    ]

    def calc_step_theo(step_shape_type, op_count):
        if step_shape_type == 0:
            return align_up(QS, lane_num) * align_up(KS, eu_num // lane_num // BF16_BYTES) * op_count
        else:
            return align_up(QS, lane_num) * op_count

    def calc_step_real(step_shape_type, op_count):
        if step_shape_type == 0:
            return QS * KS * op_count
        else:
            return QS * 1 * op_count

    vector_theo = 0
    vector_real = 0
    for name, shape_type, op_count in rmsnorm_steps:
        if name == 'mul_scale' and not has_scale:
            continue
        if name == 'add_bias' and not has_bias:
            continue
        vector_theo += calc_step_theo(shape_type, op_count)
        vector_real += calc_step_real(shape_type, op_count)
    return vector_theo, vector_real


def test_rmsnorm_comparison():
    """对比测试 RMSNorm 评估器"""

    # 测试用例: (batch_size, hidden_dim, has_scale, has_bias)
    test_cases = [
        # 基础测试
        (1, 7168, True, False),      # 单 token, DeepSeek V3 hidden_dim
        (8, 7168, True, False),      # 小 batch
        (32, 7168, True, False),     # 中 batch
        (128, 7168, True, False),    # 大 batch
        (256, 7168, True, False),    # 更大 batch

        # 不同 hidden_dim
        (32, 4096, True, False),     # Llama 7B
        (32, 5120, True, False),     # Llama 13B
        (32, 8192, True, False),     # Llama 70B

        # 带 bias
        (32, 7168, True, True),
        (32, 7168, False, False),    # 无 scale 无 bias
    ]

    # 硬件参数 (SG2260E)
    lane_num = 16
    eu_num = 512
    dtype_bytes = 2  # BF16

    print("=" * 80)
    print("RMSNorm 评估器对比测试")
    print("=" * 80)
    print(f"硬件参数: lane_num={lane_num}, eu_num={eu_num}, dtype_bytes={dtype_bytes}")
    print("-" * 80)

    all_passed = True

    for batch_size, hidden_dim, has_scale, has_bias in test_cases:
        # Tier6+ 结果
        tier6_theo, tier6_real = tier6_rmsnorm(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            lane_num=lane_num,
            eu_num=eu_num,
            dtype_bytes=dtype_bytes,
            has_scale=has_scale,
            has_bias=has_bias,
        )

        # DS_TPU 结果 (需要传入 align_up 函数)
        ds_theo, ds_real = ds_rmsnorm(
            QS=batch_size,
            KS=hidden_dim,
            lane_num=lane_num,
            eu_num=eu_num,
            BF16_BYTES=dtype_bytes,
            align_up=align_up,
            has_scale=has_scale,
            has_bias=has_bias,
        )

        # 对比
        theo_match = tier6_theo == ds_theo
        real_match = tier6_real == ds_real
        passed = theo_match and real_match

        if not passed:
            all_passed = False

        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\nRMSNorm[{batch_size}, {hidden_dim}] scale={has_scale} bias={has_bias}")
        print(f"  Tier6+: theo={tier6_theo:,}, real={tier6_real:,}")
        print(f"  DS_TPU: theo={ds_theo:,}, real={ds_real:,}")

        if not theo_match:
            diff_theo = abs(tier6_theo - ds_theo)
            print(f"  ✗ 理论值差异: {diff_theo:,}")
        if not real_match:
            diff_real = abs(tier6_real - ds_real)
            print(f"  ✗ 实际值差异: {diff_real:,}")

        utilization = tier6_real / tier6_theo if tier6_theo > 0 else 0
        print(f"  利用率: {utilization:.2%}")
        print(f"  {status}")

    print("\n" + "=" * 80)
    if all_passed:
        print("所有测试通过! RMSNorm 评估器对齐成功")
    else:
        print("存在测试失败!")
    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = test_rmsnorm_comparison()
    sys.exit(0 if success else 1)
