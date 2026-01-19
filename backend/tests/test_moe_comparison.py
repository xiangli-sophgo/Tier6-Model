#!/usr/bin/env python3
"""
MoE 专家负载查表对比测试

对比 Tier6+ 和 DS_TPU 的 get_max_expert 实现结果
"""

import sys
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

from llm_simulator.latency.moe import get_max_expert, get_max_expert_float, EXPERT_LOAD_TABLE


# DS_TPU 原版实现 (内联)
def ds_get_max_expert(batch: int, chips: int) -> float:
    """DS_TPU 原版 get_max_expert"""
    batch = min(256, batch)
    data = {
        1: {1: 8.0, 2: 5.08915, 4: 3.5096, 8: 2.5613, 16: 2.02795, 32: 1.60595, 64: 1.2973, 128: 1.1086, 256: 1.0},
        2: {1: 15.74335, 2: 9.3783, 4: 5.96155, 8: 4.07395, 16: 2.94035, 32: 2.26275, 64: 1.8285, 128: 1.38245, 256: 1.0},
        3: {1: 23.2566, 2: 13.44475, 4: 8.25905, 8: 5.38985, 16: 3.7346, 32: 2.741, 64: 2.1301, 128: 1.6721, 256: 1.0},
        4: {1: 30.5121, 2: 17.3445, 4: 10.3651, 8: 6.575, 16: 4.44845, 32: 3.18425, 64: 2.326, 128: 1.8603, 256: 1.0},
        5: {1: 37.592, 2: 21.01065, 4: 12.3875, 8: 7.68935, 16: 5.0503, 32: 3.5408, 64: 2.53475, 128: 1.9572, 256: 1.0},
        6: {1: 44.3833, 2: 24.61835, 4: 14.27665, 8: 8.73345, 16: 5.6629, 32: 3.8562, 64: 2.7497, 128: 1.9882, 256: 1.0},
        8: {1: 57.4309, 2: 31.40505, 4: 17.81475, 8: 10.6518, 16: 6.69615, 32: 4.4437, 64: 3.0746, 128: 1.9997, 256: 1.0},
        16: {1: 101.9848, 2: 54.1023, 4: 29.53585, 8: 16.7107, 16: 9.86765, 32: 6.05895, 64: 3.8144, 128: 2.0, 256: 1.0},
        20: {1: 120.3125, 2: 63.3071, 4: 34.1843, 8: 19.05355, 16: 11.00855, 32: 6.5959, 64: 3.9689, 128: 2.0, 256: 1.0},
        24: {1: 136.4634, 2: 71.4372, 4: 38.2351, 8: 21.0515, 16: 11.9747, 32: 7.01615, 64: 3.99745, 128: 2.0, 256: 1.0},
        32: {1: 163.27965, 2: 84.75245, 4: 44.79135, 8: 24.19435, 16: 13.42475, 32: 7.5834, 64: 4.0, 128: 2.0, 256: 1.0},
        40: {1: 184.0948, 2: 94.89465, 4: 49.68275, 8: 26.51455, 16: 14.41565, 32: 7.91665, 64: 4.0, 128: 2.0, 256: 1.0},
        48: {1: 200.20185, 2: 102.74895, 4: 53.3845, 8: 28.1981, 16: 15.1102, 32: 7.9938, 64: 4.0, 128: 2.0, 256: 1.0},
        64: {1: 222.417, 2: 113.352, 4: 58.32715, 8: 30.30105, 16: 15.8369, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0},
        128: {1: 251.59985, 2: 126.611, 4: 63.7934, 8: 31.9994, 16: 16.0, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0},
        256: {1: 255.9269, 2: 127.9988, 4: 64.0, 8: 32.0, 16: 16.0, 32: 8.0, 64: 4.0, 128: 2.0, 256: 1.0}
    }

    try:
        return round(data[batch][chips])
    except KeyError:
        return -1  # 不支持的组合


def test_get_max_expert():
    """测试 get_max_expert 查表函数"""
    print("=" * 80)
    print("MoE 专家负载查表对比测试")
    print("=" * 80)

    all_passed = True
    test_count = 0
    pass_count = 0

    # 测试所有有效的 batch/chips 组合
    valid_batches = [1, 2, 3, 4, 5, 6, 8, 16, 20, 24, 32, 40, 48, 64, 128, 256]
    valid_chips = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"\n测试 {len(valid_batches)} × {len(valid_chips)} = {len(valid_batches) * len(valid_chips)} 个组合...")
    print("-" * 80)

    failed_cases = []

    for batch in valid_batches:
        for chips in valid_chips:
            test_count += 1
            tier6_result = get_max_expert(batch, chips)
            ds_result = ds_get_max_expert(batch, chips)

            if tier6_result == ds_result:
                pass_count += 1
            else:
                all_passed = False
                failed_cases.append((batch, chips, tier6_result, ds_result))

    # 打印结果
    if failed_cases:
        print(f"\n失败的测试用例 ({len(failed_cases)} 个):")
        for batch, chips, tier6, ds in failed_cases[:10]:  # 只显示前10个
            print(f"  batch={batch:3d}, chips={chips:3d}: Tier6+={tier6}, DS_TPU={ds}")
        if len(failed_cases) > 10:
            print(f"  ... 还有 {len(failed_cases) - 10} 个失败用例")
    else:
        print(f"\n所有 {test_count} 个测试用例通过!")

    print(f"\n通过率: {pass_count}/{test_count} ({100*pass_count/test_count:.1f}%)")

    # 测试浮点数版本
    print("\n" + "-" * 80)
    print("测试 get_max_expert_float (浮点数版本):")

    sample_cases = [
        (1, 1), (1, 2), (8, 4), (32, 8), (128, 16), (256, 64),
    ]

    for batch, chips in sample_cases:
        float_result = get_max_expert_float(batch, chips)
        int_result = get_max_expert(batch, chips)
        expected = EXPERT_LOAD_TABLE[batch][chips]

        match = abs(float_result - expected) < 1e-6
        status = "✓" if match else "✗"
        print(f"  {status} batch={batch:3d}, chips={chips:3d}: float={float_result:.5f}, int={int_result}, expected={expected:.5f}")

    print("\n" + "=" * 80)
    if all_passed:
        print("所有测试通过! MoE 专家负载查表对齐成功")
    else:
        print(f"存在 {len(failed_cases)} 个测试失败!")
    print("=" * 80)

    return all_passed


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)

    all_passed = True

    # 测试超出范围的 batch
    print("\n超出范围的 batch (应该被截断):")
    edge_cases = [
        (0, 8, "batch=0 应该被处理"),
        (300, 8, "batch=300 应该被截断到 256"),
        (1000, 8, "batch=1000 应该被截断到 256"),
    ]

    for batch, chips, desc in edge_cases:
        try:
            result = get_max_expert(batch, chips)
            print(f"  ✓ {desc}: 返回 {result}")
        except Exception as e:
            print(f"  ✗ {desc}: 抛出异常 {e}")
            all_passed = False

    # 测试中间值的 batch (不在表中)
    print("\n中间值 batch (不在表中，应该找最接近的):")
    middle_cases = [
        (7, 8, "batch=7 应该使用 batch=8 的值"),
        (10, 8, "batch=10 应该使用 batch=16 的值"),
        (100, 8, "batch=100 应该使用 batch=128 的值"),
    ]

    for batch, chips, desc in middle_cases:
        result = get_max_expert(batch, chips)
        print(f"  batch={batch}, chips={chips}: {result} ({desc})")

    return all_passed


def main():
    """主测试函数"""
    results = []
    results.append(("get_max_expert 查表", test_get_max_expert()))
    results.append(("边界情况", test_edge_cases()))

    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
