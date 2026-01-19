#!/usr/bin/env python3
"""
通信评估器对比测试

对比 Tier6+ 和 DS_TPU 的通信评估器实现结果
"""

import sys
sys.path.insert(0, '/Users/lixiang/Documents/工作/code/Tier6+Model/backend')

from llm_simulator.evaluators.comm_eval import (
    AllReduceEval,
    AllGatherEval,
    ReduceScatterEval,
    DispatchEval,
    CombineEval,
)


# ==================== DS_TPU 原版实现 (内联) ====================

class DS_AllReduceEval:
    """DS_TPU 原版 AllReduce 评估器"""

    def __init__(self, arch):
        self.arch = arch
        self.c2c_lat = 0.15
        self.ddr_r_lat = 0.15
        self.ddr_w_lat = 0.01
        self.noc_lat = 0.05
        self.d2d_lat = 0.04
        self.rtt_tp = 0.35
        self.start_lat = 2 * self.c2c_lat + self.ddr_r_lat + self.ddr_w_lat + self.noc_lat + 2 * self.d2d_lat
        self.sync_lat = 0
        self.bw_urate = 0.95
        self.link_delay = 0

    def evaluate_raw(self, tp, bytes, comm_protocol=1):
        if tp in [8, 16, 32]:
            # Hierarchical
            group_sizes = 2 if tp == 16 else 4
            num_groups = tp // group_sizes
            comm_sizes_1 = 2 * (group_sizes - 1) / group_sizes * bytes
            comm_sizes_2 = 2 * (num_groups - 1) / num_groups * bytes
            comm_sizes_3 = bytes
            base_lat_1 = (comm_sizes_1 / self.arch.intra_bw / self.bw_urate) * 1e6 + (group_sizes - 1) * (self.start_lat + self.sync_lat)
            base_lat_2 = (comm_sizes_2 / self.arch.inter_bw / self.bw_urate) * 1e6 + (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)
            base_lat_3 = (comm_sizes_3 / self.arch.intra_bw / self.bw_urate) * 1e6 + (group_sizes - 1) * (self.start_lat + self.sync_lat)
            lat = max(base_lat_1, base_lat_2, base_lat_3)
            comm = comm_sizes_1 * num_groups + comm_sizes_2 + comm_sizes_3 * num_groups
        else:
            # Flat
            comm = 2 * (tp - 1) / tp * bytes
            lat = (comm / self.arch.intra_bw / self.bw_urate) * 1e6 + (tp - 1) * (self.start_lat + self.sync_lat)
            if comm_protocol == 2:
                lat += self.rtt_tp * 2 * (tp - 1)
            elif comm_protocol == 3:
                lat += self.rtt_tp * min(1, 2 * (tp - 1))
        return lat, comm


class DS_AllGatherEval:
    """DS_TPU 原版 AllGather 评估器"""

    def __init__(self, arch):
        self.arch = arch
        self.c2c_lat = 0.15
        self.ddr_r_lat = 0.15
        self.ddr_w_lat = 0.01
        self.noc_lat = 0.05
        self.d2d_lat = 0.04
        self.rtt_tp = 0.35
        self.start_lat = 2 * self.c2c_lat + self.ddr_r_lat + self.ddr_w_lat + self.noc_lat + 2 * self.d2d_lat
        self.sync_lat = 0
        self.bw_urate = 0.95
        self.link_delay = 0

    def evaluate_raw(self, tp, bytes, comm_protocol=1):
        if tp in [8, 16, 32]:
            # Hierarchical
            # 注意: AllGather 对所有 tp=8/16/32 都使用 group_size=4
            # 与 AllReduce/ReduceScatter 不同 (它们 tp=16 用 group_size=2)
            group_sizes = 4
            num_groups = tp // group_sizes
            comm_sizes_1 = (group_sizes - 1) * bytes
            comm_sizes_2 = (num_groups - 1) * bytes
            comm_sizes_3 = 0
            base_lat_1 = (comm_sizes_1 / self.arch.intra_bw / self.bw_urate) * 1e6 + (group_sizes - 1) * (self.start_lat + self.sync_lat)
            base_lat_2 = (comm_sizes_2 / self.arch.inter_bw / self.bw_urate) * 1e6 + (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)
            base_lat_3 = comm_sizes_3
            lat = max(base_lat_1, base_lat_2, base_lat_3)
            comm = comm_sizes_1 * num_groups + comm_sizes_2 + comm_sizes_3 * num_groups
        else:
            # Flat
            comm = (tp - 1) * bytes
            lat = (comm / self.arch.intra_bw / self.bw_urate) * 1e6 + (tp - 1) * (self.start_lat + self.sync_lat)
            if comm_protocol == 2:
                lat += self.rtt_tp * 2 * (tp - 1)
            elif comm_protocol == 3:
                lat += self.rtt_tp * min(1, 2 * (tp - 1))
        return lat, comm


class DS_ReduceScatterEval:
    """DS_TPU 原版 ReduceScatter 评估器"""

    def __init__(self, arch):
        self.arch = arch
        self.c2c_lat = 0.15
        self.ddr_r_lat = 0.15
        self.ddr_w_lat = 0.01
        self.noc_lat = 0.05
        self.d2d_lat = 0.04
        self.rtt_tp = 0.35
        self.start_lat = 2 * self.c2c_lat + self.ddr_r_lat + self.ddr_w_lat + self.noc_lat + 2 * self.d2d_lat
        self.sync_lat = 0
        self.bw_urate = 0.95
        self.link_delay = 0

    def evaluate_raw(self, tp, bytes, comm_protocol=1):
        if tp in [8, 16, 32]:
            # Hierarchical
            group_sizes = 2 if tp == 16 else 4
            num_groups = tp // group_sizes
            comm_sizes_1 = (group_sizes - 1) / group_sizes * bytes
            comm_sizes_2 = (num_groups - 1) / num_groups * bytes
            comm_sizes_3 = 0
            base_lat_1 = (comm_sizes_1 / self.arch.intra_bw / self.bw_urate) * 1e6 + (group_sizes - 1) * (self.start_lat + self.sync_lat)
            # 注意: DS_TPU ReduceScatter Stage 2 使用 intra_bw (与 AllReduce 不同)
            base_lat_2 = (comm_sizes_2 / self.arch.intra_bw / self.bw_urate) * 1e6 + (num_groups - 1) * (self.start_lat + self.sync_lat + self.link_delay)
            base_lat_3 = comm_sizes_3
            lat = max(base_lat_1, base_lat_2, base_lat_3)
            comm = comm_sizes_1 * num_groups + comm_sizes_2 + comm_sizes_3 * num_groups
        else:
            # Flat
            comm = (tp - 1) / tp * bytes
            lat = (comm / self.arch.intra_bw / self.bw_urate) * 1e6 + (tp - 1) * (self.start_lat + self.sync_lat)
            if comm_protocol == 2:
                lat += self.rtt_tp * 2 * (tp - 1)
            elif comm_protocol == 3:
                lat += self.rtt_tp * min(1, 2 * (tp - 1))
        return lat, comm


class DS_DispatchEval:
    """DS_TPU 原版 Dispatch 评估器"""

    def __init__(self, arch):
        self.arch = arch
        self.allgather_eval = DS_AllGatherEval(arch)
        self.c2c_lat = 0.15
        self.ddr_r_lat = 0.15
        self.ddr_w_lat = 0.01
        self.noc_lat = 0.05
        self.d2d_lat = 0.04
        self.start_lat = 2 * self.c2c_lat + self.ddr_r_lat + self.ddr_w_lat + self.noc_lat + 2 * self.d2d_lat
        self.switch_delay = 0.25
        self.cable_delay = 0.025
        self.link_delay = 2 * self.switch_delay + 2 * self.cable_delay
        self.cpu_fetch_delay = 0
        self.rtt_ep = 0.85
        self.topk = 8
        self.prefill_factor = 8 / 128
        self.bw_urate = 0.95

    def evaluate_raw(self, moe_tp, ep, bytes, bs, comm_protocol=1, is_prefill=False):
        t_us = (bytes / self.arch.inter_bw / self.bw_urate) * 1e6 + self.start_lat + self.cpu_fetch_delay

        if comm_protocol == 2:
            if is_prefill:
                t_us += self.rtt_ep * bs * self.topk * self.prefill_factor
            else:
                t_us += self.rtt_ep * bs * self.topk
        elif comm_protocol == 3:
            if is_prefill:
                t_us += self.rtt_ep * min(1, bs * self.topk * self.prefill_factor)
            else:
                t_us += self.rtt_ep * min(1, bs * self.topk)

        agather_lat, agather_comm = self.allgather_eval.evaluate_raw(moe_tp, bytes)
        t_us += agather_lat

        return t_us, bytes + agather_comm


class DS_CombineEval:
    """DS_TPU 原版 Combine 评估器"""

    def __init__(self, arch):
        self.arch = arch
        self.allgather_eval = DS_AllGatherEval(arch)
        self.c2c_lat = 0.15
        self.ddr_r_lat = 0.15
        self.ddr_w_lat = 0.01
        self.noc_lat = 0.05
        self.d2d_lat = 0.04
        self.start_lat = 2 * self.c2c_lat + self.ddr_r_lat + self.ddr_w_lat + self.noc_lat + 2 * self.d2d_lat
        self.switch_delay = 0.25
        self.cable_delay = 0.025
        self.link_delay = 2 * self.switch_delay + 2 * self.cable_delay
        self.cpu_fetch_delay = 0
        self.rtt_ep = 0.85
        self.topk = 8
        self.prefill_factor = 8 / 128
        self.bw_urate = 0.95

    def evaluate_raw(self, moe_tp, ep, bytes, bs, comm_protocol=1, is_prefill=False):
        t_us = (bytes / self.arch.inter_bw / self.bw_urate) * 1e6 + self.start_lat + self.cpu_fetch_delay

        if comm_protocol == 2:
            if is_prefill:
                t_us += self.rtt_ep * bs * self.topk * self.prefill_factor
            else:
                t_us += self.rtt_ep * bs * self.topk
        elif comm_protocol == 3:
            if is_prefill:
                t_us += self.rtt_ep * min(1, bs * self.topk * self.prefill_factor)
            else:
                t_us += self.rtt_ep * min(1, bs * self.topk)

        agather_lat, agather_comm = self.allgather_eval.evaluate_raw(moe_tp, bytes)
        t_us += agather_lat

        return t_us, bytes + agather_comm


# ==================== 测试函数 ====================

class MockArch:
    """模拟架构配置"""
    def __init__(self, intra_bw=504e9, inter_bw=100e9):
        self.intra_bw = intra_bw
        self.inter_bw = inter_bw


def compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm):
    """对比结果"""
    lat_diff = abs(tier6_lat - ds_lat)
    comm_diff = abs(tier6_comm - ds_comm)

    lat_match = lat_diff < 1e-6
    comm_match = comm_diff < 1e-6
    passed = lat_match and comm_match

    status = "✓ PASS" if passed else "✗ FAIL"

    print(f"\n{name}")
    print(f"  Tier6+: lat={tier6_lat:.3f} μs, comm={tier6_comm:,.0f} bytes")
    print(f"  DS_TPU: lat={ds_lat:.3f} μs, comm={ds_comm:,.0f} bytes")

    if not lat_match:
        print(f"  ✗ 延迟差异: {lat_diff:.6f} μs")
    if not comm_match:
        print(f"  ✗ 通信量差异: {comm_diff:.0f} bytes")

    print(f"  {status}")
    return passed


def test_allreduce():
    """测试 AllReduce"""
    print("\n" + "=" * 60)
    print("AllReduce 对比测试")
    print("=" * 60)

    arch = MockArch()
    tier6_eval = AllReduceEval(arch)
    ds_eval = DS_AllReduceEval(arch)

    all_passed = True

    # 测试用例: (tp, data_bytes, comm_protocol)
    test_cases = [
        # 扁平模式
        (2, 1_000_000, 1),
        (4, 1_000_000, 1),
        (4, 1_000_000, 2),
        (4, 1_000_000, 3),
        # 分层模式
        (8, 1_000_000, 1),
        (16, 1_000_000, 1),
        (32, 1_000_000, 1),
        # 不同数据量
        (8, 10_000, 1),
        (8, 100_000_000, 1),
    ]

    for tp, data_bytes, comm_protocol in test_cases:
        tier6_lat, tier6_comm = tier6_eval.evaluate_raw(tp, data_bytes, comm_protocol)
        ds_lat, ds_comm = ds_eval.evaluate_raw(tp, data_bytes, comm_protocol)

        name = f"AllReduce[tp={tp}, bytes={data_bytes:,}, proto={comm_protocol}]"
        passed = compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm)
        all_passed = all_passed and passed

    return all_passed


def test_allgather():
    """测试 AllGather"""
    print("\n" + "=" * 60)
    print("AllGather 对比测试")
    print("=" * 60)

    arch = MockArch()
    tier6_eval = AllGatherEval(arch)
    ds_eval = DS_AllGatherEval(arch)

    all_passed = True

    test_cases = [
        (2, 1_000_000, 1),
        (4, 1_000_000, 1),
        (8, 1_000_000, 1),
        (16, 1_000_000, 1),
        (32, 1_000_000, 1),
    ]

    for tp, data_bytes, comm_protocol in test_cases:
        tier6_lat, tier6_comm = tier6_eval.evaluate_raw(tp, data_bytes, comm_protocol)
        ds_lat, ds_comm = ds_eval.evaluate_raw(tp, data_bytes, comm_protocol)

        name = f"AllGather[tp={tp}, bytes={data_bytes:,}]"
        passed = compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm)
        all_passed = all_passed and passed

    return all_passed


def test_reducescatter():
    """测试 ReduceScatter"""
    print("\n" + "=" * 60)
    print("ReduceScatter 对比测试")
    print("=" * 60)

    arch = MockArch()
    tier6_eval = ReduceScatterEval(arch)
    ds_eval = DS_ReduceScatterEval(arch)

    all_passed = True

    test_cases = [
        (2, 1_000_000, 1),
        (4, 1_000_000, 1),
        (8, 1_000_000, 1),
        (16, 1_000_000, 1),
        (32, 1_000_000, 1),
    ]

    for tp, data_bytes, comm_protocol in test_cases:
        tier6_lat, tier6_comm = tier6_eval.evaluate_raw(tp, data_bytes, comm_protocol)
        ds_lat, ds_comm = ds_eval.evaluate_raw(tp, data_bytes, comm_protocol)

        name = f"ReduceScatter[tp={tp}, bytes={data_bytes:,}]"
        passed = compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm)
        all_passed = all_passed and passed

    return all_passed


def test_dispatch():
    """测试 Dispatch"""
    print("\n" + "=" * 60)
    print("Dispatch 对比测试")
    print("=" * 60)

    arch = MockArch()
    tier6_eval = DispatchEval(arch)
    ds_eval = DS_DispatchEval(arch)

    all_passed = True

    # 测试用例: (moe_tp, ep, data_bytes, batch_size, comm_protocol, is_prefill)
    test_cases = [
        (1, 8, 1_000_000, 32, 1, False),
        (2, 8, 1_000_000, 32, 1, False),
        (4, 8, 1_000_000, 32, 1, False),
        (4, 8, 1_000_000, 32, 2, False),
        (4, 8, 1_000_000, 32, 2, True),
        (4, 8, 1_000_000, 32, 3, False),
    ]

    for moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill in test_cases:
        tier6_lat, tier6_comm = tier6_eval.evaluate_raw(moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill)
        ds_lat, ds_comm = ds_eval.evaluate_raw(moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill)

        name = f"Dispatch[moe_tp={moe_tp}, ep={ep}, bs={bs}, proto={comm_protocol}, prefill={is_prefill}]"
        passed = compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm)
        all_passed = all_passed and passed

    return all_passed


def test_combine():
    """测试 Combine"""
    print("\n" + "=" * 60)
    print("Combine 对比测试")
    print("=" * 60)

    arch = MockArch()
    tier6_eval = CombineEval(arch)
    ds_eval = DS_CombineEval(arch)

    all_passed = True

    test_cases = [
        (1, 8, 1_000_000, 32, 1, False),
        (4, 8, 1_000_000, 32, 1, False),
        (4, 8, 1_000_000, 32, 2, False),
        (4, 8, 1_000_000, 32, 2, True),
    ]

    for moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill in test_cases:
        tier6_lat, tier6_comm = tier6_eval.evaluate_raw(moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill)
        ds_lat, ds_comm = ds_eval.evaluate_raw(moe_tp, ep, data_bytes, bs, comm_protocol, is_prefill)

        name = f"Combine[moe_tp={moe_tp}, ep={ep}, bs={bs}, proto={comm_protocol}, prefill={is_prefill}]"
        passed = compare_results(name, tier6_lat, tier6_comm, ds_lat, ds_comm)
        all_passed = all_passed and passed

    return all_passed


def main():
    """主测试函数"""
    print("=" * 80)
    print("通信评估器对比测试")
    print("=" * 80)
    print(f"硬件参数: intra_bw=504 GB/s, inter_bw=100 GB/s")

    results = []
    results.append(("AllReduce", test_allreduce()))
    results.append(("AllGather", test_allgather()))
    results.append(("ReduceScatter", test_reducescatter()))
    results.append(("Dispatch", test_dispatch()))
    results.append(("Combine", test_combine()))

    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 80)
    if all_passed:
        print("所有通信评估器测试通过! 对齐成功")
    else:
        print("存在测试失败!")
    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
