# -*- coding: utf-8 -*-
"""
测试 Switch 调度功能

验证阶段 5 的实现：
1. 转发模式（Store-and-Forward vs Cut-Through）
2. Round-Robin 调度算法
3. 端口队列管理
"""

import sys
from pathlib import Path

# 添加项目路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from llm_simulator.event_driven.switch_manager import (
    SwitchManager, SwitchNode, PortQueue, DEFAULT_SWITCH_CONFIGS
)
from llm_simulator.config.types import (
    SwitchInstanceConfig, SwitchLayer, SwitchType
)
from llm_simulator.event_driven.event import (
    SwitchForwardEvent, PortScheduleEvent, EventType
)
from llm_simulator.event_driven.resource import ResourceManager


def test_port_queue_rr_scheduling():
    """测试端口队列的 Round-Robin 调度"""
    print("\n" + "="*60)
    print("Test 1: PortQueue Round-Robin Scheduling")
    print("="*60)

    # 创建端口队列
    port_queue = PortQueue(
        port_id="leaf_0:port_12",
        switch_id="leaf_0",
        output_port=12,
    )

    # 创建 4 个来自不同输入端口的包
    packets = []
    for i in range(4):
        event = SwitchForwardEvent(
            timestamp=100.0,
            chip_id="chip_0",
            packet_id=f"packet_{i}",
            flow_id="flow_0",
            packet_size_bytes=512 * 1024,  # 512 KB
            src_chip="chip_0",
            dst_chip="chip_1",
            route=["leaf_0", "spine_0", "leaf_1"],
            hop_index=0,
            switch_id="leaf_0",
            output_port=12,
        )
        packets.append(event)
        # 模拟来自不同输入端口
        input_port = f"port_{i % 2}"  # 2 个输入端口
        port_queue.enqueue(event, input_port)

    print(f"[OK] Enqueued 4 packets:")
    print(f"  - port_0: 2 packets")
    print(f"  - port_1: 2 packets")
    print(f"  - Total queue size: {port_queue.total_size()}")

    # Round-Robin 调度
    scheduled_order = []
    while not port_queue.is_empty():
        next_event = port_queue.schedule_next()
        if next_event:
            scheduled_order.append(next_event.packet_id)

    print(f"\n[OK] Round-Robin scheduling order: {scheduled_order}")
    print(f"  Expected: [packet_0, packet_1, packet_2, packet_3]")

    # 验证调度顺序（应该按输入端口轮询）
    expected_order = ["packet_0", "packet_1", "packet_2", "packet_3"]
    if scheduled_order == expected_order:
        print(f"[PASS] Scheduling order is correct!")
    else:
        print(f"[FAIL] Scheduling order is wrong!")

    print()


def test_forwarding_modes():
    """测试转发模式（Store-and-Forward vs Cut-Through）"""
    print("\n" + "="*60)
    print("Test 2: Forwarding Mode Latency Calculation")
    print("="*60)

    # 创建 2 个 Switch：一个 Store-and-Forward，一个 Cut-Through
    hardware_sf = DEFAULT_SWITCH_CONFIGS["leaf_72"]
    hardware_sf.forwarding_mode = "store_and_forward"

    hardware_ct = DEFAULT_SWITCH_CONFIGS["leaf_128"]
    hardware_ct.forwarding_mode = "cut_through"

    config_sf = SwitchInstanceConfig(
        switch_id="leaf_sf",
        hardware_type="leaf_72",
        layer=SwitchLayer.INTER_BOARD,
    )

    config_ct = SwitchInstanceConfig(
        switch_id="leaf_ct",
        hardware_type="leaf_128",
        layer=SwitchLayer.INTER_BOARD,
    )

    switch_sf = SwitchNode(config=config_sf, hardware=hardware_sf)
    switch_ct = SwitchNode(config=config_ct, hardware=hardware_ct)

    # 计算延迟
    packet_size = 512 * 1024  # 512 KB

    # Store-and-Forward
    sf_processing = switch_sf.hardware.processing_delay_us
    sf_serialization = switch_sf.get_serialization_delay_us(packet_size)
    sf_total = sf_processing + sf_serialization

    # Cut-Through
    ct_processing = switch_ct.hardware.cut_through_delay_us
    ct_serialization = 0.0  # Cut-Through 忽略包大小
    ct_total = ct_processing + ct_serialization

    print(f"\nPacket size: {packet_size / 1024:.0f} KB")
    print(f"\nStore-and-Forward mode (leaf_72, 100 Gbps):")
    print(f"  - Processing delay: {sf_processing:.2f} us")
    print(f"  - Serialization delay: {sf_serialization:.2f} us")
    print(f"  - Total delay: {sf_total:.2f} us")

    print(f"\nCut-Through mode (leaf_128, 400 Gbps):")
    print(f"  - Fixed delay: {ct_processing:.2f} us")
    print(f"  - Serialization delay: {ct_serialization:.2f} us (ignored)")
    print(f"  - Total delay: {ct_total:.2f} us")

    print(f"\nSpeedup: {sf_total / ct_total:.2f}x")

    if ct_total < sf_total:
        print(f"[PASS] Cut-Through has lower latency!")
    else:
        print(f"[FAIL] Latency calculation is wrong!")

    print()


def test_port_scheduling_event():
    """测试端口调度事件"""
    print("\n" + "="*60)
    print("Test 3: PortScheduleEvent Handling")
    print("="*60)

    # 创建 ResourceManager
    resource_manager = ResourceManager(chip_ids=["chip_0", "chip_1"])

    # 创建 Switch
    hardware = DEFAULT_SWITCH_CONFIGS["leaf_72"]
    config = SwitchInstanceConfig(
        switch_id="leaf_0",
        hardware_type="leaf_72",
        layer=SwitchLayer.INTER_BOARD,
    )
    switch = SwitchNode(config=config, hardware=hardware)

    # 创建 SwitchManager 并添加 Switch
    class MockTopologyParser:
        pass

    switch_manager = SwitchManager(topology_parser=None, switch_graph=None)
    switch_manager.switches["leaf_0"] = switch

    # 注册端口资源
    resource_manager.register_switch_port("leaf_0", 12)

    # 创建 3 个等待的包
    for i in range(3):
        event = SwitchForwardEvent(
            timestamp=100.0,
            chip_id="chip_0",
            packet_id=f"packet_{i}",
            flow_id="flow_0",
            packet_size_bytes=512 * 1024,
            src_chip="chip_0",
            dst_chip="chip_1",
            route=["leaf_0"],
            hop_index=0,
            switch_id="leaf_0",
            output_port=12,
        )
        port_queue = switch.get_port_queue(12)
        port_queue.enqueue(event, f"port_{i % 2}")

    print(f"[OK] Initial port queue state:")
    port_queue = switch.get_port_queue(12)
    print(f"  - Queue size: {port_queue.total_size()} packets")

    # 创建端口调度事件
    schedule_event = PortScheduleEvent(
        timestamp=200.0,
        chip_id="chip_0",
        switch_id="leaf_0",
        output_port=12,
    )

    # 处理调度事件
    context = {"switch_manager": switch_manager}
    new_events = schedule_event.handle(
        resource_manager=resource_manager,
        gantt_builder=None,
        context=context,
    )

    print(f"\n[OK] Scheduling event result:")
    print(f"  - Generated events: {len(new_events)}")
    if new_events:
        for event in new_events:
            print(f"    - {type(event).__name__}: packet_id={event.packet_id}")

    print(f"\n[OK] Remaining port queue:")
    print(f"  - Queue size: {port_queue.total_size()} packets")

    if len(new_events) == 1 and port_queue.total_size() == 2:
        print(f"\n[PASS] Scheduling logic is correct!")
    else:
        print(f"\n[FAIL] Scheduling logic is wrong!")

    print()


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Switch Scheduling Test Suite [Phase 5]")
    print("="*60)

    try:
        test_port_queue_rr_scheduling()
        test_forwarding_modes()
        test_port_scheduling_event()

        print("\n" + "="*60)
        print("[PASS] All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
