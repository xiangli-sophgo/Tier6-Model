"""SimKernel + SimObject 单元测试"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.kernel.sim_object import SimObject


# ==================== SimKernel 测试 ====================


def test_empty_kernel():
    """空内核立即返回"""
    k = SimKernel()
    k.run()
    assert k.now() == 0.0


def test_single_event():
    """单事件调度"""
    k = SimKernel()
    results = []
    k.schedule(10.0, lambda: results.append(k.now()))
    k.run()
    assert len(results) == 1
    assert results[0] == 10.0
    assert k.now() == 10.0


def test_event_ordering():
    """事件按时间排序执行"""
    k = SimKernel()
    order = []
    k.schedule(30.0, lambda: order.append("C"))
    k.schedule(10.0, lambda: order.append("A"))
    k.schedule(20.0, lambda: order.append("B"))
    k.run()
    assert order == ["A", "B", "C"]
    assert k.now() == 30.0


def test_schedule_at():
    """绝对时间调度"""
    k = SimKernel()
    results = []
    k.schedule_at(50.0, lambda: results.append(k.now()))
    k.run()
    assert results[0] == 50.0


def test_chained_events():
    """事件中调度新事件"""
    k = SimKernel()
    trace = []

    def step1():
        trace.append(("step1", k.now()))
        k.schedule(5.0, step2)

    def step2():
        trace.append(("step2", k.now()))

    k.schedule(10.0, step1)
    k.run()
    assert trace == [("step1", 10.0), ("step2", 15.0)]


def test_same_time_stable_order():
    """同一时间事件按调度顺序执行 (FIFO)"""
    k = SimKernel()
    order = []
    k.schedule(10.0, lambda: order.append("first"))
    k.schedule(10.0, lambda: order.append("second"))
    k.schedule(10.0, lambda: order.append("third"))
    k.run()
    assert order == ["first", "second", "third"]


def test_cycle_conversion():
    """cycle <-> ns 转换"""
    k = SimKernel()
    k.add_clock("tpu", frequency_ghz=1.0)   # 1 GHz = 1 ns/cycle
    k.add_clock("ddr", frequency_ghz=0.2)   # 200 MHz = 5 ns/cycle

    assert k.cycle_to_ns(100, "tpu") == 100.0
    assert k.cycle_to_ns(100, "ddr") == 500.0
    assert k.ns_to_cycle(100.0, "tpu") == 100
    assert k.ns_to_cycle(500.0, "ddr") == 100


def test_event_count():
    """事件计数统计"""
    k = SimKernel()
    k.schedule(10.0, lambda: None)
    k.schedule(20.0, lambda: None)
    k.run()
    assert k.event_count == 2


# ==================== SimObject 测试 ====================


def test_simobject_schedule():
    """SimObject 代理 kernel.schedule()"""
    k = SimKernel()
    k.add_clock("tpu", frequency_ghz=1.0)
    obj = SimObject(kernel=k, name="test_obj", clock_name="tpu")
    trace = []
    obj.schedule(10.0, lambda: trace.append(k.now()))
    k.run()
    assert trace == [10.0]


def test_simobject_schedule_cycles():
    """SimObject cycle 级调度"""
    k = SimKernel()
    k.add_clock("tpu", frequency_ghz=1.0)
    k.add_clock("ddr", frequency_ghz=0.2)

    obj_tpu = SimObject(kernel=k, name="tpu_obj", clock_name="tpu")
    obj_ddr = SimObject(kernel=k, name="ddr_obj", clock_name="ddr")
    trace = []

    obj_tpu.schedule_cycles(100, lambda: trace.append(("tpu", k.now())))
    obj_ddr.schedule_cycles(20, lambda: trace.append(("ddr", k.now())))
    k.run()
    # tpu: 100 cycles / 1.0 GHz = 100 ns
    # ddr: 20 cycles / 0.2 GHz = 100 ns -> 同时间, 但 tpu 先调度
    assert trace == [("tpu", 100.0), ("ddr", 100.0)]


def test_simobject_cycle_now():
    """SimObject 获取当前 cycle"""
    k = SimKernel()
    k.add_clock("tpu", frequency_ghz=1.0)
    obj = SimObject(kernel=k, name="obj", clock_name="tpu")
    results = []

    def check():
        results.append(obj.cycle_now())

    k.schedule(44.0, check)
    k.run()
    assert results[0] == 44


if __name__ == "__main__":
    test_empty_kernel()
    test_single_event()
    test_event_ordering()
    test_schedule_at()
    test_chained_events()
    test_same_time_stable_order()
    test_cycle_conversion()
    test_event_count()
    test_simobject_schedule()
    test_simobject_schedule_cycles()
    test_simobject_cycle_now()
    print("[PASS] All SimKernel + SimObject tests passed")
