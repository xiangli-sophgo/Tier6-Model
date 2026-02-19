# Phase 1: SimKernel 仿真内核 + 多核扩展 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有单核 G5 事件驱动仿真器重构为 SimKernel 架构，支持多核仿真。

**Architecture:** 提取全局事件队列为 SimKernel，所有硬件模块继承 SimObject 基类。现有 `calc_*_latency()` 纯函数保留不变，SimObject 包装器调用它们并管理状态机。CoreSubsys 封装单核 4 引擎，BusModel 建模核间 Manhattan 距离延迟，SingleChip 组装多核。G5SimEngine 外部 API 不变，内部委托 SingleChip。

**Tech Stack:** Python 3.11+, heapq, dataclasses, 现有 perf_model 框架

---

## 文件结构总览

```
backend/perf_model/L4_evaluation/g5/
  kernel/                    # [新增] 仿真内核
    __init__.py
    sim_kernel.py            # 全局事件队列 + 调度循环
    sim_object.py            # SimObject 基类
  chip/                      # [新增] 芯片内模块
    __init__.py
    core_subsys.py           # 单核子系统 (封装 4 引擎)
    bus.py                   # Bus NxM 互联 (Manhattan 距离)
  top/                       # [新增] 顶层组装
    __init__.py
    single_chip.py           # 单芯片多核组装
  sim_engine.py              # [重构] 内部委托 SingleChip
```

### 现有文件保留策略

| 文件 | 策略 | 原因 |
|------|------|------|
| tiu.py | **保留不变** | `calc_tiu_latency()` 作为纯函数被 SimObject 调用 |
| dma.py | **保留不变** | `calc_dma_latency()` 作为纯函数被 SimObject 调用 |
| sdma.py | **保留不变** | `calc_sdma_latency()` 作为纯函数被 SimObject 调用 |
| hau.py | **保留不变** | `calc_hau_latency()` 作为纯函数被 SimObject 调用 |
| adapter.py | **保留不变** | SimRecord 格式不变 |
| pipeline.py | **保留不变** | 调用 G5SimEngine 接口不变 |
| memory.py | **保留不变** | lmem_budget_per_core 不变 |
| sim_engine.py | **重构内部** | 外部 API 不变，内部委托 SingleChip |

---

## Task 1: SimKernel — 全局事件队列

**Files:**
- Create: `backend/perf_model/L4_evaluation/g5/kernel/__init__.py`
- Create: `backend/perf_model/L4_evaluation/g5/kernel/sim_kernel.py`
- Test: `backend/tests/test_g5_simkernel.py`

### Step 1: 编写 SimKernel 测试

```python
# backend/tests/test_g5_simkernel.py
"""SimKernel 单元测试"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel


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


if __name__ == "__main__":
    test_empty_kernel()
    test_single_event()
    test_event_ordering()
    test_schedule_at()
    test_chained_events()
    test_same_time_stable_order()
    test_cycle_conversion()
    test_event_count()
    print("[PASS] All SimKernel tests passed")
```

### Step 2: 运行测试确认失败

```bash
cd backend && python3 tests/test_g5_simkernel.py
```
Expected: `ModuleNotFoundError: No module named 'perf_model.L4_evaluation.g5.kernel'`

### Step 3: 实现 SimKernel

```python
# backend/perf_model/L4_evaluation/g5/kernel/__init__.py
"""G5 仿真内核"""
from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.kernel.sim_object import SimObject

__all__ = ["SimKernel", "SimObject"]
```

```python
# backend/perf_model/L4_evaluation/g5/kernel/sim_kernel.py
"""全局事件驱动仿真内核

对标 SystemC sc_main / gem5 EventManager:
- 全局事件队列 (heapq)
- 时间单位 ns，支持多时钟域 cycle 转换
- 事件按 (time_ns, seq_id) 排序，保证同时间 FIFO

参考设计: docs/plans/2026-02-19-g5-full-architecture-design.md Section 5.1
"""
from __future__ import annotations

import heapq
import math
from typing import Callable


class SimKernel:
    """轻量级事件驱动仿真内核"""

    def __init__(self) -> None:
        self._current_time: float = 0.0
        self._event_queue: list[tuple[float, int, Callable]] = []
        self._seq_counter: int = 0
        self._clocks: dict[str, float] = {}  # name -> frequency_ghz
        self._event_count: int = 0

    def now(self) -> float:
        """当前仿真时间 (ns)"""
        return self._current_time

    @property
    def event_count(self) -> int:
        """已执行的事件总数"""
        return self._event_count

    def schedule(self, delay_ns: float, callback: Callable) -> None:
        """延迟调度事件"""
        if delay_ns < 0:
            raise ValueError(f"delay_ns must be >= 0, got {delay_ns}")
        time = self._current_time + delay_ns
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._event_queue, (time, seq, callback))

    def schedule_at(self, time_ns: float, callback: Callable) -> None:
        """绝对时间调度事件"""
        if time_ns < self._current_time:
            raise ValueError(
                f"Cannot schedule in the past: time_ns={time_ns}, now={self._current_time}"
            )
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._event_queue, (time_ns, seq, callback))

    def run(self) -> None:
        """主事件循环: 弹出事件 -> 推进时间 -> 执行回调"""
        while self._event_queue:
            time_ns, _seq, callback = heapq.heappop(self._event_queue)
            self._current_time = time_ns
            self._event_count += 1
            callback()

    def add_clock(self, name: str, frequency_ghz: float) -> None:
        """注册时钟域"""
        if frequency_ghz <= 0:
            raise ValueError(f"frequency_ghz must be > 0, got {frequency_ghz}")
        self._clocks[name] = frequency_ghz

    def cycle_to_ns(self, cycles: int, clock_name: str) -> float:
        """cycle -> ns"""
        freq = self._clocks.get(clock_name)
        if freq is None:
            raise KeyError(f"Clock '{clock_name}' not registered")
        return cycles / freq

    def ns_to_cycle(self, ns: float, clock_name: str) -> int:
        """ns -> cycle (向下取整)"""
        freq = self._clocks.get(clock_name)
        if freq is None:
            raise KeyError(f"Clock '{clock_name}' not registered")
        return int(ns * freq)
```

### Step 4: 运行测试确认通过

```bash
cd backend && python3 tests/test_g5_simkernel.py
```
Expected: `[PASS] All SimKernel tests passed`

### Step 5: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/kernel/ backend/tests/test_g5_simkernel.py
git commit -m "feat(perf_model): 添加 SimKernel 全局事件驱动仿真内核"
```

---

## Task 2: SimObject — 硬件模块基类

**Files:**
- Create: `backend/perf_model/L4_evaluation/g5/kernel/sim_object.py`
- Modify: `backend/tests/test_g5_simkernel.py` (追加测试)

### Step 1: 在 test_g5_simkernel.py 追加 SimObject 测试

```python
from perf_model.L4_evaluation.g5.kernel.sim_object import SimObject


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
```

### Step 2: 实现 SimObject

```python
# backend/perf_model/L4_evaluation/g5/kernel/sim_object.py
"""SimObject 硬件模块基类

所有 G5 仿真模块的基类，提供:
- kernel 引用 (全局事件队列)
- 时钟域绑定 (cycle <-> ns 转换)
- schedule / schedule_cycles 便捷方法

参考设计: docs/plans/2026-02-19-g5-full-architecture-design.md Section 5.2
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel


class SimObject:
    """硬件模块基类"""

    def __init__(self, kernel: SimKernel, name: str, clock_name: str) -> None:
        self.kernel = kernel
        self.name = name
        self.clock_name = clock_name

    def now(self) -> float:
        """当前仿真时间 (ns)"""
        return self.kernel.now()

    def cycle_now(self) -> int:
        """当前仿真 cycle (基于绑定时钟域)"""
        return self.kernel.ns_to_cycle(self.kernel.now(), self.clock_name)

    def schedule(self, delay_ns: float, callback: Callable) -> None:
        """延迟调度事件 (ns)"""
        self.kernel.schedule(delay_ns, callback)

    def schedule_cycles(self, cycles: int, callback: Callable) -> None:
        """延迟调度事件 (cycles, 自动转 ns)"""
        delay_ns = self.kernel.cycle_to_ns(cycles, self.clock_name)
        self.kernel.schedule(delay_ns, callback)

    def schedule_at(self, time_ns: float, callback: Callable) -> None:
        """绝对时间调度"""
        self.kernel.schedule_at(time_ns, callback)
```

### Step 3: 运行测试

```bash
cd backend && python3 tests/test_g5_simkernel.py
```
Expected: 全部通过

### Step 4: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/kernel/sim_object.py backend/tests/test_g5_simkernel.py
git commit -m "feat(perf_model): 添加 SimObject 硬件模块基类"
```

---

## Task 3: CoreSubsys — 单核子系统

**Files:**
- Create: `backend/perf_model/L4_evaluation/g5/chip/__init__.py`
- Create: `backend/perf_model/L4_evaluation/g5/chip/core_subsys.py`
- Test: `backend/tests/test_g5_core_subsys.py`

### Step 1: 编写 CoreSubsys 测试

```python
# backend/tests/test_g5_core_subsys.py
"""CoreSubsys 单元测试: 单核 4 引擎仿真"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreInstructions, TIUCommand, TIUOpType,
    DMACommand, DMADirection,
)
from perf_model.L4_evaluation.g5.kernel import SimKernel
from perf_model.L4_evaluation.g5.chip.core_subsys import CoreSubsys
from perf_model.L4_evaluation.g5.sim_engine import SimRecord


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def test_core_subsys_single_tiu():
    """单条 TIU 指令仿真"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")

    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0,
                op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="test_op",
            ),
        ],
        dma_cmds=[],
        sdma_cmds=[],
        hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    assert len(records) == 1
    r = records[0]
    assert r.engine == "TIU"
    assert r.cmd_id == 1
    assert r.start_ns == 0.0
    assert r.end_ns > 0.0
    assert r.flops == 2 * 16 * 8 * 32
    print(f"[PASS] single TIU: {r.end_ns:.1f} ns, {r.flops} flops")


def test_core_subsys_tiu_dma_overlap():
    """TIU 和 DMA 并行执行 (无依赖)"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")

    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0,
                op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="op0",
            ),
        ],
        dma_cmds=[
            DMACommand(
                cmd_id=1, cmd_id_dep=0,
                direction=DMADirection.DDR_TO_LMEM,
                src_addr=0, dst_addr=0,
                data_bytes=4096, elem_bytes=2,
                source_op_id="op0",
            ),
        ],
        sdma_cmds=[],
        hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    assert len(records) == 2
    tiu_r = [r for r in records if r.engine == "TIU"][0]
    dma_r = [r for r in records if r.engine == "DMA"][0]
    # 都从 t=0 开始 (无依赖, dep=0 <= sync_id=0)
    assert tiu_r.start_ns == 0.0
    assert dma_r.start_ns == 0.0
    print(f"[PASS] overlap: TIU=[0, {tiu_r.end_ns:.1f}], DMA=[0, {dma_r.end_ns:.1f}]")


def test_core_subsys_dependency_chain():
    """TIU -> DMA 依赖链"""
    chip = load_chip()
    kernel = SimKernel()
    kernel.add_clock("tpu", frequency_ghz=chip.get_tiu_frequency())

    core = CoreSubsys(kernel=kernel, chip=chip, core_id=0, clock_name="tpu")

    instr = CoreInstructions(
        core_id=0,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0,
                op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16", source_op_id="op0",
            ),
        ],
        dma_cmds=[
            # DMA dep=1: 等待 TIU cmd_id=1 完成
            DMACommand(
                cmd_id=1, cmd_id_dep=1,
                direction=DMADirection.LMEM_TO_DDR,
                src_addr=0, dst_addr=0,
                data_bytes=4096, elem_bytes=2,
                source_op_id="op0",
            ),
        ],
        sdma_cmds=[],
        hau_cmds=[],
    )
    core.load_instructions(instr)
    kernel.run()

    records = core.get_records()
    tiu_r = [r for r in records if r.engine == "TIU"][0]
    dma_r = [r for r in records if r.engine == "DMA"][0]
    # DMA 在 TIU 完成后才开始
    assert dma_r.start_ns >= tiu_r.end_ns, (
        f"DMA start {dma_r.start_ns} should be >= TIU end {tiu_r.end_ns}"
    )
    print(f"[PASS] dep chain: TIU end={tiu_r.end_ns:.1f}, DMA start={dma_r.start_ns:.1f}")


if __name__ == "__main__":
    test_core_subsys_single_tiu()
    test_core_subsys_tiu_dma_overlap()
    test_core_subsys_dependency_chain()
    print("[PASS] All CoreSubsys tests passed")
```

### Step 2: 运行测试确认失败

```bash
cd backend && python3 tests/test_g5_core_subsys.py
```
Expected: `ModuleNotFoundError`

### Step 3: 实现 CoreSubsys

CoreSubsys 封装 4 引擎的状态机，逻辑从 `sim_engine.py:_simulate_core()` 提取而来，改用 SimObject + SimKernel 事件调度。

```python
# backend/perf_model/L4_evaluation/g5/chip/__init__.py
"""G5 芯片内模块"""
from perf_model.L4_evaluation.g5.chip.core_subsys import CoreSubsys

__all__ = ["CoreSubsys"]
```

```python
# backend/perf_model/L4_evaluation/g5/chip/core_subsys.py
"""CoreSubsys: 单核子系统

封装 TIU + GDMA + SDMA + HAU 四引擎，管理同步信号和指令调度。
对标 TPUPerf TpuSubsys。

同步模型 (与现有 sim_engine.py 完全一致):
  TIU:  cmd.cmd_id_dep <= tdma_sync_id (固定依赖 DMA)
  DMA:  cmd.cmd_id_dep <= tiu_sync_id  (固定依赖 TIU)
  SDMA: cmd.dep_engine 指定依赖引擎
  HAU:  cmd.dep_engine 指定依赖引擎

参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 6.1
"""
from __future__ import annotations

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreInstructions,
    DMADirection,
)
from perf_model.L4_evaluation.g5.kernel.sim_object import SimObject
from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.sim_engine import SimRecord
from perf_model.L4_evaluation.g5.tiu import calc_tiu_latency
from perf_model.L4_evaluation.g5.dma import calc_dma_latency
from perf_model.L4_evaluation.g5.sdma import calc_sdma_latency
from perf_model.L4_evaluation.g5.hau import calc_hau_latency


class CoreSubsys(SimObject):
    """单核子系统: 4 引擎并行 + 同步信号"""

    def __init__(
        self,
        kernel: SimKernel,
        chip: ChipSpecImpl,
        core_id: int,
        clock_name: str,
        bus_delay_fn=None,
    ) -> None:
        super().__init__(kernel=kernel, name=f"core_{core_id}", clock_name=clock_name)
        self._chip = chip
        self.core_id = core_id
        self._bus_delay_fn = bus_delay_fn  # (src_core, dst_core, bytes) -> delay_ns

        # 同步信号
        self._tiu_sync_id: int = 0
        self._tdma_sync_id: int = 0
        self._sdma_sync_id: int = 0
        self._hau_sync_id: int = 0

        # 引擎忙状态
        self._tiu_busy: bool = False
        self._dma_busy: bool = False
        self._sdma_busy: bool = False
        self._hau_busy: bool = False

        # 指令队列索引
        self._tiu_idx: int = 0
        self._dma_idx: int = 0
        self._sdma_idx: int = 0
        self._hau_idx: int = 0

        # 指令队列
        self._tiu_cmds = []
        self._dma_cmds = []
        self._sdma_cmds = []
        self._hau_cmds = []

        # 仿真记录
        self._records: list[SimRecord] = []

    def load_instructions(self, instr: CoreInstructions) -> None:
        """加载指令并立即尝试发射"""
        self._tiu_cmds = list(instr.tiu_cmds)
        self._dma_cmds = list(instr.dma_cmds)
        self._sdma_cmds = list(instr.sdma_cmds)
        self._hau_cmds = list(instr.hau_cmds)
        self._tiu_idx = 0
        self._dma_idx = 0
        self._sdma_idx = 0
        self._hau_idx = 0
        # 在 t=0 (当前时间) 尝试发射
        self.schedule(0.0, self._try_issue_all)

    def get_records(self) -> list[SimRecord]:
        return list(self._records)

    def get_sync_id(self, engine: str) -> int:
        """获取指定引擎的 sync_id (供 Bus 等外部模块查询)"""
        if engine == "tdma":
            return self._tdma_sync_id
        if engine == "tiu":
            return self._tiu_sync_id
        if engine == "sdma":
            return self._sdma_sync_id
        if engine == "hau":
            return self._hau_sync_id
        return 0

    # ---- 内部: 发射逻辑 ----

    def _try_issue_all(self) -> None:
        self._try_issue_tiu()
        self._try_issue_dma()
        self._try_issue_sdma()
        self._try_issue_hau()

    def _try_issue_tiu(self) -> None:
        if self._tiu_busy or self._tiu_idx >= len(self._tiu_cmds):
            return
        cmd = self._tiu_cmds[self._tiu_idx]
        if cmd.cmd_id_dep <= self._tdma_sync_id:
            self._tiu_idx += 1
            self._tiu_busy = True
            result = calc_tiu_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="TIU", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                flops=result.flops,
                source_op_id=cmd.source_op_id,
            ))
            self.schedule_at(end, lambda: self._on_tiu_finish(cmd.cmd_id))

    def _on_tiu_finish(self, cmd_id: int) -> None:
        self._tiu_sync_id = cmd_id
        self._tiu_busy = False
        self._try_issue_all()

    def _try_issue_dma(self) -> None:
        if self._dma_busy or self._dma_idx >= len(self._dma_cmds):
            return
        cmd = self._dma_cmds[self._dma_idx]
        if cmd.cmd_id_dep <= self._tiu_sync_id:
            self._dma_idx += 1
            self._dma_busy = True
            result = calc_dma_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="DMA", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                data_bytes=result.data_bytes,
                direction=cmd.direction,
                source_op_id=cmd.source_op_id,
            ))
            self.schedule_at(end, lambda: self._on_dma_finish(cmd.cmd_id))

    def _on_dma_finish(self, cmd_id: int) -> None:
        self._tdma_sync_id = cmd_id
        self._dma_busy = False
        self._try_issue_all()

    def _try_issue_sdma(self) -> None:
        if self._sdma_busy or self._sdma_idx >= len(self._sdma_cmds):
            return
        cmd = self._sdma_cmds[self._sdma_idx]
        dep_val = self.get_sync_id(cmd.dep_engine)
        if cmd.cmd_id_dep <= dep_val:
            self._sdma_idx += 1
            self._sdma_busy = True
            result = calc_sdma_latency(cmd, self._chip)
            # Bus 距离延迟 (如有 bus_delay_fn)
            bus_delay = 0.0
            if self._bus_delay_fn is not None:
                bus_delay = self._bus_delay_fn(cmd.src_core_id, cmd.dst_core_id, cmd.data_bytes)
            start = self.now()
            end = start + result.latency_ns + bus_delay
            self._records.append(SimRecord(
                engine="SDMA", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                data_bytes=result.data_bytes,
                source_op_id=cmd.source_op_id,
            ))
            self.schedule_at(end, lambda: self._on_sdma_finish(cmd.cmd_id))

    def _on_sdma_finish(self, cmd_id: int) -> None:
        self._sdma_sync_id = cmd_id
        self._sdma_busy = False
        self._try_issue_all()

    def _try_issue_hau(self) -> None:
        if self._hau_busy or self._hau_idx >= len(self._hau_cmds):
            return
        cmd = self._hau_cmds[self._hau_idx]
        dep_val = self.get_sync_id(cmd.dep_engine)
        if cmd.cmd_id_dep <= dep_val:
            self._hau_idx += 1
            self._hau_busy = True
            result = calc_hau_latency(cmd, self._chip)
            start = self.now()
            end = start + result.latency_ns
            self._records.append(SimRecord(
                engine="HAU", cmd_id=cmd.cmd_id,
                start_ns=start, end_ns=end,
                source_op_id=cmd.source_op_id,
            ))
            self.schedule_at(end, lambda: self._on_hau_finish(cmd.cmd_id))

    def _on_hau_finish(self, cmd_id: int) -> None:
        self._hau_sync_id = cmd_id
        self._hau_busy = False
        self._try_issue_all()
```

### Step 4: 运行测试

```bash
cd backend && python3 tests/test_g5_core_subsys.py
```
Expected: `[PASS] All CoreSubsys tests passed`

### Step 5: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/chip/ backend/tests/test_g5_core_subsys.py
git commit -m "feat(perf_model): 添加 CoreSubsys 单核子系统 (SimObject 封装)"
```

---

## Task 4: BusModel — NxM 总线互联

**Files:**
- Create: `backend/perf_model/L4_evaluation/g5/chip/bus.py`
- Test: `backend/tests/test_g5_bus.py`

### Step 1: 编写 BusModel 测试

```python
# backend/tests/test_g5_bus.py
"""BusModel 单元测试"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from perf_model.L4_evaluation.g5.chip.bus import BusModel


def test_manhattan_distance():
    """Manhattan 距离计算"""
    bus = BusModel(
        core_count=64,
        mesh_dims=(8, 8),
        base_latency_cycles=45,
        frequency_ghz=1.0,
    )
    # 同核距离 = 0
    assert bus.manhattan_distance(0, 0) == 0
    # 相邻核
    d = bus.manhattan_distance(0, 1)
    assert d > 0
    print(f"[PASS] distance(0,1)={d}")


def test_same_core_zero_delay():
    """同核通信零延迟"""
    bus = BusModel(
        core_count=64,
        mesh_dims=(8, 8),
        base_latency_cycles=45,
        frequency_ghz=1.0,
    )
    delay = bus.get_delay_ns(src_core=0, dst_core=0, data_bytes=1024)
    assert delay == 0.0
    print("[PASS] same core -> 0 delay")


def test_cross_core_delay():
    """跨核通信有延迟"""
    bus = BusModel(
        core_count=64,
        mesh_dims=(8, 8),
        base_latency_cycles=45,
        frequency_ghz=1.0,
    )
    delay = bus.get_delay_ns(src_core=0, dst_core=63, data_bytes=1024)
    assert delay > 0.0
    # 对角线距离最大
    delay_adj = bus.get_delay_ns(src_core=0, dst_core=1, data_bytes=1024)
    assert delay > delay_adj, "Diagonal should be further than adjacent"
    print(f"[PASS] cross-core: adj={delay_adj:.1f}ns, diag={delay:.1f}ns")


def test_small_mesh():
    """2x2 小网格验证"""
    bus = BusModel(
        core_count=4,
        mesh_dims=(2, 2),
        base_latency_cycles=10,
        frequency_ghz=1.0,
    )
    # core 0=(0,0), core 1=(1,0), core 2=(0,1), core 3=(1,1)
    d01 = bus.manhattan_distance(0, 1)  # |1-0|+|0-0| = 1
    d03 = bus.manhattan_distance(0, 3)  # |1-0|+|1-0| = 2
    assert d01 == 1
    assert d03 == 2
    # delay = distance * base_latency_cycles / freq
    delay01 = bus.get_delay_ns(0, 1, 0)
    assert delay01 == 10.0  # 1 * 10 / 1.0
    print(f"[PASS] 2x2 mesh: d01={d01}, d03={d03}, delay01={delay01}")


if __name__ == "__main__":
    test_manhattan_distance()
    test_same_core_zero_delay()
    test_cross_core_delay()
    test_small_mesh()
    print("[PASS] All BusModel tests passed")
```

### Step 2: 实现 BusModel

```python
# backend/perf_model/L4_evaluation/g5/chip/bus.py
"""Bus NxM 总线互联模型

对标 TPUPerf simple_bus:
- 2D mesh 拓扑
- Manhattan 距离延迟: delay = distance * base_latency_cycles / frequency
- SG2262: 8x8 mesh, 45 cycles/hop, 1.0 GHz

参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 6.5
"""
from __future__ import annotations


class BusModel:
    """NxM 总线互联 (2D mesh)"""

    def __init__(
        self,
        core_count: int,
        mesh_dims: tuple[int, int],
        base_latency_cycles: int,
        frequency_ghz: float,
    ) -> None:
        self._core_count = core_count
        self._cols, self._rows = mesh_dims  # (cols, rows)
        self._base_latency_cycles = base_latency_cycles
        self._frequency_ghz = frequency_ghz

        if self._cols * self._rows < core_count:
            raise ValueError(
                f"Mesh {mesh_dims} ({self._cols * self._rows} slots) "
                f"cannot hold {core_count} cores"
            )

    def _core_to_xy(self, core_id: int) -> tuple[int, int]:
        """core_id -> (x, y) 坐标"""
        x = core_id % self._cols
        y = core_id // self._cols
        return (x, y)

    def manhattan_distance(self, src_core: int, dst_core: int) -> int:
        """两核间 Manhattan 距离"""
        sx, sy = self._core_to_xy(src_core)
        dx, dy = self._core_to_xy(dst_core)
        return abs(sx - dx) + abs(sy - dy)

    def get_delay_ns(self, src_core: int, dst_core: int, data_bytes: int) -> float:
        """计算 Bus 延迟 (ns)

        delay = manhattan_distance * base_latency_cycles / frequency_ghz
        同核通信返回 0.
        """
        if src_core == dst_core:
            return 0.0
        dist = self.manhattan_distance(src_core, dst_core)
        return dist * self._base_latency_cycles / self._frequency_ghz
```

### Step 3: 更新 `chip/__init__.py` 导出

在 `chip/__init__.py` 中添加 `BusModel` 导出。

### Step 4: 运行测试

```bash
cd backend && python3 tests/test_g5_bus.py
```
Expected: `[PASS] All BusModel tests passed`

### Step 5: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/chip/bus.py backend/perf_model/L4_evaluation/g5/chip/__init__.py backend/tests/test_g5_bus.py
git commit -m "feat(perf_model): 添加 BusModel NxM 总线互联 (Manhattan 距离延迟)"
```

---

## Task 5: SingleChip — 单芯片多核组装

**Files:**
- Create: `backend/perf_model/L4_evaluation/g5/top/__init__.py`
- Create: `backend/perf_model/L4_evaluation/g5/top/single_chip.py`
- Test: `backend/tests/test_g5_single_chip.py`

### Step 1: 编写 SingleChip 测试

```python
# backend/tests/test_g5_single_chip.py
"""SingleChip 多核组装测试"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreProgram, CoreInstructions, CommOp,
    TIUCommand, TIUOpType, DMACommand, DMADirection,
)
from perf_model.L4_evaluation.g5.top.single_chip import SingleChipSim
from perf_model.L4_evaluation.g5.sim_engine import SimRecord


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def test_single_core_via_single_chip():
    """单核 via SingleChip: 结果应与旧引擎一致"""
    chip = load_chip()

    program = CoreProgram(
        cores=[
            CoreInstructions(
                core_id=0,
                tiu_cmds=[
                    TIUCommand(
                        cmd_id=1, cmd_id_dep=0,
                        op_type=TIUOpType.MM2_NN,
                        result_addr=0, operand_addrs=[0, 0],
                        tile_m=16, tile_n=8, tile_k=32,
                        precision="BF16", source_op_id="op0",
                    ),
                ],
                dma_cmds=[
                    DMACommand(
                        cmd_id=1, cmd_id_dep=0,
                        direction=DMADirection.DDR_TO_LMEM,
                        src_addr=0, dst_addr=0,
                        data_bytes=4096, elem_bytes=2,
                        source_op_id="op0",
                    ),
                ],
                sdma_cmds=[],
                hau_cmds=[],
            ),
        ],
        comm_schedule=[],
        metadata={},
    )

    sim = SingleChipSim(chip)
    records = sim.simulate(program)

    tiu_records = [r for r in records if r.engine == "TIU"]
    dma_records = [r for r in records if r.engine == "DMA"]
    assert len(tiu_records) == 1
    assert len(dma_records) == 1
    print(f"[PASS] single core: {len(records)} records")


def test_multi_core_independent():
    """2 核独立执行: 并行, 总时间 = max(core0, core1)"""
    chip = load_chip()

    # 两核各一条 TIU, 无依赖
    program = CoreProgram(
        cores=[
            CoreInstructions(
                core_id=0,
                tiu_cmds=[
                    TIUCommand(
                        cmd_id=1, cmd_id_dep=0,
                        op_type=TIUOpType.MM2_NN,
                        result_addr=0, operand_addrs=[0, 0],
                        tile_m=16, tile_n=8, tile_k=32,
                        precision="BF16", source_op_id="core0_op",
                    ),
                ],
                dma_cmds=[], sdma_cmds=[], hau_cmds=[],
            ),
            CoreInstructions(
                core_id=1,
                tiu_cmds=[
                    TIUCommand(
                        cmd_id=1, cmd_id_dep=0,
                        op_type=TIUOpType.MM2_NN,
                        result_addr=0, operand_addrs=[0, 0],
                        tile_m=16, tile_n=8, tile_k=32,
                        precision="BF16", source_op_id="core1_op",
                    ),
                ],
                dma_cmds=[], sdma_cmds=[], hau_cmds=[],
            ),
        ],
        comm_schedule=[],
        metadata={},
    )

    sim = SingleChipSim(chip)
    records = sim.simulate(program)

    assert len(records) == 2
    # 两核都从 t=0 开始 (独立, 无 bus 交互)
    assert records[0].start_ns == 0.0
    assert records[1].start_ns == 0.0
    # 相同计算, 相同结束时间
    assert records[0].end_ns == records[1].end_ns
    print(f"[PASS] 2-core independent: {records[0].end_ns:.1f} ns each")


if __name__ == "__main__":
    test_single_core_via_single_chip()
    test_multi_core_independent()
    print("[PASS] All SingleChip tests passed")
```

### Step 2: 实现 SingleChip

```python
# backend/perf_model/L4_evaluation/g5/top/__init__.py
"""G5 顶层组装"""
from perf_model.L4_evaluation.g5.top.single_chip import SingleChipSim

__all__ = ["SingleChipSim"]
```

```python
# backend/perf_model/L4_evaluation/g5/top/single_chip.py
"""SingleChip: 单芯片多核组装

组装流程:
  1. 创建 SimKernel
  2. 创建 BusModel (2D mesh, Manhattan 距离延迟)
  3. 创建 N 个 CoreSubsys (共享 kernel + bus)
  4. 加载各核指令
  5. kernel.run()
  6. 收集所有 SimRecord

对标: TPUPerf tpuManyCore.cc
参考: docs/plans/2026-02-19-g5-full-architecture-design.md Section 8.1
"""
from __future__ import annotations

from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import CoreProgram
from perf_model.L4_evaluation.g5.kernel.sim_kernel import SimKernel
from perf_model.L4_evaluation.g5.chip.core_subsys import CoreSubsys
from perf_model.L4_evaluation.g5.chip.bus import BusModel
from perf_model.L4_evaluation.g5.sim_engine import SimRecord


# SG2262 NoC 默认参数
_DEFAULT_MESH_COLS = 8
_DEFAULT_MESH_ROWS = 8
_DEFAULT_BUS_LATENCY_CYCLES = 45


class SingleChipSim:
    """单芯片多核仿真器"""

    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip

    def simulate(self, program: CoreProgram) -> list[SimRecord]:
        """仿真 CoreProgram (多核)

        Args:
            program: 多核指令程序

        Returns:
            所有核的 SimRecord 列表 (按时间排序)
        """
        if not program.cores:
            return []

        core_count = len(program.cores)
        tiu_freq = self._chip.get_tiu_frequency()

        # 1. 创建 SimKernel + 注册时钟
        kernel = SimKernel()
        kernel.add_clock("tpu", frequency_ghz=tiu_freq)

        # 2. 创建 BusModel
        # 从芯片互联配置读取参数, 没有则用默认值
        noc_cfg = self._chip.interconnect.raw_config if self._chip.interconnect else {}
        mesh_cols = _DEFAULT_MESH_COLS
        mesh_rows = _DEFAULT_MESH_ROWS
        bus_latency_cycles = _DEFAULT_BUS_LATENCY_CYCLES

        # 对于单核仿真, mesh 大小不影响 (无跨核 SDMA)
        if core_count == 1:
            mesh_cols = 1
            mesh_rows = 1

        bus = BusModel(
            core_count=core_count,
            mesh_dims=(mesh_cols, mesh_rows),
            base_latency_cycles=bus_latency_cycles,
            frequency_ghz=tiu_freq,
        )

        # 3. 创建各核 CoreSubsys
        cores: list[CoreSubsys] = []
        for core_instr in program.cores:
            core = CoreSubsys(
                kernel=kernel,
                chip=self._chip,
                core_id=core_instr.core_id,
                clock_name="tpu",
                bus_delay_fn=bus.get_delay_ns,
            )
            core.load_instructions(core_instr)
            cores.append(core)

        # 4. 运行仿真
        kernel.run()

        # 5. 收集所有记录
        all_records: list[SimRecord] = []
        for core in cores:
            all_records.extend(core.get_records())

        # 按开始时间排序
        all_records.sort(key=lambda r: (r.start_ns, r.cmd_id))
        return all_records
```

### Step 3: 运行测试

```bash
cd backend && python3 tests/test_g5_single_chip.py
```
Expected: `[PASS] All SingleChip tests passed`

### Step 4: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/top/ backend/tests/test_g5_single_chip.py
git commit -m "feat(perf_model): 添加 SingleChipSim 单芯片多核组装"
```

---

## Task 6: 重构 G5SimEngine 委托 SingleChip

**Files:**
- Modify: `backend/perf_model/L4_evaluation/g5/sim_engine.py`
- Modify: `backend/perf_model/L4_evaluation/g5/__init__.py`

### Step 1: 重构 sim_engine.py

保留外部 API (`G5SimEngine.simulate(program) -> list[SimRecord]`)，内部委托 `SingleChipSim`。

保留 `SimRecord` 和 `EventType` 导出 (其他模块引用)。删除旧的 `_simulate_core` 方法和 `SimEvent` 类 (不再需要)。

关键变更:
```python
# sim_engine.py 重构后
class G5SimEngine:
    def __init__(self, chip: ChipSpecImpl) -> None:
        self._chip = chip

    def simulate(self, program: CoreProgram) -> list[SimRecord]:
        if not program.cores:
            return []
        sim = SingleChipSim(self._chip)
        return sim.simulate(program)
```

### Step 2: 运行现有测试回归

```bash
cd backend && python3 tests/test_g5_matmul_e2e.py && python3 tests/test_g5_moe_e2e.py
```

Expected: 两个 E2E 测试全部通过，输出结果与重构前一致。

**验证点:**
- FLOPs 数值完全一致 (纯计算逻辑不变)
- 时间数值完全一致 (同步模型不变)
- MFU/MBU 完全一致

### Step 3: 运行所有新测试

```bash
cd backend && python3 tests/test_g5_simkernel.py && python3 tests/test_g5_core_subsys.py && python3 tests/test_g5_bus.py && python3 tests/test_g5_single_chip.py
```

### Step 4: 提交

```bash
git add backend/perf_model/L4_evaluation/g5/sim_engine.py backend/perf_model/L4_evaluation/g5/__init__.py
git commit -m "refactor(perf_model): G5SimEngine 内部委托 SingleChipSim (外部 API 不变)"
```

---

## Task 7: SG2262 NoC 配置扩展

**Files:**
- Modify: `backend/perf_model/configs/chips/SG2262.yaml`
- Modify: `backend/perf_model/L4_evaluation/g5/top/single_chip.py`

### Step 1: 扩展 SG2262.yaml

在 SG2262.yaml 的 interconnect 部分添加 NoC 配置:

```yaml
# === 新增: NoC 配置 (2D mesh) ===
noc:
  topology: "2d_mesh"
  mesh_cols: 8
  mesh_rows: 8
  base_latency_cycles: 45   # Manhattan 距离系数 (cycles/hop)
```

### Step 2: SingleChipSim 读取 NoC 配置

修改 `single_chip.py` 从 `chip.interconnect` 或芯片原始配置中读取 noc 参数，取代硬编码默认值。

### Step 3: 运行全部测试回归

```bash
cd backend && python3 tests/test_g5_matmul_e2e.py && python3 tests/test_g5_moe_e2e.py && python3 tests/test_g5_simkernel.py && python3 tests/test_g5_core_subsys.py && python3 tests/test_g5_bus.py && python3 tests/test_g5_single_chip.py
```

### Step 4: 提交

```bash
git add backend/perf_model/configs/chips/SG2262.yaml backend/perf_model/L4_evaluation/g5/top/single_chip.py
git commit -m "feat(perf_model): SG2262 NoC 配置扩展 (2D mesh, 8x8, 45 cycles/hop)"
```

---

## Task 8: 多核 E2E 测试 — 多核 MatMul 验证

**Files:**
- Create: `backend/tests/test_g5_multicore_e2e.py`

### Step 1: 编写多核端到端测试

测试场景: 4 核并行执行独立 MatMul，验证:
1. 4 核结果正确收集
2. 总 FLOPs = 4 * 单核 FLOPs
3. 总时间 ≈ 单核时间 (独立并行)

```python
# backend/tests/test_g5_multicore_e2e.py
"""G5 多核 E2E 测试"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from perf_model.L2_arch.chip import ChipSpecImpl
from perf_model.L3_mapping.g5.program import (
    CoreProgram, CoreInstructions,
    TIUCommand, TIUOpType, DMACommand, DMADirection,
)
from perf_model.L4_evaluation.g5.sim_engine import G5SimEngine
from perf_model.L4_evaluation.g5.adapter import G5ResultAdapter


def load_chip(name: str = "SG2262") -> ChipSpecImpl:
    config_path = backend_dir / "perf_model" / "configs" / "chips" / f"{name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return ChipSpecImpl.from_config(name, config)


def _make_core_instrs(core_id: int) -> CoreInstructions:
    """为单核生成 MatMul 指令 (16x8x32, BF16)"""
    return CoreInstructions(
        core_id=core_id,
        tiu_cmds=[
            TIUCommand(
                cmd_id=1, cmd_id_dep=0,
                op_type=TIUOpType.MM2_NN,
                result_addr=0, operand_addrs=[0, 0],
                tile_m=16, tile_n=8, tile_k=32,
                precision="BF16",
                source_op_id=f"core{core_id}_mm",
            ),
        ],
        dma_cmds=[
            DMACommand(
                cmd_id=1, cmd_id_dep=0,
                direction=DMADirection.DDR_TO_LMEM,
                src_addr=0, dst_addr=0,
                data_bytes=4096, elem_bytes=2,
                source_op_id=f"core{core_id}_mm",
            ),
        ],
        sdma_cmds=[],
        hau_cmds=[],
    )


def test_multicore_parallel():
    """4 核并行 MatMul"""
    chip = load_chip()
    num_cores = 4

    program = CoreProgram(
        cores=[_make_core_instrs(i) for i in range(num_cores)],
        comm_schedule=[],
        metadata={"test": "multicore"},
    )

    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    # 验证: 4 核各 2 条记录 (1 TIU + 1 DMA)
    assert len(records) == num_cores * 2, f"Expected {num_cores * 2}, got {len(records)}"

    # 验证: 所有核从 t=0 开始 (独立无依赖)
    for r in records:
        assert r.start_ns == 0.0, f"Expected start=0, got {r.start_ns}"

    # 验证: 总 FLOPs = 4 * 单核
    total_flops = sum(r.flops for r in records)
    single_flops = 2 * 16 * 8 * 32
    assert total_flops == num_cores * single_flops

    # 验证: source_op_id 包含不同核
    op_ids = set(r.source_op_id for r in records)
    assert len(op_ids) == num_cores, f"Expected {num_cores} unique op_ids, got {op_ids}"

    print(f"[PASS] {num_cores}-core parallel: {len(records)} records, {total_flops} total flops")


def test_single_core_backward_compat():
    """单核回归: 与原来 G5SimEngine 行为一致"""
    chip = load_chip()
    program = CoreProgram(
        cores=[_make_core_instrs(0)],
        comm_schedule=[],
        metadata={},
    )

    engine = G5SimEngine(chip)
    records = engine.simulate(program)

    assert len(records) == 2
    tiu = [r for r in records if r.engine == "TIU"][0]
    dma = [r for r in records if r.engine == "DMA"][0]
    assert tiu.flops == 2 * 16 * 8 * 32
    assert dma.data_bytes == 4096
    print(f"[PASS] single core compat: TIU={tiu.end_ns:.1f}ns, DMA={dma.end_ns:.1f}ns")


if __name__ == "__main__":
    test_single_core_backward_compat()
    test_multicore_parallel()
    print("[PASS] All multicore E2E tests passed")
```

### Step 2: 运行测试

```bash
cd backend && python3 tests/test_g5_multicore_e2e.py
```

### Step 3: 提交

```bash
git add backend/tests/test_g5_multicore_e2e.py
git commit -m "test(perf_model): 多核 G5 E2E 测试 (4核并行 MatMul + 单核回归)"
```

---

## 完成检查清单

- [ ] Task 1: SimKernel 全局事件队列
- [ ] Task 2: SimObject 基类
- [ ] Task 3: CoreSubsys 单核子系统
- [ ] Task 4: BusModel NxM 总线
- [ ] Task 5: SingleChipSim 多核组装
- [ ] Task 6: G5SimEngine 重构委托
- [ ] Task 7: SG2262 NoC 配置
- [ ] Task 8: 多核 E2E 测试

**回归验证**: 每个 Task 完成后运行:
```bash
cd backend && python3 tests/test_g5_matmul_e2e.py && python3 tests/test_g5_moe_e2e.py
```
确保现有测试不被破坏。
