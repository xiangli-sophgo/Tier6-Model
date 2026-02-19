# 10. G5 仿真器统计框架设计

## 1. 概述

### 1.1 目标

为 G5 指令级仿真器设计一套层次化的统计信息收集框架，支持：

- **性能调优/瓶颈定位**: 每条指令的 stall 原因、流水线利用率、带宽争用等细粒度信息
- **架构探索/对比**: 不同芯片配置下的全局统计指标 (IPC、利用率、带宽效率)
- **前端可视化**: 统计数据存入数据库，通过图表展示

### 1.2 设计原则

- **借鉴 gem5 Stats 体系**: 层次化 StatGroup + 类型化统计量 (Scalar/Vector/Distribution/Formula)
- **生产者-消费者解耦**: 模块只管往自己的 StatGroup 写数据，框架负责收集和输出
- **零框架修改扩展**: 新增模块/统计量不需要改框架、adapter、数据库 schema
- **按需实现**: 先实现 Scalar + Vector 覆盖 90% 需求，Distribution 和 Formula 后续按需加

### 1.3 参考

| 仿真器 | 统计机制 | 特点 |
|--------|---------|------|
| **gem5** | Stats::Scalar/Vector/Distribution/Histogram/Formula, Stats::Group 层次化容器, ADD_STAT 宏, dump/reset | 最完善，工业标准 |
| **TPUPerf** | 自定义 Profiler, per-instruction record (start_cycle, end_cycle, stall_cycle, bank_conflict_ratio) | 轻量，面向 TPU |
| **SystemC** | sc_trace (VCD 波形), sc_report (诊断) | 无内建性能统计 |

---

## 2. 统计类型体系

### 2.1 ScalarStat — 标量统计

单个数值：计数器、累加器、峰值、最终状态值。

```python
@dataclass
class ScalarStat:
    """标量统计量"""
    name: str
    desc: str
    value: float = 0.0

    def inc(self, delta: float = 1.0) -> None:
        """累加"""
        self.value += delta

    def set_max(self, v: float) -> None:
        """取最大值"""
        self.value = max(self.value, v)
```

**使用示例**:

```python
total_cycles = ScalarStat("total_cycles", "TIU 总计算周期")
total_cycles.inc(14961)

cmd_count = ScalarStat("cmd_count", "执行的指令总数")
cmd_count.inc()  # +1

peak_queue_depth = ScalarStat("peak_queue_depth", "事件队列最大深度")
peak_queue_depth.set_max(128)
```

### 2.2 VectorStat — 向量统计

按标签分组的一组数值，等价于 `dict[str, float]`。

```python
@dataclass
class VectorStat:
    """向量统计量 (按标签分组)"""
    name: str
    desc: str
    bins: dict[str, float] = field(default_factory=dict)

    def inc(self, label: str, delta: float = 1.0) -> None:
        """指定标签累加"""
        self.bins[label] = self.bins.get(label, 0.0) + delta
```

**使用示例**:

```python
# TIU 按精度分的周期数
cycles_by_precision = VectorStat("cycles_by_precision", "按精度分的计算周期")
cycles_by_precision.inc("BF16", 12000)
cycles_by_precision.inc("INT8", 3200)
# dump → {"BF16": 12000, "INT8": 3200}

# DMA 按方向分的数据量
bytes_by_direction = VectorStat("bytes_by_direction", "按方向分的搬运字节数")
bytes_by_direction.inc("DDR_TO_LMEM", 262144)
bytes_by_direction.inc("LMEM_TO_DDR", 131072)

# SDMA 按操作类型分的次数
cmd_by_type = VectorStat("cmd_by_type", "按操作类型分的指令数")
cmd_by_type.inc("TENSOR", 1)
```

### 2.3 DistributionStat — 分布统计 (Phase 2)

记录一个指标多次采样后的分布：min/max/mean + 分桶计数，用于画直方图。

```python
@dataclass
class DistributionStat:
    """分布统计量 (直方图)"""
    name: str
    desc: str
    bucket_edges: list[float]           # 分桶边界
    _buckets: list[int] = field(...)    # 各桶计数
    _min: float = float('inf')
    _max: float = float('-inf')
    _sum: float = 0.0
    _count: int = 0

    def sample(self, value: float) -> None:
        """记录一次采样"""
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1
        # 放入对应桶
        ...
```

**使用示例**:

```python
# DMA 每次搬运的数据量分布
dma_transfer_size = DistributionStat(
    "transfer_size", "DMA 单次搬运数据量分布",
    bucket_edges=[1024, 4096, 65536, 262144]  # 1KB, 4KB, 64KB, 256KB
)
dma_transfer_size.sample(32768)   # 一次 32KB 搬运
dma_transfer_size.sample(256)     # 一次 256B 搬运
# dump → {"min": 256, "max": 32768, "mean": 16512, "count": 2, "buckets": [1, 0, 1, 0]}
```

### 2.4 FormulaStat — 公式统计 (Phase 2)

引用其他统计量通过公式计算，不直接采集。

```python
@dataclass
class FormulaStat:
    """公式统计量 (从其他统计量推导)"""
    name: str
    desc: str
    formula: Callable[[], float]   # 无参函数，引用闭包中的其他 Stat

    @property
    def value(self) -> float:
        return self.formula()
```

**使用示例**:

```python
cmd_count = ScalarStat("cmd_count", "指令总数")
total_cycles = ScalarStat("total_cycles", "总周期数")

ipc = FormulaStat(
    "ipc", "Instructions Per Cycle",
    formula=lambda: cmd_count.value / total_cycles.value if total_cycles.value > 0 else 0
)
```

### 2.5 类型总结

| 类型 | 数据形态 | 用途 | 前端图表 | 优先级 |
|------|---------|------|---------|--------|
| Scalar | 一个数 | 计数/累加/峰值 | KPI 卡片、表格 | Phase 1 |
| Vector | 一组数 (按标签) | 分类统计 | 堆叠柱状图、饼图 | Phase 1 |
| Distribution | 数值分布 | 采样直方图 | 直方图 | Phase 2 |
| Formula | 推导值 | 引用其他统计 | 同 Scalar | Phase 2 |

---

## 3. StatGroup — 层次化容器

### 3.1 设计

```python
class StatGroup:
    """统计组 — 每个模块 (SimObject) 持有一个

    支持层次化嵌套，dump() 递归收集所有子节点统计。
    """

    def __init__(self, name: str, parent: StatGroup | None = None):
        self.name = name
        self._stats: dict[str, ScalarStat | VectorStat | DistributionStat | FormulaStat] = {}
        self._children: dict[str, StatGroup] = {}
        if parent is not None:
            parent._add_child(self)

    def scalar(self, name: str, desc: str) -> ScalarStat:
        """创建并注册一个 ScalarStat"""
        stat = ScalarStat(name, desc)
        self._stats[name] = stat
        return stat

    def vector(self, name: str, desc: str) -> VectorStat:
        """创建并注册一个 VectorStat"""
        stat = VectorStat(name, desc)
        self._stats[name] = stat
        return stat

    def _add_child(self, child: "StatGroup") -> None:
        self._children[child.name] = child

    def dump(self) -> dict[str, Any]:
        """递归收集所有统计 -> 扁平化 dict

        key 格式: "parent.child.stat_name"
        """
        result: dict[str, Any] = {}
        for name, stat in self._stats.items():
            if isinstance(stat, VectorStat):
                result[f"{self.name}.{name}"] = dict(stat.bins)
            elif isinstance(stat, FormulaStat):
                result[f"{self.name}.{name}"] = stat.value
            else:
                result[f"{self.name}.{name}"] = stat.value
        for child in self._children.values():
            for key, value in child.dump().items():
                result[f"{self.name}.{key}"] = value
        return result

    def reset(self) -> None:
        """重置所有统计 (用于分阶段统计，如 prefill/decode)"""
        for stat in self._stats.values():
            if isinstance(stat, ScalarStat):
                stat.value = 0.0
            elif isinstance(stat, VectorStat):
                stat.bins.clear()
        for child in self._children.values():
            child.reset()
```

### 3.2 层次结构

```
kernel (SimKernel)
├── kernel.total_sim_time_ns          Scalar  仿真总时长
├── kernel.total_events               Scalar  事件总数
│
├── bus (BusModel)
│   ├── bus.total_transfers           Scalar  总传输次数
│   ├── bus.total_bytes               Scalar  总传输数据量
│   └── bus.avg_hop_count             Formula 平均跳数
│
├── core0 (CoreSubsys)
│   ├── core0.total_instructions      Scalar  指令总数
│   ├── core0.busy_ns                 Scalar  至少一个引擎忙的时间
│   ├── core0.idle_ns                 Scalar  全部引擎空闲的时间
│   │
│   ├── tiu (TIU Engine)
│   │   ├── core0.tiu.cmd_count       Scalar  TIU 指令数
│   │   ├── core0.tiu.compute_cycles  Scalar  纯计算周期
│   │   ├── core0.tiu.init_cycles     Scalar  流水线填充周期
│   │   ├── core0.tiu.total_flops     Scalar  总浮点运算量
│   │   └── core0.tiu.cycles_by_prec  Vector  按精度分的周期 {BF16: N, INT8: M}
│   │
│   ├── dma (DMA Engine)
│   │   ├── core0.dma.cmd_count       Scalar  DMA 指令数
│   │   ├── core0.dma.bytes_read      Scalar  DDR->LMEM 读取量
│   │   ├── core0.dma.bytes_write     Scalar  LMEM->DDR 写入量
│   │   ├── core0.dma.startup_ns      Scalar  启动延迟总和
│   │   ├── core0.dma.transfer_ns     Scalar  数据传输时间总和
│   │   └── core0.dma.bytes_by_dir    Vector  按方向分的字节数
│   │
│   ├── sdma (SDMA Engine)
│   │   ├── core0.sdma.cmd_count      Scalar  SDMA 指令数
│   │   ├── core0.sdma.total_bytes    Scalar  总传输字节数
│   │   ├── core0.sdma.bus_latency_ns Scalar  Bus 路由延迟总和
│   │   ├── core0.sdma.transfer_ns    Scalar  数据传输时间总和
│   │   ├── core0.sdma.hop_total      Scalar  总跳数
│   │   └── core0.sdma.cmd_by_type    Vector  按类型分 {TENSOR: N, GATHER: M}
│   │
│   └── hau (HAU Engine)
│       ├── core0.hau.cmd_count       Scalar  HAU 指令数
│       ├── core0.hau.total_elements  Scalar  处理的总元素数
│       ├── core0.hau.total_cycles    Scalar  总排序周期
│       └── core0.hau.cmd_by_op       Vector  按操作分 {SORT: N, TOP_K: M}
│
├── core1 (CoreSubsys)
│   └── ... (同 core0 结构)
│
└── core63 (CoreSubsys)
    └── ...
```

### 3.3 dump() 输出示例

仿真结束后调用 `kernel.stats.dump()`:

```python
{
    # 全局
    "kernel.total_sim_time_ns": 14961.10,
    "kernel.total_events": 128,

    # Bus
    "kernel.bus.total_transfers": 4,
    "kernel.bus.total_bytes": 524288,

    # Core 0
    "kernel.core0.total_instructions": 4,
    "kernel.core0.busy_ns": 14500.0,
    "kernel.core0.idle_ns": 461.1,

    # Core 0 - TIU
    "kernel.core0.tiu.cmd_count": 4,
    "kernel.core0.tiu.compute_cycles": 12000,
    "kernel.core0.tiu.init_cycles": 176,
    "kernel.core0.tiu.total_flops": 67108864,
    "kernel.core0.tiu.cycles_by_prec": {"BF16": 12000},

    # Core 0 - DMA
    "kernel.core0.dma.cmd_count": 3,
    "kernel.core0.dma.bytes_read": 262144,
    "kernel.core0.dma.bytes_write": 131072,
    "kernel.core0.dma.startup_ns": 300.0,
    "kernel.core0.dma.transfer_ns": 2900.0,
    "kernel.core0.dma.bytes_by_dir": {"DDR_TO_LMEM": 262144, "LMEM_TO_DDR": 131072},

    # Core 0 - SDMA
    "kernel.core0.sdma.cmd_count": 1,
    "kernel.core0.sdma.total_bytes": 32768,
    "kernel.core0.sdma.bus_latency_ns": 45.0,
    "kernel.core0.sdma.transfer_ns": 120.0,
    "kernel.core0.sdma.hop_total": 1,
    "kernel.core0.sdma.cmd_by_type": {"TENSOR": 1},

    # Core 0 - HAU
    "kernel.core0.hau.cmd_count": 1,
    "kernel.core0.hau.total_elements": 256,
    "kernel.core0.hau.total_cycles": 180,
    "kernel.core0.hau.cmd_by_op": {"TOP_K": 1},

    # Core 1 ... Core 63 同理
}
```

---

## 4. 各模块统计指标详细定义

### 4.1 SimKernel 全局统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `total_sim_time_ns` | Scalar | 仿真总时长 (最后事件时间 - 首事件时间) |
| `total_events` | Scalar | 事件队列处理的事件总数 |

### 4.2 BusModel (NoC) 统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `total_transfers` | Scalar | 总传输次数 |
| `total_bytes` | Scalar | 总传输数据量 (bytes) |
| `hop_total` | Scalar | 所有传输的跳数之和 |
| `hop_distribution` | Distribution (Phase 2) | 跳数分布直方图 |

### 4.3 CoreSubsys 核级统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `total_instructions` | Scalar | 该核执行的指令总数 (TIU+DMA+SDMA+HAU) |
| `busy_ns` | Scalar | 至少一个引擎在工作的时间 |
| `idle_ns` | Scalar | 所有引擎空闲的时间 |
| `tiu_busy_ns` | Scalar | TIU 引擎忙碌时间 |
| `dma_busy_ns` | Scalar | DMA 引擎忙碌时间 |
| `sdma_busy_ns` | Scalar | SDMA 引擎忙碌时间 |
| `hau_busy_ns` | Scalar | HAU 引擎忙碌时间 |
| `dependency_wait_ns` | Scalar | 等待指令依赖满足的总时间 |
| `cmd_count_by_engine` | Vector | 按引擎分的指令数 {TIU: N, DMA: M, ...} |

**Phase 2 扩展**:

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `tiu_utilization` | Formula | TIU 利用率 = tiu_busy_ns / total_sim_time |
| `dma_utilization` | Formula | DMA 利用率 |
| `ipc` | Formula | 指令吞吐 = total_instructions / total_cycles |
| `stall_by_reason` | Vector | 按原因分的 stall 时间 {dependency, resource, sync} |

### 4.4 TIU 引擎统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `cmd_count` | Scalar | TIU 指令数 |
| `compute_cycles` | Scalar | 纯计算周期总和 |
| `init_cycles` | Scalar | 流水线填充周期总和 |
| `total_flops` | Scalar | 总浮点运算量 |
| `cycles_by_prec` | Vector | 按精度分的周期 {BF16: N, INT8: M, FP32: K} |
| `cmd_by_op` | Vector | 按操作分的指令数 {MM2_NN: N, CONV: M, SFU: K} |

**计算方式** (对标 TPUPerf tiuImpl.cc):

```
MM2_NN:
  init_cycles = 44
  ch_per_cyc = cube_k (INT8) | cube_k/2 (BF16) | cube_k/4 (FP32)
  compute_cycles = ceil(M/lane_num) * ceil(N/eu_num) * ceil(K/ch_per_cyc)
  flops = 2 * M * N * K
```

**Phase 2 扩展**:

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `bank_conflicts` | Scalar | LMEM Bank 冲突次数 |
| `init_overhead_ratio` | Formula | 流水线填充占比 = init_cycles / total_cycles |
| `achieved_flops_per_cycle` | Formula | 实际算力/周期 |

### 4.5 DMA 引擎统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `cmd_count` | Scalar | DMA 指令数 |
| `bytes_read` | Scalar | DDR -> LMEM 读取字节数 |
| `bytes_write` | Scalar | LMEM -> DDR 写入字节数 |
| `startup_ns` | Scalar | 启动延迟总和 (每次搬运的固定开销) |
| `transfer_ns` | Scalar | 数据传输时间总和 |
| `bytes_by_dir` | Vector | 按方向分的字节数 {DDR_TO_LMEM, LMEM_TO_DDR} |

**Phase 2 扩展**:

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `transfer_size_dist` | Distribution | 单次搬运数据量分布 |
| `startup_overhead_ratio` | Formula | 启动开销占比 |
| `effective_bandwidth` | Formula | 有效带宽 = total_bytes / total_transfer_ns |

### 4.6 SDMA 引擎统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `cmd_count` | Scalar | SDMA 指令数 |
| `total_bytes` | Scalar | 总传输字节数 |
| `bus_latency_ns` | Scalar | Bus 路由延迟总和 (Manhattan 距离 * hop 延迟) |
| `transfer_ns` | Scalar | 数据传输时间总和 |
| `hop_total` | Scalar | 总跳数 |
| `cmd_by_type` | Vector | 按操作类型分 {TENSOR, GATHER, SCATTER} |

### 4.7 HAU 引擎统计

| 统计量 | 类型 | 含义 |
|--------|------|------|
| `cmd_count` | Scalar | HAU 指令数 |
| `total_elements` | Scalar | 处理的总元素数 |
| `total_cycles` | Scalar | 总排序/Top-K 周期 |
| `cmd_by_op` | Vector | 按操作分 {SORT, TOP_K, UNIQUE} |

---

## 5. 集成方式

### 5.1 模块集成 (各引擎如何使用)

```python
# CoreSubsys.__init__() 中:
class CoreSubsys:
    def __init__(self, kernel: SimKernel, chip: ChipSpecImpl, core_id: int, ...):
        self.stats = StatGroup(f"core{core_id}", parent=kernel.stats)
        self._tiu_stats = StatGroup("tiu", parent=self.stats)
        self._dma_stats = StatGroup("dma", parent=self.stats)
        self._sdma_stats = StatGroup("sdma", parent=self.stats)
        self._hau_stats = StatGroup("hau", parent=self.stats)

        # 核级统计
        self._stat_total_instructions = self.stats.scalar("total_instructions", "指令总数")
        self._stat_busy_ns = self.stats.scalar("busy_ns", "引擎忙碌时间")

        # TIU 统计
        self._stat_tiu_cmd = self._tiu_stats.scalar("cmd_count", "TIU 指令数")
        self._stat_tiu_compute = self._tiu_stats.scalar("compute_cycles", "计算周期")
        self._stat_tiu_init = self._tiu_stats.scalar("init_cycles", "初始化周期")
        self._stat_tiu_flops = self._tiu_stats.scalar("total_flops", "总 FLOPs")
        self._stat_tiu_by_prec = self._tiu_stats.vector("cycles_by_prec", "按精度分的周期")
        # ... DMA, SDMA, HAU 类似
```

```python
# 引擎执行时累加统计:
def _execute_tiu(self, cmd: TIUCommand) -> None:
    result = calc_tiu_latency(cmd, self._chip)

    # 记录 SimRecord (不变)
    record = SimRecord(engine="TIU", cmd_id=cmd.cmd_id, ...)
    self._records.append(record)

    # 累加统计 (新增)
    self._stat_tiu_cmd.inc()
    self._stat_tiu_compute.inc(result.compute_cycles)
    self._stat_tiu_init.inc(result.init_cycles)
    self._stat_tiu_flops.inc(result.flops)
    self._stat_tiu_by_prec.inc(cmd.precision, result.total_cycles)
    self._stat_total_instructions.inc()
```

### 5.2 SingleChipSim 收集

```python
class SingleChipSim:
    def simulate(self, program: CoreProgram) -> list[SimRecord]:
        kernel = SimKernel()
        # SimKernel 自动创建顶层 StatGroup
        # ... 创建 cores, 运行仿真 ...

        kernel.run()

        # 收集统计
        self._last_stats = kernel.stats.dump()
        return all_records

    def get_stats(self) -> dict[str, Any]:
        """获取最近一次仿真的统计数据"""
        return self._last_stats
```

### 5.3 数据流: 仿真 -> 数据库 -> 前端

```
SingleChipSim.simulate()
    ↓
kernel.stats.dump() → dict[str, Any]
    ↓
G5ResultAdapter.convert()
    ↓ 将 stats dict 存入 EngineResult.trace_meta["stats"]
EngineResult
    ↓
ReportingEngine → PerformanceReport
    ↓
存入数据库 EvaluationResult.full_result["stats"]
    ↓
前端 API 读取 → 图表组件渲染
```

### 5.4 EngineResult 集成

```python
# G5ResultAdapter.convert() 中:
def convert(self, records: list[SimRecord], stats: dict[str, Any] | None = None) -> EngineResult:
    step_metrics = self._build_step_metrics(records)
    aggregates = self._build_aggregates(step_metrics)
    return EngineResult(
        step_metrics=step_metrics,
        aggregates=aggregates,
        granularity="instruction",
        trace_meta={
            "engine": "g5",
            "stats": stats or {},   # 统计数据存入 trace_meta
        },
    )
```

---

## 6. 扩展性分析

### 6.1 新增模块统计

以未来的 IFE (指令调度引擎) 为例，只需在模块内部创建 StatGroup:

```python
class IFEEngine:
    def __init__(self, core_stats: StatGroup):
        self.stats = StatGroup("ife", parent=core_stats)
        self.stats.scalar("fifo_full_stalls", "FIFO 满停顿次数")
        self.stats.scalar("fifo_peak_depth", "FIFO 最大深度")
        self.stats.scalar("issue_count", "发射指令数")
```

不需要修改框架、adapter、数据库 schema。dump() 自动包含新字段。

### 6.2 新增统计量

在任意模块中调用 `stats.scalar()` 或 `stats.vector()` 即可:

```python
# 一行代码新增一个统计量
self._stat_new_metric = self._tiu_stats.scalar("new_metric", "新增指标描述")
```

### 6.3 改动对比

| 操作 | gem5 式 StatGroup | SimRecord 扩展方案 |
|------|-------------------|-------------------|
| 新增模块统计 | 模块内 3-5 行 | 改 4 个文件 |
| 新增一个统计量 | 1 行 `add_stat()` | 改 SimRecord + Adapter |
| 删除统计量 | 删 1 行 | 改 4 个文件 |
| 数据库 schema | 不变 (JSON) | 可能需要迁移 |
| 前端适配 | 按需读 key | 每次都要改类型定义 |

---

## 7. 前端展示规划

### 7.1 图表类型

| 图表 | 数据来源 | 用途 |
|------|---------|------|
| **引擎利用率柱状图** | core{N}.{engine}_busy_ns / total_sim_time | 各引擎资源使用率对比 |
| **Cycle 分解饼图** | tiu.init_cycles / tiu.compute_cycles | 流水线效率分析 |
| **多核热力图** | core{0..63}.tiu_busy_ns | 各核负载均衡度一览 |
| **通信分析** | sdma.bus_latency_ns / sdma.transfer_ns | Bus 路由 vs 传输时间占比 |
| **DMA 方向堆叠图** | dma.bytes_by_dir | 读写带宽对比 |
| **精度分布饼图** | tiu.cycles_by_prec | 不同精度的计算量占比 |

### 7.2 数据库存储

统计数据作为 JSON 存入 `EvaluationResult.full_result["stats"]`，无需修改数据库 schema。

前端通过 `/api/evaluation/experiments/{id}/results` 获取完整结果，从 `full_result.stats` 读取统计数据。

---

## 8. 实现计划

### Phase 1: 基础框架 + Scalar/Vector (优先)

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | 实现 ScalarStat, VectorStat, StatGroup | `kernel/stats.py` (新增) |
| 2 | SimKernel 集成顶层 StatGroup | `kernel/sim_kernel.py` |
| 3 | CoreSubsys 注册核级 + 引擎级统计 | `chip/core_subsys.py` |
| 4 | 各引擎执行逻辑中累加统计 | `chip/core_subsys.py` |
| 5 | BusModel 注册 Bus 统计 | `chip/bus.py` |
| 6 | SingleChipSim 收集 dump 结果 | `top/single_chip.py` |
| 7 | G5ResultAdapter 透传 stats | `adapter.py` |
| 8 | 单元测试 | `tests/test_g5_stats.py` |

### Phase 2: Distribution + Formula + 前端 (后续)

| 步骤 | 内容 |
|------|------|
| 1 | 实现 DistributionStat, FormulaStat |
| 2 | 各引擎添加分布统计 (DMA transfer size 等) |
| 3 | CoreSubsys 添加公式统计 (utilization, IPC) |
| 4 | 前端统计图表组件 |
| 5 | Stall 分类统计 (dependency / resource / sync) |

---

## 9. 文件结构

```
backend/perf_model/L4_evaluation/g5/
├── kernel/
│   ├── sim_kernel.py      # 集成 StatGroup 根节点
│   ├── sim_record.py      # 不变
│   └── stats.py           # [新增] ScalarStat, VectorStat, StatGroup
├── chip/
│   ├── core_subsys.py     # 注册核级 + 引擎级统计，执行时累加
│   └── bus.py             # 注册 Bus 统计
├── top/
│   └── single_chip.py     # 收集 dump 结果
├── adapter.py             # 透传 stats 到 EngineResult
├── tiu.py                 # 延迟计算返回 cycles 分解 (不变)
├── dma.py                 # 同上
├── sdma.py                # 同上
└── hau.py                 # 同上
```
