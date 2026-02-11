# L5 Reporting -- 报告与可视化层

## 功能概述

L5 是评估结果的唯一出口，负责:
- 将 L4 `EngineResult` 汇总为结构化 `ReportingReport`
- 成本分析 (服务器 + 互联 + 运营)
- 内存分解 (权重 / KV Cache / 激活)
- Roofline 性能画像 (算力/带宽瓶颈)
- Gantt 时序图 (计算/通信/等待分段)
- 链路流量分析 (c2c/b2b/r2r/p2p 利用率)
- JSON 导出

不在范围: 不做评估计算 (由 L4)，不做前端渲染。

## 模块清单

| 模块 | 职责 |
|------|------|
| `engine.py` | ReportingEngine (统一入口) |
| `assembler.py` | ReportingAssembler (指标汇总) |
| `models.py` | PerformanceSummary, BottleneckSummary, ReportingReport, OutputConfig |
| `schema.py` | SCHEMA_VERSION |
| `cost_analysis.py` | CostAnalyzer, CostBreakdown |
| `memory_analysis.py` | MemoryAnalyzer, MemoryBreakdown |
| `roofline.py` | RooflineAnalyzer, RooflineData, RooflinePoint |
| `gantt.py` | GanttChartBuilder, GanttChartData, GanttTask |
| `traffic_analysis.py` | TrafficAnalyzer, TrafficReport, LinkTraffic |
| `exporters.py` | JSONExporter, ExporterRegistry |

## 整体架构

```
L4 EngineResult (StepMetrics + Aggregates)
              |
              v
       ReportingEngine
              |
     +--------+--------+
     v        v        v
 Assembler CostAnalyzer MemoryAnalyzer
 (汇总)    (成本)      (内存)
     |        |        |
     v        v        v
 ReportingReport  CostBreakdown  MemoryBreakdown
              |
     +--------+--------+
     v        v        v
  Roofline   Gantt   Traffic
  (性能)    (时序)   (流量)
              |
              v
        JSONExporter
```

## ReportingEngine

### 接口

```python
class ReportingEngine:
    def run(
        self,
        engine_result: EngineResult,
        config: dict | None = None,
        output_config: OutputConfig | None = None,
    ) -> ReportingReport

    def build_text(self, report: ReportingReport) -> ReportText

    def export(
        self,
        report: ReportingReport,
        output_config: OutputConfig | None = None,
        filename: str = "reporting_report.json",
    ) -> str
```

### 处理流程

1. **输入校验**: 检查 EngineResult 包含有效 aggregates
2. **指标汇总**: 调用 ReportingAssembler.assemble()
3. **文本生成**: build_text() 生成可读文本报告
4. **JSON 导出**: export() 输出到文件

## ReportingAssembler

### 汇总逻辑

```python
class ReportingAssembler:
    def assemble(
        self,
        engine_result: EngineResult,
        config: dict | None = None,
        include_step_metrics: bool = True,
    ) -> ReportingReport
```

转换流程:
1. `Aggregates` -> `PerformanceSummary` (直接字段映射)
2. `StepMetrics[]` -> `BottleneckSummary` (按类型计数 + Top-5 耗时 Op)
3. 可选附带完整 `step_metrics` 列表

### 输出结构

```python
@dataclass
class PerformanceSummary:
    total_time_ms: float
    ttft_ms: float
    tpot_ms: float
    tps: float
    mfu: float
    mbu: float
    memory_peak_mb: float
    compute_time_ms: float
    comm_time_ms: float
    wait_time_ms: float
    total_flops: int
    total_bytes: int
    num_ops: int

@dataclass
class BottleneckSummary:
    compute_bound_count: int
    bw_bound_count: int
    latency_bound_count: int
    unknown_count: int
    top_ops: list[dict]       # Top-5 耗时 Op
```

## CostAnalyzer

### 成本模型

```python
class CostAnalyzer:
    def __init__(
        self,
        chip_prices: dict[str, float] | None = None,
        modules_per_server: int = 8,
        chips_per_module: int = 1,
        depreciation_years: int = 3,
    )

    def analyze(
        self,
        chip_type: str,
        chip_count: int,
        tps: float = 0.0,
        lanes_per_chip: int = 16,
    ) -> CostBreakdown

    def analyze_from_pod(
        self,
        pod: PodSpec,
        aggregates: Aggregates | None = None,
    ) -> CostBreakdown
```

### 成本公式

```
# 服务器成本 (单台)
server_cost = (chip_price * chips_per_module + 750) * modules_per_server + 12000 + 7500

# 互联成本
interconnect_cost = chip_count * modules_per_server * chips_per_module * lanes * lane_cost(chip_count)

# 总成本
total_cost = server_cost + interconnect_cost

# 百万 token 成本 (3 年折旧)
cost_per_million_tokens = total_cost / (depreciation_years * 365 * 24 * 3600 * tps) * 1e6
```

### 互联成本阶梯

| 芯片数 | 单 lane 成本 | 互联方案 |
|--------|-------------|---------|
| 1-2    | $1/lane     | PCIe 直连 |
| 8      | $55/lane    | Ethernet 交换 |
| 16     | $70/lane    | 交换 + DAC |
| 32     | $70/lane    | 交换 + DAC |
| 64     | $105/lane   | 交换 + AEC |
| 64+    | $247/lane   | 全光方案 (AOC + 光模块) |

### CostBreakdown

```python
@dataclass
class CostBreakdown:
    server_cost: float
    interconnect_cost: float
    total_cost: float
    cost_per_chip: float
    cost_per_million_tokens: float
    chip_count: int
    chip_type: str
    depreciation_years: int
```

## MemoryAnalyzer

### 内存分解

```python
class MemoryAnalyzer:
    def analyze(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        vocab_size: int,
        batch_size: int = 1,
        seq_len: int = 1024,
        num_kv_heads: int | None = None,
        tp_degree: int = 1,
    ) -> MemoryBreakdown
```

### 计算公式

```
# 权重 (单层)
attention_params = 4 * hidden_size * hidden_size   # Q/K/V/O
ffn_params = 3 * hidden_size * intermediate_size   # gate/up/down
layernorm_params = 2 * hidden_size
layer_params = attention_params + ffn_params + layernorm_params

# 总权重
weights_bytes = (layer_params * num_layers + vocab_size * hidden_size) * dtype_bytes / tp_degree

# KV Cache
kv_cache_bytes = 2 * batch * seq_len * kv_heads * head_dim * num_layers * dtype_bytes

# 激活
activations_bytes = batch * seq_len * hidden_size * dtype_bytes * activation_factor
```

### MemoryBreakdown

```python
@dataclass
class MemoryBreakdown:
    total_bytes: int
    weights_bytes: int
    kv_cache_bytes: int
    activations_bytes: int

    @property
    def total_gb(self) -> float
```

## RooflineAnalyzer

### Roofline 模型

```python
class RooflineAnalyzer:
    def __init__(
        self,
        peak_flops_gflops: float,    # 峰值算力 (GFLOPS)
        peak_bandwidth_gbps: float,   # 峰值带宽 (GB/s)
    )

    def analyze_point(
        self,
        name: str,
        flops: int,
        bytes_accessed: int,
        time_ns: float,
    ) -> RooflinePoint
```

### 关键指标

```
# 算术强度
AI = FLOPS / bytes_accessed

# 拐点
ridge_point = peak_flops / peak_bandwidth

# 可达算力
attainable = min(peak_flops, AI * peak_bandwidth)

# 瓶颈判定
if AI < ridge_point:
    bottleneck = BW_BOUND       # 带宽受限
else:
    bottleneck = COMPUTE_BOUND  # 算力受限
```

### RooflineData

```python
@dataclass
class RooflineData:
    peak_flops: float               # GFLOPS
    peak_bandwidth: float           # GB/s
    ridge_point: float              # FLOPS/Byte
    points: list[RooflinePoint]     # 数据点
    roofline_x: list[float]         # Roofline 曲线 X (AI, log scale)
    roofline_y: list[float]         # Roofline 曲线 Y (GFLOPS, log scale)
```

## GanttChartBuilder

### Gantt 图生成

```python
class GanttChartBuilder:
    def add_task(
        self,
        name: str,
        start_us: float,
        end_us: float,
        task_type: GanttTaskType,
        phase: InferencePhase,
        device_id: str = "",
        pp_stage: int = 0,
        layer_index: int | None = None,
        token_index: int | None = None,
        **attrs,
    ) -> GanttTask

    def build(self, phase_transition: float | None = None) -> GanttChartData
```

### 任务类型

| 类别 | 类型 | 颜色 |
|------|------|------|
| 计算 | compute, attention, ffn | 绿色 (#52c41a - #237804) |
| MLA | mla_q_proj, mla_kv_proj, ... | 青色 (#13c2c2 - #08979c) |
| MoE | moe_router, moe_expert, ... | 品红 (#f759ab - #c41d7f) |
| 通信 | tp_comm, pp_comm, ep_comm | 蓝/紫 (#1890ff - #722ed1) |
| 内存 | hbm_read, hbm_write, kv_cache | 橙色 (#faad14 - #d48806) |
| 空闲 | bubble, idle | 灰色 (#d9d9d9) |

### 输出格式

```python
@dataclass
class GanttChartData:
    resources: list[dict]     # PP stage 资源行 (compute + network)
    tasks: list[GanttTask]    # 任务列表
    time_range: dict          # {"start": 0, "end": us}
    phase_transition: float   # TTFT (us), prefill/decode 分界
```

## TrafficAnalyzer

### 链路流量分析

```python
class TrafficAnalyzer:
    def add_comm(
        self,
        src: str,
        dst: str,
        bytes_transferred: int,
        time_us: float,
        comm_type: str,
        phase: str,
    ) -> None

    def analyze(self, exec_plan: ExecPlan) -> TrafficReport
```

分析维度:
- **链路流量**: src -> dst 的传输量与利用率
- **设备流量**: 每芯片 send/recv 分解
- **通信类型分解**: TP_ALLREDUCE / PP_P2P / EP_ALLTOALL / ...
- **阶段分解**: prefill vs decode

### 链路类型推断

| 层级 | 类型 | 典型带宽 |
|------|------|---------|
| 同 Board | c2c | 448 GB/s |
| 同 Rack | b2b | 450 GB/s |
| 同 Pod | r2r | 200 GB/s |
| 跨 Pod | p2p | 100 GB/s |

## L0 Compat 层

`L0_entry/compat.py` 负责将 L4/L5 结果转为前端兼容格式:

### convert_to_gantt_chart()

```python
def convert_to_gantt_chart(
    step_metrics: list[dict],
    parallelism: dict,
    aggregates: dict | None = None,
    topology_config: dict | None = None,
) -> dict
```

输出格式 (前端 Gantt 组件):
```json
{
  "resources": [
    {"id": "stage0_compute", "name": "PP0 Compute", "ppStage": 0, "type": "compute"},
    {"id": "stage0_network", "name": "PP0 Network", "ppStage": 0, "type": "network"}
  ],
  "tasks": [
    {
      "id": "task_1",
      "name": "layers.5.mla",
      "resource": "stage0_compute",
      "start": 1000.0,
      "end": 5000.0,
      "type": "attention_qkv",
      "phase": "prefill",
      "color": "#389e0d"
    }
  ],
  "timeRange": {"start": 0.0, "end": 50000.0},
  "phaseTransition": 20000.0
}
```

### convert_to_stats()

```python
def convert_to_stats(
    aggregates: dict,
    step_metrics: list[dict] | None = None,
    inference_config: dict | None = None,
    parallelism: dict | None = None,
    topology_config: dict | None = None,
) -> dict
```

输出格式 (前端性能面板):
```json
{
  "prefill": {"computeTime": 120, "commTime": 25, "totalTime": 150},
  "decode": {"computeTime": 80, "commTime": 15, "totalTime": 100},
  "totalRunTime": 250,
  "ttft": 150,
  "avgTpot": 2.3,
  "dynamicMfu": 0.65,
  "dynamicMbu": 0.72,
  "linkTrafficStats": [...]
}
```

### generate_link_traffic_stats()

```python
def generate_link_traffic_stats(
    step_metrics: list[dict],
    parallelism: dict,
    topology_config: dict | None = None,
    aggregates: dict | None = None,
) -> list[dict]
```

每条链路输出:
```json
{
  "source": "chip_0",
  "target": "chip_1",
  "trafficMb": 256.0,
  "bandwidthGbps": 448.0,
  "latencyUs": 0.2,
  "utilizationPercent": 85.0,
  "linkType": "c2c",
  "taskTypeBreakdown": {"tp_allreduce": 200.0, "sp_allgather": 56.0}
}
```

## JSON 导出

### ExporterRegistry

```python
class ExporterRegistry:
    def export(
        self,
        format_name: str,            # "json"
        report: ReportingReport,
        output_path: str,
        config: OutputConfig,
    ) -> str
```

当前支持: JSONExporter。ReportingReport 通过 `dataclass.asdict()` 序列化为 JSON。

### ReportingReport 输出格式

```json
{
  "schema_version": "1.0.0",
  "timestamp": "2026-02-11T12:00:00",
  "granularity": "CHIP",
  "performance": {
    "total_time_ms": 150.5,
    "ttft_ms": 45.2,
    "tpot_ms": 2.3,
    "tps": 434.78,
    "mfu": 0.65,
    "mbu": 0.72,
    "memory_peak_mb": 64512.0,
    "compute_time_ms": 120.0,
    "comm_time_ms": 25.0,
    "wait_time_ms": 5.5,
    "total_flops": 12345678900,
    "total_bytes": 987654321,
    "num_ops": 512
  },
  "bottleneck": {
    "compute_bound_count": 200,
    "bw_bound_count": 150,
    "latency_bound_count": 50,
    "unknown_count": 112,
    "top_ops": [
      {"op_id": "layers.5.mla", "t_total_ms": 12.34, "bottleneck": "BW_BOUND"}
    ]
  },
  "config": {},
  "step_metrics": []
}
```
