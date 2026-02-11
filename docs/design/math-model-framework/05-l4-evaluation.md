# L4 Evaluation -- 评估引擎层

## 功能概述

L4 是性能评估的唯一出口，负责:
- 基于 `ExecPlan + HardwareSpec + TopologySpec` 做统一口径性能评估
- 按 `Granularity (Chip/Core/Lane)` 切换评估精度
- 按 OpType 路由到子评估器 (Compute/Comm/Fallback)
- Step 级时延分解与瓶颈归因
- 输出 TTFT/TPOT/TPS/MFU/MBU 等聚合指标
- 为 L3 TilingPlanner 提供精评估出口 (PreciseTileEvaluator)

不在范围: 不做切分/布局/调度，不新增通信节点。

## 模块清单

| 模块 | 职责 |
|------|------|
| `engine.py` | EvaluationEngine (统一入口) |
| `metrics.py` | HardwareSpec, TopologySpec, StepMetrics, Aggregates, EngineResult |
| `cost_models/base.py` | BaseCostModel |
| `cost_models/chip.py` | ChipCostModel (Roofline) |
| `cost_models/core.py` | CoreCostModel (多核并行) |
| `cost_models/comm_protocol.py` | CommProtocolCostModel (DS_TPU 口径) |
| `evaluators/compute.py` | ComputeEvaluator |
| `evaluators/comm.py` | CommEvaluator |
| `evaluators/precise.py` | PreciseTileEvaluator (精评估) |
| `evaluators/base.py` | BaseEvaluator |
| `calibration.py` | Calibration (校准) |
| `registry.py` | CostModelRegistry, OpTypeRouter |

## 整体架构

```
ExecPlan + hardware dict (merge_specs 输出)
              |
              v
        EvaluationEngine
              |
      +-------+----------+
      v       v          v
 OpTypeRouter CostModelRegistry Calibration
(类型路由)    (粒度选择)    (可选校准)
      |       |          |
      +-------+----------+
              v
    StepMetrics + Aggregates
              |
              v
          EngineResult
```

## EvaluationEngine

### 接口

```python
def evaluate(
    exec_plan: ExecPlan,
    distributed_model: DistributedModel,
    hardware: dict[str, float],        # merge_specs() 输出
    granularity: Granularity = CHIP,
    calibration: CalibrationConfig | None = None,
    output_tokens: int = 1,
    deployment_config: dict | None = None,  # ring attention 等
) -> EngineResult
```

### 计算流程

1. **口径校验**: 检查 timeline 非空、hardware 字段完整
2. **模型选择**: 按 granularity 获取 CostModel
3. **遍历 timeline**: 对每个 event:
   - 获取 Op 定义 (from DistributedModel)
   - 类型路由: op_type -> Evaluator (compute/comm/fallback)
   - Step 估时: t_compute, t_comm, t_wait
   - 记录 bottleneck (COMPUTE_BOUND / BW_BOUND / LATENCY_BOUND)
4. **MoE 重叠**: dispatch/combine 通信与相邻计算并行
5. **Ring Attention 重叠**: Attention 层计算与通信完全重叠
6. **指标聚合**: sum times -> TPS/TTFT/TPOT/MFU/MBU

### 三级 Overlap 模型

| 级别 | 位置 | 说明 |
|------|------|------|
| Tile 级 | PreciseTileEvaluator | compute vs DMA (compute_dma_overlap_rate) |
| Layer 级 | _apply_ring_attn_overlap | Attention 计算 vs 通信完全重叠 |
| Model 级 | _apply_moe_compute_overlap | MoE dispatch/combine vs 相邻计算 |

## 代价模型

### ChipCostModel

芯片当黑盒，Roofline 模型:

```python
def estimate_compute(op_type, local_shape, hardware) -> float:
    flops = estimate_flops(op_type, local_shape)
    bytes = estimate_bytes(op_type, local_shape)
    t_compute = flops / (compute_tflops * 1e9)     # FLOPs -> ms
    t_memory = bytes / (memory_bw_gbps * 1e6)      # Bytes -> ms
    return max(t_compute, t_memory)                 # Roofline

def estimate_comm(comm_bytes, path_key, participants, hardware) -> float:
    bw_gbps = hardware[f"{path_key}_bandwidth_gbps"]  # 无默认值!
    ring_factor = 2 * (N-1) / N
    return comm_bytes * ring_factor / (bw_gbps * 1e6)
```

### CoreCostModel

考虑多核并行与 SRAM 层级:

```python
def estimate_compute(op_type, local_shape, hardware) -> float:
    # 多核效率模型
    data_to_sram_ratio = total_bytes / total_sram_bytes
    efficiency = 0.9 if ratio <= 1 else 0.7 if ratio <= 2 else 0.5
    t_compute = flops / (tflops * 1e9 * efficiency)
    t_memory = bytes / (memory_bw * 1e6)
    return max(t_compute, t_memory)
```

### CommProtocolCostModel (DS_TPU 口径)

精确通信协议建模:

```python
class CommProtocolCostModel:
    def allreduce(tp, comm_bytes, comm_protocol) -> (latency_us, comm_size):
        """分层 AllReduce:
        - tp in {8, 16, 32}: 3 阶段 (板内 ring + 板间 + 板内广播)
        - 其他: 标准 ring allreduce
        """

    def allgather(tp, comm_bytes, comm_protocol) -> (latency_us, comm_size)
    def reducescatter(tp, comm_bytes, comm_protocol) -> (latency_us, comm_size)
    def dispatch(moe_tp, ep, comm_bytes, ...) -> (latency_us, comm_size)
    def combine(moe_tp, ep, comm_bytes, ...) -> (latency_us, comm_size)
```

### CommProtocolParams

```python
@dataclass
class CommProtocolParams:
    c2c_lat: float          # 片间延迟
    ddr_r_lat: float        # DDR 读延迟
    ddr_w_lat: float        # DDR 写延迟
    noc_lat: float          # NoC 延迟
    d2d_lat: float          # Die-to-Die 延迟
    sync_lat: float         # 同步延迟
    bw_urate: float         # 带宽利用率
    switch_latency: float   # 交换机延迟
    cable_latency: float    # 线缆延迟

    @classmethod
    def from_mapping(config: dict) -> CommProtocolParams:
        """从 merge_specs() 输出直接读取标准 key name"""
```

## 评估器

### ComputeEvaluator

处理计算类 Op: matmul, softmax, layernorm, attention, embedding, lmhead

```python
def evaluate(op_id, op_type, local_shape, attrs, hardware, cost_model) -> StepMetrics:
    t_compute = cost_model.estimate_compute(op_type, local_shape, hardware)
    # 使用 PreciseTileEvaluator 结果覆盖 (如果有)
    if "t_compute_ms" in attrs:
        t_compute = float(attrs["t_compute_ms"])
    return StepMetrics(t_compute=t_compute, ...)
```

### CommEvaluator

处理通信类 Op: allreduce, allgather, all2all, p2p

```python
def evaluate(...) -> StepMetrics:
    # 构建通信硬件口径 (无默认值)
    intra_bw = hardware["c2c_bandwidth_gbps"] * 1e9
    inter_bw = hardware["b2b_bandwidth_gbps"] * 1e9

    # 按 comm_type 路由
    if comm_type == "allreduce":
        latency_us, comm_size = model.allreduce(tp, comm_bytes, protocol)
    elif comm_type == "all2all":
        if "dispatch" in reason:
            latency_us, comm_size = model.dispatch(...)
        elif "combine" in reason:
            latency_us, comm_size = model.combine(...)
```

### PreciseTileEvaluator

为 L3 TilingPlanner 提供精确评估:

```python
class PreciseTileEvaluator:
    def evaluate_tile(op, tile_config, chip) -> dict:
        """计算精确 traffic/urate/执行时间
        - MatMul: 枚举 loop-order (mnk/nkm/mkn), 最小 traffic
        - Attention: FlashAttention-2 风格 Q/K/V buffer
        - Elementwise: memory-bound (2 * elements * dtype_bytes)
        """
```

## StepMetrics 与 Aggregates

### StepMetrics (per op)

```python
@dataclass
class StepMetrics:
    op_id: str
    t_compute: float = 0.0    # 计算时间 (ms)
    t_comm: float = 0.0       # 通信时间 (ms)
    t_wait: float = 0.0       # 等待时间 (ms)
    t_total: float = 0.0      # 总时间 (ms)
    bottleneck_tag: BottleneckTag = UNKNOWN
    flops: int = 0
    bytes_read: int = 0
    bytes_write: int = 0
    meta: dict = {}
```

### Aggregates (end-to-end)

```python
@dataclass
class Aggregates:
    ttft: float           # Time To First Token (ms)
    tpot: float           # Time Per Output Token (ms)
    tps: float            # Tokens Per Second
    mfu: float            # Model FLOPS Utilization
    mbu: float            # Memory Bandwidth Utilization
    memory_peak: int      # 内存峰值 (bytes)
    total_time: float     # 总时间 (ms)
    total_compute_time: float
    total_comm_time: float
    total_wait_time: float
    total_flops: int
    total_bytes: int
    num_steps: int
    bottleneck_summary: dict[str, int]
```

### 聚合公式

```python
# MFU
peak_flops = hardware["compute_tflops"] * 1e12
achieved_flops = total_flops / (total_time / 1000)
mfu = achieved_flops / peak_flops

# MBU
peak_bw = hardware["memory_bandwidth_gbps"] * 1e9
achieved_bw = total_bytes / (total_time / 1000)
mbu = achieved_bw / peak_bw

# TPS
tps = output_tokens / (decode_time / 1000)

# TTFT = prefill_time
# TPOT = decode_time / output_tokens
```

## 无默认值规则

所有 L4 模块从 hardware dict 读取参数时 **禁止使用默认值**:

```python
# [FAIL] 旧代码
bw_gbps = hardware.get("c2c_bandwidth_gbps", 400.0)

# [PASS] 新代码 -- 缺失时 KeyError
bw_gbps = hardware["c2c_bandwidth_gbps"]
```

覆盖范围:
- `chip.py`: estimate_comm 的 4 个带宽字段
- `core.py`: estimate_compute 的 4 个字段 + estimate_comm 的 5 个带宽字段
- `comm.py`: evaluate 的 4 个带宽字段
- `comm_protocol.py`: from_mapping 直接读标准 key
- `engine.py`: _aggregate_metrics 的 compute_tflops + memory_bandwidth_gbps

## Calibration (可选)

```python
@dataclass
class CalibrationConfig:
    effective_bw_factor: float = 1.0      # 有效带宽系数 (0-1)
    congestion_factor: float = 1.0        # 拥塞系数 (>=1)
    startup_overhead_ms: float = 0.0      # 启动开销 (ms)
    overlap_efficiency: float = 1.0       # 重叠效率 (0-1)
    compute_efficiency: float = 1.0       # 计算效率 (0-1)
```

修正公式:
```
t_compute = t_compute / compute_efficiency
t_comm = t_comm * congestion_factor / effective_bw_factor + startup_overhead_ms
```
