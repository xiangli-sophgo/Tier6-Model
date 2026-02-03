# 事件驱动仿真器技术文档

**创建日期**: 2025-01-28
**更新日期**: 2025-02-03
**版本**: v0.2.0 (Phase 2)
**状态**: Phase 2 Completed

---

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [已完成功能 (Phase 1)](#3-已完成功能-phase-1)
4. [已完成功能 (Phase 2)](#4-已完成功能-phase-2)
5. [待实现功能 (Phase 3-4)](#5-待实现功能-phase-3-4)
6. [使用方式](#6-使用方式)
7. [API 参考](#7-api-参考)
8. [实现细节](#8-实现细节)
9. [测试与验证](#9-测试与验证)
10. [附录](#附录)

---

## 1. 概述

### 1.1 背景

现有仿真器采用"顺序时间推进"模式，存在以下局限：

| 局限 | 具体表现 | 影响 |
|------|----------|------|
| 同步推进 | 所有芯片按相同步调推进时间 | 无法精确模拟异步执行 |
| 重叠简化 | 计算-通信重叠用简单公式估算 | 重叠收益估计不准 |
| PP 气泡粗糙 | 气泡时间基于上一 stage 完成时间 | 微批次调度无法建模 |
| 依赖隐式 | 算子间依赖通过执行顺序隐式表达 | 难以支持复杂调度策略 |

### 1.2 设计目标

事件驱动仿真器的核心目标：

1. **多芯片独立推进** - 每个芯片有独立的时间线
2. **精确重叠建模** - 计算和通信可以真正并行
3. **流水线策略支持** - GPipe / 1F1B 等调度策略
4. **精确气泡计算** - 基于资源竞争的真实等待时间

### 1.3 设计参考

借鉴了以下成熟工具的设计理念：

- **[Vidur](https://github.com/microsoft/vidur)** (Microsoft, MLSys 2024) - 事件驱动架构、三层调度器
- **[ASTRA-sim](https://github.com/astra-sim/astra-sim)** (Georgia Tech + Meta) - 分层抽象、通信建模
- **[SimPy](https://simpy.readthedocs.io/)** - Python 离散事件仿真框架

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    EventDrivenSimulator                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ EventQueue  │    │  Resource   │    │ Dependency  │         │
│  │ (优先队列)  │◀──▶│  Manager    │◀──▶│   Graph     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                   │                 │
│         ▼                  ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   Event Handlers                     │        │
│  │  ComputeStart → ComputeEnd → CommStart → CommEnd    │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              复用现有组件                             │        │
│  │  • GEMMEvaluator  • FA2Evaluator  • AllReduceEval   │        │
│  │  • GanttChartBuilder  • TopologyParser              │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
backend/llm_simulator/event_driven/
├── __init__.py           # 模块导出
├── event.py              # 事件定义（7种事件类型）
├── event_queue.py        # 优先队列（基于heapq）
├── resource.py           # 资源管理器
├── dependency.py         # 依赖图
└── simulator.py          # 事件驱动仿真器主类
```

### 2.3 事件驱动主循环

```python
def _run_event_loop(self, context):
    while self.event_queue and self.events_processed < max_events:
        # 1. 取出最早的事件
        event = self.event_queue.pop()

        # 2. 推进仿真时间
        self.current_time = event.timestamp

        # 3. 处理事件，产生新事件
        new_events = event.handle(
            self.resource_manager,
            self.gantt_builder,
            context,
        )

        # 4. 添加新事件到队列
        self.event_queue.push_many(new_events)

        self.events_processed += 1
```

---

## 3. 已完成功能 (Phase 1)

### 3.1 事件系统

#### 事件类型枚举

```python
class EventType(IntEnum):
    # 结束事件（高优先级，确保资源正确释放）
    COMPUTE_END = 1
    COMM_END = 2
    MEMORY_END = 3

    # 完成事件（中优先级）
    LAYER_COMPLETE = 10
    STAGE_READY = 11
    BATCH_COMPLETE = 12

    # 开始事件（低优先级）
    COMPUTE_START = 20
    COMM_START = 21
    MEMORY_START = 22

    # 调度事件（最低优先级）
    SCHEDULE = 30
```

#### 事件排序规则

事件按照 `(timestamp, event_type, event_id)` 排序：

1. 首先按时间戳升序
2. 相同时间按事件类型（END 优先于 START）
3. 仍相同按事件 ID（保证确定性）

#### 已实现的事件类

| 事件类 | 功能 |
|--------|------|
| `ComputeStartEvent` | 计算开始，请求计算资源，记录到 Gantt |
| `ComputeEndEvent` | 计算结束，触发后续依赖 |
| `CommStartEvent` | 通信开始，处理集合通信同步 |
| `CommEndEvent` | 通信结束，清除同步状态 |
| `LayerCompleteEvent` | 层完成，触发 PP 通信或 batch 完成 |
| `StageReadyEvent` | PP Stage 就绪，触发下一阶段 |
| `BatchCompleteEvent` | Batch 完成，记录完成时间 |

### 3.2 事件队列

基于 Python `heapq` 实现的优先队列：

```python
class EventQueue:
    def push(self, event: BaseEvent) -> None: ...      # O(log n)
    def pop(self) -> BaseEvent: ...                    # O(log n)
    def peek(self) -> Optional[BaseEvent]: ...         # O(1)
    def push_many(self, events: list[BaseEvent]): ...
    def is_empty(self) -> bool: ...
    def stats(self) -> dict: ...
```

### 3.3 资源管理器

管理每个芯片的计算和网络资源：

```python
class ResourceManager:
    def __init__(self, chip_ids: list[str]): ...

    # 资源请求
    def request_resource(
        self,
        chip_id: str,
        resource_type: ResourceType,
        requested_start: float,
        duration: float,
    ) -> tuple[float, float]: ...  # (actual_start, actual_end)

    # 气泡记录
    def record_bubble(self, chip_id, start, duration, reason): ...
    def get_total_bubble_time(self, chip_id=None) -> float: ...
    def get_bubble_breakdown(self) -> dict[str, float]: ...

    # 集合通信同步
    def record_comm_arrival(self, chip_id, comm_type, layer_index, micro_batch, arrival_time): ...
    def all_chips_ready_for_comm(self, participating_chips, comm_type, layer_index, micro_batch) -> bool: ...
```

**资源类型**：

| 类型 | 说明 |
|------|------|
| `COMPUTE` | 计算资源（Tensor Core） |
| `NETWORK` | 网络资源（NVLink/IB） |
| `MEMORY_BUS` | 内存总线 |

### 3.4 依赖图

管理算子之间的依赖关系：

```python
class DependencyGraph:
    def add_node(self, node: OperatorNode): ...
    def add_edge(self, source_key, target_key, dep_type): ...
    def mark_completed(self, key): ...
    def get_ready_successors(self, key) -> list[OperatorNode]: ...
    def is_layer_complete(self, layer_index, micro_batch) -> bool: ...
```

**依赖类型**：

| 类型 | 说明 | 示例 |
|------|------|------|
| `DATA` | 数据依赖 | GEMM 需要等 RMSNorm 完成 |
| `RESOURCE` | 资源依赖 | 下一个 GEMM 需要等当前完成 |
| `SYNC` | 同步依赖 | AllReduce 需要等所有芯片到达 |
| `COMM` | 通信依赖 | PP Stage 1 等 Stage 0 发送 |

### 3.5 仿真器主类

```python
class EventDrivenSimulator:
    def __init__(
        self,
        topology_dict: dict,
        model: LLMModelConfig,
        inference: InferenceConfig,
        parallelism: ParallelismStrategy,
        hardware: HardwareConfig,
        config: EventDrivenSimConfig = None,
        progress_callback: Callable = None,
    ): ...

    def simulate(self) -> SimulationResult: ...
```

### 3.6 复用现有组件

| 组件 | 复用方式 |
|------|----------|
| `GEMMEvaluator` | 直接使用，获取算子延迟 |
| `FA2Evaluator` | 直接使用 |
| `AllReduceEval` 等 | 直接使用通信评估器 |
| `GanttChartBuilder` | 直接使用，收集任务 |
| `TopologyParser` | 直接使用，解析拓扑和并行组 |
| 层定义 | 使用 MHALayer、MLPLayer 等构建依赖图 |

### 3.7 测试验证

```python
# 事件队列测试
queue = EventQueue()
e1 = ComputeStartEvent(timestamp=10.0, chip_id='chip_0')
e2 = ComputeStartEvent(timestamp=5.0, chip_id='chip_1')
queue.push(e1)
queue.push(e2)
assert queue.peek().timestamp == 5.0  # ✅ 时间排序正确

# 资源管理测试
rm = ResourceManager(['chip_0'])
start1, end1 = rm.request_resource('chip_0', ResourceType.COMPUTE, 0.0, 10.0)
start2, end2 = rm.request_resource('chip_0', ResourceType.COMPUTE, 5.0, 10.0)
assert start2 == 10.0  # ✅ 资源竞争正确处理

# 依赖图测试
graph = DependencyGraph()
graph.add_node(OperatorNode(name='op1', ...))
graph.add_node(OperatorNode(name='op2', ...))
graph.add_edge(op1.key, op2.key)
assert graph.get_stats()['total_edges'] == 1  # ✅ 依赖关系正确
```

---

## 4. 已完成功能 (Phase 2)

### 4.1 API 集成

事件驱动仿真器现已完全集成到后端 API，可通过 `/api/simulate` 端点调用。

#### 请求格式

```json
POST /api/simulate
{
    "topology": { ... },
    "model": { ... },
    "inference": { ... },
    "parallelism": { ... },
    "hardware": { ... },
    "use_event_driven": true,
    "event_driven_config": {
        "max_simulated_tokens": 16,
        "enable_comm_overlap": true,
        "overlap_ratio": 0.8,
        "enable_chunked_comm": true,
        "comm_chunk_size_mb": 16.0,
        "enable_tbo": true,
        "pp_schedule": "gpipe"
    }
}
```

#### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_event_driven` | bool | false | 是否使用事件驱动仿真 |
| `enable_comm_overlap` | bool | true | 是否启用计算-通信重叠 |
| `overlap_ratio` | float | 0.8 | 重叠比例（0-1） |
| `enable_chunked_comm` | bool | false | 是否启用分块传输 |
| `comm_chunk_size_mb` | float | 16.0 | 每块大小（MB） |
| `enable_tbo` | bool | true | 是否启用 MoE TBO 优化 |

### 4.2 计算-通信并行调度

实现了真正的计算-通信并行执行，利用独立的资源追踪实现精确的重叠建模。

#### 核心机制

```
传统顺序执行:
┌──────────────────┐┌──────────────────┐┌──────────────────┐
│   Compute Op     ││  TP AllReduce    ││   Next Compute   │
└──────────────────┘└──────────────────┘└──────────────────┘
        10us              8us                  10us
                                          Total: 28us

重叠执行 (overlap_ratio=0.8):
┌──────────────────┐
│   Compute Op     │
└────────┬─────────┘
         │ 在完成 20% 时启动通信
         ▼
    ┌────────────────────┐
    │   TP AllReduce     │
    └──────────┬─────────┘
               │ 通信完成
               ▼
         ┌──────────────────┐
         │   Next Compute   │
         └──────────────────┘
                          Total: ~22us (节省 ~20%)
```

#### 事件处理流程

```python
# ComputeStartEvent.handle() 中的重叠调度逻辑

def handle(self, resource_manager, gantt_builder, context):
    # 1. 请求计算资源
    actual_start, actual_end = resource_manager.request_resource(
        chip_id, ResourceType.COMPUTE, timestamp, duration
    )

    # 2. 检查是否启用重叠且后续有通信
    if context["enable_comm_overlap"] and self.has_following_comm:
        # 计算提前启动通信的时间点
        overlap_start = actual_start + duration * (1 - context["overlap_ratio"])

        # 检查网络资源是否可用
        network_idle = resource_manager.get_resource_idle_time(
            chip_id, ResourceType.NETWORK
        )

        # 创建提前的通信事件
        comm_start_time = max(overlap_start, network_idle)
        new_events.append(CommStartEvent(
            timestamp=comm_start_time,
            is_overlapped=True,  # 标记为重叠调度
            ...
        ))

    # 3. 创建计算结束事件
    new_events.append(ComputeEndEvent(
        has_overlapped_comm=True,  # 避免重复触发
        ...
    ))
```

### 4.3 分块传输 (Chunked Communication)

将大型 AllReduce 分成多个块，实现更细粒度的流水线。

#### 工作原理

```
传统 AllReduce (256MB):
┌────────────────────────────────────────┐
│           AllReduce 256MB              │
└────────────────────────────────────────┘
                                      完成后才能开始下一个计算

分块 AllReduce (16MB × 16 chunks):
┌────┐┌────┐┌────┐┌────┐...┌────┐
│ C0 ││ C1 ││ C2 ││ C3 │   │C15 │
└──┬─┘└────┘└────┘└────┘   └────┘
   │
   │ 第一块完成即可开始下一个计算
   ▼
   ┌──────────────────────────────────┐
   │        Next Compute (流水线)      │
   └──────────────────────────────────┘
```

#### 新增事件类型

```python
@dataclass
class ChunkedCommStartEvent(BaseEvent):
    """分块通信开始事件"""
    chunk_index: int = 0           # 当前块索引
    total_chunks: int = 1          # 总块数
    chunk_size_bytes: float = 0.0  # 每块大小
    duration_per_chunk_us: float   # 每块传输时间
    following_compute_op: Any      # 后续计算算子（仅第一块）

@dataclass
class ChunkedCommEndEvent(BaseEvent):
    """分块通信结束事件"""
    chunk_index: int
    total_chunks: int

    def handle(self, ...):
        # 第一块完成时，触发后续计算
        if self.chunk_index == 0 and self.following_compute_op:
            new_events.append(ComputeStartEvent(...))

        # 最后一块完成时，标记整个通信完成
        if self.chunk_index == self.total_chunks - 1:
            dependency_graph.mark_completed(...)
```

#### 启用条件

分块传输仅在以下条件下启用：
- `enable_chunked_comm = true`
- 通信类型为 `allreduce`、`allgather` 或 `reduce_scatter`
- 通信大小 > 2 × `comm_chunk_size_mb`

### 4.4 MoE TBO 优化 (Tensor-Bus Overlap)

MoE 层的 Dispatch 通信和 Expert 计算可以并行执行。

#### MoE 层结构

```
标准 MoE 层执行顺序:
Gate → Dispatch → Expert → Combine

传统顺序执行:
┌──────┐┌──────────┐┌────────────────────┐┌──────────┐
│ Gate ││ Dispatch ││      Expert        ││ Combine  │
└──────┘└──────────┘└────────────────────┘└──────────┘
                                          Total: 40us

TBO 优化后 (Dispatch 和 Expert 并行):
┌──────┐
│ Gate │
└──┬───┘
   │
   ├──────────────────────┐
   │                      │
   ▼                      ▼
┌──────────┐    ┌────────────────────┐
│ Dispatch │    │      Expert        │
└─────┬────┘    └─────────┬──────────┘
      │                   │
      └─────────┬─────────┘
                │ 等待两者都完成
                ▼
         ┌──────────┐
         │ Combine  │
         └──────────┘
                    Total: ~30us (节省 ~25%)
```

#### 依赖图构建

```python
def build_moe_layer_with_tbo(self, layer_idx, layer, enable_tbo=True):
    """构建 MoE 层的依赖图（支持 TBO 优化）"""

    if enable_tbo:
        # TBO 模式：允许并行

        # Gate → Dispatch（顺序）
        add_edge(gate.key, dispatch.key, DATA)

        # Gate → Expert（并行分支，而非 Dispatch → Expert）
        add_edge(gate.key, expert.key, DATA)

        # Combine 需要等待 Expert 和 Dispatch 都完成
        add_edge(expert.key, combine.key, DATA)
        add_edge(dispatch.key, combine.key, SYNC)
    else:
        # 非 TBO 模式：严格顺序
        for i in range(1, len(all_ops)):
            add_edge(all_ops[i-1].key, all_ops[i].key, DATA)
```

### 4.5 仿真流程更新

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EventDrivenSimulator.simulate()                   │
├─────────────────────────────────────────────────────────────────────┤
│  1. 初始化                                                           │
│     └── reset_event_counter(), event_queue.clear(), etc.           │
│                                                                      │
│  2. 构建依赖图 (支持 MoE TBO)                                         │
│     └── 识别 MoE 层 → build_moe_layer_with_tbo()                    │
│     └── Dense 层 → 标准依赖构建                                      │
│                                                                      │
│  3. 创建仿真上下文 (包含重叠配置)                                       │
│     └── enable_comm_overlap, overlap_ratio, enable_chunked_comm     │
│                                                                      │
│  4. 事件循环                                                          │
│     └── ComputeStartEvent: 检查后续通信 → 提前调度                    │
│     └── CommStartEvent: 检查是否分块 → 创建 ChunkedCommStartEvent    │
│     └── ChunkedCommEndEvent: 第一块完成 → 触发下一个计算              │
│                                                                      │
│  5. 收集结果                                                          │
│     └── 统计重叠时间、气泡时间、MFU 等                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 待实现功能 (Phase 3-4)

### 5.1 Phase 3: 精细化增强

| 功能 | 优先级 | 说明 |
|------|--------|------|
| **Kernel Launch 开销** | P1 | 每个 kernel 的启动延迟 |
| **量化支持** | P1 | INT8/FP8 量化建模 |
| **网络拥塞建模** | P2 | 多通信流竞争带宽 |
| **动态 Batching** | P2 | Continuous Batching 支持 |
| **1F1B 调度策略** | P1 | 更高效的流水线调度 |

### 5.2 Phase 4: 生产就绪

| 功能 | 优先级 | 说明 |
|------|--------|------|
| **完整测试覆盖** | P0 | 单元测试 + 集成测试 |
| **性能基准** | P0 | 与真实系统对比验证 |
| **文档完善** | P0 | API 文档 + 使用指南 |
| **前端集成** | P0 | 前端可选择仿真模式 |

### 5.3 实施路线图

```
Phase 1: 基础事件系统 ✅           Phase 2: 高级特性 ✅
┌────────────────────────┐        ┌────────────────────────┐
│ • Event & EventQueue   │        │ • API 集成 ✅          │
│ • Resource Manager     │        │ • 计算-通信重叠 ✅     │
│ • 基础 Handler         │  ──▶   │ • MoE TBO ✅           │
│ • 依赖图构建           │        │ • 分块传输建模 ✅      │
│ • 简单场景验证         │        │                        │
└────────────────────────┘        └────────────────────────┘
         │                                  │
         │                                  │
         ▼                                  ▼
Phase 3: 精细化增强                Phase 4: 生产就绪
┌────────────────────────┐        ┌────────────────────────┐
│ • Kernel Launch 开销   │        │ • 完整测试覆盖         │
│ • 量化支持             │  ──▶   │ • 性能基准             │
│ • 网络拥塞             │        │ • 文档完善             │
│ • 1F1B 调度策略        │        │ • 前端集成             │
└────────────────────────┘        └────────────────────────┘
```

---

## 6. 使用方式

### 6.1 通过 API 调用（推荐）

```bash
# 使用事件驱动仿真器
curl -X POST http://localhost:8001/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "topology": { ... },
    "model": { ... },
    "inference": { ... },
    "parallelism": { ... },
    "hardware": { ... },
    "use_event_driven": true,
    "event_driven_config": {
      "max_simulated_tokens": 16,
      "enable_comm_overlap": true,
      "overlap_ratio": 0.8,
      "enable_chunked_comm": false,
      "enable_tbo": true
    }
  }'
```

### 6.2 Python 直接调用

```python
from llm_simulator.event_driven import EventDrivenSimulator, EventDrivenSimConfig

# 配置
config = EventDrivenSimConfig(
    max_simulated_tokens=16,
    enable_data_transfer=True,
    enable_kv_cache=True,
    enable_comm_overlap=True,
    overlap_ratio=0.8,
    enable_chunked_comm=False,
    comm_chunk_size_mb=16.0,
    enable_tbo=True,
    pp_schedule="gpipe",
)

# 创建仿真器
simulator = EventDrivenSimulator(
    topology_dict=topology,
    model=model_config,
    inference=inference_config,
    parallelism=parallelism_config,
    hardware=hardware_config,
    config=config,
)

# 运行仿真
result = simulator.simulate()

# 获取结果
print(f"Total time: {result.stats.total_run_time} ms")
print(f"TTFT: {result.stats.ttft} ms")
print(f"MFU: {result.stats.dynamic_mfu:.2%}")
```

### 6.3 模式对比

| 参数 | 事件驱动模式 | 静态模式 |
|------|-------------|---------|
| `use_event_driven` | `true` | `false` (默认) |
| 精度 | 高（精确重叠建模） | 中（公式估算） |
| 性能 | 较慢 | 快 |
| 适用场景 | 详细分析、调优 | 快速评估、参数扫描 |

---

## 7. API 参考

### 7.1 EventDrivenSimConfig

```python
@dataclass
class EventDrivenSimConfig:
    # 基础配置
    max_simulated_tokens: int = 16          # 最大模拟 token 数
    enable_data_transfer: bool = True       # 是否启用数据传输
    enable_kv_cache: bool = True            # 是否启用 KV 缓存

    # 计算-通信重叠优化 (Phase 2)
    enable_comm_overlap: bool = True        # 是否启用计算-通信重叠
    overlap_ratio: float = 0.8              # 重叠比例（0-1）

    # 分块传输 (Phase 2)
    enable_chunked_comm: bool = False       # 是否启用分块传输
    comm_chunk_size_mb: float = 16.0        # 每块大小（MB）

    # MoE TBO 优化 (Phase 2)
    enable_tbo: bool = True                 # 是否启用 MoE TBO 优化

    # 评估器配置
    use_precise_evaluator: bool = True      # 是否使用精确评估器
    evaluation_granularity: str = "fine"    # 评估粒度

    # 调度策略
    pp_schedule: str = "gpipe"              # gpipe | 1f1b (1f1b 待实现)

    # 调试选项
    max_events: int = 1000000               # 最大事件数
    log_events: bool = False                # 是否记录事件日志
    max_simulation_time_us: float = 1e9     # 最大仿真时间（微秒）
```

### 7.2 HTTP API 参数

#### SimulationRequest 新增字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_event_driven` | bool | false | 是否使用事件驱动仿真 |
| `event_driven_config` | object | null | 事件驱动仿真配置 |

#### EventDrivenConfigRequest

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_simulated_tokens` | int | 16 | 最大模拟 token 数 |
| `enable_data_transfer` | bool | true | 是否启用数据传输 |
| `enable_kv_cache` | bool | true | 是否启用 KV 缓存 |
| `enable_comm_overlap` | bool | true | 是否启用计算-通信重叠 |
| `overlap_ratio` | float | 0.8 | 重叠比例（0-1） |
| `enable_chunked_comm` | bool | false | 是否启用分块传输 |
| `comm_chunk_size_mb` | float | 16.0 | 每块大小（MB） |
| `enable_tbo` | bool | true | 是否启用 MoE TBO 优化 |
| `pp_schedule` | string | "gpipe" | 流水线调度策略 |

### 7.3 EventQueue

| 方法 | 说明 |
|------|------|
| `push(event)` | 添加事件，O(log n) |
| `pop()` | 取出最早事件，O(log n) |
| `peek()` | 查看最早事件，O(1) |
| `push_many(events)` | 批量添加 |
| `is_empty()` | 检查是否为空 |
| `stats()` | 返回统计信息 |

### 7.4 ResourceManager

| 方法 | 说明 |
|------|------|
| `request_resource(chip_id, type, start, duration)` | 请求资源 |
| `get_resource_idle_time(chip_id, type)` | 获取资源空闲时间 |
| `record_bubble(chip_id, start, duration, reason)` | 记录气泡 |
| `get_total_bubble_time(chip_id)` | 获取总气泡时间 |
| `get_bubble_breakdown()` | 按原因分类气泡 |
| `record_comm_arrival(...)` | 记录通信到达 |
| `all_chips_ready_for_comm(...)` | 检查同步就绪 |

### 7.5 DependencyGraph

| 方法 | 说明 |
|------|------|
| `add_node(node)` | 添加算子节点 |
| `add_edge(src, dst, type)` | 添加依赖边 |
| `mark_completed(key)` | 标记算子完成 |
| `get_ready_successors(key)` | 获取就绪的后继 |
| `is_layer_complete(layer, mb)` | 检查层是否完成 |
| `get_entry_nodes(mb)` | 获取入口节点 |

### 7.6 DependencyGraphBuilder (Phase 2 新增)

| 方法 | 说明 |
|------|------|
| `build_moe_layer_with_tbo(layer_idx, layer, ...)` | 构建 MoE 层依赖图（支持 TBO） |
| `add_edge(src, dst, dep_type)` | 添加依赖边 |
| `get_graph()` | 获取构建完成的依赖图 |

---

## 8. 实现细节

### 8.1 事件处理流程

```
ComputeStartEvent.handle():
    1. request_resource(COMPUTE)  → 获取实际开始时间
    2. 如果有等待 → record_bubble()
    3. gantt_builder.add_task()   → 记录到 Gantt 图
    4. 创建 ComputeEndEvent       → 加入队列

ComputeEndEvent.handle():
    1. mark_completed()           → 标记算子完成
    2. get_ready_successors()     → 获取就绪的后继
    3. 为每个后继创建 StartEvent  → 加入队列
    4. 检查是否层完成             → 触发 LayerCompleteEvent
```

### 8.2 集合通信同步

```
CommStartEvent.handle():
    1. record_comm_arrival()      → 记录当前芯片到达
    2. all_chips_ready_for_comm() → 检查是否所有芯片就绪
       - 如果没有，返回空列表（等待其他芯片）
       - 如果就绪，继续
    3. actual_start = max(arrival_times)  → 使用最晚到达时间
    4. 为所有参与芯片 request_resource(NETWORK)
    5. 记录等待时间为 bubble
    6. 创建 CommEndEvent
```

### 8.3 PP Stage 转换

```
LayerCompleteEvent.handle():
    如果是当前 stage 的最后一层:
        如果不是最后一个 stage:
            → 创建 StageReadyEvent (下一 stage)
        否则:
            → 创建 BatchCompleteEvent

StageReadyEvent.handle():
    1. 计算 PP P2P 通信延迟
    2. 添加通信任务到 Gantt
    3. 创建下一层的 ComputeStartEvent
```

---

## 9. 测试与验证

### 9.1 单元测试

```bash
cd backend
python3 -c "
from llm_simulator.event_driven import *
from llm_simulator.event_driven.event import reset_event_counter

# 测试事件队列
queue = EventQueue()
reset_event_counter()
e1 = ComputeStartEvent(timestamp=10.0, chip_id='chip_0')
e2 = ComputeStartEvent(timestamp=5.0, chip_id='chip_1')
queue.push(e1)
queue.push(e2)
assert queue.peek().timestamp == 5.0
print('✅ EventQueue test passed')

# 测试资源管理
rm = ResourceManager(['chip_0'])
s1, e1 = rm.request_resource('chip_0', ResourceType.COMPUTE, 0.0, 10.0)
s2, e2 = rm.request_resource('chip_0', ResourceType.COMPUTE, 5.0, 10.0)
assert s2 == 10.0  # 应该等待
print('✅ ResourceManager test passed')

print('All tests passed!')
"
```

### 9.2 验证标准

| 场景 | 验证方法 |
|------|----------|
| 单芯片无并行 | 总时间应与顺序仿真完全相等 |
| 纯 TP 并行 | 计算时间相等，通信时间相近 |
| 纯 PP 并行 | 总时间相近，气泡比例更准确 |
| 计算-通信重叠 | Gantt 图应显示重叠区域 |
| MoE TBO | Dispatch 和 Expert 应并行执行 |

### 9.3 已知限制

- 暂不支持 1F1B 调度策略
- 暂不支持 Decode 阶段逐 token 仿真
- 分块传输仅支持 AllReduce/AllGather/ReduceScatter

---

## 附录

### A. 与 Vidur 的对比

| 方面 | Vidur | 我们的实现 |
|------|-------|-----------|
| 执行时间预测 | 基于 profile 数据 | 精确的 GEMM/FA2 评估器 ✅ |
| 通信建模 | 简化模型 | 详细的 AllReduce/P2P 评估器 ✅ |
| 硬件建模 | 预置配置 | 灵活的 5 层拓扑 ✅ |
| 事件系统 | 完善 | Phase 1 完成 ✅ |
| Pipeline 调度 | 同步 PP | Phase 2 实现 |
| MoE 支持 | 有限 | 详细的 EP/TBO 建模 ✅ |

### B. 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `event.py` | ~850 | 事件定义（含分块通信事件） |
| `event_queue.py` | ~120 | 优先队列 |
| `resource.py` | ~320 | 资源管理（含空闲时间查询） |
| `dependency.py` | ~400 | 依赖图（含 MoE TBO） |
| `simulator.py` | ~800 | 仿真器主类（含重叠配置） |
| `__init__.py` | ~65 | 模块导出 |
| **总计** | ~2555 | |

**API 相关文件变更**:

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `web/api.py` | 修改 | 添加事件驱动仿真调用 |
| `web/schemas.py` | 修改 | 添加 EventDrivenConfigRequest |

### C. 更新日志

- **2025-02-03**: Phase 2 完成
  - **API 集成**: 事件驱动仿真器可通过 `/api/simulate` 端点调用
  - **计算-通信并行**: 实现真正的计算-通信重叠调度
    - 新增 `overlap_ratio` 配置参数
    - ComputeStartEvent 支持提前触发通信事件
  - **分块传输**: 实现 AllReduce 分块传输
    - 新增 ChunkedCommStartEvent、ChunkedCommEndEvent 事件
    - 第一块完成即可触发后续计算
  - **MoE TBO 优化**: 实现 Dispatch 和 Expert 并行执行
    - 新增 `build_moe_layer_with_tbo()` 方法
    - Combine 等待 Expert 和 Dispatch 双重依赖
  - **文档更新**: 更新架构文档至 v0.2.0

- **2025-01-28**: Phase 1 完成
  - 实现事件系统（7种事件类型）
  - 实现事件队列（基于 heapq）
  - 实现资源管理器（计算/网络资源）
  - 实现依赖图（算子依赖关系）
  - 实现仿真器主类
  - 复用现有评估器和 Gantt 构建器
