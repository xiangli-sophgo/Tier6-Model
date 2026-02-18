# G5 指令级仿真器 - 架构设计与实施计划

## Context

当前 Tier6+Model 拥有完整的代数性能建模能力（L0-L5 分层框架，回归测试对齐到 0.67%），但缺乏**指令级周期精确仿真**能力。G5 模式将对标 TPUPerf，通过事件驱动仿真引擎，在同一框架内提供更高精度的性能预测。

**设计文档参考**: `docs/design/instruction-level-simulator/` (01-09 共 10 篇)

**核心目标**:
- 在现有 L0-L5 框架中增加 G5 仿真模式
- 复用共享层（L0/L1/L2/L5），L3/L4 内部三分法（common/math/g5）
- 统一输出 EngineResult，前端无需区分模式
- 支持两种指令来源：从 LLM 模型自动生成 / 解析 TPUPerf 二进制文件
- 完整建模 TPUPerf 五大引擎：TIU + GDMA + SDMA + HAU + IFE
- NoC 互连预留 CrossRing 集成接口，v1 先用简化 Bus 模型

---

## 一、整体架构

### 1.1 层级数据流

```
两种模式共享:  L0 → L1 → L2 → L3.common.parallelism → DistributedModel
                                                             │
代数模式:      DistributedModel → L3.math → L4.math → EngineResult
G5 模式:       DistributedModel → L3.g5  → L4.g5  → EngineResult
                                                             │
两种模式共享:                                          EngineResult → L5 → 前端
```

### 1.2 共享/分叉边界

| 层级 | 共享程度 | 说明 |
|------|---------|------|
| L0 入口 | **完全共享** | EvalConfig 增加 mode 字段 |
| L1 工作负载 | **完全共享** | WorkloadIR 是两种模式的共同输入 |
| L2 硬件规格 | **共享 + 扩展** | ChipSpec 扩展 G5 所需的微架构参数 |
| L3.common | **完全共享** | ParallelismPlanner → DistributedModel |
| L3.math / L3.g5 | **各自独立** | tiling 粒度本质不同 |
| L4.common | **完全共享** | CostModel + EngineResult 定义 |
| L4.math / L4.g5 | **各自独立** | 代数评估 vs 事件驱动仿真 |
| L5 报告 | **完全共享** | 统一消费 EngineResult |

### 1.3 为什么 L3/L4 不能共享

**L3 Tiling**:
- Math: 输出 `TilePlan {tile_m/k/n, traffic, t_compute_ms}` — 统计摘要
- G5: 输出 `CoreProgram {tiu_cmds[], dma_cmds[], deps[]}` — 可执行指令序列
- 粒度差 1-2 个数量级，强行统一会导致抽象泄漏

**L4 Evaluation**:
- Math: 无状态遍历 op 列表，O(N)
- G5: 有状态事件驱动仿真（TIU/DMA 状态机、内存 bank 状态、bus 仲裁），O(M*C)
- 执行模型完全不同

---

## 二、目录结构

### 2.1 顶层重命名

`backend/math_model/` → `backend/perf_model/`

含义: performance model，统一包含代数模式和 G5 模式

### 2.2 完整目录

```
backend/perf_model/
    L0_entry/                      # 共享入口
        api.py                     # 路由，增加 mode 参数
        engine.py                  # 编排，根据 mode 选 math/g5 pipeline
        eval_config.py             # EvalConfig，增加 mode 字段
        compat.py                  # 前端兼容层
        websocket.py               # WebSocket 推送
        database.py                # 数据库 ORM

    L1_workload/                   # 共享模型 IR
        ir.py                      # WorkloadIR 协议与 Model 实现
        graph.py                   # WorkloadGraph
        models/                    # 模型构建器 (DeepSeek, Qwen 等)
        layers/                    # Layer 定义 (MLA/FFN/MoE)
        operators/                 # Op 定义 (matmul/softmax 等)

    L2_arch/                       # 共享硬件规格
        chip.py                    # ChipSpecImpl (G5 扩展参数)
        memory.py                  # MemoryHierarchy
        protocols.py               # ChipSpec/CoreSpec 协议

    L3_mapping/
        common/                    # === 共享 ===
            parallelism/           # ParallelismPlanner
                planner.py
                pattern_rules.py
            protocols.py           # DistributedModel, DistributedOp, CommType
        math/                      # === 代数模式 ===
            tiling/                # TilingPlanner (tile 打分选优)
                planner.py
                evaluators.py
            scheduling/            # Scheduler (抽象时间线)
                scheduler.py
        g5/                        # === G5 模式 ===
            instruction_tiler.py   # 指令级 tiling (精确地址布局)
            instruction_emitter.py # 指令发射 (生成 TIU/DMA/HAU cmd 序列)
            program.py             # CoreProgram 数据结构
            binary_parser.py       # .BD/.GDMA/.HAU/.SDMA 二进制解析

    L4_evaluation/
        common/                    # === 共享 ===
            cost_models/           # 成本计算 (芯片 + 互联)
                base.py
                comm_protocol.py
            metrics.py             # EngineResult, StepMetrics, Aggregates
        math/                      # === 代数模式 ===
            evaluators/            # 代数评估器
                precise.py
                rmsnorm_eval.py
                softmax_eval.py
            engine.py              # MathEvaluationEngine
        g5/                        # === G5 模式 ===
            sim_engine.py          # 事件驱动仿真调度器
            tiu.py                 # TIU 计算引擎
            dma.py                 # GDMA 搬运引擎 (LMEM <-> DDR)
            sdma.py                # SDMA 核间通信引擎 (GMEM <-> GMEM)
            hau.py                 # HAU 硬件辅助单元 (Sort/Top-K)
            ife.py                 # IFE 指令调度模型 (FIFO + 依赖管理)
            memory.py              # LMEM + DDR 模型
            noc_adapter.py         # NoC 适配器 (v1: SimpleBus / v2: CrossRing)
            interconnect.py        # 多芯片互连 (C2C + CDMA)
            adapter.py             # 仿真事件 → EngineResult

    L5_reporting/                  # 共享报告生成
        engine.py
        gantt.py
        cost_analysis.py
        memory_analysis.py
        traffic_analysis.py

    configs/                       # 配置文件
        chips/ models/ topologies/ benchmarks/
```

---

## 三、关键接口设计

### 3.1 分叉点：DistributedModel（L3.common 输出）

```python
@dataclass
class DistributedModel:
    """ParallelismPlanner 输出，两种模式共用"""
    ops: list[DistributedOp]
    op_map: dict[str, DistributedOp]
    tp: int; pp: int; ep: int
    stages: list[list[str]]
    parallel_groups: dict[str, list[list[int]]]
    chip_assignments: dict[str, list[int]]

@dataclass
class DistributedOp:
    op_id: str
    op_type: str
    role: NodeRole              # COMPUTE / COMM
    local_shape: dict[str, int] # 切分后的 shape
    parallel_spec: ParallelSpec
    comm_type: CommType | None
    comm_bytes: int
    topology_path_key: str
    chip_ids: list[int]
    deps: list[str]
```

### 3.2 G5 模式中间接口：CoreProgram（L3.g5 → L4.g5）

```python
@dataclass
class CoreProgram:
    """G5 模式下 L3 传给 L4 的指令序列"""
    cores: list[CoreInstructions]
    comm_schedule: list[CommOp]
    metadata: dict              # 总指令数、核数等统计

@dataclass
class CoreInstructions:
    core_id: int
    tiu_cmds: list[TIUCommand]
    dma_cmds: list[DMACommand]
    sdma_cmds: list[SDMACommand]
    hau_cmds: list[HAUCommand]

@dataclass
class TIUCommand:
    cmd_id: int
    cmd_id_dep: int             # 依赖的 DMA cmd_id
    op_type: str                # CONV / MM2_NN / MM2_NT / MM2_TT / SFU / AR
    result_addr: int            # LMEM 地址
    operand_addrs: list[int]
    shape: dict                 # n/c/h/w
    precision: str              # INT8 / BF16 / FP32
    source_op_id: str           # 对应 DistributedOp 的 op_id（用于聚合到 StepMetrics）

@dataclass
class DMACommand:
    """GDMA: LMEM <-> DDR 数据搬运"""
    cmd_id: int
    cmd_id_dep: int             # 依赖的 TIU cmd_id
    direction: str              # DDR_TO_LMEM / LMEM_TO_DDR / LMEM_TO_LMEM
    src_addr: int
    dst_addr: int
    shape: tuple[int, ...]      # (n, c, h, w)
    stride: tuple[int, ...]     # (n_s, c_s, h_s, w_s)
    elem_bytes: int
    source_op_id: str

@dataclass
class SDMACommand:
    """SDMA: 核间通信 (GMEM <-> GMEM)"""
    cmd_id: int
    cmd_id_dep: int             # 依赖的 HAU/TIU cmd_id
    cmd_type: str               # TENSOR / GENERAL / GATHER / SCATTER / CW_TRANS / SYS
    src_addr: int               # 源 GMEM 地址
    dst_addr: int               # 目标 GMEM 地址
    src_core_id: int            # 源核 ID
    dst_core_id: int            # 目标核 ID
    shape: tuple[int, ...]      # (n, c, h, w)
    elem_bytes: int
    source_op_id: str

@dataclass
class HAUCommand:
    """HAU: 硬件辅助排序/Top-K"""
    cmd_id: int
    cmd_id_dep: int             # 依赖的 TIU/DMA cmd_id
    op_type: str                # SORT / SORT_INDEX / TOP_K / UNIQUE
    src_addr: int               # 输入数据 LMEM 地址
    dst_addr: int               # 输出数据 LMEM 地址
    num_elements: int           # 元素数量
    top_k: int                  # Top-K 中的 K 值 (仅 TOP_K 类型)
    descending: bool            # 是否降序
    data_format: str            # FP32 / BF16 / INT32
    msg_action: str             # NONE / SEND / WAIT (触发 SDMA 通信)
    msg_id: int                 # 消息 ID (与 SDMA 联动)
    source_op_id: str

@dataclass
class CommOp:
    type: str                   # CDMA_SEND / CDMA_RECV / CDMA_ALLREDUCE
    participants: list[int]     # core_ids
    data_bytes: int
    source_op_id: str
```

### 3.3 汇合点：EngineResult（L4 → L5 统一输出）

```python
@dataclass
class EngineResult:
    """两种模式的统一输出"""
    step_metrics: list[StepMetrics]
    aggregates: Aggregates
    granularity: Granularity
    trace_meta: dict[str, Any]

@dataclass
class StepMetrics:
    op_id: str
    t_compute: float
    t_comm: float
    t_wait: float
    t_total: float
    bottleneck_tag: BottleneckTag
    flops: int
    bytes_read: int
    bytes_write: int
    meta: dict[str, Any]

@dataclass
class Aggregates:
    ttft: float; tpot: float; tps: float
    mfu: float; mbu: float
    memory_peak: int
    total_time: float
    total_compute_time: float
    total_comm_time: float
    total_wait_time: float
```

---

## 四、G5 模式内部设计

### 4.1 L3.g5 指令生成流程

```
DistributedModel
    │
    ├─ 遍历每个 DistributedOp (per chip/core)
    │      │
    │      ├─ [1] Tiling 决策 (instruction_tiler.py)
    │      │   输入: op.local_shape (M/K/N) + ChipSpec (sram_size, lane_num)
    │      │   输出: tile_m/k/n + 循环嵌套顺序
    │      │   约束: A_tile + B_tile + C_tile <= SRAM * utilization
    │      │
    │      ├─ [2] 地址布局 (instruction_tiler.py)
    │      │   输入: tile 大小 + lane_num + per_lane_lmem_size
    │      │   输出: LMEM buffer 地址 (A/B/C 各两份用于 double buffering)
    │      │   逻辑: lane 映射、bank conflict 避免
    │      │
    │      └─ [3] 指令发射 (instruction_emitter.py)
    │          输入: tile 方案 + 地址布局
    │          输出: TIU 指令 + DMA 指令
    │          逻辑: 生成 cmd_id, 设置 cmd_id_dep 依赖
    │          关键: double buffering 下 DMA 提前预取下一 tile
    │
    ├─ 处理 MoE routing op (per chip/core)
    │      │
    │      ├─ [1] Gating 计算 → TIU 指令 (Softmax)
    │      ├─ [2] Top-K 选择 → HAU 指令 (SORT/TOP_K)
    │      │      HAU.msg_action = SEND → 触发 SDMA
    │      └─ [3] Token 分发 → SDMA 指令 (GATHER/SCATTER)
    │             src_core → dst_core (expert 所在核)
    │
    ├─ 处理核间通信 op (AllReduce/P2P/AllToAll)
    │      ├─ 芯片内通信 → SDMA 指令 (核间 GMEM 搬运)
    │      └─ 跨芯片通信 → CommOp (CDMA send/recv/allreduce)
    │
    └─ 输出: CoreProgram (含 tiu_cmds + dma_cmds + sdma_cmds + hau_cmds)
```

### 4.2 L4.g5 事件驱动仿真架构

```
EventScheduler (sim_engine.py)
    │  时间堆 + delta cycle + 信号/FIFO/信号量
    │  6 个时钟域: clk, clk_tiu, clk_gdma, clk_sdma, clk_ife, clk_ddr
    │
    ├── IFE (ife.py) ─────────────── 指令调度层
    │   职责: 指令解码 → 分发到 TIU/GDMA/SDMA/HAU 的 FIFO
    │   FIFO 深度限制: TIU FIFO (32KB), GDMA FIFO (8320B)
    │   依赖管理: cmd_id 计数器, CHUNK 级读写依赖
    │   调度延迟: pack(1cyc) + decode(1cyc) + dispatch(1cyc)
    │   注: 自动生成模式下由 instruction_emitter 替代取指功能,
    │       但 FIFO 背压和调度延迟仍需建模
    │
    ├── TIU (tiu.py) ─────────────── 张量计算引擎
    │   状态机: Init → Compute → Finish
    │   Init:   检查 cmd_id_dep <= dma_sync_id
    │   Compute: cycle_count 倒计时 (查延迟表)
    │   Finish:  写 tiu_sync_id，通知 DMA/SDMA
    │   指令类型: CONV / MM2_NN / MM2_NT / MM2_TT / SFU / AR
    │
    ├── GDMA (dma.py) ─────────────── 全局 DMA (LMEM <-> DDR)
    │   5 级流水: 调度 → 分段 → 总线请求 → 响应处理 → 完成跟踪
    │   分段: tensor shape → 1D segments (连续性判断)
    │   地址: lane 映射 (lane_idx * per_lane_size + offset)
    │   Outstanding 控制: rd=512, wr=512 (信号量)
    │   同步: tdma_sync_id ←→ TIU cmd_id_dep
    │
    ├── SDMA (sdma.py) ─────────────── 系统 DMA (核间通信)
    │   职责: GMEM → GMEM 跨核数据搬运
    │   用途: AllReduce/AllGather/P2P/MoE expert 数据分发
    │   指令类型: TENSOR / GENERAL / GATHER / SCATTER / CW_TRANS / SYS
    │   同步: sdma_sync_id (独立于 TIU/GDMA)
    │   触发: HAU msg_action (SEND/WAIT) 或直接调度
    │   端口: 独立 GMEM socket, 通过 Bus/NoC 访问其他核的 DDR
    │   与 GDMA 区别:
    │     - GDMA: 本核 LMEM <-> DDR (计算数据搬运)
    │     - SDMA: 跨核 GMEM <-> GMEM (核间通信)
    │
    ├── HAU (hau.py) ─────────────── 硬件辅助单元
    │   职责: 硬件排序/Top-K/Unique (MoE routing 关键路径)
    │   状态机: Init → Compute → Finish → [MsgAction]
    │   指令类型: SORT / SORT_INDEX / TOP_K / UNIQUE
    │   延迟计算: f(num_elements, data_format) (TPUPerf 当前硬编码为 1, 需修正)
    │   消息联动: msg_action=SEND → 触发 SDMA 发送
    │             msg_action=WAIT → 等待 SDMA 消息到达
    │   MoE 推理流程: Gating(TIU) → Top-K(HAU) → 数据分发(SDMA) → Expert(TIU)
    │
    ├── Memory (memory.py) ─────────── 存储子系统
    │   LMEM: 29ns 基础延迟 + bank conflict 检测 (16 banks)
    │         容量: 2MB/核 (SG2262), 按 lane 分布
    │   DDR:  150ns 基础延迟 + 地址映射 (col/bank_group/bank/row)
    │         + bank conflict (同 bank 不同 row 惩罚)
    │         + outstanding 控制 (FIFO 128)
    │
    ├── NoC (noc_adapter.py) ────────── 片内互连 (Die 内)
    │   v1 (SimpleBus, G5 初版):
    │     NxM 互联, 距离延迟 = hop_latency * manhattan_distance
    │     Master: 64 GDMA + 64 SDMA = 128 端口
    │     Slave: 64 DDR (通过 ARE)
    │     仲裁: 轮询, 带宽竞争
    │   v2 (CrossRing 集成, 后续):
    │     通过 NoCInterface 协议对接 CrossRing 仿真器
    │     支持 req/rsp/data 三网络并行
    │     支持 D2D 跨 Die 通信
    │     详见 "4.6 NoC 接口设计"
    │
    └── Interconnect (interconnect.py) ── 多芯片互连 (Die 间)
        C2C:  链路带宽模型, delay = data_bytes / bw (11.2 GB/s)
        CDMA: 信用流控 (RECV 分配缓冲 → 发 credit → SEND 等 credit → 传数据)
```

### 4.2.1 五引擎同步关系

```
            ┌─────────────────────────────────────────┐
            │              IFE (指令调度)                │
            │  cmd 分发 → TIU FIFO / GDMA FIFO / ...   │
            └──┬──────────┬──────────┬──────────┬──────┘
               │          │          │          │
               ▼          ▼          ▼          ▼
            ┌─────┐  ┌──────┐  ┌──────┐  ┌─────┐
            │ TIU │  │ GDMA │  │ SDMA │  │ HAU │
            └──┬──┘  └──┬───┘  └──┬───┘  └──┬──┘
               │        │         │          │
    同步信号:  tiu_sync_id  tdma_sync_id  sdma_sync_id
               │        │         │          │
               ├────────┤         │          │
               │ cmd_id_dep       │          │
               │ (TIU 等 GDMA     │          │
               │  GDMA 等 TIU)    │          │
               │                  │          │
               │                  ├──────────┤
               │                  │ msg_action│
               │                  │ HAU SEND → SDMA 发送
               │                  │ HAU WAIT ← SDMA 消息
               │                  │          │
    MoE 流程:  │                  │          │
    Gating ────┘                  │          │
    Top-K ────────────────────────┼──────────┘
    Dispatch ─────────────────────┘
    Expert ────┘
```

### 4.3 HAU 延迟计算

```
TPUPerf 现状: cal_cycle() 硬编码返回 1 (不准确)

G5 需要根据数据量建模:

SORT (升序/降序排序):
  // 基于硬件比较网络, 类似 bitonic sort
  total = init_cycle + ceil(N / sort_width) * ceil(log2(N)) * compare_cycle

TOP_K (Top-K 查询):
  // 部分排序, 只需找到前 K 个
  total = init_cycle + ceil(N / sort_width) * ceil(log2(K)) * compare_cycle

UNIQUE (去重):
  total = init_cycle + ceil(N / sort_width) * scan_cycle

注: 具体参数需从 TPUPerf 硬件 spec 或实测校准获取
    init_cycle, sort_width, compare_cycle 为芯片特定常量
```

### 4.4 TIU 延迟计算（核心公式）

```
MM2.nn:
  init_cycle = 44
  total = ceil(C/lane_num) * ceil(W/eu_num) * (ceil(K/ch_per_cyc) + bank_conflict + bias) + init

CONV:
  sync_cycle = 23
  total = sync + N * ceil(C_out/lane_num) * ceil(H_out*W_out/eu_num)
          * (kh*kw*ceil(C_in/ch_per_cyc) + shift_round + psum)

Bank Conflict:
  res_bank  = (res_addr - LMEM_START) >> bank_width
  opd_bank  = (opd_addr - LMEM_START) >> bank_width
  conflict  = (res_bank == opd0_bank) + (res_bank == opd1_bank)
```

### 4.5 SDMA 延迟计算

```
SDMA 延迟 = 调度延迟 + 分段延迟 + NoC 传输延迟

调度延迟:
  dispatch = cmd_decode(1cyc) + dep_check(1cyc)

分段延迟 (与 GDMA 类似):
  segment_count = tensor_4d_to_1d_segments(shape, stride)
  segment_latency = foreach segment: bus_request + bus_response

NoC 传输延迟:
  v1 (SimpleBus):
    latency = data_bytes / link_bw + hop_latency * manhattan_distance(src, dst)
  v2 (CrossRing):
    latency = noc_adapter.simulate(traffic_flow)  // 委托 CrossRing

与 GDMA 的关键区别:
  - GDMA 走 LMEM <-> DDR 路径, 延迟由 DDR 主导 (150ns base)
  - SDMA 走 GMEM <-> GMEM 路径, 延迟由 NoC 距离主导
  - SDMA 有独立 gmem_socket, 不与 GDMA 竞争端口
  - SDMA 的 outstanding 控制独立于 GDMA
```

### 4.6 NoC 接口设计（CrossRing 集成预留）

```python
# === 抽象协议 (noc_adapter.py) ===

class NoCInterface(Protocol):
    """NoC 仿真器抽象接口 - v1/v2 均实现此协议"""

    def configure(self, spec: NoCTopologySpec) -> None:
        """配置 NoC 拓扑参数"""
        ...

    def inject_traffic(self, flows: list[TrafficFlow]) -> None:
        """注入流量请求 (来自 GDMA/SDMA 的总线请求)"""
        ...

    def step(self, cycles: int) -> None:
        """推进仿真 N 个周期"""
        ...

    def query_flow_result(self, flow_id: int) -> FlowResult | None:
        """查询单条流量的完成状态和延迟"""
        ...

    def get_metrics(self) -> NoCMetrics:
        """获取全局性能统计"""
        ...


@dataclass
class NoCTopologySpec:
    """NoC 拓扑配置"""
    rows: int                       # mesh 行数
    cols: int                       # mesh 列数
    flit_size_bytes: int            # flit 大小 (64/128)
    network_freq_ghz: float         # 网络时钟频率
    ip_freq_ghz: float              # IP 核时钟频率
    version: str                    # "v1" (SimpleBus) | "v2" (CrossRing)
    # v2 专用参数
    iq_depth: int = 4               # 注入队列深度
    eq_depth: int = 4               # 弹出队列深度
    arbitration: str = "round_robin" # 仲裁策略


@dataclass
class TrafficFlow:
    """单条流量请求"""
    flow_id: int                    # 唯一标识
    src_core: int                   # 源核 ID
    dst_core: int                   # 目标核 ID
    src_type: str                   # "gdma" / "sdma"
    dst_type: str                   # "ddr" / "lmem"
    op_type: str                    # "R" (读) / "W" (写)
    burst_length: int               # flit 数量
    start_cycle: int                # 注入时刻
    source_cmd_id: int              # 关联的 GDMA/SDMA cmd_id


@dataclass
class FlowResult:
    """单条流量结果"""
    flow_id: int
    start_cycle: int
    end_cycle: int
    latency_cycles: int
    hops: int
    contentions: int                # 仲裁竞争次数


@dataclass
class NoCMetrics:
    """全局 NoC 性能统计"""
    avg_latency_cycles: float
    peak_latency_cycles: int
    total_flits: int
    bandwidth_utilization: float    # 0.0 ~ 1.0
    congestion_events: int


# === v1 实现: SimpleBus (G5 初版) ===

class SimpleBusModel(NoCInterface):
    """Manhattan 距离延迟模型, 对标 TPUPerf SimpleBus<128,64>"""

    def __init__(self):
        self.hop_latency: int = 45  # cycles per hop
        self.link_bw_gbps: float = 128.0
        self.master_ports: int = 128  # 64 GDMA + 64 SDMA
        self.slave_ports: int = 64    # 64 DDR (via ARE)

    # NoCInterface 方法实现...


# === v2 实现: CrossRing 适配器 (后续集成) ===

class CrossRingAdapter(NoCInterface):
    """对接 CrossRing Python 仿真器
    CrossRing 项目位置: code/CrossRing/
    支持: v1 (IQ/EQ/RB) 和 v2 (RingStation) 两种架构
    三网络并行: req_network / rsp_network / data_network
    D2D 通信: 6 阶段跨 Die 流程, Tracker 资源管理
    """

    def __init__(self, crossring_path: str):
        # 导入 CrossRing 的 BaseModel
        # 配置转换: NoCTopologySpec → CrossRing YAML config
        # 流量转换: TrafficFlow → CrossRing TXT format
        pass

    # CrossRing 特有功能:
    # - configure() 生成 kcin_*.yaml 配置
    # - inject_traffic() 生成 traffic.txt 文件
    # - step() 调用 BaseModel.step()
    # - get_metrics() 从 CrossRing Result/ 提取结果
```

**CrossRing 集成要点**:

| 维度 | G5 侧 | CrossRing 侧 | 转换 |
|------|--------|-------------|------|
| 拓扑 | NoCTopologySpec | kcin_*.yaml | rows/cols → TOPO_TYPE |
| 流量 | TrafficFlow list | traffic.txt | cycle,die,src,dst,op,burst |
| 时钟 | 统一 ns 时间轴 | IP 1GHz + Network 2GHz | freq ratio 换算 |
| 结果 | FlowResult | Flit.arrival_cycle | 逐 flow 提取 |
| D2D | InterconnectModel | D2D_Model | 跨芯片请求路由 |

### 4.7 结果适配（adapter.py）

```
仿真事件流:
  [{core_id, type: TIU/DMA, cmd_id, start_cycle, end_cycle, source_op_id}, ...]

聚合为 StepMetrics:
  按 source_op_id 分组 → 每组:
    t_compute = max(tiu_end) - min(tiu_start)    (cycle → ms)
    t_comm    = 对应 CommOp 的实际延迟
    t_wait    = idle gaps (TIU 等 DMA 或反之)

聚合为 Aggregates:
  total_cycles → ttft (prefill) / tpot (decode)
  total_tiu_flops / (total_cycles / freq) / peak_flops → mfu
```

---

## 五、L0 编排逻辑

```python
# L0_entry/engine.py
def run_evaluation(config: EvalConfig):
    # === 共享阶段 ===
    model_ir = build_workload_ir(config)                    # L1
    chip_spec = build_chip_spec(config)                      # L2
    distributed = ParallelismPlanner.plan(                   # L3.common
        model_ir, chip_spec, config.deployment
    )

    # === 模式分叉 ===
    if config.mode == "math":
        tile_plan = MathTilingPlanner.plan(distributed, chip_spec)
        exec_plan = MathScheduler.schedule(distributed, tile_plan)
        result = MathEngine.evaluate(exec_plan, distributed, hw)

    elif config.mode == "g5":
        program = InstructionEmitter.emit(distributed, chip_spec)
        result = G5SimEngine.simulate(program, chip_spec)

    elif config.mode == "g5_binary":
        # 跳过 L1/L3，直接从二进制文件构造 CoreProgram
        program = BinaryParser.parse(config.binary_files)
        result = G5SimEngine.simulate(program, chip_spec)

    # === 共享阶段 ===
    cost = CostModel.calculate(config, result)               # L4.common
    report = ReportingEngine.generate(result, cost, config)   # L5
    return report
```

---

## 六、L2 扩展参数

G5 模式需要 ChipSpec 扩展以下微架构参数：

| 参数 | 用途 | 示例值 (SG2262) |
|------|------|----------------|
| clock_freq_ghz | 仿真时钟基准 | 1.0 |
| lmem_read_latency_ns | LMEM 访问延迟 | 29 |
| ddr_read_latency_ns | DDR 访问延迟 | 150 |
| ddr_bus_width_bytes | DDR 总线宽度 | 64 |
| ddr_outstanding | DDR outstanding 数 | 128 |
| lmem_bank_num | LMEM bank 数 | 16 |
| cache_line_size_bytes | Cache 行大小 | 128 |
| cache_ways | Cache 关联度 | 4 |
| bus_grid_size | Bus 网格尺寸 | (8, 8) |
| bus_hop_latency_cycles | Bus 每跳延迟 | 45 |
| gdma_outstanding_rd | GDMA 读 outstanding 数 | 512 |
| gdma_outstanding_wr | GDMA 写 outstanding 数 | 512 |
| sdma_outstanding_rd | SDMA 读 outstanding 数 | 256 |
| sdma_outstanding_wr | SDMA 写 outstanding 数 | 256 |
| ife_tiu_fifo_bytes | IFE TIU FIFO 容量 | 32768 |
| ife_gdma_fifo_bytes | IFE GDMA FIFO 容量 | 8320 |
| hau_sort_width | HAU 硬件排序宽度 | 16 |
| hau_compare_cycles | HAU 单次比较周期 | 1 |

**NoC 参数 (v2 CrossRing 集成时使用)**:

| 参数 | 用途 | 示例值 (SG2262) |
|------|------|----------------|
| noc_version | NoC 版本 | "v1" (SimpleBus) / "v2" (CrossRing) |
| noc_rows | NoC mesh 行数 | 5 |
| noc_cols | NoC mesh 列数 | 4 |
| noc_flit_size_bytes | Flit 大小 | 128 |
| noc_network_freq_ghz | 网络时钟频率 | 2.0 |
| noc_ip_freq_ghz | IP 核时钟频率 | 1.0 |
| noc_iq_depth | 注入队列深度 | 4 |
| noc_eq_depth | 弹出队列深度 | 4 |
| noc_arbitration | 仲裁策略 | "round_robin" |

已有参数可直接复用：cube_m/k/n, sram_size_kb, lane_num, align_bytes, compute_dma_overlap_rate

---

## 七、实施步骤

### Step 1: 目录重构
- `backend/math_model/` → `backend/perf_model/`
- L3_mapping/ 内部文件移入 common/ 和 math/ 子目录
- L4_evaluation/ 内部文件移入 common/ 和 math/ 子目录
- 更新所有 import 路径
- 验证现有 math 模式功能不受影响

### Step 2: L3.g5 指令生成
- 实现 `program.py` (CoreProgram/TIUCommand/DMACommand/SDMACommand/HAUCommand 数据结构)
- 实现 `instruction_tiler.py` (指令级 tiling + 地址布局)
- 实现 `instruction_emitter.py` (从 DistributedOp → 指令序列, 含 TIU/GDMA/SDMA/HAU)
  - MatMul → TIU + GDMA 指令
  - MoE routing → HAU (Top-K) + SDMA (数据分发) 指令
  - AllReduce/P2P → SDMA (芯片内) 或 CommOp (跨芯片) 指令
- 用 MatMul 单 op 验证指令生成正确性

### Step 3: L4.g5 仿真内核 (TIU + GDMA 核心)
- 实现 `sim_engine.py` (事件调度器 + delta cycle + 信号机制 + 多时钟域)
- 实现 `ife.py` (指令调度模型: FIFO 深度限制 + 调度延迟 + 背压)
- 实现 `tiu.py` (状态机 + MM2/CONV/SFU/AR 延迟计算)
- 实现 `dma.py` (GDMA 5 级流水 + 分段逻辑 + outstanding 控制)
- 实现 `memory.py` (LMEM bank conflict + DDR 模型)
- 单核单 op 端到端验证 (TIU + GDMA 协同)

### Step 4: SDMA + HAU 扩展
- 实现 `sdma.py` (核间通信: GMEM→GMEM, 独立同步 ID, 独立 outstanding)
- 实现 `hau.py` (硬件排序: Sort/Top-K/Unique, 消息联动 SDMA)
- 验证 MoE Top-K → SDMA 分发流程
- 验证 SDMA AllReduce 通信延迟

### Step 5: 多核互连与 NoC
- 实现 `noc_adapter.py` (NoCInterface 协议 + SimpleBus v1 实现)
  - SimpleBus: Manhattan 距离延迟, 128 master + 64 slave 端口, 带宽仲裁
- 实现 `interconnect.py` (多芯片互连: C2C + CDMA 信用流控)
- 多核并行仿真 (64 核 TpuSubsys 实例)
- AllReduce/P2P 通信建模 (芯片内走 SDMA+NoC, 跨芯片走 CDMA+C2C)

### Step 6: 系统集成
- L0 编排逻辑（mode 切换: math / g5 / g5_binary）
- L2 参数扩展 (NoC 参数 + SDMA/HAU/IFE 参数)
- adapter.py（仿真事件 → EngineResult, 含 SDMA/HAU 事件聚合）
- 前端 mode 选择 UI
- 端到端验证（Math vs G5 结果交叉对比）

### Step 7: 二进制解析
- 实现 `binary_parser.py` (.BD/.GDMA/.HAU/.SDMA 四类二进制文件解析)
- g5_binary 模式集成 (跳过 L1/L3, 直接构造 CoreProgram)

### Step 8 (后续): CrossRing NoC 集成
- 实现 `CrossRingAdapter` (NoCInterface v2 实现)
- 配置转换: NoCTopologySpec → CrossRing YAML
- 流量转换: TrafficFlow → CrossRing TXT
- 结果提取: CrossRing Result → FlowResult
- D2D 跨 Die 通信路由
- 对比验证: SimpleBus vs CrossRing 精度差异

---

## 八、验证策略

| 级别 | 验证内容 | 精度要求 |
|------|---------|---------|
| V1 | TIU 单指令延迟 vs TPUPerf | <= 2 cycles |
| V2 | GDMA 单次搬运延迟 vs TPUPerf | <= 5% |
| V3 | HAU Sort/Top-K 延迟 vs 实测 | 量级正确 (TPUPerf 硬编码, 需实测校准) |
| V4 | SDMA 核间通信延迟 | <= 5% |
| V5 | IFE 调度开销 (FIFO 满时背压) | 定性验证 |
| V6 | 单核单层延迟 (TIU+GDMA 协同) | <= 1% |
| V7 | 多核单层延迟 (含 NoC 竞争) | <= 5% |
| V8 | MoE 完整流程 (HAU+SDMA+TIU) | <= 10% |
| V9 | Math vs G5 交叉验证 | 趋势一致 |
| V10 | 端到端 DeepSeek-V3 | 与 Math 模式偏差分析 |
| V11 | SimpleBus vs CrossRing 对比 | 差异分析 (后续) |

---

## 九、关键文件清单

### 需要修改的文件（目录重构）
- `backend/math_model/` → `backend/perf_model/` (重命名)
- `L3_mapping/` 内所有文件 → 移入 common/ 或 math/
- `L4_evaluation/` 内所有文件 → 移入 common/ 或 math/
- `L0_entry/engine.py` — 增加 mode 分叉逻辑
- `L0_entry/eval_config.py` — 增加 mode 字段
- `L2_arch/chip.py` — 扩展 G5 微架构参数
- 所有 import 语句更新

### 需要新增的文件（G5 模式）
- `L3_mapping/g5/program.py` — CoreProgram 数据结构 (含 TIU/DMA/SDMA/HAU 指令)
- `L3_mapping/g5/instruction_tiler.py` — 指令级 tiling
- `L3_mapping/g5/instruction_emitter.py` — 指令生成 (含 MoE routing 的 HAU+SDMA 指令)
- `L3_mapping/g5/binary_parser.py` — 二进制解析 (.BD/.GDMA/.HAU/.SDMA)
- `L4_evaluation/g5/sim_engine.py` — 事件驱动仿真调度器
- `L4_evaluation/g5/ife.py` — IFE 指令调度模型 (FIFO + 依赖管理)
- `L4_evaluation/g5/tiu.py` — TIU 计算引擎
- `L4_evaluation/g5/dma.py` — GDMA 搬运引擎 (LMEM <-> DDR)
- `L4_evaluation/g5/sdma.py` — SDMA 核间通信引擎 (GMEM <-> GMEM)
- `L4_evaluation/g5/hau.py` — HAU 硬件辅助单元 (Sort/Top-K)
- `L4_evaluation/g5/memory.py` — 内存子系统 (LMEM + DDR)
- `L4_evaluation/g5/noc_adapter.py` — NoC 适配器 (v1: SimpleBus / v2: CrossRing)
- `L4_evaluation/g5/interconnect.py` — 多芯片互连 (C2C + CDMA)
- `L4_evaluation/g5/adapter.py` — 仿真事件 → EngineResult

### 外部依赖（CrossRing 集成, 后续）
- `code/CrossRing/` — NoC 仿真器 (纯 Python)
  - 通过 `CrossRingAdapter` 导入其 `BaseModel`
  - 配置格式: YAML (kcin_*.yaml)
  - 流量格式: TXT (cycle, die, src, dst, op, burst)
  - 结果: Result/ 目录下的统计数据

---

## 附录 A: TPUPerf 五引擎 vs G5 模块对照

| TPUPerf 模块 | TPUPerf 实现 | G5 对应模块 | LLM 推理用途 | 优先级 |
|-------------|-------------|-----------|-------------|-------|
| **TIU** | `tpu/tiu*.cpp` — 3态状态机, cal_cycle() 逐指令类型计算 | `tiu.py` | MatMul/CONV/Softmax/RMSNorm 所有计算 | 必须 |
| **GDMA (Tdma)** | `tpu/tdma*.cpp` — SystemC 多线程流水, 分段+outstanding | `dma.py` | 权重/激活值搬运 (LMEM <-> DDR) | 必须 |
| **SDMA** | `tpu/tdma*.cpp` (复用 Tdma 类, 独立实例) | `sdma.py` | AllReduce/P2P 核间通信 (GMEM <-> GMEM) | 高 |
| **HAU** | `tpu/hau*.cpp` — SortCmd 80位指令, 硬编码1周期 | `hau.py` | MoE Top-K routing, 消息触发 SDMA | 中高 (MoE 必需) |
| **IFE** | `spec/*/ife*.cpp` — 指令取指+解码+FIFO分发 | `ife.py` | 指令调度, FIFO 背压, 依赖管理 | 高 (影响流水效率) |
| **LMEM** | `memory/local_mem*.cpp` — 16 bank, conflict 检测 | `memory.py` | 片上缓存, bank conflict 影响 TIU 延迟 | 必须 |
| **DDR/ARE** | `memory/ddr*.cpp` + `ip/are*.cpp` — bank group, outstanding | `memory.py` | 全局存储, 权重/KV cache 存放 | 必须 |
| **SimpleBus** | `framework/simple_bus.h` — NxM, Manhattan 距离 | `noc_adapter.py` (v1) | 片内核间互连 (SG2262: 128M+64S) | 必须 |
| **NoC** | `noc/` — Ring/Mesh 路由 (非 SG2262) | `noc_adapter.py` (v2) | 片内核间互连 (高级芯片) | 后续 |
| **CDMA** | `ip/cdma*.cpp` — 信用流控 | `interconnect.py` | 跨芯片通信 | 必须 (多芯片) |
| **C2C** | `ip/c2c*.cpp` — 链路带宽 11.2 GB/s | `interconnect.py` | Die 间互连 | 必须 (多芯片) |
| **GsCache** | `ip/gs_cache*.cpp` — L2 cache | 暂不建模 | 缓存加速, 对 LLM 推理影响较小 | 低 |

## 附录 B: MoE 推理中 HAU+SDMA 协作流程

```
单层 MoE FFN 执行流程 (以 DeepSeek-V3 为例: 256 experts, top-8):

1. [TIU] Gating 计算
   input: hidden_state (bs, seq, hidden_dim)
   op: MatMul(hidden_state, gate_weight) → gating_scores (bs*seq, 256)
   output: 每个 token 对 256 个 expert 的分数

2. [HAU] Top-K 选择
   input: gating_scores (bs*seq, 256)
   op: TOP_K(gating_scores, K=8) → top_indices (bs*seq, 8), top_weights (bs*seq, 8)
   output: 每个 token 激活的 8 个 expert ID 和权重
   msg_action: SEND → 通知 SDMA 开始数据分发

3. [SDMA] Token 分发 (Dispatch)
   input: hidden_state + top_indices
   op: SCATTER → 将每个 token 发送到对应 expert 所在的核
   路径: src_core.GMEM → NoC → dst_core.GMEM
   注: EP=8 时, 每个核持有 32 个 expert, token 需跨核路由

4. [TIU] Expert 计算 (per expert)
   input: 路由到本核的 token 子集
   op: FFN (up_proj → activation → down_proj)
   output: expert_output

5. [SDMA] 结果收集 (Combine)
   input: expert_output + top_weights
   op: GATHER → 将 expert 输出按权重加权后收集回源核
   路径: expert_core.GMEM → NoC → src_core.GMEM

6. [TIU] 加权求和
   input: 8 个 expert 的加权输出
   op: weighted_sum → final_output
```

## 附录 C: CrossRing NoC 关键参数参考

```yaml
# CrossRing 典型配置 (SG2262, 5x4 mesh)
# 来源: code/CrossRing/config/topologies/kcin_5x4.yaml

TOPO_TYPE: 5x4                    # 20 节点 mesh
FLIT_SIZE: 128                    # 128 字节/flit
NETWORK_FREQUENCY: 2.0            # 网络域 2GHz
IP_FREQUENCY: 1.0                 # IP 域 1GHz
KCIN_VERSION: v1                  # IQ/EQ/RB 架构

# 缓冲
IQ_CH_FIFO_DEPTH: 4               # 注入队列深度
EQ_CH_FIFO_DEPTH: 4               # 弹出队列深度
RN_RDB_SIZE: 256                  # 读数据缓冲
RN_WDB_SIZE: 256                  # 写数据缓冲

# 仲裁
arbitration:
  default:
    type: round_robin

# 三通道
NETWORK_CHANNEL_CONFIG:
  req:  { num_channels: 1 }       # 请求通道
  rsp:  { num_channels: 1 }       # 响应通道
  data: { num_channels: 1 }       # 数据通道

# Ring 参数
SLICE_PER_LINK_HORIZONTAL: 7      # 横向环分片数
SLICE_PER_LINK_VERTICAL: 7        # 纵向环分片数

# D2D (多 Die 时启用)
# D2D_SN/RN Tracker 资源管理
# 6 阶段通信流程: GDMA→D2D_SN→D2D_RN→DDR→D2D_RN→D2D_SN→GDMA
```
