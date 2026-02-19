# G5 指令级仿真器 — 完整架构设计

> 日期: 2026-02-19
> 状态: 设计阶段
> 前置: G5-instruction-level-simulator.md (Step 1-6 已完成)

## 1. 设计目标

在现有 G5 单核仿真器基础上，扩展为**完整的多核多芯片 cycle 级行为仿真器**，覆盖：

- **芯片内**: 多核 (TIU/GDMA/SDMA/HAU) + Bus互联 + LMEM/DDR 存储层级
- **Die间**: D2D (Die-to-Die) 互联
- **芯片间**: CDMA + PAXI + RC Link + CESOC + 物理链路
- **交换机**: L2 Switch (12-stage pipeline, VOQ, iSLIP)

精度目标：cycle 级（时间单位 ns，基于频率换算 cycle），对标 TPUPerf SystemC 实现。

## 2. SG2262 硬件层级

```
算子 (MatMul, Attention, FFN, ...)
  | (L3 mapping: 算子 -> 指令序列)
  v
Core (核心): TIU + GDMA + SDMA + HAU + LMEM
  | (SDMA 走 Bus, GDMA 走 Bus -> DDR)
  v
NoC / Bus: 片上互联 (NxM 总线, Manhattan 距离延迟, FCFS 仲裁)
  |
  v
Die: 硅片单元 (SG2262: 每Die最多32核)
  | (D2D: Die-to-Die 互联, 通过 NoC D2D group)
  v
Chip: 芯片 (SG2262: 2 Die = 64核)
  | (CDMA: 跨芯片 DMA, 4个/Die, 8线程/CDMA)
  v
C2C 链路: CDMA -> PAXI -> RC Link -> CESOC -> SerDes -> 物理链路
  | (直连模式: 两芯片直接 SerDes 相连)
  | (Switch模式: 经 L2 交换机转发)
  v
L1 Layer: Cluster (最多32芯片, 支持 all2all/ring/torus/mesh/clos)
  |
  v
L2 Layer: 多Cluster互联 (仅 clos 拓扑, 最多1024芯片)
```

### 关键组件职责

| 组件 | 层级 | 职责 |
|------|------|------|
| TIU | Core 内 | 矩阵计算引擎 |
| GDMA | Core 内 | 全局DMA, LMEM <-> DDR |
| SDMA | Core 内 | 串行DMA, 核间小粒度搬运 (走Bus) |
| HAU | Core 内 | 硬件排序/Top-K |
| Bus (NoC) | Die 内 | NxM 总线, 核间互联, Manhattan 距离延迟 |
| D2D | Chip 内 | Die 间互联 |
| CDMA | Chip 级 | 跨芯片 DMA 引擎, Send/Recv/AllReduce, 信用流控 |
| PAXI Core | C2C 链路 | 事务层: AXI <-> Flit 编解码 |
| RC Link | C2C 链路 | 传输层: Go-Back-N 重传, CBFC/PFC 流控 |
| CESOC | C2C 链路 | 数据链路层: MAC/PCS/FEC |
| SerDes | C2C 链路 | 物理层: 112G PAM4 |
| L2 Switch | 网络 | 12-stage pipeline, VOQ, iSLIP 调度 |

## 3. 与 L0-L5 框架的映射

```
L0 (Entry)     -- 不变: EvalConfig + mode="g5" 路由
L1 (Workload)  -- 不变: ModelConfig -> WorkloadIR
L2 (Arch)      -- 扩展: ChipSpec 增加 PAXI/Switch/CDMA 参数
L3 (Mapping)   -- 扩展: InstructionEmitter 生成多核+CDMA 指令
L4 (Evaluation)-- 核心重构: SimEngine -> 全局事件驱动仿真内核
L5 (Reporting) -- 不变: EngineResult -> 报表
```

## 4. L4 仿真内核架构

### 4.1 设计理念

采用 **全局事件队列 + 模块化 SimObject** 模式，对标 SystemC / gem5：

- 所有硬件模块继承 `SimObject` 基类
- 共享一个全局事件队列 (heapq)
- 时间单位: ns，基于芯片频率换算 cycle
- 事件驱动: 仅在有活动时推进时间（跳过空闲 cycle）
- 支持多时钟域 (TPU clock, Bus clock, DDR clock)

参考框架:
- [SystemC Simulation Kernel](https://users.ece.utexas.edu/~gerstl/ee382v_f14/lectures/SE100-SimulationEngine-2.0.pdf)
- [gem5 Event-driven Programming](https://www.gem5.org/documentation/learning_gem5/part2/events/)
- [gem5 Garnet NoC Model](https://www.gem5.org/documentation/general_docs/ruby/garnet-2/)
- [SuperSim Cycle-Accurate Network Simulator](https://github.com/ssnetsim/supersim)

### 4.2 TPUPerf 对标

```
TPUPerf (SystemC)                    G5 仿真器 (Python)
==================                   ==================

sc_main()                            SimKernel (全局事件队列)
  |                                    |
  +-- TpuSubsys[0..N]                 +-- CoreSubsys[0..N]
  |     +-- Tiu (SC_METHOD)           |     +-- TIUEngine(SimObject)
  |     +-- Tdma/GDMA (SC_THREAD)     |     +-- GDMAEngine(SimObject)
  |     +-- Tdma/SDMA (SC_THREAD)     |     +-- SDMAEngine(SimObject)
  |     +-- Hau                        |     +-- HAUEngine(SimObject)
  |     +-- lmem                       |     +-- LMEMModel(SimObject)
  |                                    |
  +-- simple_bus (NxM)                 +-- BusModel(SimObject)
  +-- ARE + DDR[0..N]                  +-- DDRModel(SimObject)
  +-- gs_cache                         +-- CacheModel(SimObject)
  +-- CDMA (16 SC_THREAD)              +-- CDMAEngine(SimObject)
  +-- C2C (6 SC_THREAD, 简化)          +-- PAXICore + RCLink + CESOC (细化!)
  +-- FakeChip (stub)                  +-- L2Switch(SimObject)
```

关键差异: TPUPerf 的 C2C 是简化的 `delay = data/bandwidth` 模型，我们要做 **cycle 级行为建模**。

### 4.3 文件结构

```
backend/perf_model/L4_evaluation/g5/
  |
  +-- kernel/                        # [新增] 仿真内核
  |     __init__.py
  |     sim_kernel.py                # 全局事件队列 + 调度循环
  |     sim_object.py                # SimObject 基类
  |
  +-- chip/                          # [新增] 芯片内部模块
  |     __init__.py
  |     core_subsys.py               # 单核子系统 (TIU+GDMA+SDMA+HAU+LMEM)
  |     tiu.py                       # TIU 引擎 (重构自现有 tiu.py)
  |     dma.py                       # GDMA 引擎 (重构自现有 dma.py)
  |     sdma.py                      # SDMA 引擎 (重构自现有 sdma.py)
  |     hau.py                       # HAU 引擎 (重构自现有 hau.py)
  |     lmem.py                      # LMEM 存储模型 (bank 级)
  |     ddr.py                       # DDR 存储模型
  |     bus.py                       # Bus NxM 互联
  |     cache.py                     # Cache 模型 (可选)
  |
  +-- interconnect/                  # [新增] 芯片间互联模块
  |     __init__.py
  |     cdma.py                      # CDMA 引擎
  |     paxi_core.py                 # PAXI 事务层
  |     rc_link.py                   # RC Link 传输层
  |     cesoc.py                     # CESOC MAC/PCS/FEC
  |     c2c_link.py                  # 物理链路
  |     l2_switch.py                 # L2 交换机
  |
  +-- top/                           # [新增] 顶层组装
  |     __init__.py
  |     single_chip.py               # 单芯片组装 (多核)
  |     multi_chip.py                # 多芯片组装 (含 CDMA/C2C/Switch)
  |
  +-- sim_engine.py                  # [重构] 现有 -> 使用 kernel
  +-- adapter.py                     # [保留] 结果适配
  +-- pipeline.py                    # [保留] G5 管线封装
  +-- memory.py                      # [保留] 现有内存模型
```

## 5. 仿真内核详细设计 (kernel/)

### 5.1 SimKernel

```python
class SimKernel:
    """轻量级 SystemC-like 仿真内核

    核心数据结构:
      event_queue: heapq[(time_ns, seq_id, callback)]
      current_time: float   # 当前仿真时间 (ns)
      objects: dict          # 注册的所有 SimObject
      clocks: dict           # 时钟域定义
      seq_counter: int       # 事件序号 (解决同时间事件的稳定排序)

    核心方法:
      schedule(delay_ns, callback)     # 延迟调度事件
      schedule_at(time_ns, callback)   # 绝对时间调度事件
      now() -> float                   # 获取当前仿真时间
      run() -> None                    # 主事件循环
      register(obj: SimObject)         # 注册模块
      cycle_to_ns(cycles, clock_name)  # cycle -> ns 转换
      ns_to_cycle(ns, clock_name)      # ns -> cycle 转换

    时钟域:
      tpu_clk:  1.0 GHz (1 ns/cycle)
      bus_clk:  可配置 (128/153/192 GB/s)
      ddr_clk:  200 MHz (5 ns/cycle)

    Delta Cycle:
      同一 time_ns 的事件按 seq_id 排序执行
      信号写入在当前事件结束后生效
      依赖该信号的模块在下一个 delta 被唤醒
    """
```

### 5.2 SimObject

```python
class SimObject:
    """所有硬件模块的基类

    属性:
      kernel: SimKernel      # 全局内核引用
      name: str              # 模块名称 (调试/profiling用)
      clock_name: str        # 绑定的时钟域

    便捷方法:
      schedule(delay_ns, callback)   # 代理到 kernel.schedule()
      schedule_cycles(cycles, cb)    # 按本模块时钟域换算 ns
      now() -> float                 # 当前时间
      cycle_now() -> int             # 当前 cycle 数

    子类实现具体硬件行为，不强制接口
    模块间通过事件和方法调用交互
    """
```

### 5.3 与现有 SimEngine 的重构关系

现有 `sim_engine.py` 的 `G5SimEngine._simulate_core()` 方法：
- heapq + SimEvent 机制 -> 提取为 `SimKernel`
- `try_issue_tiu/dma/sdma/hau` -> 各 SimObject 内部方法
- `tiu_sync_id` 等同步变量 -> SimObject 状态
- `EventType` 枚举 -> 由各 SimObject 自行管理事件类型

重构后的调用链:
```
pipeline.py
  -> SimKernel.create()
  -> SingleChip/MultiChip 组装 SimObject
  -> kernel.run()
  -> adapter.convert(kernel.records) -> EngineResult
```

## 6. 芯片内模块详细设计 (chip/)

### 6.1 CoreSubsys

```
CoreSubsys(SimObject):
  对标: TPUPerf TpuSubsys

  属性:
    core_id: int
    tiu: TIUEngine
    gdma: GDMAEngine
    sdma: SDMAEngine
    hau: HAUEngine
    lmem: LMEMModel
    bus_master_gdma_id: int   # Bus 上的 master 端口 ID
    bus_master_sdma_id: int   # Bus 上的 master 端口 ID

  同步信号:
    tiu_sync_id: int    # TIU 完成时写入 cmd_id
    tdma_sync_id: int   # GDMA 完成时写入 cmd_id
    sdma_sync_id: int   # SDMA 完成时写入 cmd_id
    hau_sync_id: int    # HAU 完成时写入 cmd_id

  指令加载:
    load_instructions(core_instr: CoreInstructions)
```

### 6.2 TIUEngine

```
TIUEngine(SimObject):
  对标: TPUPerf tiu (SC_METHOD, 每周期)

  状态机: IDLE -> WAIT_DEP -> COMPUTING -> FINISH -> IDLE

  流程:
    1. IDLE: 检查指令队列是否有待执行指令
    2. WAIT_DEP: 检查 cmd.cmd_id_dep <= parent.tdma_sync_id
    3. COMPUTING: 计算延迟 = calc_tiu_latency(cmd, chip)
       schedule_cycles(latency_cycles, on_finish)
    4. FINISH: 更新 parent.tiu_sync_id = cmd.cmd_id
       记录 SimRecord, 返回 IDLE

  延迟计算: 复用现有 calc_tiu_latency() 函数
```

### 6.3 GDMAEngine

```
GDMAEngine(SimObject):
  对标: TPUPerf tdma/GDMA (SC_THREAD, 5级流水)

  简化版 (当前):
    延迟 = startup + data_bytes / bandwidth

  精确版 (后续):
    5 级流水:
      1. 分段 (tensor -> 1D segments)
      2. Fabric 转换 (segment -> AXI burst)
      3. Bus 请求 (地址解码, 仲裁等待)
      4. 内存访问 (LMEM 或 DDR, 含 bank conflict)
      5. 响应处理 (ROB 有序提交, 更新 tdma_sync_id)
    Outstanding 控制: 最大并发事务数
```

### 6.4 SDMAEngine

```
SDMAEngine(SimObject):
  对标: TPUPerf tdma/SDMA (SC_THREAD)

  核间搬运 (Die 内):
    1. 检查依赖 (dep_engine 指定)
    2. 通过 Bus 发送请求到目标核的 DDR/LMEM
    3. Bus 延迟 = 距离延迟 + 仲裁等待 + 数据传输
    4. 完成后更新 sdma_sync_id
```

### 6.5 BusModel

```
BusModel(SimObject):
  对标: TPUPerf simple_bus (NxM)

  属性:
    num_masters: int   # = 2 * core_num (每核 GDMA口 + SDMA口)
    num_slaves: int    # = core_num (每核 DDR/ARE)
    latency_matrix: float[N][M]   # 距离延迟
    slave_busy: bool[M]           # slave 占用状态
    wait_queues: deque[M]         # slave 等待队列

  延迟矩阵初始化 (8x8 2D mesh):
    for i in range(num_masters):
      for j in range(num_slaves):
        core_i = master_to_core(i)
        core_j = slave_to_core(j)
        latency_matrix[i][j] = 45 * manhattan_distance(core_i, core_j)

  核心行为:
    1. 收到 master 请求 (master_id, target_addr, size, type)
    2. 地址解码 -> slave_id (通过 SAM 映射)
    3. 仲裁: slave 空闲则立即服务, 否则排队 (FCFS)
    4. 传输延迟 = latency_matrix[master_id][slave_id] + size/bandwidth
    5. 完成后释放 slave, 唤醒等待队列下一个请求

  核坐标 (SG2262 8x8):
    baseX = [0, 4, 0, 4, 0, 4, 0, 4]
    baseY = [0, 0, 2, 2, 4, 4, 6, 6]
    core(i) = (baseX[i//8] + i%4, baseY[i//8] + (i%8)//4)
```

### 6.6 LMEMModel / DDRModel

```
LMEMModel(SimObject):
  容量: 由 chip config 指定
  访问延迟: 29 ns (典型值)
  Bank conflict: 同一 bank 同时访问需排队
  简化: 初期不建模 bank conflict, 用固定延迟

DDRModel(SimObject):
  容量: 由 chip config 指定
  访问延迟: 150 ns (典型值)
  Bank conflict: 可选建模
  页模式: page hit (~50ns) / page miss (~150ns)
```

## 7. 芯片间互联详细设计 (interconnect/)

### 7.1 CDMAEngine

```
CDMAEngine(SimObject):
  对标: TPUPerf cdma.cc (~1,191行, 16 SC_THREAD)

  规格 (SG2262):
    CDMA数/Die: 4
    Thread数/CDMA: 8
    总Thread数/Die: 32
    带宽上限/CDMA: 64 GB/s
    总带宽/Chip: 512 GB/s (> C2C 448 GB/s)

  命令类型:
    SEND:
      1. 等待 credit (从 credit_fifo 读取, 可能阻塞)
      2. 解析 credit 获得对端接收缓冲地址
      3. 读取本地数据 (通过 Bus -> LMEM/DDR)
      4. 通过 PAXI/C2C 写入远端地址
      5. 完成

    RECV:
      1. 分配本地接收缓冲区
      2. 创建 credit 包 (含目标地址信息)
      3. 通过 PAXI/C2C 发送 credit 到对端
      4. 阻塞等待数据到达
      5. 完成

    ALL_REDUCE (伪延迟模式):
      delay = 4*alpha + 3*beta + 2*data_size / bw
      alpha = 0.25 us (启动延迟)
      beta = 0.3 us (同步延迟)

  信用流控:
    credit_fifo: FIFO[Credit]
    credit_capacity: 2 (可配置)

  Thread Arbiter:
    选择非 barrier 线程 -> 共享 CDMA datapath
    barrier 指令阻塞本线程, 不阻塞其他线程

  fence 保序:
    fence 指令确保: fence 后的搬运在 fence 前全部完成后才执行
```

### 7.2 PAXICore (事务层)

```
PAXICore(SimObject):
  对标: PAXI SUE2.0 Core (docs/design/PAXI/)

  TX 方向 (AXI -> Flit):
    1. AXI Slave 接收事务 (来自 CDMA 通过 NoC)
    2. 地址解码: 确定目标 DA (Destination Address)
    3. 事务编码为 Flit 序列:
       - head flit: 含路由信息 (MAC DA, VC, QP)
       - body flit: 携带数据负载
       - tail flit: 标记事务结束
    4. OST 管理: 记录 Outstanding Transaction
       - 限制: 最大 512 OST
       - OST 满时: 背压, 阻塞新事务
    5. 多播处理 (可选):
       - 检查是否为多播目标
       - 复制 flit 到多个输出 VC
    6. 输出 Flit 到 RC Link TX

  RX 方向 (Flit -> AXI):
    1. 从 RC Link RX 接收 Flit 序列
    2. Flit 解码为 AXI 事务
    3. AXI Master 发出事务 (写入本地 DDR)
    4. OST 释放 (收到响应后)

  关键延迟参数:
    axi_to_flit_latency: ~10 cycles (编码处理)
    flit_to_axi_latency: ~10 cycles (解码处理)
    ost_limit: 512

  支持的报文格式:
    Standard: 完整 IP/UDP 头, 支持 ECN, 兼容 L2 Switch
    AFH_GEN1: 简化头
    AFH_GEN2_16b: 16字节头
    AFH_Lite: 最小头, 低延迟, 无 E2E 重传
```

### 7.3 RCLink (传输层)

```
RCLink(SimObject):
  对标: RC Link (docs/design/PAXI/PAXI_MODELING_ANALYSIS.md)

  TYPE1 可靠传输:
    Go-Back-N 重传:
      send_window: 512 OST
      timeout: 可配置 (us)
      seq_num: 顺序号
      ack_tracker: 已确认的最大 seq_num

      发送流程:
        1. 从 PAXI 接收 Flit
        2. 分配 seq_num
        3. 缓存到重传缓冲 (直到收到 ACK)
        4. 发送到 CESOC

      ACK 处理:
        1. 收到 ACK(seq_n) -> 释放 seq <= n 的缓冲
        2. 收到 NACK(seq_n) -> 从 seq_n 开始重传
        3. 超时未收到 ACK -> 从最早未确认开始重传

  TYPE2 不可靠传输:
    直接发送, 无重传, 用于多播

  CBFC (Credit-Based Flow Control):
    mode: "cbfc" (与 PFC 互斥)
    vc_count: 8 (Virtual Channel)

    per-VC 状态:
      credits[vc]: int          # 可用 credit 数
      credit_max[vc]: int       # credit 上限

    发送时:
      if credits[vc] > 0:
        credits[vc] -= 1
        发送 flit
      else:
        阻塞, 等待 credit_return

    收到 credit_return:
      credits[vc] += return_count
      唤醒阻塞的发送

  PFC (Priority Flow Control):
    mode: "pfc" (与 CBFC 互斥)

    per-端口状态:
      pfc_state[priority]: XOFF | XON

    收到 PFC XOFF(priority):
      pfc_state[priority] = XOFF
      停止发送该优先级的 flit

    收到 PFC XON(priority):
      pfc_state[priority] = XON
      恢复发送

    本地 PFC 生成:
      if rx_buffer_level > xoff_threshold:
        发送 PFC XOFF 到对端
      if rx_buffer_level < xon_threshold:
        发送 PFC XON 到对端

  速率控制:
    per-QP token bucket:
      rate: 配置的速率限制
      tokens: 当前可用 token
      每发送 1 flit 消耗 1 token
      按配置速率补充 token
```

### 7.4 CESOCModel (MAC/PCS/FEC)

```
CESOCModel(SimObject):
  对标: CESOC MAC/PCS/FEC 层

  TX:
    1. 从 RC Link 接收 flit/帧
    2. MAC 组帧 (添加前导码, SFD, CRC)
    3. PCS 编码 (64b/66b 等)
    4. FEC 编码 (RS-FEC)
    5. 输出到 SerDes

  RX:
    1. 从 SerDes 接收数据
    2. FEC 解码 + 纠错
    3. PCS 解码
    4. MAC 解帧 (CRC 校验)
    5. 输出到 RC Link

  延迟参数:
    mac_latency_ns: ~5 ns
    fec_encode_latency_ns: ~50 ns (RS-FEC)
    fec_decode_latency_ns: ~100 ns (RS-FEC, 含纠错)
    pcs_latency_ns: ~2 ns

  简化选项:
    可将整个 CESOC 建模为固定延迟:
    total_latency_ns = mac + fec_encode + pcs (TX)
                     + pcs + fec_decode + mac (RX)
                     ~ 160 ns 往返
```

### 7.5 C2CLink (物理链路)

```
C2CLink(SimObject):
  物理链路模型

  参数:
    bandwidth_gbps: 448 (SG2262: 8 x4 @ 112G)
    propagation_delay_ns: 取决于线缆长度
      PCB trace (~10cm): ~0.5 ns
      DAC cable (~1m): ~5 ns
      AOC cable (~10m): ~50 ns

  延迟:
    transfer_delay_ns = data_bytes / (bandwidth_gbps / 8)
    total_delay_ns = propagation_delay_ns + transfer_delay_ns
```

### 7.6 L2Switch (交换机)

```
L2Switch(SimObject):
  对标: docs/design/switch-modeling/SWITCH_BEHAVIOR_ANALYSIS.md

  属性:
    num_ports: int              # 端口数 (与连接的芯片数对应)
    pipeline_stages: 12         # 流水线级数
    clock_period_ns: float      # 交换机时钟周期

  12-Stage Pipeline:
    Stage 1:  帧接收 (从入端口读取)
    Stage 2:  帧解析 (提取 MAC DA/SA, VLAN, QoS)
    Stage 3:  MAC 查表 (确定出端口)
    Stage 4:  QoS 分类 (映射到优先级队列)
    Stage 5:  VOQ 入队 (ingress_port x egress_port)
    Stage 6:  iSLIP Request (各 VOQ 向目标出端口发请求)
    Stage 7:  iSLIP Grant (出端口仲裁, 授权一个入端口)
    Stage 8:  iSLIP Accept (入端口确认授权)
    Stage 9:  Crossbar 传输 (通过交叉开关矩阵传输数据)
    Stage 10: Egress 出队 (出端口缓冲)
    Stage 11: 帧整形 + PFC/ECN 处理
    Stage 12: 帧发送 (到出端口)

  VOQ (Virtual Output Queue):
    voq[ingress_port][egress_port]: FIFO
    消除 Head-of-Line (HOL) blocking
    每个 VOQ 有独立缓冲

  iSLIP 调度器:
    N 轮迭代 (N = num_ports):
      1. Request: 每个非空 VOQ 向对应出端口发请求
      2. Grant: 每个出端口用轮询 (Round-Robin) 选择一个请求
      3. Accept: 每个入端口用轮询选择一个 grant
    保证公平性和最大匹配

  共享缓冲 + 动态阈值 (DT):
    total_buffer: 共享缓冲总量 (bytes)
    alpha: DT 系数 (典型值 1/8 ~ 8)

    per_queue_threshold(queue_i):
      remaining = total_buffer - sum(all_queue_usage)
      threshold = alpha * remaining
      return threshold

    入队决策:
      if queue_usage[i] < per_queue_threshold(i):
        允许入队
      else:
        丢弃 (或触发 PFC)

  PFC 状态机 (每入端口):
    监控入端口接收缓冲水位:
      if buffer_level > xoff_threshold:
        发送 PFC XOFF 帧到上游
        xoff_sent = True
      if buffer_level < xon_threshold and xoff_sent:
        发送 PFC XON 帧到上游
        xoff_sent = False

  ECN 标记:
    基于出端口队列深度:
      if queue_depth > ecn_threshold:
        标记 ECN CE bit (Congestion Experienced)

  延迟模型:
    零负载延迟: 12 * clock_period_ns + SerDes 延迟
    有负载延迟: 零负载 + 排队延迟 (取决于拥塞程度)
    Incast 场景: 多入端口同时发往一个出端口
      排队延迟 = f(入端口数, 数据量, 出端口带宽)
      典型值: 5-50 us
```

## 8. 顶层组装设计 (top/)

### 8.1 SingleChip (单芯片, 多核)

```
SingleChip:
  创建流程:
    1. 创建 SimKernel
    2. 创建 N 个 CoreSubsys (N = chip.core_count)
    3. 创建 BusModel (2N masters, N slaves)
    4. 创建 N 个 DDRModel
    5. 配置 Bus 延迟矩阵 (Manhattan 距离)
    6. 绑定端口:
       CoreSubsys[i].gdma -> Bus.master[2*i]
       CoreSubsys[i].sdma -> Bus.master[2*i+1]
       Bus.slave[i] -> DDRModel[i]
    7. 加载指令到各核
    8. kernel.run()
    9. 收集 SimRecord

  对标: TPUPerf tpuManyCore.cc
```

### 8.2 MultiChip (多芯片, 含 C2C/Switch)

```
MultiChip:
  创建流程:
    1. 创建 SimKernel
    2. 创建 M 个 SingleChip (共享同一 kernel)
    3. 创建 M 个 CDMAEngine (每芯片一个)
    4. 创建 C2C 链路组件:
       - M 个 PAXICore (每芯片一个)
       - M 个 RCLink (每芯片一个)
       - M 个 CESOCModel (每芯片一个)
    5. 创建物理链路 C2CLink (芯片间连接)
    6. 创建 L2Switch (可选, 根据拓扑配置)
    7. 按拓扑配置连接:
       直连模式: CESOC[A] <-> C2CLink <-> CESOC[B]
       Switch模式: CESOC[A] <-> C2CLink <-> L2Switch <-> C2CLink <-> CESOC[B]
    8. 加载指令 + CDMA 命令
    9. kernel.run()
    10. 收集所有芯片的 SimRecord

  拓扑配置: 从 L2 ChipSpec + TopologyConfig 读取
  连接关系: 根据 interconnect 配置生成
```

## 9. 数据流与事件交互

### 9.1 Die 内核间通信 (SDMA)

```
Core A (SDMA)                Bus              DDR[B]           Core B
    |                         |                  |                |
    |-- send_request -------->|                  |                |
    |                         |-- arbitrate      |                |
    |                         |-- distance_delay |                |
    |                         |-- write_request->|                |
    |                         |                  |-- access       |
    |                         |<--- response ----|                |
    |<--- complete -----------|                  |                |
    |                                                             |
    |  更新 sdma_sync_id                                           |
```

### 9.2 芯片间通信 (CDMA + C2C)

```
CDMA[A]    PAXI[A]   RCLink[A]  CESOC[A]  Link  CESOC[B]  RCLink[B]  PAXI[B]   CDMA[B]
  |          |          |          |        |       |          |          |          |
  |          |          |          |        |       |          |          |          |
  |-- SEND --|          |          |        |       |          |          |-- RECV --|
  |   等credit|          |          |        |       |          |          |  发credit|
  |          |          |          |        |       |          |          |          |
  |<-credit--|<---------|<---------|<-------|-------|----------|----------|----------|
  |          |          |          |        |       |          |          |          |
  |--写数据->|--编码Flit->|--加seq/FC->|--组帧FEC->|--传输-->|--解帧FEC->|--解FC/seq->|--解码Flit->|--写DDR
  |          |          |          |        |       |          |          |          |
  |          |          |          |        |       |          |<--ACK----|          |
  |          |          |<---------|<-------|-------|----------|          |          |
  |          |          |  释放缓冲  |        |       |          |          |          |
  |--完成    |          |          |        |       |          |          |--完成    |
```

### 9.3 经 Switch 的通信

```
CESOC[A]  Link  L2Switch             Link  CESOC[B]
  |        |       |                   |       |
  |--帧---->|------>| Stage1: 接收      |       |
  |        |       | Stage2: 解析      |       |
  |        |       | Stage3: MAC查表   |       |
  |        |       | Stage4: QoS分类   |       |
  |        |       | Stage5: VOQ入队   |       |
  |        |       | Stage6: iSLIP Req |       |
  |        |       | Stage7: iSLIP Gnt |       |
  |        |       | Stage8: iSLIP Acc |       |
  |        |       | Stage9: Crossbar  |       |
  |        |       | Stage10: Egress   |       |
  |        |       | Stage11: 整形/PFC |       |
  |        |       | Stage12: 发送     |       |
  |        |       |---------->|------>|       |
  |        |       |           |       |--帧-->|
```

## 10. L2/L3 层的扩展

### 10.1 L2 ChipSpec 扩展

ChipSpec 需要新增以下配置:

```yaml
# SG2262.yaml 扩展
# === 现有 ===
name: SG2262
frequency_ghz: 1.0
cores:
  count: 64
  lanes_per_core: 16
# ...

# === 新增: 芯片内互联 ===
noc:
  topology: "2d_mesh"
  dimensions: [8, 8]
  base_latency_cycles: 45  # Manhattan 距离系数
  bus_bandwidth_gbps: 192

# === 新增: Die 配置 ===
die:
  count: 2
  cores_per_die: 32
  d2d_bandwidth_gbps: 256
  d2d_latency_ns: 5

# === 新增: CDMA 配置 ===
cdma:
  engines_per_die: 4
  threads_per_engine: 8
  bandwidth_per_engine_gbps: 64
  alpha_us: 0.25
  beta_us: 0.3

# === 新增: C2C 链路 ===
c2c:
  ports: 8
  lanes_per_port: 4
  serdes_rate_gbps: 112
  total_bandwidth_gbps: 448

# === 新增: PAXI 配置 ===
paxi:
  ost_limit: 512
  vc_count: 8
  axi_to_flit_cycles: 10
  flit_to_axi_cycles: 10
  flow_control: "cbfc"   # cbfc | pfc
  packet_format: "standard"  # standard | afh_lite

# === 新增: RC Link 配置 ===
rc_link:
  type1_enabled: true     # Go-Back-N 可靠传输
  send_window: 512
  timeout_us: 10.0

# === 新增: CESOC 配置 ===
cesoc:
  mac_latency_ns: 5
  fec_encode_latency_ns: 50
  fec_decode_latency_ns: 100
```

### 10.2 L3 InstructionEmitter 扩展

需要为多核 + CDMA 生成指令:

```
现有: DistributedOp -> CoreInstructions (单核 TIU/DMA/SDMA/HAU)

扩展:
  DistributedOp (计算类) -> 分配到各核的 TIU/DMA 指令
  DistributedOp (通信类) -> CDMA 命令 (SEND/RECV/ALL_REDUCE)

  CoreProgram 扩展:
    cores: list[CoreInstructions]    # 多核指令
    cdma_cmds: list[CDMACommand]     # CDMA 命令 (新增)
    comm_schedule: list[CommOp]      # 通信调度 (细化)
```

## 11. 分阶段实现计划

### Phase 1: 仿真内核 + 多核扩展

目标: 把现有单核 G5 扩展为多核, 引入 SimKernel

```
新增文件:
  kernel/sim_kernel.py
  kernel/sim_object.py
  chip/core_subsys.py
  chip/bus.py
  top/single_chip.py

重构文件:
  sim_engine.py (使用 SimKernel)
  tiu.py, dma.py, sdma.py, hau.py (适配 SimObject)

验证:
  多核 MatMul 仿真, 对比单核结果
  Bus 距离延迟验证
```

### Phase 2: LMEM/DDR 存储层级

目标: 加入 bank 级内存建模

```
新增文件:
  chip/lmem.py
  chip/ddr.py
  chip/cache.py (可选)

验证:
  Bank conflict 对性能影响
  对比 TPUPerf 结果
```

### Phase 3: CDMA + C2C 基础链路

目标: 支持两芯片直连通信

```
新增文件:
  interconnect/cdma.py
  interconnect/c2c_link.py
  top/multi_chip.py

简化: C2C 先用 alpha-beta 延迟模型 (与 TPUPerf 对齐)

验证:
  双芯片 SEND/RECV
  AllReduce 延迟
```

### Phase 4: PAXI + RC Link cycle 级建模

目标: 把 C2C 从简化模型升级为 cycle 级行为模型

```
新增文件:
  interconnect/paxi_core.py
  interconnect/rc_link.py
  interconnect/cesoc.py

验证:
  CBFC 流控行为
  Go-Back-N 重传行为
  各报文格式延迟对比
```

### Phase 5: L2 Switch

目标: 加入交换机 cycle 级建模

```
新增文件:
  interconnect/l2_switch.py

验证:
  零负载延迟 (~300ns)
  Incast 场景排队延迟
  PFC 背压行为
  iSLIP 吞吐率
```

### Phase 6: 前端集成 + 端到端验证

目标: 前端支持多芯片 G5 仿真

```
修改:
  L2 ChipSpec YAML 格式扩展
  L3 InstructionEmitter 多核+CDMA 支持
  前端拓扑配置 UI 支持 PAXI/Switch 参数

验证:
  端到端: 前端配置 -> 多芯片 G5 仿真 -> 结果展示
```

## 12. 与 Math Model 的关系

```
                 L0-L3 (共享)
                    |
          +---------+---------+
          |                   |
    mode="math"          mode="g5"
          |                   |
  L3.math: Tiling      L3.g5: InstructionEmitter
  L4.math: EvalEngine  L4.g5: SimKernel (全局事件驱动)
          |                   |
          +---------+---------+
                    |
               L5 (共享)
            EngineResult
```

Math Model 不需要修改。两种模式在 L3 分叉, L5 汇合。

## 13. 风险与注意事项

### 性能风险
- Python 事件循环比 SystemC 慢 50-100x
- 多核+C2C 事件数量巨大
- 缓解: Cython 关键路径, 多进程, C 扩展

### 精度风险
- cycle 级建模需要精确的硬件参数
- PAXI/RC Link 内部行为文档可能不完整
- 缓解: 与 TPUPerf 结果交叉验证

### 复杂度风险
- 全局事件队列 + 多模块交互调试困难
- 缓解: 分阶段实现, 每阶段有独立验证
