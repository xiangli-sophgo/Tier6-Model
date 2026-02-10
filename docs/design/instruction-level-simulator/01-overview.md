# 01 - 整体架构与设计目标

## 1. 设计目标

### 核心目标
在 Tier6+Model 中实现一套 Python 指令级动态仿真引擎, 对标 TPUPerf 的 SystemC 实现, 支持:
- 读取 TPUPerf 编译器产出的二进制指令文件 (.BD/.GDMA)
- 从 LLM 模型结构自动生成虚拟指令序列
- 周期精确的 TIU 计算延迟建模
- 事务级的 DMA 数据搬运建模
- 完整的内存层级建模 (LMEM + DDR + Cache + Bank Conflict)
- 多核互连建模 (Bus + C2C + CDMA)

### 芯片范围
仅支持 SG 系列芯片 (SG2260/SG2262), 与 TPUPerf 对齐。

### 精度目标
- TIU: 指令级周期精确 (与 TPUPerf 的 tiuImpl.cc 计算公式一致)
- DMA: 事务级近似周期精确 (分段 + outstanding + 带宽受限)
- Memory: DDR bank conflict + Cache hit/miss
- Bus: 距离相关延迟 + 仲裁

## 2. 整体架构

```
+------------------------------------------------------------------+
|                        Python 仿真引擎                             |
|                                                                    |
|  +--------------------+    +----------------------------------+   |
|  | Command Source      |    | Event-Driven Simulation Kernel  |   |
|  |                    |    |                                  |   |
|  | - Binary Parser    |    |  EventScheduler                 |   |
|  |   (.BD/.GDMA)      |    |  - 时间推进                      |   |
|  |                    |    |  - 事件队列 (heapq)              |   |
|  | - Instruction Gen   |    |  - Process 调度                 |   |
|  |   (LLM -> TIU/DMA) |    |  - 信号/等待机制                 |   |
|  +--------+-----------+    +--------+-------------------------+   |
|           |                         |                             |
|           v                         v                             |
|  +------------------------------------------------------------------+
|  |                     Single Core Model                             |
|  |                                                                    |
|  |  +----------+  cmd_id sync  +----------+                          |
|  |  |   TIU    |<------------>|   TDMA   |                          |
|  |  | Engine   |               | (GDMA)   |                          |
|  |  +----+-----+               +----+-----+                          |
|  |       |                          |                                 |
|  |       | compute                  | read/write                     |
|  |       v                          v                                 |
|  |  +----------+    +----------+  +----------+                       |
|  |  |  LMEM    |    |  Cache   |  |   DDR    |                       |
|  |  | (banks)  |    | (4-way)  |  | (banks)  |                       |
|  |  +----------+    +----+-----+  +----------+                       |
|  |                       |              ^                             |
|  |                       +--------------+                             |
|  +------------------------------------------------------------------+
|           |                                                          |
|           | (多核时)                                                  |
|           v                                                          |
|  +------------------------------------------------------------------+
|  |                    Multi-Core Interconnect                        |
|  |                                                                    |
|  |  +----------+    +----------+    +----------+                     |
|  |  |   Bus    |    |   C2C    |    |   CDMA   |                     |
|  |  | (NxM)    |    |  (link)  |    | (cross)  |                     |
|  |  +----------+    +----------+    +----------+                     |
|  +------------------------------------------------------------------+
|                                                                      |
|  +------------------------------------------------------------------+
|  |                    Profiler & Output                               |
|  |  - Cycle 统计                                                     |
|  |  - Gantt 数据生成 (复用现有 gantt.py)                              |
|  |  - 性能指标计算 (TPS/TPOT/TTFT/MFU)                               |
|  +------------------------------------------------------------------+
+----------------------------------------------------------------------+
```

## 3. TPUPerf 架构映射

### TPUPerf 的分层结构

```
TPUPerf (SystemC):
  sc_main()
    +-- TpuSubsys[0..N]           --> SingleCoreModel
    |     +-- Tiu                  --> TIUEngine
    |     +-- Tdma (GDMA)         --> DMAEngine
    |     +-- Tdma (SDMA)         --> DMAEngine (第二个实例)
    |     +-- Hau                  --> (可选, 低优先级)
    |     +-- lmem                 --> LMEMModel
    |
    +-- simple_bus                 --> BusModel
    +-- ARE + DDR[0..N]            --> DDRModel
    +-- gs_cache                   --> CacheModel
    +-- C2C                        --> C2CLink
    +-- CDMA                       --> CDMAEngine
    +-- FakeChip                   --> RemoteChipStub
```

### Python 实现的对应结构

```
instruction_simulator/
  +-- core/
  |     +-- event_scheduler.py     # 事件驱动内核 (替代 SystemC 调度器)
  |     +-- signal.py              # 信号/端口抽象 (替代 sc_signal)
  |     +-- process.py             # 进程抽象 (替代 SC_THREAD/SC_METHOD)
  |
  +-- engines/
  |     +-- tiu_engine.py          # TIU 计算引擎
  |     +-- tiu_delay.py           # TIU 指令延迟计算 (从 tiuImpl.cc 翻译)
  |     +-- dma_engine.py          # DMA 搬运引擎 (GDMA/SDMA)
  |     +-- dma_delay.py           # DMA 延迟计算 (从 tdmaDelayImpl.cc 翻译)
  |     +-- hau_engine.py          # HAU 引擎 (低优先级)
  |
  +-- memory/
  |     +-- lmem.py                # Local Memory 模型
  |     +-- ddr.py                 # DDR 模型 (bank conflict)
  |     +-- cache.py               # Cache 模型 (4-way set-associative)
  |     +-- address_map.py         # 系统地址映射
  |
  +-- interconnect/
  |     +-- bus.py                 # NxM 总线模型
  |     +-- c2c.py                 # Chip-to-Chip 链路
  |     +-- cdma.py                # Cross-chip DMA
  |
  +-- command/
  |     +-- binary_parser.py       # .BD/.GDMA 二进制解析
  |     +-- instruction_gen.py     # 从 LLM 模型自动生成指令
  |     +-- types.py               # 指令类型定义
  |
  +-- top/
  |     +-- single_core.py         # 单核组装
  |     +-- multi_core.py          # 多核组装
  |     +-- config_loader.py       # 芯片/拓扑配置加载
  |
  +-- profiler/
        +-- profiler.py            # 性能数据收集
        +-- gantt_adapter.py       # 适配现有 Gantt 系统
        +-- reporter.py            # 结果报告生成
```

## 4. 关键设计决策

### 4.1 事件驱动引擎选择

**决策**: 自实现轻量级事件调度器, 不使用 simpy。

**理由**:
- TPUPerf 的 SystemC 调度模型比较特殊: SC_METHOD (每周期调用) + SC_THREAD (可阻塞)
- simpy 是纯协程模型, 不直接支持 SC_METHOD 的语义
- 自实现可以精确匹配 SystemC 的 delta cycle 和信号更新语义
- 代码量不大 (~300行), 且更可控

### 4.2 TIU/DMA 并行模型

**决策**: 使用协程 (generator/async) 模拟 SC_THREAD, 使用回调模拟 SC_METHOD。

TPUPerf 的核心并行模式:
```
TIU (SC_METHOD, 每周期):  Init -> Compute(countdown) -> Finish -> Init
DMA (SC_THREAD, 事件驱动): dispatch -> segment -> fabric -> memory -> response -> done
```

Python 等价实现:
```
TIU: 注册为 clock_sensitive method, 每周期执行状态机
DMA: 注册为 coroutine, 通过 yield wait_event() 挂起/恢复
```

### 4.3 TLM 事务建模

**决策**: 简化 TLM-2.0 为请求-响应模型, 不实现完整的 4 phase 协议。

TPUPerf 使用 TLM-2.0 的 nb_transport (BEGIN_REQ/END_REQ/BEGIN_RESP/END_RESP)。Python 中简化为:
- `send_request(addr, size, type)` -> 返回 Future
- `await response` -> 获取延迟和数据

这保留了 outstanding 控制和延迟建模的能力, 去掉了 TLM 的协议细节。

### 4.4 数据精度

**决策**: 不模拟实际数据内容, 只模拟时序。

TPUPerf 的 LMEM 使用 sparse_array 存储实际数据, 用于功能验证。指令级仿真引擎只关心性能, 不需要数据正确性验证, 因此:
- LMEM/DDR 只追踪地址和大小, 不存储实际数据
- 可以节省大量内存, 提升仿真速度
- 如果未来需要功能验证, 可以加回数据层

## 5. 性能预期

| 场景 | TPUPerf (SystemC) | Python 仿真引擎 | 加速方案 |
|------|-------------------|-----------------|---------|
| 单核 ResNet50 | ~2s | ~30-60s | Cython TIU delay |
| 8核 DeepSeek-V3 单层 | ~10s | ~150-300s | 多进程并行 |
| 64核 全模型 | ~5min | ~1-2h | C 扩展关键路径 |

**初期不做性能优化**, 先保证功能正确。后续可通过以下手段加速:
1. Cython 编译 TIU 延迟计算 (最热路径)
2. 多进程并行多核仿真
3. C 扩展事件调度器
4. 去掉不必要的 Python 对象创建
