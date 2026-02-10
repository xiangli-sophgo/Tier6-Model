# 指令级动态仿真引擎设计文档

## 概述

本设计文档描述在 Tier6+Model 中新增**指令级动态仿真引擎**的架构设计。该引擎对标 TPUPerf (SystemC C++ 实现), 用 Python 实现等价的周期精确仿真能力。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-overview.md](./01-overview.md) | 整体架构与设计目标 |
| [02-event-engine.md](./02-event-engine.md) | 事件驱动仿真内核 |
| [03-tiu-engine.md](./03-tiu-engine.md) | TIU 计算引擎 |
| [04-dma-engine.md](./04-dma-engine.md) | DMA 数据搬运引擎 |
| [05-memory-subsystem.md](./05-memory-subsystem.md) | 内存子系统 (LMEM/DDR/Cache) |
| [06-multicore-interconnect.md](./06-multicore-interconnect.md) | 多核互连 (Bus/C2C/CDMA) |
| [07-command-parser.md](./07-command-parser.md) | 指令解析与生成 |
| [08-integration.md](./08-integration.md) | 与现有系统集成 + TPUPerf 直接接入方案 |
| [09-implementation-plan.md](./09-implementation-plan.md) | 实现计划与工作量评估 |

## TPUPerf 模块对照表

| TPUPerf 模块 | 代码量 | 本项目对应模块 | 优先级 |
|-------------|--------|--------------|--------|
| `tiu.cc` + `tiuImpl.cc` | ~6,000行 | TIU Engine | P0 |
| `tdma.cc` + `tdmaDelayImpl.cc` | ~7,300行 | DMA Engine | P0 |
| `lmem.cpp` | ~400行 | LMEM Model | P0 |
| `ddr_wrapper.cpp` | ~550行 | DDR Model | P0 |
| `tpu.cc` + `tpu_subsys.cc` | ~200行 | Core Assembler | P0 |
| `tpuManyCore.cc` | ~830行 | Multi-Core Top | P1 |
| `simple_bus.h` | ~960行 | Bus Model | P1 |
| `gs_cache.cpp` | ~550行 | Cache Model | P1 |
| `c2c.cc` | ~310行 | C2C Link | P1 |
| `cdma.cc` | ~1,200行 | CDMA Engine | P1 |
| `cmodel_common.cpp` | ~1,600行 | Command Parser | P0 |
| `profiler.cc` + `utility.cc` | ~2,700行 | Framework | P0 |

## 与现有系统的关系

```
用户配置 (模型 + 拓扑 + 芯片)
         |
         +-- [快速模式] 数学建模 (现有 simulator.py) --> 毫秒级结果
         |
         +-- [精确模式] 指令级仿真 (新引擎) --> 周期级结果
                |
                +-- 指令来源: 二进制文件 (.BD/.GDMA) 或 自动生成
                +-- 单核仿真: TIU + DMA + LMEM + DDR
                +-- 多核: Bus + C2C + CDMA
                +-- 输出: 复用现有 Gantt/图表/结果系统
```
