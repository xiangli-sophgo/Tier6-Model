# 09 - 实现计划与工作量评估

## 1. 整体计划

### 阶段划分

```
Phase 0: TPUPerf 快速接入 (方案 B)     [3-5 周]
  -> 最快获得指令级仿真能力

Phase 1: Python 仿真内核               [3-4 周]
  -> 事件引擎 + 单核框架

Phase 2: TIU/DMA 引擎                  [4-6 周]
  -> 核心计算和搬运建模

Phase 3: 内存子系统                     [2-3 周]
  -> LMEM + DDR + Cache

Phase 4: 多核互连                       [2-3 周]
  -> Bus + C2C + CDMA

Phase 5: 指令生成器                     [3-4 周]
  -> LLM 模型 -> 指令序列

Phase 6: 系统集成                       [2-3 周]
  -> API + 前端 + 结果适配

总计: 约 17-28 周 (4-7 个月, 1人)
```

## 2. Phase 0: TPUPerf 快速接入

**目标**: 最快速度获得指令级仿真能力, 验证集成可行性。

| 任务 | 工作量 | 说明 |
|------|--------|------|
| pybind11 接口设计 | 2天 | 定义 Python-C++ 交互接口 |
| sc_main 重构 | 3天 | 提取核心逻辑为可调用函数 |
| 全局状态管理 | 2天 | reset 机制, 解决多次调用问题 |
| profiler 输出适配 | 2天 | 将 CSV 输出转为 Python dict |
| CMake + pybind11 构建 | 2天 | 编译 .so 模块 |
| Python wrapper | 2天 | 封装为 Tier6+Model 接口 |
| Gantt 数据适配 | 2天 | 适配到现有图表系统 |
| 测试验证 | 3天 | 对比 TPUPerf 直接运行的结果 |

**风险**:
- SystemC 的 sc_main 只能调一次 -> 可能需要子进程方案
- 全局状态清理不彻底 -> 多次仿真结果不一致
- 跨平台编译 (macOS/Linux/Windows) 复杂度

**交付物**:
- `tpu_perf.so` Python 模块
- API: `POST /api/instruction-simulate` (binary 模式)
- Gantt 可视化显示指令级事件

## 3. Phase 1: Python 仿真内核

**目标**: 实现事件驱动调度器和基础框架。

| 任务 | 工作量 | 说明 |
|------|--------|------|
| EventScheduler | 3天 | 时间推进, 事件队列, delta cycle |
| Signal/Port | 2天 | 信号模型, 延迟更新 |
| Method/Thread | 3天 | SC_METHOD/SC_THREAD 等价物 |
| FIFO/Semaphore | 2天 | 同步原语 |
| Clock | 1天 | 多时钟域支持 |
| 单核骨架 | 3天 | TIU + DMA 空壳 + 信号连接 |
| 单元测试 | 3天 | 调度器正确性验证 |

**验证方式**:
- 简单的 TIU+DMA 并行场景, 验证 cmd_id 同步
- 多时钟域场景, 验证时钟对齐

## 4. Phase 2: TIU/DMA 引擎

**目标**: 完成核心计算和数据搬运建模。

### TIU 引擎 (2-3 周)

| 任务 | 工作量 | 说明 |
|------|--------|------|
| TIU 状态机 | 2天 | Init/Compute/Finish |
| cmd_id 同步 | 1天 | TIU<->DMA 依赖 |
| 消息同步 | 1天 | send_msg/wait_msg |
| CONV 延迟 | 3天 | 从 tiuImpl.cc 翻译 |
| MM2.nn/nt/tt 延迟 | 3天 | 三种矩阵乘模式 |
| MM v1 延迟 | 1天 | 旧版矩阵乘 |
| SFU/AR/LIN 延迟 | 2天 | 辅助指令 |
| Bank conflict | 1天 | LMEM bank 冲突 |
| 对比验证 | 2天 | 与 TPUPerf 逐指令对比 cycle |

### DMA 引擎 (2-3 周)

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 命令调度 | 2天 | cmd_dispatch + 依赖检查 |
| Tensor 分段 | 3天 | NCHW 连续性分析 + 分段 |
| Matrix 分段 | 2天 | 矩阵搬运 + 转置 |
| Fabric 请求生成 | 2天 | AXI burst 拆分 |
| Outstanding 控制 | 1天 | 信用池 |
| 完成跟踪 | 2天 | ROB + tdmaSyncId |
| 对比验证 | 2天 | 与 TPUPerf 对比 DMA 延迟 |

## 5. Phase 3: 内存子系统

| 任务 | 工作量 | 说明 |
|------|--------|------|
| LMEM 模型 | 3天 | Lane 映射 + bank conflict |
| DDR 模型 | 4天 | 地址映射 + bank conflict + outstanding |
| Cache 模型 | 4天 | 4-way + LRU + miss 合并 |
| SAM 地址映射 | 2天 | SG2260/SG2262 地址空间 |
| 集成测试 | 2天 | LMEM-DDR-Cache 联合验证 |

## 6. Phase 4: 多核互连

| 任务 | 工作量 | 说明 |
|------|--------|------|
| Bus 模型 | 3天 | NxM 仲裁 + 距离延迟 |
| C2C 链路 | 3天 | 带宽建模 + ROB + 信用 |
| CDMA 引擎 | 4天 | send/recv + credit + AllReduce |
| 多核组装 | 3天 | N 核 + Bus + DDR + C2C |
| 终止检测 | 1天 | 全核 idle 检测 |
| 多核验证 | 2天 | 与 TPUPerf 8核结果对比 |

## 7. Phase 5: 指令生成器

| 任务 | 工作量 | 说明 |
|------|--------|------|
| 计算图构建 | 3天 | Transformer 层 -> 操作列表 |
| Tiling 策略 | 4天 | MatMul/Conv 的 LMEM 分块 |
| 指令生成 | 3天 | 操作 -> TIU/DMA 指令 |
| cmd_id 生成 | 2天 | 依赖关系分析 |
| MoE 支持 | 2天 | Expert routing + AllToAll |
| MLA 支持 | 2天 | KV 压缩 + LoRA projection |
| 验证 | 3天 | 生成结果 vs 真实编译器输出 |

## 8. Phase 6: 系统集成

| 任务 | 工作量 | 说明 |
|------|--------|------|
| API 端点 | 2天 | /api/instruction-simulate |
| 结果适配器 | 2天 | 转换为标准格式 |
| 前端: 模式切换 | 2天 | 数学建模 / 指令级仿真 |
| 前端: 文件上传 | 1天 | .BD/.GDMA 上传 |
| 前端: Gantt 增强 | 2天 | 指令级事件显示 |
| 前端: 新面板 | 2天 | TIU/DMA 利用率统计 |
| 评估任务集成 | 2天 | 任务队列 + 数据库存储 |
| 端到端测试 | 2天 | 完整流程验证 |

## 9. 推荐执行顺序

### 如果有 1 个人

```
月份 1: Phase 0 (TPUPerf 接入) -> 快速获得能力
月份 2: Phase 1 (仿真内核)
月份 3: Phase 2 (TIU 引擎)
月份 4: Phase 2 续 (DMA 引擎) + Phase 3 (内存)
月份 5: Phase 4 (多核) + Phase 5 前半 (指令生成)
月份 6: Phase 5 续 + Phase 6 (集成)
```

### 如果有 2 个人

```
人员 A (仿真引擎):
  月份 1: Phase 1 (仿真内核)
  月份 2: Phase 2 (TIU 引擎)
  月份 3: Phase 2 续 (DMA) + Phase 3 (内存)
  月份 4: Phase 4 (多核)

人员 B (集成 + 生成器):
  月份 1: Phase 0 (TPUPerf 接入)
  月份 2: Phase 5 (指令生成器)
  月份 3: Phase 5 续 + Phase 6 (集成)
  月份 4: 端到端验证 + 优化
```

## 10. 验证策略

### 逐级验证

```
Level 1: 单指令验证
  - 每种 TIU 指令的 cycle 计算与 TPUPerf 对比
  - 误差 <= 2 cycles

Level 2: 单核验证
  - 同一组 .BD/.GDMA, Python vs TPUPerf 总 cycle 对比
  - 误差 <= 1%

Level 3: 多核验证
  - 8核场景, 对比总 cycle 和 per-core cycle
  - 误差 <= 5% (总线仲裁可能有差异)

Level 4: 端到端验证
  - LLM 模型 (DeepSeek-V3), 对比性能指标
  - 与数学建模结果交叉验证
```

### 基准测试集

```
从 TPUPerfBenchZoo 选取:
  1. ResNet50 (单核, 基础验证)
  2. BERT Large (单核, Transformer)
  3. GPT-2 (单核, 生成模型)
  4. FC_QKV (8核, 多核验证)
  5. DeepSeek-V3 层 (指令生成验证)
```

## 11. 风险和缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Python 性能不足 | 高 | 中 | Cython 加速关键路径, 或方案 B 兜底 |
| tiuImpl 翻译错误 | 中 | 高 | 逐指令与 TPUPerf 对比验证 |
| SystemC sc_main 限制 | 中 | 中 | 子进程方案 |
| 指令生成器 Tiling 不准 | 中 | 中 | 与真实编译器输出对比 |
| DMA 分段逻辑复杂 | 低 | 高 | 优先实现最常用类型 (TENSOR) |
| 多核同步 bug | 中 | 高 | 先单核稳定, 再逐步增加核数 |
