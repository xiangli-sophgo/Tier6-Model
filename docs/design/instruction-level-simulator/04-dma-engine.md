# 04 - DMA 数据搬运引擎

## 1. 对标: TPUPerf TDMA 实现

### TPUPerf 中的 TDMA

- **文件**: `c_model/src/sg2260/tdma.cc` (~3,031行) + `tdmaDelayImpl.cc` (~4,276行)
- **模型**: 1 个 SC_METHOD (命令调度) + 32 个 SC_THREAD (数据通路)
- **复杂度**: 这是 TPUPerf 中最复杂的模块

### 流水线架构 (5 级)

```
cmd_dispatch_mth (SC_METHOD)
    |
    v
[第1级: 命令分段] rd/wr_cmd_segmentation_th (16个线程)
    |  tensor shape -> 1D segments
    v
[第2级: Fabric转换] rd/wr_tensor_to_fab_th (2个线程)
    |  segments -> AXI bursts
    v
[第3级: Cache/路由] rd/wr_ca_mux_th (3个线程)
    |  地址解码 -> GMEM/LMEM 路由
    v
[第4级: 响应处理] fab_rrsp/wrsp_mux_th + dp_update (4个线程)
    |  数据路径更新
    v
[第5级: 完成跟踪] fab_to_tensor_th + cmd_collect_th (2个线程)
    |  ROB 有序完成, 更新 tdmaSyncId
```

## 2. Python DMA Engine 设计

### 2.1 整体结构

将 TPUPerf 的 32 线程流水线简化为**逻辑等价**的模型:

```python
class DMAEngine:
    """DMA 搬运引擎 - 对应 TPUPerf 的 Tdma 类

    不需要完全复刻 32 个线程, 而是保留功能等价的处理阶段:
    1. 命令调度 (cmd_dispatch) - Method
    2. 命令分段 (segmentation) - Thread
    3. 总线请求 (fabric_request) - Thread
    4. 响应处理 (response_handler) - Thread
    5. 完成跟踪 (completion_tracker) - Thread
    """
```

### 2.2 命令调度 (对标 cmd_dispatch_mth)

这是一个 SC_METHOD, 敏感于 `transfer_done` 和 `tiuSyncId`:

```
调度逻辑:
  if 命令队列空 && 无活跃命令:
    idle = True
    return

  if 当前命令是 SYS_END:
    标记结束
    return

  if 当前命令是 SYS_SEND_MSG:
    message_queue.send_msg(msg_id, wait_cnt)
    弹出命令, 继续下一条

  if 当前命令是 SYS_WAIT_MSG:
    if not message_queue.wait_msg(msg_id, send_cnt):
      return  # 阻塞等待

  if 当前命令的 cmd_id_dep > tiu_sync_id:
    return  # 等待 TIU 完成依赖

  # 依赖满足, 分发命令
  根据命令类型分发:
    TENSOR      -> 标准读写路径
    SCATTER     -> scatter 特殊路径
    GATHER      -> gather 特殊路径
    RANDOM_MASK -> randmask 路径
    ...
```

### 2.3 命令分段 (对标 rd/wr_cmd_segmentation_th)

这是 DMA 建模的核心: 将 4D tensor 操作拆分为 1D 连续数据块。

**分段策略**:

```
输入: tensor shape (N, C, H, W) + strides + base_addr

连续性判断:
  is_h_continuous = (h_stride == w_size * w_stride)
  is_c_continuous = (c_stride == h_size * w_size * w_stride) && is_h_continuous

分段:
  if is_c_continuous:
    # 整个 NCHW 在内存中连续, 每个 N 一个 segment
    for n in range(N):
      yield Segment(addr=base + n*n_stride, size=C*H*W*elem_size)

  elif is_h_continuous:
    # H*W 连续, 每个 (N,C) 一个 segment
    for n in range(N):
      for c in range(C):
        yield Segment(addr=base + n*n_stride + c*c_stride, size=H*W*elem_size)

  else:
    # 逐行, 每个 (N,C,H) 一个 segment
    for n in range(N):
      for c in range(C):
        for h in range(H):
          yield Segment(addr=base + ..., size=W*elem_size)
```

**LMEM 地址计算** (lane 映射):
```
对于 LMEM 访问, 需要额外的 lane 映射:
  lane_idx = (start_lane + c_idx) % NPU_NUM
  addr = lane_idx * PER_LANE_LMEM_SIZE + offset
```

### 2.4 总线请求 (对标 rd/wr_tensor_to_fab_th)

将 segment 进一步拆分为 AXI burst 级请求:

```
输入: Segment(addr, size)

拆分为 burst:
  while remaining > 0:
    burst_size = min(remaining, bus_width * max_burst_length)

    # 地址对齐检查
    alignment = addr % bus_width
    if alignment != 0:
      burst_size = min(burst_size, bus_width - alignment)

    yield BurstRequest(addr, burst_size, type=READ/WRITE)
    addr += burst_size
    remaining -= burst_size
```

**Outstanding 控制**:
```
发送请求前:
  yield semaphore.acquire(rd_outstanding)  # 获取 outstanding 信用

收到响应后:
  semaphore.release(rd_outstanding)        # 释放 outstanding 信用
```

### 2.5 地址路由

根据地址范围决定请求发往 LMEM 还是 GMEM (DDR):

```
if LMEM_START <= addr < LMEM_END:
  -> lmem_port
else:
  -> gmem_port (通过 Bus 到 DDR)
```

### 2.6 完成跟踪 (对标 cmd_collect_th)

使用 ROB 保证按序完成:

```
命令完成条件:
  1. 该命令的所有读 segment 都已响应
  2. 该命令的所有写 segment 都已响应
  3. 数据缓冲区已清空

完成后:
  - tdma_sync_id.write(cmd.cmd_id)  # 通知 TIU
  - profiler.record_dma(cmd, start_cycle, end_cycle)
  - transfer_done.notify()           # 触发下一条命令调度
```

## 3. DMA 延迟计算

### 3.1 延迟来源

DMA 操作的总延迟由以下部分组成:

```
total_latency = segmentation_delay     # 分段开销 (通常 0)
              + fabric_transfer_delay  # 总线传输
              + memory_access_delay    # 内存访问 (LMEM/DDR)
              + pipeline_overhead      # 流水线启动/排空
```

其中 `memory_access_delay` 是主要部分, 取决于目标:

**LMEM 访问**:
```
delay = lmem_latency + burst_length  (cycles)
# SG2260: lmem_latency = 29ns -> 约 58 cycles @ 2GHz
```

**DDR 访问**:
```
delay = ddr_latency + burst_length + alignment_penalty  (cycles)
# SG2260: ddr_latency = 150ns -> 约 300 cycles @ 2GHz
# alignment_penalty: 非对齐访问额外 10 cycles
```

### 3.2 不同 DMA 类型

| 命令类型 | 说明 | 分段方式 |
|---------|------|---------|
| TENSOR | 标准 tensor 搬运 | NCHW 连续性分段 |
| MATRIX | 矩阵搬运 (可能转置) | 按 row+section 分段 |
| CW_TRANS | C-W 转置 | 交换 C/W 维度后分段 |
| BROADCAST | 广播 (C=1) | 单次读, 多次写 |
| DISTRIBUTE | 小块分发 | 每次 128B |
| SCATTER | 散射写 | 按 index 分散 |
| GATHER | 聚集读 | 按 index 收集 |

### 3.3 实现优先级

**P0 (必须)**:
- TENSOR: 最常用的 DMA 操作
- MATRIX: MatMul 的权重加载
- Outstanding 控制

**P1 (重要)**:
- CW_TRANS: 转置操作
- BROADCAST: 广播
- DISTRIBUTE: 数据分发

**P2 (可延后)**:
- SCATTER/GATHER: 特殊索引操作
- RANDOM_MASK: 随机掩码
- NONZERO/MASKED_SEL: 稀疏操作

## 4. TIU-DMA 并行执行模型

这是指令级仿真的核心价值: 精确模拟 TIU 和 DMA 的并行执行。

```
时间轴:
  TIU: |--CMD0--|  |--CMD1--|           |--CMD2--|
  DMA:    |---CMD0---|  |------CMD1------|

  CMD0: TIU 和 DMA 并行, 无依赖
  CMD1: TIU CMD1 等待 DMA CMD0 完成 (cmd_id_dep)
  CMD2: TIU CMD2 等待 DMA CMD1 完成

Gantt 图数据:
  TIU_CMD0: {start: 0, end: 100, type: "CONV"}
  DMA_CMD0: {start: 20, end: 150, type: "TENSOR_LOAD"}
  TIU_CMD1: {start: 150, end: 250, type: "MM2"}  # 等到 DMA_CMD0 完成
  DMA_CMD1: {start: 152, end: 400, type: "TENSOR_STORE"}
  TIU_CMD2: {start: 400, end: 500, type: "CONV"}  # 等到 DMA_CMD1 完成
```

这个并行模式直接决定了:
- **Parallelism** 指标: (TIU_busy + DMA_busy) / Total_time, 理想值 200%
- **Bottleneck**: TIU-bound 还是 DMA-bound
- **优化方向**: double buffering, 预取策略等
