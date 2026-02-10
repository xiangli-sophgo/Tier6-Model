# 06 - 多核互连

## 1. 对标: TPUPerf 多核架构

### TPUPerf 的多核拓扑

```
SG2260 (8核):
  tpuEightCore.cc -> 8 个 TpuSubsys + Bus + CDMA + C2C

SG2262 (最多64核):
  tpuManyCore.cc -> N 个 TpuSubsys + Bus + CDMA + C2C + FakeChip
  CORE_NUM 编译时确定, 布局为 8x8 2D mesh
```

### 系统级连接关系

```
+--------+  +--------+       +--------+
| Core 0 |  | Core 1 |  ...  | Core N |
| TIU    |  | TIU    |       | TIU    |
| GDMA   |  | GDMA   |       | GDMA   |
| SDMA   |  | SDMA   |       | SDMA   |
| LMEM   |  | LMEM   |       | LMEM   |
+---+----+  +---+----+       +---+----+
    |            |                 |
    |  GDMA port |  SDMA port     |
    v            v                 v
+----------------------------------------------+
|              Simple Bus (NxM)                |
|  N masters = CoreNum*2 (GDMA + SDMA)        |
|  M slaves  = CoreNum (DDR/ARE per core)      |
+----------------------------------------------+
    |            |                 |
    v            v                 v
+--------+  +--------+       +--------+
| ARE[0] |  | ARE[1] |       | ARE[N] |
| DDR[0] |  | DDR[1] |       | DDR[N] |
+--------+  +--------+       +--------+

            +--------+    +----------+
            |  CDMA  |<-->|   C2C    |<--link-->  FakeChip
            +--------+    +----------+            (远端芯片)
```

## 2. Bus 模型

### 2.1 对标: `c_model/include/fabric/simple_bus.h` (~960行)

**架构**:
- 模板化 NxM 互联总线 (`simple_bus<128, 64>`)
- N 个 master 端口 (initiator), M 个 slave 端口 (target)
- TLM-2.0 仲裁: 先到先服务
- 可配置 per-master-per-slave 延迟矩阵

**关键参数**:
```
param_rsp_rd_lat[i][j]  # master i 到 slave j 的读延迟 (cycle)
param_rsp_wr_lat[i][j]  # master i 到 slave j 的写延迟 (cycle)
```

**距离相关延迟** (tpuManyCore.cc 中配置):
```
核的坐标 (8x8 网格):
  baseX = {0, 4, 0, 4, 0, 4, 0, 4}  (cluster 内)
  baseY = {0, 0, 2, 2, 4, 4, 6, 6}  (cluster 间)

延迟 = 45 * (|x1 - x2| + |y1 - y2|)  cycles

例: Core 0 (0,0) -> Core 7 (4,6) = 45 * (4+6) = 450 cycles
```

### 2.2 Python Bus 实现

```python
class BusModel:
    """NxM 总线模型

    核心行为:
    1. 接收 master 请求 (master_id, target_addr, size, type)
    2. 地址解码 -> slave_id (通过 SAM)
    3. 仲裁 (多 master 争用同一 slave)
    4. 添加距离延迟: lat[master_id][slave_id]
    5. 转发到 slave
    6. slave 响应后添加返回延迟
    7. 返回给 master

    仲裁策略: FCFS (先到先服务)
    """

    # 延迟矩阵初始化:
    # for i in range(num_masters):
    #   for j in range(num_slaves):
    #     core_i = master_to_core(i)
    #     core_j = slave_to_core(j)
    #     distance = manhattan_distance(core_i, core_j)
    #     latency[i][j] = 45 * distance
```

## 3. C2C 链路模型

### 3.1 对标: `c_model/src/sg2260/c2c.cc` (~308行)

**架构**:
- 6 个 SC_THREAD: tx, rx, slave req/rsp
- 带宽受限链路模型
- ROB 有序提交
- 信用流控

**关键参数**:
```
link_bw = 256 / 8 / 2 * 0.7 = 11.2 GB/s
internal_delay = 0 cycles
rd_outstanding = 128
wr_outstanding = 128
```

**传输延迟**:
```
delay = data_length / link_bw  (ns)
```

### 3.2 Python C2C 实现

```python
class C2CLink:
    """Chip-to-Chip 链路模型

    核心行为:
    1. 发送: tx 端接收请求, 计算链路延迟, 发到远端
    2. 接收: rx 端接收响应, 通过 ROB 有序提交
    3. 信用: 接收 credit 类型消息, 传递给 CDMA

    带宽建模:
      delay_ns = data_bytes / bw_GBps
      delay_cycles = delay_ns * frequency_GHz

    ROB 有序提交:
      rob: deque[(request, response)]
      每次只提交 rob 头部已完成的请求
    """
```

## 4. CDMA 模型

### 4.1 对标: `c_model/src/sg2260/cdma.cc` (~1,191行)

**架构**:
- 16 个 SC_THREAD
- 支持 SEND/RECV/FAKE_ALL_REDUCE 命令
- 信用流控: recv 发信用 -> send 等信用
- 伪延迟模式: alpha-beta 模型

**关键参数**:
```
pseudo_cmd = true          # 使用伪延迟 (简化)
credit_capacity = 2        # 信用容量
alpha_us = 0.25            # 启动延迟 (us)
beta_us = 0.3              # 同步延迟 (us)
c2c_bw = 11.2 GB/s         # C2C 带宽
```

**命令类型和延迟**:

| 命令 | 延迟公式 | 说明 |
|------|---------|------|
| SEND | alpha + data_size / (2 * bw) | 单向发送 |
| RECV | alpha + data_size / (2 * bw) | 单向接收 |
| FAKE_ALL_REDUCE | 4*alpha + 3*beta + 2*data_size / bw | 模拟 AllReduce |

### 4.2 CDMA 信用流控机制

```
RECV 端:
  1. 收到 RECV 命令
  2. 分配本地接收缓冲区
  3. 创建 credit 包 (包含目标地址信息)
  4. 通过 C2C 发送 credit 到对端
  5. 等待数据到达

SEND 端:
  1. 收到 SEND 命令
  2. 等待 credit (从 credit_fifo 读取)
  3. 获取 credit 中的目标地址
  4. 读取本地数据 (通过 Bus -> LMEM/DDR)
  5. 通过 C2C 写入远端地址
  6. 完成
```

### 4.3 Python CDMA 实现

```python
class CDMAEngine:
    """跨芯片 DMA 引擎

    核心行为:
    1. 命令调度 (类似 TDMA 的 cmd_dispatch)
    2. SEND: 等 credit -> 读本地 -> 写远端
    3. RECV: 发 credit -> 等数据到达
    4. FAKE_ALL_REDUCE: 伪延迟模型

    信用流控:
      credit_fifo: FIFO  # 存放远端发来的 credit
      credit_port: C2CLink 的 credit 通道

    有序提交:
      commit_fifo: deque[(cmd, {sema, rd_done, wr_done})]
    """
```

## 5. 多核组装

### 5.1 对标: `c_model/src/top/tpuManyCore.cc` (~830行)

**组装流程**:

```
1. 创建 N 个 TpuSubsys (含 TIU + GDMA + SDMA + LMEM)
2. 创建 N 个 DDR (每核一个, 通过 ARE)
3. 创建 Bus (2N masters, N slaves)
4. 创建 CDMA + C2C + FakeChip (芯片间通信)
5. 绑定端口:
   - TpuSubsys[i].gdma_port -> Bus.master[i]
   - TpuSubsys[i].sdma_port -> Bus.master[N+i]
   - Bus.slave[i] -> ARE[i] -> DDR[i]
   - CDMA -> C2C -> FakeChip
6. 配置时钟:
   - tpu_clock, tiu_clock, bus_clock, ddr_clock
7. 配置延迟矩阵:
   - Bus.latency[i][j] = 45 * manhattan_distance(i, j)
8. 加载指令:
   - TpuSubsys[i].tiu.load_commands(tiu_buf[i])
   - TpuSubsys[i].gdma.load_commands(gdma_buf[i])
```

### 5.2 Python MultiCore 实现

```python
class MultiCoreSimulator:
    """多核仿真器顶层

    组装流程:
    1. 根据芯片配置创建核心数组
    2. 创建共享基础设施 (Bus, DDR, Cache)
    3. 创建芯片间通信 (C2C, CDMA)
    4. 配置延迟参数
    5. 加载指令
    6. 运行仿真
    7. 收集结果

    终止条件:
      所有核的 TIU 和 DMA 都 idle 且持续 100 cycles
    """
```

## 6. 时钟域配置

```
SG2260/SG2262 时钟域:

TPU 主时钟:     1.0 GHz (1ns)
TIU 计算时钟:   1.0 GHz (与主时钟相同或独立)
Bus 时钟:       根据带宽配置
  128GB/s -> bus_clk = tpu_clk
  153GB/s -> bus_clk = tpu_clk / 1.25
  192GB/s -> bus_clk = tpu_clk / 1.5
DDR 时钟:       bus_width / bw_per_instance (ns)
  例: 64B / 12.8 GB/s = 5ns -> 200MHz
```

Python 中通过多时钟域支持实现:

```python
# 注册多个时钟
scheduler.add_clock("tpu_clk", period=1)      # 1ns
scheduler.add_clock("bus_clk", period=0.67)    # 1.5GHz for 192G
scheduler.add_clock("ddr_clk", period=5)       # 200MHz

# 绑定模块到时钟域
tiu_engine.bind_clock("tpu_clk")
dma_engine.bind_clock("bus_clk")
ddr_model.bind_clock("ddr_clk")
```
