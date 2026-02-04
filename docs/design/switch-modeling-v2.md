# 交换机建模设计文档 v2.0

## 文档信息

- **版本**: v2.0
- **创建日期**: 2026-02-04
- **状态**: 设计阶段
- **目标**: 为Tier6仿真系统提供cycle级精确的网络交换机建模方案

---

## 一、背景与动机

### 1.1 现状分析

当前Tier6仿真系统主要建模芯片间**直连互联**（Chip-to-Chip NVLink/PCIe），但在实际大规模部署中，存在以下场景需要通过**网络交换机**进行互联：

| 场景 | 互联层级 | 典型设备 | 带宽需求 | 延迟要求 |
|------|---------|---------|---------|---------|
| **Rack内互联** | Board-to-Board | ToR交换机 (Top-of-Rack) | 400-800 Gbps/端口 | < 1 μs |
| **Pod内互联** | Rack-to-Rack | 汇聚交换机 (Aggregation) | 400-1600 Gbps/端口 | 1-3 μs |
| **跨Pod互联** | Pod-to-Pod | 核心交换机 (Core) | 800-3200 Gbps/端口 | 3-10 μs |

### 1.2 设计目标

1. **Cycle级精确性**: 建模流水线、缓冲区、调度器等微架构组件，提供ns级延迟精度
2. **性能评估能力**: 准确预测吞吐量、排队延迟、丢包率等关键指标
3. **灵活配置**: 支持不同交换机规格（端口数、缓冲深度、调度算法）
4. **与拓扑集成**: 无缝融入现有的Pod-Rack-Board-Chip层级拓扑模型

### 1.3 非目标 (Out of Scope)

- ❌ 数据包级模拟（packet content simulation）
- ❌ 完整TCP/IP协议栈建模
- ❌ 硬件功耗建模（Phase 1）
- ❌ 故障注入和容错机制

---

## 二、交换机分类与应用场景

### 2.1 交换机类型定义

#### **Type 1: ToR交换机 (Top-of-Rack Switch)**

**部署位置**: Rack内部，连接多个Board（服务器/节点）

**典型规格**:
- 端口数: 32-64端口
- 端口速率: 400G/800G Ethernet
- 背板带宽: 25.6-51.2 Tbps
- 缓冲深度: 16-64 MB（共享缓冲池）
- 延迟: 500 ns - 1 μs (端口到端口)

**应用场景**:
```
Rack 0
├── Board 0 (8 chips) ──┐
├── Board 1 (8 chips) ──┤
├── Board 2 (8 chips) ──┼──> ToR Switch (32 ports)
├── Board 3 (8 chips) ──┤
└── Board 4 (8 chips) ──┘
```

**典型产品**: Broadcom Tomahawk 4/5, Cisco Nexus 93xxx

#### **Type 2: 汇聚交换机 (Aggregation Switch)**

**部署位置**: Pod内部，连接多个ToR交换机（跨Rack互联）

**典型规格**:
- 端口数: 64-128端口
- 端口速率: 400G/800G Ethernet
- 背板带宽: 51.2-102.4 Tbps
- 缓冲深度: 64-256 MB
- 延迟: 1-3 μs

**应用场景**:
```
Pod 0
├── Rack 0 (ToR Switch) ──┐
├── Rack 1 (ToR Switch) ──┤
├── Rack 2 (ToR Switch) ──┼──> Aggregation Switch
├── Rack 3 (ToR Switch) ──┤
└── Rack 4 (ToR Switch) ──┘
```

**典型产品**: Arista 7800R3, Cisco Nexus 9500

### 2.2 与现有拓扑层级的关系

```
拓扑层级                     互联方式
─────────────────────────────────────────
Pod (集群)
 │                          ┌─ 核心交换机 (Core Switch, 未来扩展)
 ├─ Rack 0 ←──────────────→ 汇聚交换机 (Aggregation Switch)
 │   ├─ Board 0 ←─────────→ ToR交换机 (Top-of-Rack Switch)
 │   │   ├─ Chip 0 ←──────→ NVLink/PCIe (直连, 现有模型)
 │   │   ├─ Chip 1
 │   │   └─ Chip N
 │   ├─ Board 1
 │   └─ Board M
 └─ Rack 1
```

**集成原则**:
- 芯片间 (Chip-to-Chip): 优先使用NVLink/PCIe直连
- Board间 (Board-to-Board): 可选直连或通过ToR交换机
- Rack间 (Rack-to-Rack): 必须通过汇聚交换机
- Pod间 (Pod-to-Pod): 通过核心交换机（Phase 2）

---

## 三、微架构设计

### 3.1 整体架构

交换机微架构采用**输入缓冲 + 虚拟输出队列 (VOQ) + Crossbar交换矩阵**的经典架构：

```
┌────────────────────────────────────────────────────────────┐
│                      Network Switch                        │
│                                                            │
│  Input Port 0        ┌──────────────────┐      Output Port 0
│  ┌──────────┐        │                  │      ┌──────────┐
│  │ Input    │        │    Crossbar      │      │  Output  │
│  │ Buffer   │───────>│    Switching     │─────>│  Queue   │
│  │ (VOQ)    │        │    Fabric        │      │          │
│  └──────────┘        │    (N × N)       │      └──────────┘
│       │              │                  │           │
│       │              └──────────────────┘           │
│       │                      ▲                      │
│       │                      │                      │
│       └──────────> Scheduler/Arbiter <──────────────┘
│                    (iSLIP/PIM)                          │
│                                                            │
│  Input Port N-1                             Output Port N-1
│  ┌──────────┐                               ┌──────────┐  │
│  │ Input    │                               │  Output  │  │
│  │ Buffer   │───────────────────────────────>│  Queue   │  │
│  │ (VOQ)    │                               │          │  │
│  └──────────┘                               └──────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 关键组件定义

#### 3.2.1 输入缓冲区 (Input Buffer)

**结构**: 虚拟输出队列 (VOQ) 架构
```
Input Port i:
  ├─ VOQ[0] → 目标输出端口0的队列
  ├─ VOQ[1] → 目标输出端口1的队列
  ├─ VOQ[2] → 目标输出端口2的队列
  └─ VOQ[N-1] → 目标输出端口N-1的队列
```

**关键参数**:
- `num_ports`: 端口总数 (N)
- `voq_depth`: 每个VOQ的深度 (flits/packets)
- `buffer_sharing_mode`: 缓冲共享模式
  - `dedicated`: 每VOQ独立缓冲
  - `shared_pool`: 共享缓冲池 (推荐)

**性能特性**:
- ✅ 消除Head-of-Line (HOL) 阻塞
- ✅ 理论吞吐量: 100% (vs 简单输入队列的58.6%)
- ⚠️ 内存开销: O(N²) 个队列

#### 3.2.2 交换矩阵 (Crossbar Switching Fabric)

**功能**: 在任意输入端口和输出端口间建立无阻塞连接

**实现方式**:
1. **无缓冲Crossbar** (Unbuffered)
   - 优点: 硬件开销小
   - 缺点: 需要复杂调度器 (iSLIP)
   - 适用: ToR交换机

2. **缓冲Crossbar** (Buffered, 可选)
   - 每个交叉点 (crosspoint) 配备小缓冲 (2-8 packets)
   - 优点: 简化调度，提升突发吸收能力
   - 缺点: 高SRAM成本
   - 适用: 汇聚/核心交换机

**关键参数**:
- `crossbar_type`: "unbuffered" | "buffered"
- `crosspoint_buffer_size`: 交叉点缓冲大小 (仅buffered模式)
- `internal_speedup`: 内部加速比 (1.0 = 无加速, 2.0 = 2倍速)

#### 3.2.3 调度器/仲裁器 (Scheduler/Arbiter)

**目标**: 每个时间槽为输入-输出端口对分配crossbar连接

**核心算法**: **iSLIP** (Iterative Round-Robin Matching)

**算法流程** (3步迭代):
```
Iteration k (k = 1, 2, ..., max_iterations):

Step 1: Request
  每个输入端口i向所有非空VOQ的目标输出端口j发送请求

Step 2: Grant
  每个输出端口j从收到的请求中，选择优先级最高的输入端口
  (使用round-robin指针，起始位置 = grant_ptr[j])
  如果授权，grant_ptr[j] 移动到下一个位置

Step 3: Accept
  每个输入端口i从收到的grant中，选择优先级最高的输出端口
  (使用round-robin指针，起始位置 = accept_ptr[i])
  如果接受，accept_ptr[i] 移动到 (j+1) mod N

输出: 输入-输出匹配对 (i, j)
```

**关键参数**:
- `arbiter_algorithm`: "islip" | "pim" | "wwfa"
- `max_iterations`: 最大迭代次数 (1-4, 典型值=2)
- `speedup`: 调度器加速比 (支持每cycle多次调度)

**性能特性**:
- 单次迭代: 吞吐量 ~63%
- 2次迭代: 吞吐量 ~99%
- 4次迭代: 吞吐量 ~99.9%

#### 3.2.4 输出队列 (Output Queue, 可选)

**作用**: 平滑输出流量，处理短期突发

**配置**:
- `output_queue_depth`: 输出队列深度 (packets)
- `output_scheduling`: 输出端口调度策略
  - `fifo`: 先进先出
  - `priority`: 优先级队列 (支持QoS)
  - `wfq`: 加权公平队列

---

## 四、流水线与延迟模型

### 4.1 简化流水线模型 (推荐)

基于**包级调度**的3阶段流水线：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Stage 1   │───>│   Stage 2   │───>│   Stage 3   │
│   Ingress   │    │  Scheduling │    │   Egress    │
│  Buffering  │    │  & Switching│    │ Transmission│
└─────────────┘    └─────────────┘    └─────────────┘
     │                   │                   │
     v                   v                   v
  写入VOQ          iSLIP仲裁           通过Crossbar
  (1 cycle)      (1-4 iterations)       + 输出发送
                  (2-8 cycles)          (N cycles)
```

**阶段详解**:

| 阶段 | 操作 | 周期数 | 说明 |
|------|------|--------|------|
| **Ingress** | 包到达 → 路由查找 → 写入VOQ | 1-2 | 包括解析和分类 |
| **Scheduling** | iSLIP仲裁 → Crossbar配置 | 2-8 | 迭代次数×2 |
| **Egress** | Crossbar传输 → 输出发送 | 变长 | 取决于包大小 |

### 4.2 延迟计算公式

#### **端到端延迟组成**

$$
T_{\text{total}} = T_{\text{ingress}} + T_{\text{queue}} + T_{\text{schedule}} + T_{\text{xbar}} + T_{\text{egress}} + T_{\text{serialization}}
$$

#### **各项延迟定义**

**1. Ingress处理延迟** $T_{\text{ingress}}$
```
T_ingress = cycles_ingress × clock_period
```
- `cycles_ingress`: Ingress流水线级数 (1-2 cycles)
- `clock_period`: 时钟周期 (典型值: 2 ns @ 500 MHz)

**2. 排队延迟** $T_{\text{queue}}$
```python
# 基于M/M/1队列模型
λ = arrival_rate  # 包到达率 (packets/s)
μ = service_rate  # 服务率 (packets/s)
ρ = λ / μ         # 利用率

T_queue = 1 / (μ - λ)  # 平均排队时间 (Little's Law)
```

**实际计算** (考虑VOQ):
```
# 每个VOQ独立排队
T_queue[i,j] = current_voq_depth[i,j] × packet_transmission_time
```

**3. 调度延迟** $T_{\text{schedule}}$
```
T_schedule = num_iterations × 2 × clock_period
```
- `num_iterations`: iSLIP迭代次数 (1-4)
- 每次迭代2个cycle (request + grant/accept)

**4. Crossbar传输延迟** $T_{\text{xbar}}$
```
T_xbar = 1 × clock_period  # 单cycle传输
```

**5. Egress处理延迟** $T_{\text{egress}}$
```
T_egress = cycles_egress × clock_period
```
- `cycles_egress`: Egress流水线级数 (1 cycle)

**6. 串行化延迟** $T_{\text{serialization}}$
```
T_serialization = packet_size_bits / link_bandwidth_bps
```

**示例计算** (ToR交换机):
```
给定条件:
- 时钟频率: 500 MHz (clock_period = 2 ns)
- 包大小: 1500 Bytes (12000 bits)
- 链路带宽: 400 Gbps
- iSLIP迭代: 2次
- VOQ当前深度: 10 packets

计算:
T_ingress = 2 cycles × 2 ns = 4 ns
T_queue = 10 packets × (12000 bits / 400×10^9 bps) = 10 × 30 ns = 300 ns
T_schedule = 2 iterations × 2 cycles × 2 ns = 8 ns
T_xbar = 1 cycle × 2 ns = 2 ns
T_egress = 1 cycle × 2 ns = 2 ns
T_serialization = 12000 bits / 400×10^9 bps = 30 ns

总延迟 = 4 + 300 + 8 + 2 + 2 + 30 = 346 ns
```

#### **零负载延迟** (Zero-Load Latency)

当交换机空闲时（无排队）:
```
T_zero_load = T_ingress + T_schedule + T_xbar + T_egress + T_serialization
            = 4 + 8 + 2 + 2 + 30 = 46 ns
```

这是交换机的**最小延迟**，用于配置文件中的 `latency_us` 参数。

---

## 五、吞吐量与缓冲区模型

### 5.1 吞吐量建模

#### **理论最大吞吐量**

对于N×N交换机:
```
Throughput_max = N × port_bandwidth
```

例如: 32端口 × 400 Gbps = 12.8 Tbps (背板带宽)

#### **实际吞吐量限制因素**

1. **VOQ调度效率**
   - iSLIP (2次迭代): ~99% 理论吞吐
   - 实际: 95-98% (考虑碰撞和重调度)

2. **输入端口竞争**
   - 多个输入同时请求同一输出 → 仲裁失败 → 吞吐下降

3. **缓冲溢出**
   - VOQ满 → 反压 (backpressure) → 上游限速

#### **吞吐量计算公式**

```python
# 单端口吞吐量
throughput_per_port = successfully_transmitted_bits / simulation_time

# 整体吞吐量
total_throughput = sum(throughput_per_port[i] for i in range(N))

# 吞吐率 (归一化)
throughput_ratio = total_throughput / (N × port_bandwidth)
```

### 5.2 缓冲区管理

#### **共享缓冲池模型** (推荐)

```
全局缓冲池: Total_Buffer_Size (例: 64 MB)
  ├─ 静态分配: Reserved_Buffer_Per_VOQ (例: 512 KB)
  └─ 动态共享: Shared_Pool = Total - N² × Reserved
       └─ 动态阈值算法 (DT):
           max_share[i,j] = α × (Shared_Pool - Current_Usage)
```

**动态阈值 (DT) 算法**:
```python
def can_enqueue(voq, packet_size):
    static_quota = voq.reserved_buffer
    current_usage = sum(voq.size for all voqs)
    shared_available = total_buffer - current_usage

    # α: 阈值系数 (0.5 - 2.0)
    dynamic_quota = alpha × shared_available

    max_allowed = static_quota + dynamic_quota
    return voq.current_size + packet_size <= max_allowed
```

**参数配置**:
```yaml
buffer_management:
  total_buffer_mb: 64
  sharing_mode: "dynamic_threshold"
  reserved_per_voq_kb: 512
  alpha: 1.0  # 阈值系数
```

#### **缓冲溢出处理**

**策略1: Tail Drop** (默认)
```
if VOQ is full:
    drop incoming packet
    update drop_counter
```

**策略2: ECN Marking** (可选)
```
if VOQ_usage > ecn_threshold:
    mark packet with ECN flag
    forward to destination
```

### 5.3 流量控制 (Flow Control)

#### **Credit-Based Flow Control** (端口间)

```
发送端:
  credits[neighbor] = neighbor_buffer_depth  # 初始化

  每发送1个packet:
    credits[neighbor] -= 1

  每收到1个credit反馈:
    credits[neighbor] += 1

  发送条件:
    can_send = (credits[neighbor] > 0) and (packet in VOQ)

接收端:
  每接收1个packet并从buffer移除:
    send_credit_to_sender()
```

**配置参数**:
```yaml
flow_control:
  type: "credit_based"
  credit_delay_cycles: 4  # Credit返回延迟
```

---

## 六、配置格式定义

### 6.1 拓扑配置扩展

在现有的 `topologies/*.yaml` 中新增交换机节点：

```yaml
name: "P1-R4-B32-C256-with-switches"
pod_count: 1
racks_per_pod: 4

# 新增: 交换机定义
switches:
  - name: "tor_switch_rack0"
    type: "top_of_rack"
    location:
      pod: 0
      rack: 0
    preset_id: "tor_400g_32port"  # 引用预设配置

  - name: "tor_switch_rack1"
    type: "top_of_rack"
    location:
      pod: 0
      rack: 1
    preset_id: "tor_400g_32port"

  - name: "agg_switch_pod0"
    type: "aggregation"
    location:
      pod: 0
    preset_id: "agg_800g_64port"

# 现有: Rack配置
rack_config:
  boards:
    - chips:
        - name: "SG2262"
          count: 8
      count: 32  # 32 boards per rack

# 修改: 连接定义 (新增经由交换机的路径)
connections:
  # Board内芯片直连 (NVLink, 保持不变)
  - from: {pod: 0, rack: 0, board: 0, chip: 0}
    to: {pod: 0, rack: 0, board: 0, chip: 1}
    type: "direct"
    bandwidth_gbps: 448
    latency_us: 0.2

  # Board间通过ToR交换机 (新)
  - from: {pod: 0, rack: 0, board: 0}
    to: {pod: 0, rack: 0, board: 1}
    type: "via_switch"
    switch_name: "tor_switch_rack0"
    bandwidth_gbps: 400
    latency_us: 0.001  # 1 μs

  # Rack间通过汇聚交换机 (新)
  - from: {pod: 0, rack: 0}
    to: {pod: 0, rack: 1}
    type: "via_switch"
    switch_name: "agg_switch_pod0"
    bandwidth_gbps: 800
    latency_us: 0.003  # 3 μs
```

### 6.2 交换机预设配置

新增目录: `backend/configs/switch_presets/*.yaml`

**示例: ToR交换机预设**
```yaml
# switch_presets/tor_400g_32port.yaml
name: "ToR 400G 32-Port Switch"
type: "top_of_rack"

# 基本参数
num_ports: 32
port_bandwidth_gbps: 400
backplane_bandwidth_tbps: 12.8

# 微架构参数
microarchitecture:
  clock_frequency_mhz: 500
  pipeline_stages: 3

  # 输入缓冲
  input_buffer:
    architecture: "voq"  # Virtual Output Queuing
    total_buffer_mb: 16
    sharing_mode: "dynamic_threshold"
    reserved_per_voq_kb: 128
    alpha: 1.0

  # 交换矩阵
  crossbar:
    type: "unbuffered"
    internal_speedup: 1.0

  # 调度器
  scheduler:
    algorithm: "islip"
    max_iterations: 2
    scheduling_cycle: 2  # cycles per scheduling decision

  # 输出队列 (可选)
  output_queue:
    enabled: false

# 延迟参数
latency:
  zero_load_ns: 50  # 零负载延迟
  ingress_cycles: 2
  scheduling_cycles: 4  # 2 iterations × 2 cycles
  egress_cycles: 1

# 流量控制
flow_control:
  type: "credit_based"
  credit_delay_cycles: 4

# 性能特性
performance:
  max_throughput_ratio: 0.98  # 98% of theoretical max
  buffer_overflow_policy: "tail_drop"
```

**示例: 汇聚交换机预设**
```yaml
# switch_presets/agg_800g_64port.yaml
name: "Aggregation 800G 64-Port Switch"
type: "aggregation"

num_ports: 64
port_bandwidth_gbps: 800
backplane_bandwidth_tbps: 51.2

microarchitecture:
  clock_frequency_mhz: 600
  pipeline_stages: 3

  input_buffer:
    architecture: "voq"
    total_buffer_mb: 64  # 更大缓冲
    sharing_mode: "dynamic_threshold"
    reserved_per_voq_kb: 256
    alpha: 1.5

  crossbar:
    type: "buffered"  # 使用缓冲crossbar
    crosspoint_buffer_packets: 4
    internal_speedup: 1.2

  scheduler:
    algorithm: "islip"
    max_iterations: 3  # 更多迭代保证高吞吐
    scheduling_cycle: 2

  output_queue:
    enabled: true
    depth_packets: 128
    scheduling_policy: "wfq"  # 加权公平队列

latency:
  zero_load_ns: 150
  ingress_cycles: 2
  scheduling_cycles: 6  # 3 iterations × 2 cycles
  egress_cycles: 1

flow_control:
  type: "credit_based"
  credit_delay_cycles: 8  # 更长距离

performance:
  max_throughput_ratio: 0.99
  buffer_overflow_policy: "ecn_marking"  # 使用ECN
  ecn_threshold_ratio: 0.7  # 70%缓冲使用率触发ECN
```

### 6.3 硬件参数集成

在现有 `hardware_params` 中新增交换机参数：

```yaml
hardware_params:
  # 现有芯片配置
  chips:
    SG2262:
      name: "SG2262"
      compute_tflops_fp8: 768
      # ... (保持不变)

  # 新增: 交换机配置
  switches:
    tor_400g_32port:
      preset_id: "tor_400g_32port"  # 引用预设

    agg_800g_64port:
      preset_id: "agg_800g_64port"

  # 现有互联配置 (保持不变)
  interconnect:
    c2c:
      bandwidth_gbps: 448
      latency_us: 0.2
    # ...
```

---

## 七、性能指标与输出

### 7.1 交换机级指标

仿真结束后，为每个交换机输出以下指标：

```python
{
    "switch_name": "tor_switch_rack0",
    "type": "top_of_rack",

    # 吞吐量指标
    "throughput": {
        "total_gbps": 11520.5,  # 总吞吐
        "per_port_gbps": [380.2, 395.1, ...],  # 每端口吞吐
        "utilization_ratio": 0.90,  # 90% 利用率
        "backpressure_ratio": 0.05  # 5% 时间被反压
    },

    # 延迟指标
    "latency": {
        "average_ns": 285.3,
        "p50_ns": 120.5,
        "p95_ns": 680.2,
        "p99_ns": 1250.8,
        "max_ns": 3200.0
    },

    # 缓冲区指标
    "buffer": {
        "peak_usage_mb": 12.5,
        "average_usage_mb": 4.8,
        "overflow_count": 125,  # 溢出次数
        "drop_rate": 0.001  # 0.1% 丢包率
    },

    # 调度器指标
    "scheduler": {
        "total_scheduling_cycles": 1500000,
        "average_iterations": 1.85,  # 平均iSLIP迭代次数
        "match_efficiency": 0.982  # 匹配效率
    },

    # VOQ统计
    "voq_stats": {
        "max_depth_observed": [
            {"input": 0, "output": 5, "depth": 45},
            {"input": 2, "output": 3, "depth": 38},
            # ...
        ],
        "hotspot_pairs": [
            {"input": 0, "output": 5, "traffic_gb": 125.3},
            # ...
        ]
    }
}
```

### 7.2 路径延迟分解

对于经由交换机的通信路径，提供详细的延迟分解：

```python
{
    "path": "Chip(0,0,0,0) -> Chip(0,0,5,2)",
    "hops": [
        {
            "type": "chip_to_switch",
            "from": "Chip(0,0,0,0)",
            "to": "Switch(tor_switch_rack0)",
            "latency_breakdown": {
                "serialization_ns": 30.0,
                "propagation_ns": 5.0,
                "switch_ingress_ns": 4.0,
                "switch_queue_ns": 250.0,
                "switch_schedule_ns": 8.0,
                "switch_xbar_ns": 2.0,
                "switch_egress_ns": 2.0,
                "total_ns": 301.0
            }
        },
        {
            "type": "switch_to_chip",
            "from": "Switch(tor_switch_rack0)",
            "to": "Chip(0,0,5,2)",
            "latency_breakdown": {
                "serialization_ns": 30.0,
                "propagation_ns": 8.0,
                "total_ns": 38.0
            }
        }
    ],
    "total_latency_ns": 339.0
}
```

### 7.3 可视化增强

在现有Gantt图基础上，新增交换机活动可视化：

```
Timeline (横轴: 时间)
─────────────────────────────────────────────────────
Chip 0     [Compute]──┐
                      │
                      ├──[Send]──>┐
                                  │
Switch 0                          ├─[Queue]─[Schedule]─[Xbar]─>┐
                                                                │
Chip 5                                                          ├──[Recv]──[Compute]
                                                                │
Legend:
  [Queue]    : 排队延迟 (红色)
  [Schedule] : 调度延迟 (橙色)
  [Xbar]     : 交换矩阵传输 (绿色)
```

---

## 八、与现有系统的集成

### 8.1 拓扑解析器扩展

**修改点**: `backend/llm_simulator/topology.py`

新增类:
```python
class Switch:
    """交换机节点"""
    def __init__(self, name, type, location, config):
        self.name = name
        self.type = type  # "top_of_rack" | "aggregation" | "core"
        self.location = location  # {"pod": 0, "rack": 0}
        self.config = config  # SwitchConfig对象

        # 微架构状态
        self.voq_buffers = {}  # VOQ缓冲区
        self.scheduler = iSLIPScheduler(config)
        self.crossbar = Crossbar(config.num_ports)

class InterconnectGraph:
    """互联拓扑图 (现有类扩展)"""
    def __init__(self):
        self.chips = []      # 现有
        self.switches = []   # 新增: 交换机节点列表
        self.edges = []      # 现有 (边包括chip-chip和chip-switch)

    def add_switch(self, switch: Switch):
        """添加交换机节点"""
        self.switches.append(switch)

    def find_path(self, src_chip, dst_chip):
        """查找路径 (支持经由交换机)"""
        # Dijkstra最短路径 (考虑延迟权重)
        pass
```

### 8.2 仿真器集成

**修改点**: `backend/llm_simulator/simulator.py`

新增交换机事件类型:
```python
class EventType(str, Enum):
    # 现有事件
    COMPUTE_START = "compute_start"
    COMPUTE_END = "compute_end"
    COMM_START = "comm_start"
    COMM_END = "comm_end"

    # 新增: 交换机事件
    SWITCH_PACKET_ARRIVE = "switch_packet_arrive"
    SWITCH_PACKET_ENQUEUE = "switch_packet_enqueue"
    SWITCH_SCHEDULE = "switch_schedule"
    SWITCH_XBAR_TRANSFER = "switch_xbar_transfer"
    SWITCH_PACKET_DEPART = "switch_packet_depart"

class Simulator:
    def simulate_communication_via_switch(
        self, src_chip, dst_chip, data_size_gb, switch
    ):
        """
        通过交换机的通信仿真
        """
        # 1. 计算包数量
        packet_size_bytes = 1500  # MTU
        num_packets = ceil(data_size_gb * 1e9 / packet_size_bytes)

        # 2. 为每个包生成事件序列
        for packet_id in range(num_packets):
            t_arrive = current_time + packet_id × inter_packet_gap

            # Event 1: 包到达交换机
            self.add_event(Event(
                time=t_arrive,
                type=EventType.SWITCH_PACKET_ARRIVE,
                switch=switch,
                packet=packet_id
            ))

            # Event 2: 写入VOQ
            t_enqueue = t_arrive + switch.config.ingress_delay
            self.add_event(Event(
                time=t_enqueue,
                type=EventType.SWITCH_PACKET_ENQUEUE,
                switch=switch,
                packet=packet_id,
                voq=(src_port, dst_port)
            ))

        # 3. 调度器周期性仲裁
        self.schedule_switch_arbiter(switch)
```

### 8.3 延迟计算集成

**修改点**: `backend/llm_simulator/latency.py`

新增函数:
```python
def calc_switch_latency(
    data_size_gb: float,
    switch_config: SwitchConfig,
    src_chip: Chip,
    dst_chip: Chip,
    current_load: float = 0.0
) -> SwitchLatencyResult:
    """
    计算经由交换机的通信延迟

    Returns:
        SwitchLatencyResult:
            - ingress_latency_ns
            - queue_latency_ns
            - schedule_latency_ns
            - xbar_latency_ns
            - egress_latency_ns
            - serialization_latency_ns
            - total_latency_ns
    """
    packet_size_bytes = 1500
    num_packets = ceil(data_size_gb * 1e9 / packet_size_bytes)

    # 1. Ingress延迟
    clock_period_ns = 1e3 / switch_config.clock_frequency_mhz
    ingress_latency = switch_config.ingress_cycles * clock_period_ns

    # 2. 排队延迟 (基于当前负载)
    # 使用M/M/1模型近似
    rho = current_load  # 利用率
    avg_service_time = packet_size_bytes * 8 / (switch_config.port_bandwidth_gbps * 1e9)
    queue_latency = avg_service_time * rho / (1 - rho) if rho < 0.95 else float('inf')

    # 3. 调度延迟
    schedule_latency = switch_config.scheduling_cycles * clock_period_ns

    # 4. Crossbar延迟
    xbar_latency = 1 * clock_period_ns

    # 5. Egress延迟
    egress_latency = switch_config.egress_cycles * clock_period_ns

    # 6. 串行化延迟
    serialization_latency = packet_size_bytes * 8 / (switch_config.port_bandwidth_gbps * 1e9) * 1e9

    total = (ingress_latency + queue_latency + schedule_latency +
             xbar_latency + egress_latency + serialization_latency)

    return SwitchLatencyResult(
        ingress_latency_ns=ingress_latency,
        queue_latency_ns=queue_latency,
        schedule_latency_ns=schedule_latency,
        xbar_latency_ns=xbar_latency,
        egress_latency_ns=egress_latency,
        serialization_latency_ns=serialization_latency,
        total_latency_ns=total
    )
```

---

## 九、验证与测试

### 9.1 单元测试场景

**测试1: 零负载延迟验证**
```python
def test_zero_load_latency():
    """验证空闲交换机的最小延迟"""
    switch = create_test_switch(preset="tor_400g_32port")
    packet = create_test_packet(size=1500)

    latency = simulate_single_packet(switch, packet)

    expected = (
        switch.config.ingress_cycles +
        switch.config.scheduling_cycles +
        1 +  # xbar
        switch.config.egress_cycles
    ) * switch.config.clock_period_ns + serialization_delay

    assert abs(latency - expected) < 1.0  # 1ns误差范围
```

**测试2: VOQ HOL阻塞消除**
```python
def test_voq_eliminates_hol_blocking():
    """验证VOQ架构消除HOL阻塞"""
    switch = create_test_switch(num_ports=4)

    # 流量模式: 输入0,1,2均发往输出3 (热点)
    #          输入3发往输出0 (空闲)
    traffic = [
        (0, 3, 100),  # (src, dst, packets)
        (1, 3, 100),
        (2, 3, 100),
        (3, 0, 100)
    ]

    result = simulate_traffic(switch, traffic)

    # 输出0应该能及时接收来自输入3的流量
    # 不应被输入3中其他目标输出的包阻塞
    assert result.port_throughput[0] > 0.95
```

**测试3: iSLIP公平性**
```python
def test_islip_fairness():
    """验证iSLIP调度器的公平性"""
    switch = create_test_switch(
        num_ports=8,
        scheduler="islip",
        max_iterations=2
    )

    # 全排列流量 (所有输入到所有输出)
    traffic = generate_uniform_traffic(num_flows=64, duration=1000)

    result = simulate_traffic(switch, traffic)

    # 检查Jain公平性指数
    throughputs = [result.flow_throughput[i] for i in range(64)]
    fairness_index = jain_fairness_index(throughputs)

    assert fairness_index > 0.95  # 高公平性
```

### 9.2 集成测试场景

**场景1: Rack内Board间通信 (经由ToR)**
```yaml
# 测试配置
topology: 1 Pod, 1 Rack, 8 Boards, 64 Chips
switch: ToR 32-port 400G
traffic: AllReduce across 8 boards (TP=8)

预期结果:
- AllReduce延迟: < 10 μs (8 boards × ~1μs switch latency)
- 交换机吞吐: > 95% (11.52 Tbps / 12.8 Tbps)
- 无丢包
```

**场景2: Rack间通信 (经由汇聚交换机)**
```yaml
topology: 1 Pod, 4 Racks, 32 Boards, 256 Chips
switches:
  - 4× ToR (每Rack 1个)
  - 1× Aggregation (连接4个ToR)
traffic: Pipeline Parallelism (PP=4, 跨Rack)

预期结果:
- P2P延迟: < 5 μs (ToR + Agg + ToR)
- 端到端带宽: > 350 Gbps (400G链路的87.5%)
```

### 9.3 性能基准测试

**Benchmark 1: 吞吐-延迟曲线**
```python
def benchmark_throughput_latency():
    """测试不同负载下的延迟变化"""
    switch = create_test_switch()
    loads = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    results = []
    for load in loads:
        traffic = generate_poisson_traffic(load_ratio=load)
        result = simulate_traffic(switch, traffic)
        results.append({
            "load": load,
            "avg_latency_ns": result.latency.average,
            "p99_latency_ns": result.latency.p99
        })

    plot_throughput_latency_curve(results)
```

**Benchmark 2: 缓冲区占用分析**
```python
def benchmark_buffer_occupancy():
    """测试突发流量下的缓冲区行为"""
    switch = create_test_switch(buffer_mb=16)

    # 突发流量: 1秒稳定 → 0.1秒突发10倍 → 1秒稳定
    traffic = generate_burst_traffic(
        base_rate_gbps=100,
        burst_rate_gbps=1000,
        burst_duration_ms=100
    )

    result = simulate_traffic(switch, traffic)

    assert result.buffer.peak_usage_mb < 16  # 不溢出
    assert result.drop_count == 0
```

---

## 十、实施路线图

### Phase 1: 核心建模 (2周)

**目标**: 实现基本的交换机仿真能力

- [ ] 定义交换机配置数据结构 (`SwitchConfig`, `Switch`类)
- [ ] 实现简化延迟模型 (公式计算，无cycle级仿真)
- [ ] 扩展拓扑解析器支持交换机节点
- [ ] 实现 `calc_switch_latency()` 函数
- [ ] 单元测试: 零负载延迟验证

**交付物**:
- `backend/llm_simulator/switch.py` (新文件)
- `switch_presets/tor_400g_32port.yaml` (示例配置)
- 更新 `topology.py`, `latency.py`

### Phase 2: VOQ与调度器 (3周)

**目标**: 实现cycle-accurate的VOQ和iSLIP调度器

- [ ] 实现VOQ缓冲区数据结构
- [ ] 实现iSLIP仲裁算法 (`iSLIPScheduler`类)
- [ ] 集成到事件驱动仿真器 (`simulator.py`)
- [ ] 实现排队延迟动态计算
- [ ] 单元测试: VOQ HOL阻塞消除, iSLIP公平性

**交付物**:
- `backend/llm_simulator/schedulers/islip.py`
- 更新 `simulator.py` 支持交换机事件
- 测试用例: `tests/test_switch_scheduling.py`

### Phase 3: 缓冲区管理与流控 (2周)

**目标**: 实现动态缓冲分配和Credit流控

- [ ] 实现共享缓冲池 + 动态阈值算法
- [ ] 实现Credit-based流控机制
- [ ] 支持Tail Drop和ECN Marking策略
- [ ] 单元测试: 缓冲溢出处理, 反压机制

**交付物**:
- `backend/llm_simulator/flow_control.py`
- 更新交换机配置支持缓冲管理参数
- 测试用例: `tests/test_buffer_management.py`

### Phase 4: 性能指标与可视化 (2周)

**目标**: 输出交换机性能指标和可视化

- [ ] 实现交换机级性能统计收集
- [ ] 扩展Gantt图显示交换机活动
- [ ] 实现路径延迟分解输出
- [ ] 集成测试: Rack内/Rack间通信场景

**交付物**:
- 更新 `gantt.py` 支持交换机可视化
- 更新API返回交换机指标
- 前端更新: 显示交换机性能面板

### Phase 5: 验证与优化 (2周)

**目标**: 全面测试和性能优化

- [ ] 完成所有Benchmark测试
- [ ] 与理论值对比验证 (吞吐量, 延迟)
- [ ] 性能优化 (仿真速度)
- [ ] 文档完善 (用户手册, API文档)

**交付物**:
- 完整测试报告
- 性能优化报告
- 用户配置指南

---

## 十一、参考资料

### 学术论文

1. **iSLIP算法**
   - McKeown, N. (1999). "The iSLIP scheduling algorithm for input-queued switches." IEEE/ACM Transactions on Networking, 7(2), 188-201.
   - 经典论文，详细描述iSLIP算法及性能分析

2. **Virtual Output Queuing**
   - Tamir, Y., & Frazier, G. (1992). "Dynamically-allocated multi-queue buffers for VLSI communication switches." IEEE Transactions on Computers, 41(6), 725-737.
   - VOQ架构的理论基础

3. **HOL阻塞分析**
   - Karol, M., Hluchyj, M., & Morgan, S. (1987). "Input versus output queueing on a space-division packet switch." IEEE Transactions on Communications, 35(12), 1347-1356.
   - 证明输入队列的58.6%吞吐上限

4. **Buffered Crossbar**
   - Nabeshima, M. (2000). "Performance evaluation of a combined input-and crosspoint-queued switch." IEICE Transactions on Communications, E83-B(3), 737-741.
   - 缓冲crossbar的性能建模

### 技术文档

5. **Broadcom Tomahawk白皮书**
   - https://elegantnetwork.github.io/posts/A-Summary-of-Network-ASICs/
   - 商用交换机ASIC的微架构分析

6. **BookSim 2.0用户手册**
   - https://github.com/mikedw/Booksim-2.0
   - Cycle-accurate仿真器实现参考

7. **gem5 Garnet文档**
   - https://www.gem5.org/documentation/general_docs/ruby/garnet-2/
   - 另一种cycle-accurate网络模型

### 在线资源

8. **ipSpace.net博客**
   - https://blog.ipspace.net/2022/06/data-center-switching-asic-tradeoffs/
   - 数据中心交换机技术深度分析

9. **Packet Pushers: Understanding Crossbar Fabrics**
   - https://packetpushers.net/blog/understanding-crossbar-fabrics-the-islip-algorithm/
   - Crossbar和iSLIP的工程实践

---

## 附录A: 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| **VOQ** | Virtual Output Queuing | 虚拟输出队列，为每个输出端口维护独立队列 |
| **HOL阻塞** | Head-of-Line Blocking | 队首包阻塞导致的性能下降 |
| **Crossbar** | Crossbar Switching Fabric | N×N全交叉交换矩阵 |
| **iSLIP** | Iterative SLIP | 迭代轮询匹配调度算法 |
| **Flit** | Flow Control Unit | 流控单元，最小传输单位 |
| **Credit** | Credit-Based Flow Control | 基于信用的流控机制 |
| **ToR** | Top-of-Rack | 机架顶部交换机 |
| **Backplane** | Backplane Bandwidth | 背板带宽，交换机内部总带宽 |
| **Speedup** | Internal Speedup | 内部加速比，相对端口速率 |

---

## 附录B: 常见问题 (FAQ)

**Q1: 为什么选择VOQ而不是输出队列？**

A: 输出队列理论性能最优(100%吞吐)，但需要N倍速内部结构，硬件成本极高。VOQ只需1倍速(或1.2倍)即可达到近100%吞吐。

**Q2: iSLIP的迭代次数如何选择？**

A:
- 1次迭代: 吞吐~63%, 延迟最低
- 2次迭代: 吞吐~99%, 延迟适中 (推荐)
- 4次迭代: 吞吐~99.9%, 延迟较高

**Q3: 交换机延迟 vs 芯片直连延迟？**

A:
- 芯片直连(NVLink): 0.2-0.5 μs
- ToR交换机: 0.5-1 μs (零负载), 可能数μs (高负载)
- 汇聚交换机: 1-3 μs

设计建议: 关键路径(如TP组内AllReduce)使用直连，跨Rack通信使用交换机。

**Q4: 如何处理交换机丢包？**

A:
1. 短期: 增大缓冲深度
2. 长期: 上游限速(credit流控)或使用ECN标记反馈拥塞

**Q5: 支持异构交换机吗？**

A: 支持。可在同一拓扑中配置不同规格的ToR和汇聚交换机，通过preset_id引用不同配置。

---

## 附录C: 配置示例完整版

```yaml
# 完整示例: 4-Rack系统，带ToR和汇聚交换机
name: "P1-R4-B32-C256-hierarchical-switching"
pod_count: 1
racks_per_pod: 4

# 交换机定义
switches:
  # ToR交换机 (每Rack 1个)
  - name: "tor_rack0"
    type: "top_of_rack"
    location: {pod: 0, rack: 0}
    preset_id: "tor_400g_32port"

  - name: "tor_rack1"
    type: "top_of_rack"
    location: {pod: 0, rack: 1}
    preset_id: "tor_400g_32port"

  - name: "tor_rack2"
    type: "top_of_rack"
    location: {pod: 0, rack: 2}
    preset_id: "tor_400g_32port"

  - name: "tor_rack3"
    type: "top_of_rack"
    location: {pod: 0, rack: 3}
    preset_id: "tor_400g_32port"

  # 汇聚交换机 (连接4个ToR)
  - name: "agg_pod0"
    type: "aggregation"
    location: {pod: 0}
    preset_id: "agg_800g_64port"

# Rack配置
rack_config:
  boards:
    - chips:
        - name: "SG2262"
          count: 8
      count: 8  # 每Rack 8个Board

# 硬件参数
hardware_params:
  chips:
    SG2262:
      name: "SG2262"
      compute_tflops_fp8: 768
      memory_capacity_gb: 64
      memory_bandwidth_gbps: 12000
      cube_m: 16
      cube_k: 32
      cube_n: 8
      sram_size_kb: 2048

  switches:
    tor_400g_32port:
      preset_id: "tor_400g_32port"
    agg_800g_64port:
      preset_id: "agg_800g_64port"

  interconnect:
    # 芯片间NVLink (直连)
    c2c:
      bandwidth_gbps: 448
      latency_us: 0.2

    # Board间 (经由ToR)
    b2b:
      type: "via_switch"
      bandwidth_gbps: 400
      latency_us: 0.001  # 基础1μs + 动态排队延迟

    # Rack间 (经由汇聚)
    r2r:
      type: "via_switch"
      bandwidth_gbps: 800
      latency_us: 0.003

# 通信延迟配置
comm_latency_config:
  allreduce_algorithm: "ring"
  enable_compute_comm_overlap: true
  network_efficiency: 0.85
```

---

**文档结束**

如有问题或需要澄清，请联系文档维护者。
