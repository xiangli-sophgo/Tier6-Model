# AI 场景网络需求与协议

## 一、AI 训练/推理网络的特殊需求

### 1.1 与传统数据中心的差异

| 维度 | 传统数据中心 | AI 训练/推理集群 |
|------|-------------|----------------|
| **流量模式** | 多对多、随机 | 集合通信、周期性、同步 |
| **数据量** | KB-MB 级 | GB 级 (模型梯度、激活值) |
| **延迟容忍** | 10-100 ms | < 5 ms |
| **丢包容忍** | 可容忍 (TCP 重传) | **零容忍 (RDMA 断连)** |
| **拥塞特征** | 偶发、局部 | 周期性突发 (AllReduce) |
| **关键指标** | 吞吐量、可用性 | **尾延迟 (P99)、无丢包** |

### 1.2 三种集合通信模式

#### AllReduce (梯度同步 -- DP/TP)

```
训练迭代流程:
  Forward Pass -> Backward Pass -> 计算本地梯度
       |
  === AllReduce ===
  所有 GPU 交换梯度，每个 GPU 得到平均梯度
       |
  更新模型参数 -> 下一个 iteration

通信特征:
  - 全对全通信，周期性 (每个 iteration)
  - 同步阻塞 (计算等通信完成)
  - 数据量: 模型参数量 x 2 (FP16/BF16)
    DeepSeek-V3 671B: ~1.3 TB
```

**Ring AllReduce**: 通信量 = 2(N-1)/N * Data_Size, 延迟 = O(N)
**Tree AllReduce**: 通信量 = 2 * Data_Size, 延迟 = O(log N)

#### All-to-All (MoE 专家路由 -- EP)

```
MoE Layer:
  输入 Tokens -> Router 决策 (每个 Token 选 Top-K 专家)
       |
  === All-to-All Scatter === (把 Token 分发到对应 GPU 的专家)
       |
  各 GPU 上的专家并行计算
       |
  === All-to-All Gather === (把结果收集回原 GPU)

通信特征:
  - 全对全排列通信
  - Token 分布不均 (热门专家 vs 冷门专家)
  - 数据量: Hidden_Size x Active_Tokens
```

#### P2P (Pipeline 并行 -- PP)

```
GPU0 (Layer 0-10) -> GPU1 (Layer 11-20) -> GPU2 (Layer 21-30)
     |                    |                    |
 Forward 激活值       Forward 激活值       Forward 激活值
 Backward 梯度 <---  Backward 梯度 <---  Backward 梯度

通信特征:
  - 点对点 (相邻 stage)
  - 流水线化 (Micro-batch 并行)
  - 频繁 (每个 Micro-batch)
```

### 1.3 三大核心需求

**需求 1: 无损网络** -- 零丢包 (< 10^-12 丢包率)

RDMA 网络中丢包会导致连接断开 (RNR - Receiver Not Ready)，进而导致整个训练 Job 失败。

**需求 2: 极低延迟**
- ToR 交换机: < 1 us
- Spine 交换机: < 3 us
- 端到端 (GPU 到 GPU): < 10 us

**需求 3: 高带宽与突发吸收**

AllReduce 开始时所有 GPU 同时发送，瞬时流量远超平均流量，需要深缓冲吸收。

---

## 二、RDMA 与 RoCE v2

### 2.1 RDMA 原理

```
传统 TCP/IP:
  应用 -> 系统调用 -> 内核 TCP/IP 栈 -> 网卡驱动 -> 网卡
  延迟: 10-50 us, CPU 开销: 每 GB 消耗 1 个 CPU 核心

RDMA (远程直接内存访问):
  应用 -> RDMA Verbs -> RNIC -> 网络 -> RNIC -> 直接 DMA 到 GPU 内存
  延迟: 1-3 us, CPU 开销: 接近 0
```

关键优势: 零拷贝、内核旁路、CPU 卸载、低延迟

### 2.2 RoCE v2 协议栈

```
+-------------------------------------------+
|         应用层 (PyTorch, NCCL)            |
+-------------------------------------------+
|       IB Verbs API (ibv_post_send)        |
+-------------------------------------------+
|       InfiniBand Transport Layer          |
|  (Reliable Connection, Queue Pairs)       |
+-------------------------------------------+
|            UDP (Port 4791)                |  <-- RoCE v2
+-------------------------------------------+
|            IP (可路由)                     |  <-- RoCE v2
+-------------------------------------------+
|       Ethernet (with PFC + ECN)           |
+-------------------------------------------+
```

### 2.3 RoCE v2 数据包结构

```
+----------+----------+----------+
| Eth Hdr  | IP Header| UDP Hdr  |
| 14 bytes | 20 bytes | 8 bytes  |
+----------+----------+----------+
+----------------------------------------------+
|     InfiniBand Transport Header (BTH)        |
|                  12 bytes                     |
|  OpCode: RDMA_WRITE / RDMA_READ / SEND       |
|  PSN: 包序号 (24-bit)                        |
|  QP: 队列对标识                               |
+----------------------------------------------+
+----------------------------------------------+
|              RDMA Payload                     |
|              0 - 4096 bytes                   |
+----------------------------------------------+
+----------+----------+
|   ICRC   |   FCS    |
| 4 bytes  | 4 bytes  |
+----------+----------+
```

关键字段:
- **PSN (Packet Sequence Number)**: 接收端检测乱序/丢失，连续丢失导致连接断开
- **QP (Queue Pair)**: 唯一标识 RDMA 连接
- **ICRC (Invariant CRC)**: 端到端校验，检测中间交换机的数据损坏

---

## 三、无损网络三大支柱

### 3.1 PFC (Priority Flow Control, IEEE 802.1Qbb)

**目标**: 避免接收端缓冲溢出导致丢包

```
接收端缓冲区使用率 > 85%
       |
  发送 PFC PAUSE 帧 (指定优先级)
       |
交换机收到 PAUSE 帧
       |
  暂停向该端口发送特定优先级的流量
       |
  接收端处理积压数据，缓冲降低
       |
  停止 PAUSE 或发送 RESUME
       |
  恢复发送
```

**AI 训练优先级分配**:
- Priority 3: RDMA 数据流量 (启用 PFC)
- Priority 5: RoCE 控制流量 (启用 PFC)
- Priority 0: 管理流量 (Best-Effort，不启用 PFC)

**PFC 的问题**:

1. **PFC 死锁**: 循环依赖导致网络挂起。解决: PFC Watchdog 超时恢复、避免环路拓扑
2. **PFC 风暴**: 单点触发的 PAUSE 雪崩传播到整个子网。解决: 限制传播深度、使用 ECN 主动避免拥塞

### 3.2 ECN (Explicit Congestion Notification, RFC 3168)

**目标**: 在拥塞发生前主动降速，避免丢包和 PFC

```
Step 1: 交换机检测拥塞
  端口队列深度 > ECN 阈值 (如 50%)
       |
  标记 IP 包的 ECN 字段: CE (Congestion Experienced)
       |
  继续转发包 (不丢弃)

Step 2: 接收端检测标记
  GPU B 收到带 CE 标记的 IP 包
       |
  RDMA 网卡生成 CNP (Congestion Notification Packet)
       |
  发送 CNP 给发送端

Step 3: 发送端降速 (DCQCN 算法)
  GPU A 收到 CNP
       |
  降低发送速率 (如从 100 Gbps 降到 80 Gbps)
       |
  逐渐探测增加速率
```

**ECN 阈值配置**:

```
队列深度:
  0% -------- 50% --------- 80% --------- 85% --- 100%
  |           |              |              |        |
 空闲     ECN标记开始     全部标记ECN    PFC触发   丢包
         (概率线性增长)
```

### 3.3 DCQCN (Data Center QCN) 算法

```
收到 CNP 时:
  RC = RC * (1 - alpha/2)   # 快速降速
  RT = RT * (1 - alpha)     # 目标速率降低
  alpha ~= 0.5

未收到 CNP 时 (恢复阶段):
  每个 RTT: RC = RC + alpha_i  # 加性增
  如果 RC > RT: RC = (RC + RT) / 2  # 快速增

参数:
  alpha:   降速因子 (典型 0.5)
  alpha_i: 增速因子 (典型 40 Mbps)
  MIN_RATE: 最低速率 (典型 100 Mbps)
```

### 3.4 深缓冲

**微突发吸收计算**:

```
8 个输入端口 x 400 Gbps = 3200 Gbps
1 个输出端口 = 400 Gbps
突发持续时间 = 10 us

需要缓冲 = (3200 - 400) Gbps x 10 us = 3.5 MB
```

**实际交换机缓冲深度**:

| 类型 | 缓冲深度 | 适用规模 |
|------|---------|---------|
| 通用数据中心 | 16-32 MB | 小规模 |
| AI 优化 ToR | 64-128 MB | 128-256 GPU |
| AI 优化 Spine | 128-256 MB | 512-1024 GPU |

---

## 四、AI 交换机特殊行为

### 4.1 SHARP (网内聚合加速)

```
传统 Ring AllReduce (256 GPU):
  延迟 = 255 * (Data/256) / Bandwidth ~= 26 ms

SHARP 网内聚合:
  GPU0-3 ---> Switch 内 SHARP Engine (SUM) ---> 广播结果
  延迟 = log2(256) * Switch_Latency ~= 8 us (理论)
```

- 需要专用硬件 (NVIDIA Spectrum 系列)
- 只支持特定拓扑 (Fat-Tree)
- 支持操作: SUM, MIN, MAX, AND, OR (FP16, BF16, FP32, INT32)

### 4.2 自适应路由

传统 ECMP: Hash(5-tuple) % N 选择路径，一旦选定不变
AI 交换机: 实时监控每条路径的队列深度/延迟，动态选择最优路径

```
Path 1: 队列 80%, 延迟 15 us -> Score 20
Path 2: 队列 20%, 延迟 2 us  -> Score 90 <-- 选择
Path 3: 队列 50%, 延迟 8 us  -> Score 50
```

使用 Flowlet (同一 Flow 内间隔 > Threshold 的包组) 为粒度，避免乱序。

### 4.3 拥塞树抑制

Incast 导致单点拥塞，PFC 向上游传播形成 "拥塞树"，阻塞无关流量。

缓解策略:
1. **选择性丢包**: 主动丢弃低优先级包，避免 PFC 传播
2. **动态缓冲调整**: 从空闲端口 "借用" 缓冲
3. **早期 ECN 反馈**: 低阈值标记 ECN，提前降速

### 4.4 INT 遥测 (In-band Network Telemetry)

在数据包中嵌入每跳路径信息:

```
每个交换机追加:
  - Switch ID
  - Ingress/Egress Port
  - Queue Depth
  - Timestamp
  - 在本交换机的延迟
```

用于故障定位、路径追踪、逐跳延迟分解。

---

## 五、RDMA 数据包在交换机中的完整生命周期

### 5.1 发送端

```
Step 1: NCCL 调用 ncclSend()
Step 2: RNIC 构造 RoCE v2 包，从 GPU 内存 DMA 读取 (GPUDirect RDMA)
Step 3: 设置优先级 PCP=3，计算 ICRC
Step 4: 发送到物理端口
```

### 5.2 交换机处理 (Cycle 级)

```
Cycle 0-10:  物理层接收 (光信号 -> 电信号 -> SerDes 解串)
Cycle 10-12: Ingress 处理 (解析 Eth/IP/UDP, 识别 RoCE, 分类到优先级 3)
Cycle 12-20: 队列管理 (选择 VOQ[in][out][pri=3], 检查缓冲空间, 入队)
Cycle 20-30: iSLIP 调度 (Request -> Grant -> Accept)
Cycle 30-35: ECN 标记 (检查队列深度，超阈值则标记 CE)
Cycle 35-40: Crossbar 传输 (输入 -> 输出)
Cycle 40-N:  Egress 发送 (串行化到物理端口, 监控 Credit)

典型总延迟: 150-600 ns (单跳)
```

### 5.3 接收端

```
Step 1: 解析 RoCE v2 包
Step 2: ICRC 校验 (不匹配则丢弃并断连)
Step 3: PSN 检查 (乱序/丢失则可能断连)
Step 4: ECN 检查 (如果 CE, 生成 CNP 发回发送端)
Step 5: DMA 写入 GPU 内存 (GPUDirect RDMA)
Step 6: 最后一个包完成后通知应用层
```

---

## 六、AI 交换机核心特征总结

### 必需特性

1. RoCE v2 支持 -- RDMA 传输
2. PFC -- 无损网络
3. ECN + DCQCN -- 主动拥塞控制
4. 深缓冲 (64-256 MB) -- 吸收突发
5. 低延迟 (< 1 us) -- Cut-Through 转发
6. 高带宽 (400G/800G) -- 大规模并行

### 增强特性

- SHARP (集合通信加速)
- 自适应路由
- INT 遥测
- 拥塞树抑制
- PFC 死锁检测

### 与通用交换机的本质区别

| 维度 | 通用交换机 | AI 交换机 |
|------|-----------|---------|
| 设计目标 | 灵活性 | 性能、无损 |
| 丢包处理 | 可容忍 | 零容忍 |
| 延迟 | 毫秒级 | 微秒级 |
| 流控 | 可选 | 必需 (PFC) |
| 缓冲 | 浅 (MB) | 深 (数十-数百 MB) |
| 协议 | TCP/IP | RDMA (RoCE) |
| 拥塞控制 | WRED | ECN + DCQCN |
| 优化目标 | 平均吞吐 | 尾延迟 (P99) |
