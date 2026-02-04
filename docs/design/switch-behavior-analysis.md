# 以太网交换机行为详细分析

## 文档信息

- **版本**: v1.0
- **创建日期**: 2026-02-04
- **目标**: 详细分析以太网交换机的内部工作机制和数据包转发行为

---

## 一、以太网交换机基础

### 1.1 交换机的角色

以太网交换机工作在**OSI第2层（数据链路层）**，主要功能：
- 根据**MAC地址**转发以太网帧
- 维护**MAC地址表**（也称转发表、CAM表）
- 隔离冲突域，每个端口是独立的冲突域
- 提供**全双工通信**，同时发送和接收

### 1.2 与Hub的区别

| 特性 | Hub（集线器） | Switch（交换机） |
|------|--------------|----------------|
| 工作层次 | 物理层（L1） | 数据链路层（L2） |
| 转发方式 | 广播到所有端口 | 根据MAC地址选择性转发 |
| 冲突域 | 共享冲突域 | 每端口独立冲突域 |
| 带宽 | 所有端口共享 | 每端口独享 |
| 智能性 | 无 | MAC学习、过滤 |

---

## 二、以太网帧结构

### 2.1 标准以太网帧格式（IEEE 802.3）

```
┌─────────────┬─────────────┬──────┬──────────┬─────────────┬─────┐
│  Preamble   │     SFD     │ Dest │  Source  │ Type/Length │ ... │
│   7 bytes   │   1 byte    │ MAC  │   MAC    │   2 bytes   │     │
│             │             │6 byte│  6 bytes │             │     │
└─────────────┴─────────────┴──────┴──────────┴─────────────┴─────┘
        ↓              ↓         ↓        ↓          ↓
    同步信号      帧起始定界   目标地址   源地址    协议类型
    (不进入交换机处理)

┌──────────────────────────────────────┬─────────┬─────────────┐
│            Payload                   │   FCS   │     IFG     │
│         46-1500 bytes                │ 4 bytes │  12 bytes   │
│                                      │  CRC    │ (帧间隙)    │
└──────────────────────────────────────┴─────────┴─────────────┘
              ↓                            ↓           ↓
         实际数据                      校验和      帧间隔
```

**关键字段解析**：

1. **Destination MAC (6 bytes)**
   - 目标设备的物理地址
   - 格式：`AA:BB:CC:DD:EE:FF`
   - 特殊地址：
     - `FF:FF:FF:FF:FF:FF` - 广播地址
     - `01:00:5E:xx:xx:xx` - 多播地址（IPv4）

2. **Source MAC (6 bytes)**
   - 发送设备的物理地址
   - 交换机用于学习的关键信息

3. **EtherType / Length (2 bytes)**
   - `≥ 0x0600` (1536): 表示上层协议类型
     - `0x0800`: IPv4
     - `0x0806`: ARP
     - `0x86DD`: IPv6
     - `0x8100`: 802.1Q VLAN标签
   - `< 0x0600`: 表示Payload长度

4. **FCS (Frame Check Sequence, 4 bytes)**
   - CRC-32校验和
   - 用于检测传输错误

### 2.2 VLAN标签帧（IEEE 802.1Q）

```
┌──────┬──────┬────────────┬──────┬─────────────┬─────────┬─────┐
│ Dest │ Src  │  802.1Q    │ Type │   Payload   │   FCS   │ IFG │
│ MAC  │ MAC  │   Tag      │      │             │         │     │
│6 byte│6 byte│  4 bytes   │2 byte│ 46-1500 B   │ 4 bytes │12 B │
└──────┴──────┴────────────┴──────┴─────────────┴─────────┴─────┘
                    ↓
          ┌─────────────────────┐
          │ TPID (0x8100, 2B)  │ - Tag Protocol ID
          ├─────────┬───────────┤
          │ PCP(3b) │ DEI (1b) │ - Priority, Drop Eligible
          ├─────────┴───────────┤
          │   VLAN ID (12 bits) │ - 0-4095
          └─────────────────────┘
```

**VLAN标签字段**：
- **PCP (Priority Code Point, 3 bits)**: QoS优先级（0-7）
- **DEI (Drop Eligible Indicator, 1 bit)**: 拥塞时可丢弃标记
- **VLAN ID (12 bits)**: VLAN编号（0-4095，实际可用1-4094）

---

## 三、交换机核心行为：MAC地址学习

### 3.1 MAC地址表结构

交换机内部维护一张**MAC地址表**（也称CAM表 - Content Addressable Memory）：

```
┌──────────────────┬──────────┬─────────────┬──────────┬─────────┐
│   MAC Address    │   Port   │   VLAN ID   │   Age    │  Type   │
├──────────────────┼──────────┼─────────────┼──────────┼─────────┤
│ AA:BB:CC:11:22:33│    1     │     100     │   120s   │ Dynamic │
│ DD:EE:FF:44:55:66│    5     │     100     │    85s   │ Dynamic │
│ 11:22:33:AA:BB:CC│    8     │     200     │     0s   │ Static  │
│ FF:FF:FF:FF:FF:FF│   All    │     All     │    -     │ Special │
└──────────────────┴──────────┴─────────────┴──────────┴─────────┘
```

**表项字段**：
- **MAC Address**: 48位物理地址
- **Port**: 该MAC地址对应的出端口号
- **VLAN ID**: 所属VLAN（如果支持）
- **Age**: 老化时间（典型：300秒）
- **Type**:
  - Dynamic: 自动学习的，会老化
  - Static: 手工配置的，不老化

### 3.2 学习过程（Learning）

**步骤详解**：

```
┌─────────────────────────────────────────────────────────────┐
│                   Frame到达端口3                            │
│   Source MAC: AA:BB:CC:11:22:33                            │
│   Dest MAC:   DD:EE:FF:44:55:66                            │
└─────────────────────────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │  Step 1: 学习Source MAC        │
         └───────────────────────────────┘
                         ↓
    查找MAC表中是否已有 AA:BB:CC:11:22:33
              ↙                    ↘
         已存在                    不存在
              ↓                        ↓
    更新表项:                   新建表项:
    - Port = 3                 - MAC = AA:BB:CC:11:22:33
    - 刷新Age计时器             - Port = 3
                               - VLAN = 入端口的VLAN
                               - Age = 0
                               - Type = Dynamic
```

**学习规则**：
1. **只学习Source MAC**，不学习Destination MAC
2. 如果Source MAC已存在但端口不同：
   - 更新端口号（设备可能移动了）
   - 刷新老化计时器
3. 如果表满了：
   - 删除最老的Dynamic表项
   - 或拒绝学习新地址（取决于策略）

### 3.3 老化机制（Aging）

**目的**：删除不再活跃的MAC地址，释放表空间

**老化计时器**：
- 每个Dynamic表项有独立的Age计数器
- 每次该MAC作为Source出现时，Age重置为0
- Age超过阈值（如300秒）→ 删除表项

**为什么需要老化**：
- 设备可能断开连接
- 设备可能移动到其他端口
- MAC地址表空间有限（典型：8K-128K表项）

---

## 四、数据包转发决策

### 4.1 转发流程总览

```
Frame到达 → 学习Source MAC → 查找Dest MAC → 转发决策
                ↓                  ↓              ↓
            更新MAC表          查CAM表         Unicast
                                 ↓             Multicast
                            找到 / 未找到       Broadcast
                              ↓       ↓           ↓
                           Unicast  Flooding   Flooding
                           (单播)    (泛洪)     (泛洪)
```

### 4.2 三种转发场景

#### **场景1: 单播 (Unicast)**

**条件**：Destination MAC在MAC表中找到

```
Frame: Dest MAC = DD:EE:FF:44:55:66
          ↓
查MAC表: DD:EE:FF:44:55:66 → Port 5
          ↓
转发决策:
  - 如果入端口 == 5 → 丢弃（不转发回同一端口）
  - 如果入端口 ≠ 5 → 仅从端口5转发出去
```

**关键特性**：
- **选择性转发**：只发往目标端口
- **带宽隔离**：其他端口不受影响
- **回路抑制**：不发回入端口

#### **场景2: 未知单播泛洪 (Unknown Unicast Flooding)**

**条件**：Destination MAC不在MAC表中

```
Frame: Dest MAC = 99:88:77:66:55:44 (表中没有)
          ↓
查MAC表: 未找到
          ↓
转发决策:
  → 泛洪到所有端口（除了入端口）
  → 类似广播行为
```

**为什么需要泛洪**：
- 目标设备存在但交换机尚未学习到其MAC
- 通过泛洪，目标设备会收到帧并回复
- 回复帧会触发交换机学习该MAC

**性能影响**：
- 消耗所有端口带宽
- 可能导致广播风暴（如果有环路）

#### **场景3: 广播/多播 (Broadcast/Multicast)**

**广播条件**：Dest MAC = `FF:FF:FF:FF:FF:FF`

```
Frame: Dest MAC = FF:FF:FF:FF:FF:FF
          ↓
识别为广播帧
          ↓
转发决策:
  → 泛洪到同一VLAN的所有端口（除入端口）
```

**多播条件**：Dest MAC最高字节的最低位 = 1
- IPv4多播：`01:00:5E:xx:xx:xx`
- IPv6多播：`33:33:xx:xx:xx:xx`

**多播处理**：
- 如果支持**IGMP Snooping**：
  - 只转发给订阅该多播组的端口
- 如果不支持：
  - 泛洪到所有端口（类似广播）

### 4.3 VLAN隔离

**VLAN的作用**：逻辑分割广播域

```
交换机端口分配:
  Port 1-8:  VLAN 100 (销售部)
  Port 9-16: VLAN 200 (技术部)

Frame从Port 3到达 (VLAN 100):
  Dest MAC = FF:FF:FF:FF:FF:FF (广播)
          ↓
转发决策:
  → 只泛洪到 Port 1-8 (同VLAN)
  → Port 9-16 不会收到 (不同VLAN)
```

**VLAN间通信**：
- 需要**三层设备**（路由器或三层交换机）
- 交换机本身不转发跨VLAN流量

---

## 五、转发模式详解

### 5.1 存储转发 (Store-and-Forward)

**工作原理**：
```
Frame到达 → 完整接收到缓冲区 → CRC校验 → 查表转发 → 发送
            (接收全部字节)      (检查FCS)   (查MAC表)
```

**时序分析**：
```
入端口接收:  |===========帧接收===========| (1500 bytes @ 1 Gbps = 12 μs)
                                         ↓
缓冲存储:                                |存储| (几十ns)
                                              ↓
CRC校验:                                      |校验| (几十ns)
                                                   ↓
查表决策:                                          |查表| (1 cycle)
                                                        ↓
出端口发送:                                             |===========帧发送===========|
```

**优点**：
- ✅ **错误检测**：丢弃损坏的帧（CRC错误）
- ✅ **速率适配**：入端口1G，出端口100M → 缓冲调节
- ✅ **完整处理**：可检查整个帧（QoS、ACL）

**缺点**：
- ❌ **延迟高**：必须接收完整帧
  - 1500字节 @ 1 Gbps = 12 μs
  - 9000字节 @ 10 Gbps = 7.2 μs

**适用场景**：
- 企业网络（需要错误过滤）
- 速率不匹配的端口
- 需要深度包检测（DPI）

### 5.2 直通转发 (Cut-Through)

**工作原理**：
```
Frame到达 → 读取前14字节 → 查表决策 → 立即转发 → 边收边发
            (Dest MAC)      (查MAC表)
```

**时序分析**：
```
入端口接收:  |前14字节|==========剩余部分==========|
                 ↓
查表决策:        |查表|
                     ↓
出端口发送:          |前14字节|==========剩余部分==========|
                      ↑
                  重叠开始（流水线）
```

**延迟计算**：
- 只需等待**Destination MAC接收完成** = 6字节
- @ 10 Gbps: 6 bytes = 4.8 ns
- @ 1 Gbps: 6 bytes = 48 ns

**优点**：
- ✅ **超低延迟**：纳秒级（vs 存储转发的微秒级）
- ✅ **流水线**：接收和发送重叠

**缺点**：
- ❌ **无错误检测**：损坏的帧也会转发
- ❌ **速率限制**：出端口速率必须 ≥ 入端口
- ❌ **无速率适配**：不能从1G转到100M

**适用场景**：
- 数据中心（追求极致低延迟）
- 高频交易（HFT）
- 均匀速率网络（所有端口同速）

### 5.3 无碎片转发 (Fragment-Free)

**工作原理**：折中方案
```
Frame到达 → 接收前64字节 → 查表转发 → 边收边发
            (最小帧长)
```

**为什么是64字节**：
- 以太网最小帧长 = 64字节
- 冲突（Collision）只发生在前64字节
- 接收64字节后，帧不会再是碎片（Runt Frame）

**优点**：
- ✅ 过滤掉碎片帧（< 64字节）
- ✅ 延迟低于存储转发

**缺点**：
- ⚠️ 无法检测CRC错误（如果错误在64字节之后）

### 5.4 三种模式对比

| 特性 | Store-and-Forward | Fragment-Free | Cut-Through |
|------|------------------|---------------|-------------|
| **延迟** | 高（12 μs @ 1G） | 中（512 ns @ 1G） | 低（48 ns @ 1G） |
| **错误检测** | 完整CRC校验 | 过滤碎片 | 无 |
| **速率适配** | 支持 | 有限支持 | 不支持 |
| **内存需求** | 大（完整帧） | 小（64字节） | 最小（14字节） |
| **适用网络** | 企业、混合速率 | 通用 | 数据中心、同速 |

---

## 六、缓冲和排队机制

### 6.1 缓冲架构类型

#### **输入缓冲 (Input Buffering)**

```
Port 1 → [Queue] ─┐
Port 2 → [Queue] ─┤
Port 3 → [Queue] ─┼→ Crossbar → Output Ports
Port 4 → [Queue] ─┤
Port N → [Queue] ─┘
```

**特点**：
- 每个入端口有独立缓冲队列
- 简单实现，内存需求低
- **问题**：Head-of-Line (HOL) 阻塞
  - 队首帧等待忙碌输出时，阻塞后续所有帧
  - 理论吞吐上限：58.6%

#### **输出缓冲 (Output Buffering)**

```
                         ┌→ [Queue] → Port 1
                         ├→ [Queue] → Port 2
Input Ports → Crossbar ──┼→ [Queue] → Port 3
                         ├→ [Queue] → Port 4
                         └→ [Queue] → Port N
```

**特点**：
- 每个出端口有独立缓冲队列
- **理论性能最优**：100%吞吐量
- **代价高**：需要N倍速的内部交换矩阵
  - 最坏情况：N个输入同时发往1个输出

#### **虚拟输出队列 (VOQ - Virtual Output Queuing)**

```
Port 1 → [VOQ to Port 1] ─┐
         [VOQ to Port 2]   │
         [VOQ to Port 3]   │
         ...               │
         [VOQ to Port N]   ├→ Crossbar → Output Ports
                           │
Port N → [VOQ to Port 1]   │
         [VOQ to Port 2]   │
         ...               │
         [VOQ to Port N] ──┘
```

**特点**：
- 每个输入为每个输出维护独立队列（N²个队列）
- **消除HOL阻塞**
- 吞吐量接近100%（配合iSLIP调度器）
- **数据中心交换机主流架构**

### 6.2 共享缓冲池 (Shared Buffer Pool)

**架构**：
```
              ┌─────────────────────────┐
              │   Shared Memory Pool    │
              │      (64 MB SRAM)       │
              └─────────────────────────┘
                   ↑    ↑    ↑    ↑
                   │    │    │    │
            ┌──────┴────┴────┴────┴──────┐
            │  Dynamic Allocation Logic  │
            └──────┬────┬────┬────┬──────┘
                   ↓    ↓    ↓    ↓
                 Port1 Port2 ... PortN
```

**动态分配策略**：

1. **静态保留 (Static Reservation)**
   ```
   每个端口保留: Reserved_Size
   例如: 64 MB ÷ 32 ports = 2 MB/port
   ```

2. **动态阈值 (Dynamic Threshold)**
   ```
   可用共享空间 = Total_Buffer - Current_Usage
   端口可额外使用 = α × 可用共享空间

   α = 1.0: 平等分配
   α > 1.0: 允许临时超分配（吸收突发）
   α < 1.0: 保守分配（防止饥饿）
   ```

3. **优先级差异分配**
   ```
   高优先级队列: α_high = 2.0
   低优先级队列: α_low = 0.5

   → 高优先级获得更多缓冲空间
   ```

### 6.3 队列调度算法

#### **FIFO (First-In-First-Out)**

```
队列: [Pkt1] → [Pkt2] → [Pkt3] → [Pkt4] → Output
        ↑                              ↓
      入队顺序                      出队顺序
```

- 最简单，无优先级区分
- 适用于尽力而为（Best-Effort）流量

#### **严格优先级 (Strict Priority)**

```
High Priority Queue:  [Pkt_H1] → [Pkt_H2] → ┐
                                             ├→ Output
Low Priority Queue:   [Pkt_L1] → [Pkt_L2] → ┘
                           ↑
                     只在High为空时才发送
```

- 高优先级队列完全优先
- **问题**：低优先级可能饥饿（Starvation）

#### **加权轮询 (WRR - Weighted Round Robin)**

```
Queue 1 (权重=3): [A] [B] [C] ─┐
Queue 2 (权重=2): [D] [E] ─────┼→ 发送顺序: A, B, C, D, E, A, B, C, D, E, ...
Queue 3 (权重=1): [F] ─────────┘             ↑     ↑     ↑
                                            Q1占3  Q2占2  Q3占1
```

- 按权重比例分配带宽
- 避免饥饿

#### **加权公平队列 (WFQ - Weighted Fair Queueing)**

```
每个包分配虚拟完成时间:
  Finish_Time[i] = Start_Time + (Packet_Size / Weight)

按Finish_Time排序发送
```

- 基于包大小和权重的公平性
- 考虑帧大小差异

#### **缺省加权轮询 (DWRR - Deficit Weighted Round Robin)**

```
每个队列有Deficit Counter (DC):
  Round开始: DC += Quantum (基于权重)
  发送帧: DC -= Frame_Size
  DC < 0: 跳过该队列
  Round结束: DC保留到下一轮
```

- 解决WRR的帧大小问题
- **数据中心常用**

---

## 七、QoS（服务质量）机制

### 7.1 流量分类（Classification）

**基于802.1p优先级**（VLAN标签中的PCP字段）：
```
PCP值  优先级名称         典型应用
──────────────────────────────────────
  7    网络控制 (NC)     路由协议、LACP
  6    网际控制 (IC)     -
  5    语音 (VO)         VoIP
  4    视频 (VI)         视频会议
  3    关键应用 (CA)     业务关键数据
  2    优秀努力 (EE)     -
  1    尽力而为 (BE)     普通数据
  0    背景 (BK)         批量下载
```

**基于DSCP**（IP头部的Differentiated Services字段）：
```
DSCP值   名称             映射到PCP
────────────────────────────────────
 46      EF (加速转发)    5
 34      AF41             4
 26      AF31             3
  0      BE (尽力而为)    0
```

### 7.2 流量整形（Traffic Shaping）

#### **令牌桶算法 (Token Bucket)**

```
        Token Generator
             ↓ (速率: CIR)
       ┌─────────────┐
       │ Token Bucket│ (容量: CBS)
       │   [tokens]  │
       └─────────────┘
             ↓
       到达的数据包
             ↓
       消耗tokens发送
       (1 token = 1 byte)

参数:
  CIR (Committed Information Rate): 承诺速率 (如100 Mbps)
  CBS (Committed Burst Size): 突发容量 (如1 MB)
```

**行为**：
- 有token → 立即发送，消耗token
- 无token → 排队等待或丢弃

#### **漏桶算法 (Leaky Bucket)**

```
       入流量（突发）
             ↓
       ┌──────────┐
       │   Bucket │
       │    ████  │ (缓冲队列)
       └──────────┘
             ↓ (固定速率流出)
       出流量（平滑）
```

**特点**：
- 强制平滑输出速率
- 吸收突发流量

### 7.3 拥塞管理

#### **WRED (Weighted Random Early Detection)**

```
队列深度
   ↑
100%├─────────────────────┐ Drop all
    │                    /│
    │                  /  │
    │                /    │
 Max├──────────────/      │ Random drop (线性增长)
    │            /        │
    │          /          │
 Min├────────/            │ No drop
    │      /              │
    └────────────────────→ 队列使用率
      0   Min    Max    100%

Drop概率:
  < Min: 0%
  Min~Max: 线性增长 (如0% → 10%)
  > Max: 100%
```

**权重差异**：
- 高优先级流量：Min=80%, Max=90%
- 低优先级流量：Min=40%, Max=60%

#### **ECN (Explicit Congestion Notification)**

**工作流程**：
```
交换机端:
  队列深度 > 阈值
       ↓
  标记IP包的ECN字段 (而非丢弃)
       ↓
  转发给接收端

接收端:
  检测到ECN标记
       ↓
  通知发送端（通过TCP ACK或其他机制）

发送端:
  收到ECN反馈
       ↓
  降低发送速率（类似收到丢包信号）
```

**ECN的IP头部字段**：
```
IP Header的ToS字段 (8 bits):
  Bit 6-7: ECN字段
    00: Not-ECT (不支持ECN)
    01: ECT(1) (支持ECN)
    10: ECT(0) (支持ECN)
    11: CE (拥塞已发生)
```

**优势**：
- 无损网络（RDMA over Ethernet需要）
- 提前拥塞通知，避免丢包

---

## 八、流量控制（Flow Control）

### 8.1 以太网流控（IEEE 802.3x）

#### **PAUSE帧机制**

**工作原理**：
```
接收端交换机:
  缓冲区将满 (如85%使用率)
       ↓
  发送PAUSE帧给发送端
       ↓
发送端交换机:
  收到PAUSE帧
       ↓
  暂停发送 (持续时间由PAUSE帧指定)
       ↓
  等待Resume或超时
       ↓
  恢复发送
```

**PAUSE帧格式**：
```
┌──────┬──────┬───────┬──────────┬─────┐
│ Dest │ Src  │ Type  │ Opcode   │ ... │
│ MAC  │ MAC  │0x8808 │ 0x0001   │     │
│(固定)│      │(PAUSE)│(PAUSE)   │     │
└──────┴──────┴───────┴──────────┴─────┘
           ↓
    Dest MAC = 01:80:C2:00:00:01 (预留多播地址)

Data字段:
  ┌──────────────┬─────┐
  │ Pause Time   │ ... │
  │  (2 bytes)   │     │
  └──────────────┴─────┘
     ↓
  单位: 512 bit-times
  例: 0xFFFF = 最大暂停时间
```

**问题**：
- 全端口暂停（粗粒度）
- 高优先级流量也被阻塞

### 8.2 优先级流控（PFC - Priority Flow Control, IEEE 802.1Qbb）

**改进**：支持**每优先级独立暂停**

```
接收端:
  优先级3的队列将满
       ↓
  发送PFC帧: PAUSE优先级3
       ↓
发送端:
  只暂停优先级3的流量
  其他优先级(0,1,2,4,5,6,7)继续发送
```

**PFC帧格式**：
```
Data字段:
  ┌──────────────┬─────────────────────┐
  │ Enable Bits  │ Pause Time [0-7]    │
  │  (8 bits)    │  (8 × 2 bytes)      │
  └──────────────┴─────────────────────┘
       ↓                  ↓
  每bit对应一个      每个优先级的暂停时间
  优先级(0-7)
```

**应用**：
- **RoCE (RDMA over Converged Ethernet)** 必需
- 无损以太网（Lossless Ethernet）
- 数据中心存储网络

### 8.3 Credit-Based Flow Control（数据中心常用）

**原理**：
```
接收端:
  初始化: Credits = Buffer_Depth
       ↓
  每接收1个包: Credits -= Packet_Size
       ↓
  每从缓冲移除1个包:
       ↓
  发送Credit返回给发送端

发送端:
  维护Credit计数器
       ↓
  Credits > 0 → 可以发送
  Credits = 0 → 必须等待
       ↓
  每收到Credit反馈: Credits += 1
```

**优势**：
- 细粒度控制（逐包）
- 无丢包（发送前保证有缓冲空间）
- 适合高速互联（InfiniBand、NVLink等也用此机制）

---

## 九、高级功能

### 9.1 生成树协议（STP - Spanning Tree Protocol）

**目的**：防止二层环路（Loop）导致的广播风暴

**工作原理**：
```
物理拓扑（有冗余链路）:
    Switch A ─────┐
       │          │
       │          │
    Switch B ── Switch C

逻辑拓扑（STP阻塞冗余链路）:
    Switch A ─────X (Blocked)
       │
       │
    Switch B ── Switch C
```

**端口状态**：
1. **Forwarding**: 正常转发数据
2. **Blocking**: 阻塞（只接收BPDU，不转发数据）
3. **Listening**: 监听BPDU，确定拓扑
4. **Learning**: 学习MAC地址，不转发
5. **Disabled**: 人工关闭

**BPDU (Bridge Protocol Data Unit)**：
- 交换机间交换拓扑信息
- 选举根桥（Root Bridge）
- 计算到根桥的最短路径

### 9.2 链路聚合（LAG - Link Aggregation）

**LACP (Link Aggregation Control Protocol, IEEE 802.3ad)**：

```
         ┌─ Link 1 (1 Gbps) ─┐
Switch A ├─ Link 2 (1 Gbps) ─┤ Switch B
         └─ Link 3 (1 Gbps) ─┘
              ↓
      聚合成一个逻辑链路 (3 Gbps)
```

**负载均衡算法**：
- 基于源MAC
- 基于目标MAC
- 基于源+目标MAC（最常用）
- 基于IP地址
- 基于TCP/UDP端口

**Hash计算**：
```
hash = (src_mac XOR dst_mac) % num_links
     → 选择物理链路
```

### 9.3 镜像端口（Port Mirroring / SPAN）

**功能**：将某端口的流量复制到监控端口

```
配置:
  Mirror Source: Port 5 (被监控)
  Mirror Destination: Port 24 (连接分析器)

行为:
  Port 5的所有流量 → 正常转发
                   → 同时复制到Port 24
```

**应用**：
- 网络监控
- 流量分析（Wireshark等工具）
- 入侵检测系统（IDS）

### 9.4 访问控制列表（ACL）

**基于MAC的ACL**：
```
Rule 1: Permit   src_mac=AA:BB:CC:*
Rule 2: Deny     dst_mac=DD:EE:FF:*
Rule 3: Permit   any
```

**基于IP的ACL（需要三层功能）**：
```
Rule 1: Permit   src_ip=192.168.1.0/24, dst_port=80
Rule 2: Deny     src_ip=10.0.0.0/8
```

**执行位置**：
- Ingress ACL: 入端口检查（进入前）
- Egress ACL: 出端口检查（转发前）

---

## 十、数据中心交换机特性

### 10.1 RDMA over Ethernet (RoCE)

**需求**：
- **无损网络**（Zero Packet Loss）
- 极低延迟（< 1 μs）
- PFC支持

**RoCE v2 (Routable)**：
```
┌────────────────────────────────────┐
│     RDMA Application               │
├────────────────────────────────────┤
│     InfiniBand Verbs               │
├────────────────────────────────────┤
│     RoCE (RDMA over Ethernet)      │
├────────────────────────────────────┤
│     UDP / IP                       │
├────────────────────────────────────┤
│     Ethernet (with PFC)            │
└────────────────────────────────────┘
```

**交换机要求**：
- PFC（优先级流控）
- ECN（显式拥塞通知）
- 深缓冲（吸收微突发）
- DCQCN（数据中心QCN）拥塞控制

### 10.2 Jumbo Frame（巨型帧）

**标准以太网帧**：
- 最大payload: 1500 bytes
- 最大帧长: 1518 bytes (不含VLAN)

**Jumbo Frame**：
- 最大payload: 9000 bytes（常见）
- 最大帧长: 9018 bytes

**优势**：
- 减少帧数量 → 降低CPU开销
- 提升吞吐量（减少帧头开销）

**要求**：
- 全路径支持（交换机、网卡、链路）

### 10.3 VXLAN（虚拟可扩展局域网）

**目的**：突破VLAN 4096限制，支持多租户

**VXLAN封装**：
```
Original Ethernet Frame:
  [Eth][IP][TCP][Data]

VXLAN Encapsulation:
  [Outer Eth][Outer IP][UDP][VXLAN Header][Inner Eth][IP][TCP][Data]
              ↑         ↑          ↑
        Overlay IP    UDP 4789   VNI (24 bits)
                                 → 16M VLAN空间
```

**VNI (VXLAN Network Identifier)**：
- 24位，支持16,777,216个虚拟网络
- 替代传统VLAN ID（12位，4096个）

**VTEP (VXLAN Tunnel Endpoint)**：
- 负责封装/解封装
- 通常在交换机或服务器实现

---

## 十一、性能指标

### 11.1 吞吐量（Throughput）

**理论最大吞吐量**：
```
对于N端口交换机，每端口速率R:
  理论吞吐 = N × R

例: 48端口 × 10 Gbps = 480 Gbps
```

**实际吞吐量影响因素**：
- 帧大小（小帧 → 更多处理开销）
- 流量模式（单播 vs 广播）
- 缓冲溢出

**帧速率（pps - Packets Per Second）**：
```
最小帧: 64 bytes (512 bits)
@ 10 Gbps:
  Max pps = 10 Gbps / (512 bits + 96 bits IFG + 64 bits Preamble)
          = 10 Gbps / 672 bits
          = 14.88 Mpps (每端口)
```

### 11.2 延迟（Latency）

**延迟组成**：
```
Total Latency = Store Delay + Processing Delay + Serialization Delay + Propagation Delay
```

**典型值**：
| 交换机类型 | 延迟范围 |
|-----------|---------|
| 企业级L2交换机 | 5-50 μs (Store-and-Forward) |
| 数据中心ToR | 300 ns - 1 μs (Cut-Through) |
| 数据中心Spine | 500 ns - 3 μs |
| 高频交易专用 | < 100 ns |

### 11.3 缓冲深度（Buffer Depth）

**计算公式**（基于BDP - Bandwidth-Delay Product）：
```
Required Buffer = Bandwidth × RTT

例: 10 Gbps链路，RTT=100 μs
  Buffer = 10 Gbps × 100 μs = 125 MB
```

**实际交换机缓冲**：
- 企业级: 4-16 MB
- 数据中心: 16-256 MB
- 长距离/高BDP: 1 GB+

---

## 十二、故障场景与处理

### 12.1 广播风暴（Broadcast Storm）

**原因**：
- 网络拓扑有环路
- STP未启用或配置错误
- 大量ARP请求

**现象**：
- CPU占用率100%
- 网络瘫痪
- 交换机崩溃

**防护**：
- 启用STP/RSTP
- 限制广播速率（Storm Control）
- 端口隔离

### 12.2 MAC表溢出攻击

**攻击原理**：
```
攻击者发送大量伪造源MAC的帧
    ↓
交换机不断学习新MAC
    ↓
MAC表填满
    ↓
交换机进入Hub模式（泛洪所有流量）
    ↓
攻击者可嗅探所有流量
```

**防护**：
- Port Security（限制每端口学习的MAC数量）
- MAC地址绑定
- 802.1X认证

### 12.3 VLAN跳跃攻击

**攻击原理**：
```
攻击者发送双层VLAN标签的帧:
  [Eth][VLAN 100][VLAN 200][IP][Data]
       ↑         ↑
    外层标签   内层标签

交换机处理:
  1. 剥离外层VLAN 100
  2. 转发到VLAN 100的端口
  3. 目标交换机看到VLAN 200标签
  4. 转发到VLAN 200（绕过隔离）
```

**防护**：
- 禁用Trunk端口的Native VLAN
- 显式标记所有VLAN

---

## 总结

以太网交换机的核心行为可以总结为：

1. **学习（Learning）**：从源MAC学习端口映射
2. **过滤（Filtering）**：不回发到同一端口
3. **转发（Forwarding）**：根据目标MAC选择性转发
4. **泛洪（Flooding）**：未知MAC/广播/多播时泛洪

**关键机制**：
- **MAC地址表**：快速查找（CAM）
- **VLAN隔离**：逻辑分割广播域
- **QoS调度**：优先级和带宽管理
- **流量控制**：防止拥塞丢包

**数据中心特性**：
- **低延迟**：Cut-Through, 浅缓冲
- **无损网络**：PFC + ECN
- **高吞吐**：深缓冲, VOQ架构
- **大规模**：VXLAN, Spine-Leaf拓扑
