# 01. SG2262 C2C 方案总览

## 1.1 设计目标

**[DOC]** SG2262 基于 Ethernet 二层协议实现最多 **1024 颗芯片**互联，其核心设计特征：

- SerDes 速率支持 **56Gbps / 112Gbps**
- 链路组合支持 **x4 / x8**
- 每个芯片出 **8 组 x4 Link**（C2C 端口）
- 外加 **1 个 x16 Link**（PCIe 端口）
- 支持多种拓扑结构：clos、all2all、torus、mesh、ring、cube

**[DOC]** 系统支持两种模式，均走同一套 spc.c2c 协议：

| 系统 | 描述 | 规模 |
|------|------|------|
| 系统一 | 通过交换机互联 | 最多 1024 芯片 |
| 系统二 | 通过 Ethernet 点到点互联 | 小规模直连 |

## 1.2 支持的拓扑类型

SG2262 支持六种互联拓扑，分为**单层互联**和**两层互联**两大类：

### 单层互联

| 拓扑 | 描述 | 规模 |
|------|------|------|
| **clos** | 所有芯片通过交换机互联，交换机可为多层 clos | 最多 1024 芯片 |
| **cube** | 低成本服务器形态，芯片直连 | 最多 8 芯片 |

### 两层互联 (L1 + L2)

两层互联的基本模型：**L1 形成 cluster（最多 32 芯片）**，**L2 通过交换机连接 cluster**。

| 拓扑 | L1 (cluster 内) | L2 (cluster 间) | 特点 |
|------|-----------------|-----------------|------|
| **all2all + clos** | all2all/torus/ring/mesh | 交换机 | 标准两层 |
| **clos + clos** | 交换机 | 交换机 | 全交换机方案 |
| **低成本 all2all + clos** | all2all/torus/ring/mesh | 单轨交换机 | L2 成本低 |
| **低成本 clos + clos** | 交换机 | 单轨交换机 | 全交换机低成本 |

**[推导]** 关键架构选择：
- **L1 (cluster 内)**: 支持任意拓扑（点对点直连或交换机），最多 32 芯片
- **L2 (cluster 间)**: 仅支持交换机(clos)拓扑
- 路由原则：**先做 L2 路由，再做 L1 路由**（L2 流量更均匀，L1 路由表更简单）

## 1.3 核心 Feature List

### 数据传输能力

| Feature | 说明 |
|---------|------|
| 常规读写 | 读写其他芯片内存 |
| Write + MSG Sync | 写完成后触发接收方 msg 同步 |
| Write + Atomic | 写完成后触发接收方 atomic 写操作（不支持 atomic 读） |
| All Reduce | 支持在网计算 (scale + reduce_sum) |
| Send/Receive | 点对点配对通信（每 CDMA Thread 一对一） |
| 芯片转发 | MAC 帧通过 SG2262 转发到其他芯片 |
| 广播/多播 | 支持（待确认） |
| 概率拼包 | 小报文合并发送 |

### 保序与同步

| Feature | 说明 |
|---------|------|
| fence 指令 | CDMA 通过 fence 建立屏障，确保前后执行顺序 |
| 保序窗口 | 可配置 8/12/32 个地址区域的写保序 |
| post_write | CHS 模式：硬件沿路保序 |
| non_post_write | CFS 模式：fence 指令保序 |
| Bresp 拼包 | c2c_sys_top 支持响应报文合并 |

### 可靠性

| Feature | 说明 |
|---------|------|
| LLR | Link Level Retry |
| E2E Retry | 端到端重传 |
| CBFC | Credit-Based Flow Control |
| CRC 校验 | 所有 C2C 报文添加 CRC |
| Memory Protect | 非法地址访问中断上报 |
| 隔离模式 | 错误芯片可隔离不扩散 |

### All Reduce 操作码

**[DOC]** 支持的 Reduce 操作：

| 操作 | opcode | 描述 |
|------|--------|------|
| nop | 4'b0 | 无操作 |
| max | 4'b10 | 取最大值 |
| min | 4'b11 | 取最小值 |
| add | 4'b100 | 加法 |

支持的数据类型：fp32, fp20, fp16, bp16

## 1.4 硬件约束与限制

**[DOC]** 存在如下限制：

1. **路径唯一**: 不支持 ECMP 等多路径算法，C2C 跨网络路径唯一且严格保序
2. **AXI 边界**: AXI 的 4KB boundary 和 MAC 的 1.5KB MTU 限制（当前仅支持 512B/256B AXI 报文）
3. **Send/Receive 限制**: 每个 PCIe Link 只支持同时与一颗芯片的一个 PCIe Link 做 Send/Receive 交互
4. **无负载均衡**: 当前不支持负载均衡（未来可通过多路径+网络负载监测实现）
5. **Read 性能**: C2C Read 高带宽代价较大且扩展性受限（软件尽量用 Write 替代）

## 1.5 封装与出 Pin 方案

**[DOC]** 封装原则：

- PCIe 卡形态：C2C Link 与 PCIe Link 出在相反方向
- OAM 形态：C2C Link 与 PCIe Link 均匀出在芯片两端
- 只出一种封装方案，优先照顾 OAM 形态
- 建议 AI Die 尽量做扁，利于做成正方形，节省封装面积

物理链路配置：
- **8 组 x4 C2C Link**（每组 4 lane，每 lane 56G/112G）
- **1 组 x16 PCIe Link**

**[推导]** 单芯片 C2C 总带宽计算：
- 8 组 x4 @ 112Gbps = 8 x 4 x 112 = **3584 Gbps = 448 GB/s**（单向）
- 8 组 x4 @ 56Gbps = 8 x 4 x 56 = **1792 Gbps = 224 GB/s**（单向）

## 1.6 术语定义

| 术语 | 全称 | 说明 |
|------|------|------|
| CLE | Chip Address Lookup Engine | 芯片地址查找引擎（路由） |
| CMAP | Chip-level MAC ID Assignment Plan | 以芯片为单位分配 MAC ID |
| DMAP | Die-level MAC ID Assignment Plan | 以 Die 为单位分配 MAC ID |
| L1 | Layer 1 | 第一层互联（cluster 内） |
| L2 | Layer 2 | 第二层互联（cluster 间） |
| CHS | C2C post-write Hardware Sequence | 硬件保序方案 |
| CFS | C2C Fence Sequence | fence 指令保序方案 |
| CDMA | - | 跨芯片 DMA 引擎 |
| Datagram | - | 原始以太网帧收发模式 |
