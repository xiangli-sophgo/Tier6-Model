# 04. 虚拟通道与仲裁机制

## 4.1 8 CBFC/PFC虚拟通道

**[DOC]** PAXI SUE2.0 Features:

> "Support 8 CBFC and PFC VC."

**[DOC]** 来自SUE2.0 2.2 CBFC & PFC Virtual Channel:

> "MAC's CBFC/PFC supports 8 Virtual Channels (VCs). The user transmits the VC ID via the User field of the AXI interface. PAXI encapsulates the AXI Flit with VC ID/DA and routes it to RC Link to encapsulates MAC Frame."

### 与旧版的对比

| 特性 | 旧版 (v2R0p6) | SUE2.0 |
|------|---------------|--------|
| VC数量 | 3 (Management/Control/Data) | 8 (CBFC/PFC VC) |
| 仲裁方式 | 优先级 + WRR | 通道权重 + CBFC/PFC |
| 独立复位 | 3个通道各自独立 | 通过VC映射统一管理 |
| 流控机制 | Per-DA Credit | Per-VC CBFC/PFC |

## 4.2 两种VC映射模式

**[DOC]** 来自SUE2.0 2.2:

> "PAXI supports two modes of VC mapping."

### Mode 0: PAXI默认映射

**[DOC]**:

> "The first mode is VC-DA mapping by PAXI, where DAs are evenly distributed across two VC channels based on the number of DAs."

Ctrl Register (0x00C) bit8 = 0 (默认):

| VC编号 | 用途 |
|--------|------|
| VC0, VC2 | Single-cast request (REQ) |
| VC1, VC3 | Single-cast response (RSP) |
| VC4 | Multi-cast request |
| VC0~VC3 | APB/Msg request/response (可配置) |

**[DOC]** DA分配示例:

> "Take request for example, with 8 DAs, DA0-3 are assigned to VC0, and DA4-7 are assigned to VC2."

DA按数量均匀分配到两个VC通道: 前半DA分配到VC0(REQ)/VC1(RSP), 后半DA分配到VC2(REQ)/VC3(RSP)。

### Mode 1: 用户自定义映射

**[DOC]**:

> "Set vc mode to 1 in Ctrl Register to activates this mode. User transmit mapping info through VC/DA bit in AXI USER field."

Ctrl Register (0x00C) bit8 = 1:

用户通过AXI USER field的VC位直接指定VC通道。

**[DOC]** 约束:

> "In this mode, REQ channels can only be assigned to VC0 and VC2, while RSP channels are restricted to VC1 and VC3. APB transactions can be allocated to any available VC."

| 事务类型 | 可用VC |
|---------|--------|
| REQ (AW+W, AR) | VC0, VC2 |
| RSP (R, B) | VC1, VC3 |
| APB | 任意VC |

### APB VC配置

**[DOC]** Ctrl Register (0x00C) bits[11:9]:

> "APB VC: Virtual channel of APB."

APB事务的VC通道可通过Ctrl Register的APB VC字段配置。

## 4.3 死锁预防规则

**[DOC]** 来自SUE2.0 2.2:

> "When using the user VC/DA mapping mode, to prevent out-of-order and deadlocks, the following principles must be followed when assigning Virtual Channels (VCs) at the user side:"

### 规则

1. **REQ和RSP必须分配到不同VC**

   **[DOC]**: REQ只能用VC0/VC2, RSP只能用VC1/VC3, 物理上隔离。

2. **跨VC无序列保证**

   **[DOC]**:
   > "PAXI does not guarantee ordering for data across different virtual channels. The upper layer must maintain the order or support out-of-order operations."

**[推导]** 这些规则的原因:
- REQ/RSP分离到不同VC避免了经典的请求-响应死锁: 如果REQ和RSP共享同一VC, 当VC缓冲满时, RSP无法返回, 导致死锁
- 上层需要自行管理跨VC的数据顺序, 这是VC机制的固有特性

## 4.4 CBFC机制

**[DOC]** 来自RCLINK Spec 5.5 CBFC:

CBFC (Credit-Based Flow Control) 是一种端到端的流量控制机制, 由RC Link实现, 与PAXI的VC映射配合工作。

### 核心参数 (per-VC)

| 参数 | 说明 |
|------|------|
| credit_size | 单个Credit代表的字节数 (32/64/128/256/1024/2048) |
| credit_limit (CL) | 该VC的最大Credit数量 |
| pkt_ovhd | CBFC协议中的PktOvhd值 (有符号数) |
| credit_uf_limit | Credit下限 (1~7), 单位为最大报文消耗的Credit数 |

### 工作流程

1. **初始化阶段**: MAC配置各VC的credit_size, credit_limit, pkt_ovhd
2. **工作阶段**:
   - 发送方: 根据报文长度和credit_size计算消耗的Credit数量
   - 接收方: 数据从缓冲弹出时归还Credit
   - 当剩余Credit低于credit_uf_limit时, 阻塞该VC的发送

### RC Link中的VC映射

**[DOC]** 来自RCLINK Spec:

RC Link内部有7路物理流量, 映射到8个VC:

| 流量类型 | 映射关系 |
|---------|---------|
| TYPE1_REQ (Bank0~3) | 可映射到1/2/4个VC (按QPID低位划分) |
| TYPE1_ACK + CNP | 同一个VC |
| TYPE2 | 一个独立VC |
| TYPE3 | 一个独立VC |

**[推导]** PAXI的VC0~VC4对应RC Link的不同流量类型:
- VC0/VC2 (REQ) -> TYPE1_REQ (不同Bank)
- VC1/VC3 (RSP) -> TYPE1_ACK
- VC4 (多播) -> TYPE2

## 4.5 PFC机制

PFC (Priority Flow Control) 是CBFC的替代方案, 两者互斥。

**[DOC]** Ctrl Register (0x00C) bit7:

> "CBFC_EN: Set 1 to enable CFBC mode. Set 0 to enable PFC mode."

### PFC工作方式

基于RX Buffer水位线触发:
- 当接收数据超过高水位线时, 触发PFC帧, 暂停远端对应VC的发送
- 当数据降到低水位线以下时, 解除PFC限制

水位线配置:

| Buffer | 寄存器 | 默认高水位 | 默认低水位 |
|--------|--------|-----------|-----------|
| REQ | RX REQ Buffer Water Mark (0x058) | 32帧+1RTT | 1RTT |
| RSP | RX RSP Buffer Water Mark (0x05C) | 32帧+1RTT | 1RTT |
| MUL | RX MUL Buffer Water Mark (0x060) | 8帧+1RTT | 1RTT |

### CBFC vs PFC对比

| 特性 | CBFC | PFC |
|------|------|-----|
| 控制粒度 | Per-VC Credit计数 | Per-VC水位线 |
| 实现位置 | RC Link端到端 | PAXI RX Buffer |
| 参数配置 | credit_size, credit_limit, pkt_ovhd | 高/低水位线 |
| 软件流控 | 支持 (software_ctrl_cbfc_vc_status) | 不支持 |
| 动态下限 | 支持 (dyn_uf_limit_cbfc_en, 仅TYPE1 REQ) | 不支持 |
| 使能控制 | Ctrl Register bit7 = 1 | Ctrl Register bit7 = 0 |

**不能同时使用CBFC和PFC。**

## 4.6 通道权重配置

**[DOC]** 来自SUE2.0 3.2.29/3.2.30:

### Channel Weight Register (0x070)

| Bits | 字段 | 说明 | 默认值 |
|------|------|------|--------|
| 31:16 | REQ channel Weight | REQ通道权重 | 0x0008 |
| 15:0 | RSP channel Weight | RSP通道权重 | 0x0008 |

### Channel Weight Register2 (0x074)

| Bits | 字段 | 说明 | 默认值 |
|------|------|------|--------|
| 31:16 | MUL channel Weight | 多播通道权重 | 0x0100 |
| 15:0 | APB channel Weight | APB通道权重 | 0x0008 |

**[推导]** 权重决定了不同类型流量的仲裁优先级和带宽分配:
- REQ和RSP默认权重相同 (0x0008)
- MUL通道默认权重较高 (0x0100), 确保多播数据优先传输
- APB通道默认权重与REQ/RSP相同

## 4.7 仲裁整体行为

**[推导]** SUE2.0的仲裁模型:

```
TX方向:
  REQ (VC0/VC2) -----+
  RSP (VC1/VC3) -----+--- Channel Weight Arbiter ---> RC Link TX
  MUL (VC4) ---------+
  APB ----------------+

其中:
  - 各通道按Channel Weight进行加权轮询
  - RC Link内部进一步按CBFC Credit或PFC进行流控
  - TYPE1 REQ内部按Bank轮转 (tx_bank_rr_en)
```
