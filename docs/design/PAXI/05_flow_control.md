# 05. 流控机制

## 5.1 流控概述

**[DOC]** PAXI SUE2.0 Features:

> "Base-on credit flow control with RC Link."

SUE2.0的流控体系与旧版有根本性变化:
- 旧版: PAXI内部管理per-DA Credit (128/256 OST), 配合PFC水位线
- SUE2.0: 流控职责下放到RC Link层, 使用CBFC (Credit-Based Flow Control) 或PFC, 以VC为粒度

**[推导]** 旧版的per-DA Credit机制 (DA MAP寄存器体系, Credit Threshold, Credit Update协商) 在SUE2.0中已被移除, 由RC Link的CBFC机制替代。

## 5.2 CBFC (Credit-Based Flow Control)

**[DOC]** 来自RCLINK Spec 5.5:

CBFC是一种端到端的流量控制机制, 部署在RC Link层, 支持8个独立的VC通道。

### 核心原理

**[DOC]**:

发送方在协商阶段获取对端接收方的总Credit数量, 工作过程中主动计算每个报文消耗的Credit数量(按VC粒度), 跟踪接收方的可用缓冲空间。若剩余Credit不足, 则禁止该VC调度数据包。

接收方维护本地Credit数量, 定期将可用Credit同步给发送端。数据包从缓冲弹出时归还Credit。

### Per-VC参数

| 参数 | 说明 | 配置来源 |
|------|------|---------|
| credit_size | 单个Credit代表的字节数 | MAC配置, 支持32/64/128/256/1024/2048 |
| credit_limit (CL) | 该VC的最大Credit数量 | MAC配置 |
| pkt_ovhd | CBFC协议的PktOvhd值 (有符号数) | MAC配置 |
| credit_uf_limit | Credit下限 (1~7) | 寄存器配置 |

### 工作流程

**[DOC]**:

**初始化阶段**:
- MAC通过 MAC2TX_CBFC_RST_I[31:0] 信号配置各VC
- 低8bit指示哪些VC需要初始化
- 配置CREDIT_SIZE, CREDIT_LIMIT, PKT_OVHD_LEN
- 支持各VC独立配置
- 初始化建议在RC Link工作前完成

**工作阶段**:
- 发送方: 报文仲裁发出时, 计算消耗的Credit数量
- 接收方: 通过 MAC2TX_CBFC_VLD_I 返还Credit
- 当剩余Credit低于 credit_uf_limit 时, 阻塞该VC

### 高级功能

**[DOC]**:

1. **Credit下限配置** (credit_uf_limit):
   - 可配范围: 1~7, 单位为一个最大报文消耗的Credit数
   - 剩余Credit低于下限时停止该VC流量
   - 必须至少配置为1, 否则CBFC失效

2. **动态下限调节** (dyn_uf_limit_cbfc_en):
   - 剩余Credit低于水线但仍大于一个最大包长时, 可动态再下发一个报文
   - 直到剩余Credit不足以下发一个最大报文
   - 仅TYPE1 REQ支持, TYPE2/TYPE3/ACK不支持

3. **软件流控** (software_ctrl_cbfc_vc_status):
   - 配置对应VC为0可强制停止该VC流量, 无视Credit数量
   - 不支持在Credit低于下限时通过该寄存器强制打开VC

### VC与流量类型映射

**[DOC]**:

| 流量类型 | VC映射寄存器 | 使能寄存器 |
|---------|-------------|-----------|
| TYPE1_REQ Bank0~3 | type1_req_bank0/1/2/3_cbfc[2:0] | rx_type1_req_bank0/1/2/3_cbfc_en |
| TYPE1_ACK + CNP | type1_ack_cbfc[2:0] | rx_type1_ack_cbfc_en |
| TYPE2 | type2_cbfc[2:0] | rx_type2_cbfc_en |
| TYPE3 | type3_cbfc[2:0] | rx_type3_cbfc_en |

**[DOC]** 约束:
- 不同流量类型不可映射到同一VC
- CL值必须大于下限所代表的Credit数量, 否则VC会被一直阻塞
- 全局控制: stop_cbfc_en 禁用全局CBFC (最高优先级)

### 使能控制

**[DOC]** PAXI Ctrl Register (0x00C) bit7:

> "CBFC_EN: Set 1 to enable CFBC mode. Set 0 to enable PFC mode."

## 5.3 PFC (Priority Flow Control)

PFC是CBFC的替代方案, 两者互斥 (Ctrl Register bit7控制)。

### 工作原理

基于PAXI RX Buffer水位线触发:
- 数据量超过高水位线 -> 触发MAC发送PFC帧, 暂停远端对应VC发送
- 数据量降到低水位线以下 -> 解除PFC限制

### 水位线寄存器

**[DOC]** 来自SUE2.0 3.2.23~3.2.25:

| 寄存器 | 地址 | 说明 | 默认值 |
|--------|------|------|--------|
| RX REQ Buffer Water Mark | 0x058 | REQ通道水位线 | {UVIP_PAXI_REQ_HIGH_WATERMARK, UVIP_PAXI_REQ_LOW_WATERMARK} |
| RX RSP Buffer Water Mark | 0x05C | RSP通道水位线 | {UVIP_PAXI_RESP_HIGH_WATERMARK, UVIP_PAXI_RESP_LOW_WATERMARK} |
| RX MUL Buffer Water Mark | 0x060 | 多播通道水位线 | {UVIP_PAXI_MUL_HIGH_WATERMARK, UVIP_PAXI_MUL_LOW_WATERMARK} |

**[推导]** 与旧版对比: 旧版有DAXI/CAXI两个水位线寄存器 (0x058/0x05C), SUE2.0改为REQ/RSP/MUL三个, 反映了从Data/Ctrl通道到REQ/RSP/MUL VC的架构变化。

### 默认水位线

| Buffer | 默认高水位 | 默认低水位 | 背压预留 |
|--------|-----------|-----------|---------|
| REQ MEM | 32帧 + 1RTT | 1RTT | 1RTT (吸收背压数据) |
| RSP MEM | 32帧 + 1RTT | 1RTT | 1RTT |
| MUL MEM | 8帧 + 1RTT | 1RTT | 1RTT |

### RX方向CBFC与PFC共用

**[DOC]** 来自RCLINK Spec:

> "由于CBFC的反压与PFC共用接口, 因此在RX方向需要将PFC TX方向的各VC使能寄存器 tx_typeX_pfc_en 置1, 否则CBFC不能通过PFC接口反压。"

即使使用CBFC模式, RX方向的反压信号 TX_PFC_REQ_O[7:0] 仍然被使用。

## 5.4 CBFC vs PFC对比

| 特性 | CBFC | PFC |
|------|------|-----|
| 控制层级 | 端到端 (RC Link) | 链路级 (PAXI RX) |
| 控制粒度 | Per-VC Credit计数 | Per-VC水位线 |
| 参数 | credit_size, credit_limit, pkt_ovhd | 高/低水位线 |
| 软件流控 | 支持 | 不支持 |
| 动态下限 | 支持 (仅TYPE1 REQ) | 不支持 |
| 使能 | Ctrl Register bit7 = 1 | Ctrl Register bit7 = 0 |
| 同时使用 | **不可以** | **不可以** |

## 5.5 多播流量控制

**[DOC]** 来自RCLINK Spec 5.6:

多播报文有独立的基于Credit的流量管理, 与CBFC不同:
- 该Credit仅反映下游MAC TX方向的多播Buffer空间
- CREDIT_SIZE和CREDIT_LIMIT与Buffer尺寸相关
- TYPE2多播报文同时受CBFC和该流量控制的双重约束
- 复用CBFC控制器模块, 但不支持软件流控和动态下限
- 受独立的 multi_credit_ctrl_en 寄存器控制

## 5.6 RX OST约束

**[DOC]** Ctrl Register (0x00C) bit12:

> "rx ost constr en: 1: RX master will stop sending request when outstanding number request send without response return. 0: RX master will send request regardless of outstanding number constraint."

设为1时, RX Master在outstanding请求数达到限制前会停止发送新请求, 防止上层AXI fabric的缓冲溢出。
