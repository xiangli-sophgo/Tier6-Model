# 12. RC Link 传输层

> 基于 RCLINK_AFH_SPEC_v2.4 (Doc rev:2.4, July 2025) 文档分析

**标注约定:**
- **[DOC]** - 直接引用自官方文档原文
- **[推导]** - 基于文档信息的合理推导
- **[行业]** - 基于行业通用知识的补充说明

---

## 12.1 概述

**[DOC]** RCLINK-Stream IP 是 UniVista 实现的 SUE 协议传输层，支持最高可达 1024 个队列对。

### 核心规格

| 特性 | 规格 |
|------|------|
| MAC 接口 | 400G / 200G (1024bit / 512bit MAC 数据位宽) |
| 最大队列数 | 1024 个 QP |
| Outstanding 数量 | TYPE1: 512, TYPE2: 16 |
| 最大报文尺寸 | 4KB |
| 传输类型 | 固定到 RC (Reliable Connection) 发送 |
| 数据总线位宽 | 512 bit |
| 时钟频率 | 最大 1GHz |

### 支持特性

**[DOC]** 主要功能特性包括:

- **三种数据类型**: TYPE1 (可靠单播)、TYPE2 (不可靠多播)、TYPE3 (原始以太网)
- **四种报文格式**: Standard、AFH_GEN1、AFH_GEN2_16b、AFH_Lite
- **显式拥塞通知 (ECN)**: 检测 ECN 数据包自动回复 CNP，接收 CNP 时通过 CC 接口上报
- **优先级流控制 (PFC)**: 4 路数据流独立流控 (TYPE1_REQ / TYPE1_ACK / TYPE2 / TYPE3)，其中 TYPE1_REQ 分为四个独立逻辑 BANK 进行独立流控
- **基于信用的流量控制 (CBFC)**: TYPE1/TYPE2/TYPE3 独立流控，支持 8 个独立 VC 通道
- **MB 精度队列速率控制**: 设置时间窗口大小 (最小值 4us) 和每个时窗可通过数据字节数
- **PSN 错误重传**: 检测到 PSN 错误时支持数据包重传
- **时戳插入**: 支持时戳插入反馈 RTT 测量

### 接口概览

**[DOC]** TYPE1/TYPE2 接口:
- 配置: AXI-Lite 64 位接口
- TX Din: 带有信用有效性的 512 位数据流至 MAC
- TX Dout: 从 MAC 接收的带有信用有效性的 512 位数据流
- RX Din/Dout: 同上

**[DOC]** TYPE3 接口:
- 数据: AXI4 512 位写/读接口
- 配置: AXI Lite 64 位接口

**[DOC]** MAC 接口: AXI-Stream 512/1024 位接口，支持 200G/400G CEMAC

### 复位结构

**[DOC]** RC_LINK_IP 内部产生复位节点:
- `A15_NON_DMA/u_reset/TX_RESET_O`
- `A15_NON_DMA/u_reset/RX_RESET_O`

**[DOC]** v2.4 版本增加了软件复位 TYPE1/TYPE2 的功能。

---

## 12.2 三种数据类型

### TYPE1: 可靠传输

**[DOC]** TYPE1 支持端到端保护，支持 Go-Back-N 端到端重传。可以进行单独调度，支持单播报文传输。

| 属性 | 规格 |
|------|------|
| 端到端保护 | 支持 |
| 重传机制 | Go-Back-N |
| 调度 | 独立调度 |
| 传输模式 | 单播 |
| Outstanding | 512 (TYPE1_OST_N) |
| 最大 Payload | 1344 字节 (TYPE1_PKT_LEN) |
| VC 通道数 | 4 (TYPE1_VC_NUM) |
| CRC 保护 | 可选，由 `type1_remove_icrc_en` 寄存器控制 |

**[推导]** TYPE1 是核心数据传输通道，512 个 outstanding 配合 Go-Back-N 重传机制，适用于对数据完整性要求高的场景 (如 RDMA 数据传输)。

### TYPE2: 不可靠传输

**[DOC]** TYPE2 不支持端到端保护，不支持 Go-Back-N 端到端重传。可以进行单独调度，支持多播报文传输。

| 属性 | 规格 |
|------|------|
| 端到端保护 | 不支持 |
| 重传机制 | 无 |
| 调度 | 独立调度 |
| 传输模式 | 多播 |
| Outstanding | 16 (TYPE2_OST_N) |
| 最大 Payload | 1344 字节 (TYPE2_PKT_LEN) |
| Credit 初值 | 2048 (TYPE2_CRD_RST_NUM) |
| CRC 保护 | 不支持 |

**[推导]** TYPE2 主要用于多播场景 (如集合通信中的 Broadcast)，不需要端到端可靠性保证，因此 outstanding 数量较小。

### TYPE3: 原始以太网报文

**[DOC]** TYPE3 可以当作普通 ETH 报文传输，使用 Memory Descriptor (MD) 机制进行数据管理。

| 方向 | Buffer 大小 | 最大报文长度 |
|------|-------------|-------------|
| TX | 512bit(64byte) x 64 = 4096 byte | 4096 byte |
| RX | 1024bit(128byte) x 64 = 8192 byte | 8192 byte |

**[DOC]** MD (Memory Descriptor) 是一个 128 位的数据结构，定义如下:
- **Buffer head address**: 内存缓冲区地址。接收路径 `[5:0]` 固定为 0，地址需对齐到 64 字节边界；传输路径地址无对齐要求
- **type** `='b00`: 表示当前内存描述符，其他位保留
- **Buffer size** `[63:48]`: 系统填充的缓冲区大小 (以字节为单位)。接收路径需对齐到 64 字节
- **Data size** `[19:4]`: 缓冲区内有效数据的大小 (以字节为单位)。接收端由 TYPE3 填充，传输端由本地系统填充
- **ctrl/status**:
  - `bit[20]` SF: 数据包起始标志，传输路径上应总是 1
  - `bit[21]` LF: 数据包结束标志，传输路径上应总是 1
  - `bit[22]`: CRC 校验错误 (仅接收路径有效，LF=1 时)
  - `bit[23]`: 长度校验错误 (仅接收路径有效，LF=1 时)
  - `bit[47:24]`: 保留位

**[DOC]** MD 使用规则:
1. MD 地址需要对齐到 64 字节边界
2. 每个 MD 块只能包含一个数据包的内容
3. MD 块中的数据与块的开始对齐，并连续填充
4. 如果是 frame_end，可能没有完全填充
5. 对于传输端，一个数据包应在一个 MD 块中
6. 对于接收路径，一个数据包可以分布在多个 MD 块中

**[DOC]** 超长异常: TX 方向填充 data size 大于 4096 byte 会触发 tx oversize 中断; RX 方向接收 data size 大于 8096 byte 会触发 rx oversize 中断。

---

## 12.3 报文格式

### 格式总览

**[DOC]** 报文格式主要有四种: Standard Format、AFH_GEN1 Format、AFH_GEN2_16b Format 和 AFH_Lite Format。

格式选择由以下寄存器信号控制:

| 寄存器信号 | 说明 |
|------------|------|
| `tx_standard_en` | 标准报头格式使能 |
| `tx_afh_gen1_en` | AFH_GEN1 报头格式使能 |
| `tx_afh_gen1_tc_en` | AFH_GEN1 下 TC 域使能，表示报头携带 TC 域 |
| `tx_afh_gen2_16b_en` | AFH_GEN2_16b 压缩报头格式使能 |
| `afh_lite_en` | AFH_LITE 报文使能 |
| `vlan_en` | VLAN 使能 |

**[DOC]** 格式特性说明:
1. AFH_GEN2 Format 为 16bit 地址压缩格式 (AFH_GEN2_16b)
2. Standard、AFH_GEN1、AFH_GEN2 均携带 Eth_type 域，根据 `vlan_en` 选择是否插入 VLAN。Standard 携带 IP 和 UDP 报头
3. AFH_GEN1 根据 `afh_gen1_tc_en` 选择是否插入 TC 域。若打开 CBFC 功能且 AFH_GEN1 不携带 TC 域时，必须强制打开 VLAN
4. TYPE1 中 CRC 保护由 `type1_remove_icrc_en` 决定，TYPE2 不支持 CRC 保护

### Standard Format

**[DOC]** Standard Format 的报文结构 (以 No VLAN 为例):

```
[MAC Header] [IP Header] [UDP Header] [RH] [Payload] [ICRC(可选)]
```

带 VLAN 时:

```
[MAC Header] [VLAN_TAG] [IP Header] [UDP Header] [RH] [Payload] [ICRC(可选)]
```

**[DOC]** MAC/UDP/IP 报头均为标准格式报头，其中 UDP 校验和固定 `0xFFFF` 不进行 UDP 层校验。TYPE1 的 MAC 报头中的 DA 域需要进行校验，TYPE2 不需要对 DA 域校验。

### AFH_GEN1 Format

**[DOC]** MAC 报头为 AFH_GEN1 格式报文头。TYPE1 的 MAC 报头中的 DA 域需要进行校验，TYPE2 不需要。

### AFH_GEN2_16b Format

**[DOC]** MAC 报头为 AFH_GEN2 格式报文头，其中 TYPE1_ACK 中的 RSV 域填充 0。TYPE1 的 MAC 报头中的 DA 域需要进行校验，TYPE2 不需要。

### AFH_Lite Format

**[DOC]** MAC 报头为 AFH_GEN2_16b 压缩格式报文头，根据 AFH_GEN2_16b 中的 Traffic Class 域区分 TYPE1 和 TYPE2。不支持 CRC 保护。

---

### 12.3.1 MAC 报文头

**[DOC]** 定义 `Key_words[7:0] = {W, V, Rsv, Rsv, Z, Y, X, M}`

不同报文格式下 DA/SA 填充方式:

#### Standard Format

| 字段 | 填充方式 |
|------|----------|
| MAC_SA[47:0] | `[47:0]`: Source Addr |
| MAC_DA[47:0] | `[47:0]`: Dest Addr |

#### AFH_GEN1 Format

| 字段 | 填充方式 |
|------|----------|
| MAC_SA[47:0] | `[47:40]`: Traffic Class; `[39:0]`: Source Addr |
| MAC_DA[47:0] | `[47:0]`: Dest Addr |

#### AFH_GEN2_16b Format

| 字段 | 填充方式 |
|------|----------|
| MAC_SA[47:0] | `[47:40]`: Key_words; `[39:16]`: REQ 时填 SUE_payload，ACK 时填 Reserve; `[15:0]`: Source Addr |
| MAC_DA[47:0] | `[47:40]`: Traffic Class; `[39:16]`: REQ 时填 SUE_payload，ACK 时填 Reserve; `[15:0]`: Dest Addr |

**[DOC]** Key_words 各标志位定义:

| 位 | 名称 | 说明 |
|----|------|------|
| [7] | W | 0 表示含跳数和熵值的 AFH_GEN2_32b 格式，1 表示不含跳数和熵值的 AFH_GEN2_16b 格式。默认为 1 |
| [6] | V | 版本号格式标志位，默认为 0 |
| [5:4] | Rsv | 保留 |
| [3:2] | Z, Y | 默认设置 00，基于 SLAP 协议的 AAI 编码 |
| [1] | X | 0 表示全局分配 (DA/SA 使用全局 MAC 地址)，1 表示本地分配地址。默认为 1 |
| [0] | M | 0 表示单播，1 表示多播。TYPE1 时为 0，TYPE2 时为 1 |

**[DOC]** 主要寄存器信号:

| 信号 | 说明 |
|------|------|
| `mac_addr[47:0]` | Local device MAC Address，填充 Source_addr 域 |
| `dest_mac_addr[47:0]` | MAC address of the remote device，TYPE1 使用 `dest_mac_addr`，TYPE2 使用 `dest_mac_addr0`，填充 Dest_addr 域 |
| `TRAFFIC_CLASS` | 填充 Traffic Class[7:0] 域，其中 `[7:2]` 体现流量类型及优先级，使能 CBFC 时体现 VC 信息 |

**[DOC]** Traffic Class 寄存器组:
- `type1_req_tc_bank0[7:0]` / `type1_req_tc_bank1[7:0]` / `type1_req_tc_bank2[7:0]` / `type1_req_tc_bank3[7:0]`
- `type1_ack_tc[7:0]`
- `type2_req_tc[7:0]`

### 12.3.2 IP 报文头

**[DOC]** IP 报文头各字段定义:

| 字段 | 位域 | 值/来源 |
|------|------|---------|
| 版本号 (Version) | [3:0] | `4'd4` (IPv4) |
| 首部长度 (Header Length) | [7:4] | `4'd5` (5 word = 20 Byte) |
| 服务类型 (Type of Service) | [15:8] | 来自寄存器 TRAFFIC_CLASS，由 `{DSCP[5:0], ECN[1:0]}` 组成。使能 CBFC 时 DSCP 域体现 VC 信息。接收侧 ECN 标志为低 2bit 为 `2'b11` |
| 总长度 (Total Length) | [31:16] | 以字节为单位的 IP 包长度 (包括头部和数据)，随数据流 |
| 标识 (Identifier) | [47:32] | 来自寄存器静态配置 `IP_ADDR bit[47:32]` |
| 标志 (Flags) | [50:48] | 固定值 `3'b010` |
| 片偏移 (Fragment Offset) | [63:51] | 固定值 `13'h0` |
| 生存时间 (TTL) | [71:64] | 来自寄存器 `IP_ADDR bit[55:48]`，建议缺省值 64 |
| 协议 (Protocol) | [79:72] | `8'h11` (UDP，协议号 17) |
| 首部校验和 (Header Checksum) | [95:80] | 硬件产生，16 bit |
| 源 IP (Source Address) | [127:96] | 静态寄存器配置 `IP_ADDR[31:0]` |
| 目的 IP (Destination Address) | [159:128] | 静态配置 `DEST_IP_MEM[31:0]` |

**[行业]** 协议号 `0x11` 即十进制 17，对应 UDP 协议，这与 RoCEv2 标准一致，RoCEv2 将 RDMA 报文封装在 UDP/IPv4 中传输。

### 12.3.3 UDP 报文头

**[DOC]** UDP 报文头字段:

| 字段 | 长度 | 值/来源 |
|------|------|---------|
| 源端口 (source port) | 2B | TYPE1: 来自寄存器 `UDP_SRC_PORT_MEM`; TYPE2: `UDP_PORT_MEMx` |
| 目的端口 (dest port) | 2B | TYPE1: 固定值 `16'd4791` (`0x12B7`, RoCE V2); TYPE2: `UDP_PORT_MEMx` |
| 长度 | 2B | 根据 MTU 切分，最后一个包不满 MTU，包长在 12KB 以内。最小值为 8 (UDP 报文头长度) |
| 校验和 (checksum) | 2B | 固定值 `0x0`，不校验 |

**[行业]** 目的端口 4791 (`0x12B7`) 是 IANA 分配给 RoCEv2 的标准端口号，表明 TYPE1 协议层与 RoCEv2 兼容。

### 12.3.4 VLAN_TAG

**[DOC]** VLAN_TAG 字段:

| 字段 | 位宽 | 来源 |
|------|------|------|
| CFI | 1b | 来自寄存器 `VLAN_CTRL` |
| VID | 12b | 来自寄存器 `VLAN_CTRL` |
| PRI | 3b | CBFC 和 PFC 共用一套寄存器 |
| TPID | 16b | 来自寄存器 `VLAN_CTRL` |

**[DOC]** PRI 字段来自以下寄存器 (CBFC/PFC 共用):
- `type1_req_vc_bank0_cbfc[2:0]` / `type1_req_vc_bank1_cbfc[2:0]` / `type1_req_vc_bank2_cbfc[2:0]` / `type1_req_vc_bank3_cbfc[2:0]`
- `type1_ack_vc_cbfc[2:0]`
- `type2_vc_cbfc[2:0]`
- `type3_vc_cbfc[2:0]`

**[DOC]** PRI 同时表示 CBFC 和 PFC 优先级映射。对于 TYPE1_REQ 分为 4 个 BANK 区，根据 `SRC_QPID[QP_AW-1:0]` 最低 2bit 进行区分 (要求 `SRC_QPID[1:0] = DEST_QPID[1:0]`):
- bank0: 00
- bank1: 01
- bank2: 10
- bank3: 11

### 12.3.5 RH (Routing Header)

**[DOC]** RH 结构 (TYPE1 REQ 和 ACK 共用):

| 位域 | 字段 | 说明 |
|------|------|------|
| [12:0] | Dest_QP | QP 序列号 |
| [24:13] | PSN | 数据包序列号 |
| [31:25] | REQ: Pkt_len / ACK: Aeth_syndrom | REQ: 指示 RH+payload 长度，<=127 时为实际长度，>127 时为 0; ACK: 用于指示 ACK/NAK 类型及 RNR Time |
| [39:32] | P_key | Partition Key，用于 InfiniBand 网络隔离，只有具有同样 PKEY 的 QP 才可以通信 |
| [55:40] | Timestamp | 时间戳信号，支持 RTT 测量 |
| [56] | Timestamp_en | 时间戳有效标志信号 |
| [58:57] | Rsv | 保留 |
| [60:59] | Pad | Payload 长度是 4Byte 对齐的，当不足整 4Byte 时通过 Pad 指示 |
| [61] | Fack | REQ 报文 FACK 标记，带标记的请求强制对端回复 ACK; ACK 报文中填充 0 |
| [63:62] | Opcode | `00`: SEND Req; `01`: ACK/NAK; `10`: CNP; `11`: Reserve |

**[DOC]** 时间戳规则: 在不同报头格式下，TYPE1 REQ 报文均插入时间戳。对于 TYPE1 ACK/FACK 报文，根据寄存器信号 `tx_ack_time_en` 决定:
- `tx_ack_time_en=1`: ACK/FACK 均携带时间戳
- `tx_ack_time_en=0`: 只有 FACK 携带时间戳

---

## 12.4 速率控制

**[DOC]** 对应框图中的 `crdt_ctrl` 模块。功能是基于软件配置的发送速率信息，控制对应 QP 的发送流量。

### 控制原理

**[DOC]** 具体实现方法是设置一个软件可配置的时间窗口 (CC_WINDOW)，软件将发送速率转换为这个时间窗口内允许通过的字节数 (`length_thr`)，通过 `SET_CC_RATE` 接口设置。

**[DOC]** 时间窗口可配值: **4.096 / 8.192 / 16.384 / 32.768 / 65.536 us** (由寄存器配置)。

### ACC_CNT 累加机制

**[DOC]** 模块统计每个 QP 在每个时间窗口内发送 WQE 对应数据量的累加值 `acc_cnt`:
- 当 WQE 有效时，加上 WQE 对应的数据量
- 当时间窗口复位 (`cc_win_reset`) 时，减去 `length_thr` (下饱和到 0)

**[DOC]** 控制逻辑:
- 当 `acc_cnt >= length_thr` 时: SET `ARB_REQ_MASK_O`，mask 对应 QP 的请求 (阻止发送)
- 当 `acc_cnt < length_thr` 时: CLR `ARB_REQ_MASK_O`，unmask 对应 QP 的请求 (允许发送)

### 工作示例

**[DOC]** 以 QP2 为例，控制时间窗口内发送数据量在 `0x30` bytes 以下:

1. QP2 的第一个 `cc_win_rst` 到来后，从 `ACC_CNT_MEM` 和 `SPEED_MEM` 中读出 `acc_cnt=0` 和 `speed=0xFF`，将 `max(acc_cnt-speed, 0)=0` 写入 `ACC_CNT_MEM`
2. 驱动降速，更新速率为一个时窗最多发出 `0x30` bytes 数据
3. TX 第一个消息: `length=0x10`，`acc_cnt_wdata=0+0x10=0x10`，`0x10 < 0x30`，不改变 `ARB_REQ_MASK[2]`
4. TX 第二个消息: `length=0x30`，`acc_cnt_wdata=0x10+0x30=0x40`，`0x40 > 0x30`，将 `ARB_REQ_MASK[2]` 置位，该时窗内 QP2 不能再发包
5. 第二个 `cc_win_rst` 到来: `max(0x40-0x30, 0)=0x10` 写入 `ACC_CNT_MEM`，`0x10 < 0x30`，清零 `ARB_REQ_MASK[2]`
6. TX 第三个消息: `length=0x20`，`acc_cnt_wdata=0x10+0x20=0x30`，`0x30 >= 0x30`，置位 `ARB_REQ_MASK[2]`

**[推导]** 该速率控制机制通过时间窗口+累加计数器实现 per-QP 粒度的带宽管控，跨窗口的"欠账"会被带入下一个窗口 (减去 `length_thr` 后的剩余部分)，确保长期平均速率不超过配置值。

### TYPE2 速率控制

**[DOC]** TYPE2 也支持独立速率控制，通过 `SET_TYPE2_RATE_VLD_I` 和 `SET_TYPE2_RATE_DAT_I[21:0]` 接口设置窗口期内最大发送字节数。

### 速率控制接口

| 信号 | 方向 | 说明 |
|------|------|------|
| `SET_CC_RATE_VLD_I` | I | TYPE1 通道目标速率更新指示 |
| `SET_CC_RATE_DAT_I[21:0]` | I | 窗口期内最大发送字节数 |
| `SET_CC_RATE_QPID_I[QP_AW-1:0]` | I | 需要更新速率的队列号 |
| `SET_TYPE2_RATE_VLD_I` | I | TYPE2 通道速率更新指示 |
| `SET_TYPE2_RATE_DAT_I[21:0]` | I | 窗口期内最大发送字节数 |
| `CC_TYPE1_QP_EN_SET_O` | O | 允许发送 TYPE1 对应队列数据 |
| `CC_TYPE1_QP_EN_CLR_O` | O | 禁止发送 TYPE1 对应队列数据 |
| `CC_TYPE1_QP_ID_O[QP_AW-1:0]` | O | TYPE1 对应队列号 |

**[DOC]** `CC_TYPE1_QP_EN_SET_O` 和 `CC_TYPE1_QP_EN_CLR_O` 不会同时为高。这组信号的目的是给 TYPE1 数据源端提供参考，避免当某个 QP 控速速率较低时，IP 内部控速模块已经停止发送该 QP 数据，源端仍然在持续输出该 QP 的报文，使 IP 内部 buffer 被此 QP 数据占用，导致其他 QP 发送被阻塞。

---

## 12.5 CBFC 流量控制

### 基本原理

**[DOC]** CBFC 是一种端到端的流量控制机制。发送方在协商阶段获取对端接收方的总 Credit 数量，并在工作过程中主动地计算每个报文消耗的 Credit 数量 (按 VC 粒度)，以跟踪接收方的可用缓冲空间。若剩余的 Credit 不足 (缓冲空间总容量 = CREDIT 数量 x 单个 CREDIT 对应的存储空间)，此时数据包调度程序禁止该 VC 从 lossless 队列调度数据包进行传输。

**[DOC]** 接收方维护本地 CREDIT 数量，并定期将可用 CREDIT 数量同步给发送端，以补偿丢包造成的 Credit 泄露。接收方在收到数据包时根据 CREDIT_SIZE 和数据包长来计算需要消耗的 CREDIT；当数据包从缓冲区弹出时归还对应数量的 CREDIT。

### VC 通道与物理流量映射

**[DOC]** RCLINK 支持 8 个虚拟通道 (VC)，各 VC 维护各自的剩余 CREDIT 计数器，不同 VC 的流量彼此独立、分别控制。RCLINK 中共有 **7 路实际的物理流量**，需保证物理流量与 VC 之间一一映射。

**[DOC]** 物理流量到 VC 的映射:

| 物理流量 | VC 映射寄存器 | CBFC 使能寄存器 | 说明 |
|----------|-------------|----------------|------|
| TYPE1_REQ Bank0 | `type1_req_bank0_cbfc[2:0]` | `rx_type1_req_bank0_cbfc_en` | 按 QPID 低 2bit=00 划分 |
| TYPE1_REQ Bank1 | `type1_req_bank1_cbfc[2:0]` | `rx_type1_req_bank1_cbfc_en` | 按 QPID 低 2bit=01 划分 |
| TYPE1_REQ Bank2 | `type1_req_bank2_cbfc[2:0]` | `rx_type1_req_bank2_cbfc_en` | 按 QPID 低 2bit=10 划分 |
| TYPE1_REQ Bank3 | `type1_req_bank3_cbfc[2:0]` | `rx_type1_req_bank3_cbfc_en` | 按 QPID 低 2bit=11 划分 |
| TYPE1_ACK + CNP | `type1_ack_cbfc[2:0]` | `rx_type1_ack_cbfc_en` | ACK 和 CNP 映射到同一 VC |
| TYPE2 | `type2_cbfc[2:0]` | `rx_type2_cbfc_en` | |
| TYPE3 | `type3_cbfc[2:0]` | `rx_type3_cbfc_en` | |

**[DOC]** TYPE1_REQ 通过参数配置划分为 1、2、4 个 BANK，按 QPID 的低位划分，可映射到 1、2、4 个 VC。不同 BANK 不可映射到同一 VC 上。暂不支持将多个流量映射给同一 VC。

### 初始化阶段

**[DOC]** 初始化阶段接收 MAC 的配置信息，主要配置包括:
- **CREDIT_SIZE**: 单个 Credit 对应的字节数，可选值: 32, 64, 128, 256, 1024, 2048 byte
- **CREDIT_LIMIT (CL)**: 一个 VC 可拥有的最大 Credit 数量
- **PKT_OVHD_LEN**: PktOvhd 值，为有符号数 (bit[7:9] 为符号位)

**[DOC]** 支持各 VC 独立配置，若多次配置则以最后一次配置为准。`MAC2TX_CBFC_RST_I[31:0]` 的低 8bit 用来指示配置信息有效。可以单拍拉高所有 VC 的 RST 信号将所有 VC 初始化为同一配置，或每拍拉高对应 VC 的 RST 位对每个 VC 单独初始化。初始化通常建议在 RCLINK 工作前完成。

### 工作阶段

**[DOC]** 工作阶段:
- **Credit 消耗**: 计算下发包消耗的 CREDIT 数量
- **Credit 返还**: 通过 `MAC2TX_CBFC_VLD_I[31:0]` 的低 8bit 指示有释放行为的 VC (必须 one-hot)，`MAC2TX_CBFC_NUM_I` 信号指示释放 CREDIT 的数量 (默认位宽 15)
- **流控触发**: 当剩余 CREDIT 数量低于 `CREDIT_UF_LIMIT` 时，拉低对应 VC 的状态位，阻塞该 VC 的报文参与发射仲裁

### Credit 下限配置

**[DOC]** 通过寄存器 `credit_uf_limit` 配置对应报文类型的 CBFC Credit 下限:
- 可配范围: **1~7**
- 单位: 一个最大报文消耗的 Credit 数量
- **禁止配置为 0** (会导致 Credit 泄露溢出)

**[DOC]** 寄存器地址:
- `type1_credit_uf_limit_cbfc`: `0x0160[38:36]`
- `type2_credit_uf_limit_cbfc`: `0x0160[41:39]`
- `type3_credit_uf_limit_cbfc`: `0x0160[44:42]`
- `ack_credit_uf_limit_cbfc`: `0x0160[47:45]`

### 动态下限调节

**[DOC]** 由 `dyn_uf_limit_cbfc_en` 寄存器配置。当剩余 CREDIT 数量低于水线但仍大于一个最大包长时，可以动态地再下发一个该 VC 的报文，直到剩余 CREDIT 不足以下发一个最大报文。

**[DOC]** 注意: 仅 TYPE1 REQ 类型的报文支持该功能，TYPE2、TYPE3、ACK 类型报文不支持。

寄存器地址:
- `type1_dyn_uf_limit_cbfc_en`: `0x0160[48]` (配置为 1)
- `type2_dyn_uf_limit_cbfc_en`: `0x0160[49]` (必须配置为 0)
- `type3_dyn_uf_limit_cbfc_en`: `0x0160[50]` (必须配置为 0)
- `ack_dyn_uf_limit_cbfc_en`: `0x0160[51]` (必须配置为 0)

### 软件流控

**[DOC]** 可通过配置 `software_ctrl_cbfc_vc_status` (`0x0160[59:52]`) 配置对应 VC 的流控信号:
- 配置为 0: 强制停止该 VC 流量，无视 Credit 数量
- 配置为 1: 使能该通道流量

**[DOC]** 注意: 不支持在对应 VC 的 Credit 数量低于下限时，通过配置该寄存器打开该 VC 流量。

### 配置约束

**[DOC]** CBFC 配置注意事项:
1. 需要保证 CL 值至少大于下限所代表的 Credit 数量，否则该 VC 会被一直阻塞
2. 若想打开某个 VC 的 CBFC 流量控制，需保证全局的 `STOP_CBFC_EN` 为 0 (`0x0160[5]`)，各 VC 的 CBFC 使能信号配置为 1
3. 需要至少配置 `credit_uf_limit` 为 1，配置为 0 会导致该 VC 的 CBFC 流控失效
4. CBFC 和 PFC 不支持同时使用
5. 使能 CBFC 时，网络中所有节点的 VCID 或 PRIORITY 配置必须保持一致
6. 需要使能 CBFC 和 PFC 时，建链双方需为相同 BANK 的 QP，不支持跨 BANK 的 QP 之间建链
7. 不支持同一个 VCID 或 PRIORITY 配置给不同的数据流

### RX 方向反压

**[DOC]** 在 RX 方向，共用 `TX_PFC_REQ_O[7:0]` 端口反压 MAC 端，每 bit 代表一个 VC。反压信号由 RCLINK 的 RX 端 Buffer 水限和上游客户逻辑产生的 `FLOW_STOP` 信号共同产生。

**[DOC]** 使能 CBFC 时，客户 `FLOW_STOP` 仅产生 `TX_PFC_REQ_O[7:0]` 信号表示向 MAC 的反压。由于 CBFC 的反压与 PFC 共用接口，在 RX 方向需要将 PFC TX 方向的各 VC 使能寄存器 `tx_typeX_pfc_en` 置 1，否则 CBFC 不能通过 PFC 接口反压。

**[DOC]** 通过寄存器 `stop_cbfc_en` (`0x0160[5]`) 可禁用全局的 CBFC，该控制信号拥有最高优先级。

---

## 12.6 多播报文流量控制

**[DOC]** 在 RCLINK 中，对于多播报文有额外的基于 Credit 的流量管理，与 CBFC 流量控制不同，该 Credit 仅反映**下游 MAC TX 方向的多播 Buffer 的空间**。

**[DOC]** 关键差异:
- CREDIT_SIZE 和 CREDIT_LIMIT 与 MAC TX Buffer 尺寸相关 (而非端到端接收缓存)
- TYPE2 多播报文同时受 CBFC 和该流量控制的**双重约束**
- 多播报文流量管理**复用 CBFC 控制器模块**
- **不支持**软件流控和动态调节下限功能
- **不受** MAC 强制复位的影响
- 受到独立的 `multi_credit_ctrl_en` 寄存器控制

**[DOC]** 通过与 MAC TX 接口完成 credit 配置和返还，时序与 CBFC 相同。CREDIT_SIZE 和 CREDIT_LIMIT 均通过参数直接配置。

**[DOC]** MAC 接口信号:

| 信号 | 说明 |
|------|------|
| `MAC2TX_TYPE2_CRD_VLD_I` | MAC 释放多播 Credit 有效指示 |
| `MAC2TX_TYPE2_CRD_NUM_I[CRD_NUM-1:0]` | 释放的 Credit 数量，Credit cell size 为 128 Byte |
| `MAC2TX_TYPE2_CREDIT_RST_I` | 恢复 Credit 到初始值 (TYPE2_CRD_RST_NUM) |

---

## 12.7 ACK MERGE

**[DOC]** 对应框图中的 `ACK_MERGE` 模块。

### 实现机制

**[DOC]** 设置一个深度为 `2^QP_AW` 的寄存器 Buffer:
- 当 `SET_EN_I` 有效时，将对应的 `Buffer[SET_QPID_I]` 位置 1
- 当 `CLEAR_EN_I` 有效时，将对应的 `Buffer[CLEAR_QPID_I]` 置 0

**[DOC]** 维护一个指针 Pointer，以 **4 组为单位**从初始位置检查 Buffer 相应位是否为 1。如果 Pointer 指向的对应位置为 1，则握手成功后将相应的 QPID 发出，然后将 `Buffer[QPID]` 置 0。

**[推导]** ACK MERGE 的作用是合并同一 QP 的多个 ACK 请求，减少 ACK 报文的数量。由于以 4 组为单位轮询检查，可以在多个 QP 之间公平调度 ACK 发送。

---

## 12.8 CNP MERGE

**[DOC]** 对应框图中的 `data_filter` 模块。该模块的功能是过滤掉在一段时间内连续到来的同一个 QPID 的 CNP 报文。

### 实现机制

**[DOC]** QPID_Buffer 深度为 **8**，表示最多记录 8 个 QPID。

**[DOC]** 工作流程:
1. 当 CNP MERGE 功能未打开时，QPID 及对应数据直接透传
2. 当功能开启时:
   - 如果 QPID 不在 `QPID_Buffer` 内: 直接透传
   - 如果 QPID 在 `QPID_Buffer` 内: 检查当前 QPID 出现的时间与 `TIMER_Buffer` 内相同 QPID 的 `qpid_time` 的时间差
     - 时间差 **<=** `TIMER_THR_I`: 将该 QPID 进行屏蔽处理 (过滤)
     - 时间差 **>** `TIMER_THR_I`: 将该 QPID 进行透传，并将其放入 `ID_BUFFER` 中

**[DOC]** 这保证了在 `TIMER_THR_I` 时间窗口内，过滤掉连续到来的同一个 QPID 的 CNP 报文。

**[推导]** CNP MERGE 功能的设计目的是避免拥塞通知风暴。当网络发生拥塞时，短时间内可能收到大量针对同一 QP 的 CNP，若不合并则会造成 CC (Congestion Control) 模块过度降速。

---

## 12.9 TYPE1 发送仲裁

### Slot 状态机

**[DOC]** 上游下发的 TYPE1 数据报文会顺序进入一个队列，队列中的 slot 有四种状态:

```
            新数据写入           被仲裁发出
  EMPTY  -----------> WAIT_GRANT -----------> WAIT_ACK
    ^                     ^    \                  |
    |                     |     \  被控速阻塞     |
    |        重传请求     |      v                |
    |  被 ACK 确认        |    WAIT_CRD           |
    +---------------------+       (控速撤销后     |
                                   返回 WAIT_GRANT)|
```

**[DOC]** 状态转换规则:
- **EMPTY**: 未被使用的 slot
- **EMPTY -> WAIT_GRANT**: 有上游下发新的数据报文写入队列
- **WAIT_GRANT -> WAIT_ACK**: 对应 slot 报文被仲裁发出
- **WAIT_ACK -> WAIT_GRANT**: 有重传请求
- **WAIT_ACK -> EMPTY**: slot 中的报文被 ACK 确认
- **WAIT_GRANT -> WAIT_CRD**: 当报文被控速模块阻塞时 (set `cc_mask`)
- **WAIT_CRD -> WAIT_GRANT**: 控速模块阻塞撤销 (clear `cc_mask`)

### 仲裁规则

**[DOC]** 所有 WAIT_GRANT 状态的报文按照进入队列的**先进先出顺序**仲裁，先进入队列的优先输出。

### Bank 轮转机制

**[DOC]** 当 `tx_bank_rr_en` 置 1 时，按照 QPID 低位将报文分为 `TYPE1_VC_NUM` 个 bank，bank 的报文轮流发出，每个 bank 内的 WAIT_GRANT 状态报文仍然按照进入队列的顺序发出。

**[DOC]** 实现方法: 每次发出一个报文后，记录发出报文 `qpid_msb`，检查目前队列中各个 slot 中报文的 `qpid[1:0]`，与 `qpid_msb` 相同的 slot 对应的 mask 置 1，mask 置 1 的 slot 不参与下一次的发送仲裁。当队列中仅存在其中一个 bank 的报文时，slot 的 mask 不生效。

**[推导]** Bank 轮转机制确保了不同 VC 通道之间的公平性，防止单个 bank 的报文独占发送带宽，同时在只有单个 bank 有报文时不会造成带宽浪费。

---

## 12.10 LITE 模式组网

**[DOC]** 在 LITE 模式下，只支持数据调度发送与接收，**不进行端对端的重传保护**。

### QPID 编码方式

**[DOC]** 上游将 XPU_ID 和虚拟通道信息传输至 RCLINK:

```
Source_qpid = {XPU_ID, BANK}
```

**[DOC]** 根据 `TYPE1_VC_NUM` 划分不同 BANK 区间，BANK 信息与 VC 信息存在映射关系。

### 报文封装

**[DOC]** 通过查找表索引出 `Dest_qpid`，然后:
1. 将 XPU_ID 信息封装至报头中 `MAC_SA` 字段高位
2. 报头中 `Traffic Class[7:0]` 携带 VC 信息

### 接收解析

**[DOC]** 远端接收到 LITE 报文后:
1. 解析出 XPU_ID 信息
2. 通过 `Traffic Class[7:0]` 映射出 BANK 信息
3. 组成完整的 QPID 信息

### 接收报文识别

**[DOC]** RX 方向根据以下逻辑判断接收报文属性:

```
if (Eth_type == 16'h0800)       // IP
    if (ip_protocol == 8'h11)   // UDP
        case (udp_dport)
            type1_udp_dport: TYPE1
            type2_udp_port:  TYPE2
            default:         TYPE3
else
    case (Eth_type)
        MAC_TYPE1: TYPE1
        MAC_TYPE2: TYPE2
        default:   TYPE3
```

---

## 12.11 硬件接口

### Clock/Reset

| 信号 | 方向 | 位宽 | 说明 |
|------|------|------|------|
| `CLK_I` | I | 1 | 最大 1GHz |
| `RST_N_I` | I | 1 | 全局复位信号，低有效 |

**[DOC]** RC_LINK 采用全同步设计，整个 IP 只有一个时钟域，除静态配置信号外，其他信号均保持 `CLK_I` 同步。

### 配置接口

| 信号 | 方向 | 说明 |
|------|------|------|
| `CFG_S_AXI*` | I/O | AXI-Lite 寄存器访问接口 |
| `IS_400G_I` | I | 支持 200/400 两种 MAC 接口，分别采用 512b/1024b MAC 数据位宽 |
| `TX_WAIT_MAC_CRD_O` | O | 表示当前有数据在等待 MAC 返回 credit |
| `TX_MAC_FREE_O` | O | 表示当前 MAC TX 方向无数据发送 |

### 12.11.1 TX_S: 上游数据发送接口

**[DOC]** 本地上层逻辑通过 TX_S 口将待发送报文交给 IP。第一个 byte 在 `[511:504]`，tkeep 有气泡时气泡在低位 `[0]`。

**Control 信号:**

| 信号 | 说明 |
|------|------|
| `TX_S_VLD_I` | 数据有效指示 |
| `TX_S_CRD_O[3:0]` | 令牌反馈信号，为高时标识有一个包缓冲区释放。`[0]`-type1, `[1]`-rsv, `[2]`-rsv, `[3]`-type2 |
| `TX_S_CRD_QPID_O[QP_AW-1:0]` | 正常返回的 Credit 对应的 QPID (不包括 oversize、qp disable 的情况) |
| `TX_S_CRD_QPID_VLD_O` | 指示当前 QPID 为有效的、正常返回的 Credit 对应的 QPID |

**Data 信号:**

| 信号 | 说明 |
|------|------|
| `TX_S_FS_I` | 包头指示 |
| `TX_S_FE_I` | 包尾指示 |
| `TX_S_DAT_I[DW-1:0]` | 包数据，高字节先发送，数据量最小为 64 bytes |
| `TX_S_QPID_I[QP_AW-1:0]` | RC 队列号。`TYPE1_VC_NUM=1` 时无 VC 编号; `=2` 时 `[0]` 作为 bank 编号; `=4` 时 `[1:0]` 作为 bank 编号 |
| `TX_S_FACK_I` | 报文 FACK 标记，带标记的请求对端将尽快回复 ACK |
| `TX_S_KEEP_I[DW/8-1:0]` | strb 信号，一定是连续的 |
| `TX_S_ETYPE_I[1:0]` | `0`-type1, `1`-rsv, `2`-rsv, `3`-type2 |
| `TX_S_TYPE2_GROUP_INDEX_I[2:0]` | Multicast group number |

### 12.11.2 RX_M: 接收数据接口

**[DOC]** 接收报文通过 RX_M 接口交给本地系统。TYPE1/TYPE2 为独立数据流。`FLOW_STOP` 电平信号由客户逻辑产生。

| 信号 | 说明 |
|------|------|
| `TYPE1_2_RX_M_VLD_O` | 数据有效指示 |
| `TYPE1_2_RX_M_RDY_I` | 接收就绪 |
| `TYPE1_2_RX_M_FS_O` | 包头指示 |
| `TYPE1_2_RX_M_FE_O` | 包尾指示 |
| `TYPE1_2_RX_M_DAT_O[DW-1:0]` | 包数据 |
| `TYPE1_2_RX_M_QPID_O[QP_AW-1:0]` | QP 队列号 |
| `TYPE1_2_RX_M_KEEP_O[DW/8-1:0]` | 字节有效指示 |
| `TYPE1_2_RX_M_ETYPE_O[1:0]` | `00`-type1, `01`-rsv, `10`-rsv, `11`-type2 |
| `TYPE1_W_RX_M_LEN_O[12:0]` | Payload length |
| `TYPE1_2_RX_M_SA_O[9:0]` | MAC source addr |

### 12.11.3 NONR_MD_AXI: TYPE3 MD 读写接口

**[DOC]** 最大 outstanding 是读写各 128，数据位宽 256bit，地址位宽 64bit。

**AR Channel:**

| 信号 | 说明 |
|------|------|
| `NONR_MD_AXI_ARVALID_O` / `ARREADY_I` | 握手信号 |
| `NONR_MD_AXI_ARADDR_O[63:0]` | 读地址 |
| `NONR_MD_AXI_ARLEN_O[7:0]` | 固定为 0 |
| `NONR_MD_AXI_ARSIZE_O[2:0]` | 4 或 5 |
| `NONR_MD_AXI_ARBURST_O[1:0]` | 固定为 `2'h1` (Burst INC) |
| `NONR_MD_AXI_ARID_O[6:0]` | 0~127，未完成传输请求的 ID 不会重复 |

**AW/W Channel:**

| 信号 | 说明 |
|------|------|
| `NONR_MD_AXI_AWLEN_O[7:0]` | 固定为 0 |
| `NONR_MD_AXI_AWSIZE_O[2:0]` | 固定为 4 (16 byte) |
| `NONR_MD_AXI_AWBURST_O[1:0]` | 固定为 `2'h1` (Burst INC) |
| `NONR_MD_AXI_WSTRB_O[31:0]` | 只更新 MD 中需要写回部分: `31'h003f_0000` 或 `31'h0000_003f` |
| `NONR_MD_AXI_BUSY_O` | `1'b1` 表示当前 MD_AXI 有未完成传输 |

### 12.11.4 NONR_DATA_AXI: TYPE3 数据接口

**[DOC]** 最大 outstanding 是读写各 128，数据位宽 512bit，地址位宽 64bit。发送方向最大 burst len 为 256 byte，接收方向最大 burst len 为 512 byte。

| 参数 | AR Channel | AW Channel |
|------|-----------|-----------|
| ID 范围 | 0~127 | 0~127 |
| ARLEN/AWLEN 最大值 | 3 (4 beat) | 7 (8 beat) |
| SIZE | 固定 6 (64 byte) | 固定 6 (64 byte) |
| BURST | 固定 `2'h1` (INC) | 固定 `2'h1` (INC) |

### 12.11.5 MAC TX 接口 (CEMAC_TX)

| 信号 | 说明 |
|------|------|
| `MAC2TX_CREDIT_I` | MAC 释放 1 个 credit |
| `MAC2TX_CREDIT_RST_I` | 恢复 credit 到初始值 |
| `TX2MAC_DATA_O[MAC_DW-1:0]` | 发送数据流 |
| `TX2MAC_DATA_BE_O[MAC_BE-1:0]` | 数据字节使能 |
| `TX2MAC_VLD_O` | 数据有效指示 |
| `TX2MAC_FS_O` | 帧起始 |
| `TX2MAC_FE_O` | 帧结束 |
| `TX2MAC_VCID_O[2:0]` | 数据 VCID |
| `TX2MAC_FCS_O` | 固定为 1 |
| `TX2MAC_ERR_O` | 固定为 0 |
| `TX2MAC_CBFC_VCID_O[VC_CH_N-1:0]` | 指示 VC Channel ID 给 MAC |
| `MAC2TX_CBFC_RESET_I` | 清除所有 CBFC 相关寄存器和计数器，设 CBFC 为阻塞状态 |
| `TX2MAC_MAC_HDR_TYPE_O[1:0]` | `00`-Standard, `01`-AFH_GEN1, `10`-AFH_GEN2_16b, `11`-AFH_LITE |
| `TX2MAC_MULTICAST_EN_O` | 多播标志 |
| `TX2MAC_VLAN_EN_O` | VLAN 使能 |

### 12.11.6 MAC RX 接口 (CEMAC_RX)

| 信号 | 说明 |
|------|------|
| `RX2MAC_RDY_O` | 接收就绪 |
| `MAC2RX_VLD_I` | 数据有效指示 |
| `MAC2RX_FS_I` | 帧起始 |
| `MAC2RX_FE_I` | 帧结束 |
| `MAC2RX_DATA_I[MAC_DW-1:0]` | 接收数据 |
| `MAC2RX_DATA_BE_I[MAC_BE-1:0]` | 字节使能，前 N-1 拍为全 0，FE 拍显示尾拍有效字节数 |
| `MAC2RX_FCS_ERR_I[1:0]` | FCS 错误指示 |

### 12.11.7 流量控制接口

| 组 | 信号 | 说明 |
|----|------|------|
| PFC | `TX_PFC_REQ_O[7:0]` | 请求 MAC 发送流控帧，停止入站数据流 |
| PFC | `RX_PFC_REQ_I[7:0]` | MAC 收到流控帧，停止本地出站数据流 |
| FLOW_STOP | `TYPE1_FLOW_STOP_I[3:0]` | TYPE1 按 QP 号最多分 4 个 bank，客户可分别停流 |
| FLOW_STOP | `TYPE2_PFC_REQ_I` | 客户 TYPE2 接收侧反压信号 |
| RECV_CNP | `RECV_CNP_VLD_O` | 收到 CNP 报文指示 |
| RECV_CNP | `RECV_CNP_QPID_O[QP_AW-1:0]` | CNP 报文队列号 |
| RECV_ACK | `RECV_ACK_VLD_O` | 收到 ACK/NAK 指示 (只有包含时戳信息的 ACK 才会上报) |
| RECV_ACK | `RECV_ACK_QPID_O[QP_AW-1:0]` | ACK/NAK 报文队列号 |
| RECV_ACK | `RECV_ACK_RTT_O[31:0]` | ACK/NAK 报文的 RTT 延时 |
| RECV_ACK | `RECV_NUM_ACKED_O[9:0]` | ACK/NAK 确认报文数量 |
| RECV_ACK | `RECV_NAK_FLG_O` | 收到的 response 中包含 NAK 指示 |
| RETRY_TMO | `RETRY_TMO_VLD_O` | 发生超时重传 |
| RETRY_TMO | `RETRY_TMO_QPID_O[QP_AW-1:0]` | 重传队列号 |
| NOCNP_TIMER | `NOCNP_TIMER_VLD_O` | 没有收到 CNP 报文定时上报 (timer 间隔 1~256us 可配置) |
| NOCNP_TIMER | `NOCNP_TIMER_QPID_O[QP_AW-1:0]` | 上报的队列号 |
| NOCNP_TIMER | `NOCNP_TIMER_BYTE_CNT_O[31:0]` | 从上次上报到此次上报期间 TX 发送的总字节数 (为 0 时不上报，收到 CNP 后对应 QP 的 byte counter 会清零) |

### DFX 接口

| 信号 | 位宽 | 说明 |
|------|------|------|
| `DFX_TYPE1_OST_FREE_LVL_O` | 10 | TX OST Buffer 的空闲表项数量 |
| `DFX_CBFC_CF_O` | 7*16 | 7 个流量的可用 Credit 数量，16bit 为单位。从高到低: Type3、Type2、Ackcnp、Type1Bank3~Type1Bank0 |
| `RCLINK_IDLE_O` | 2 | `bit1`: TX IDLE Status; `bit0`: RX IDLE Status |

---

## 12.12 中断系统

### 12.12.1 MSI 中断接口

**[DOC]** MSI 消息接口:

| 信号 | 位宽 | 说明 |
|------|------|------|
| `IRQ_VLD_O` | 1 | 中断请求有效 |
| `IRQ_INFO_O[63:0]` | 64 | 中断信息 |

**[DOC]** 中断信息编码:

```
IRQ_INFO_O[63:33]: Reserved
IRQ_INFO_O[32:31]: 数据类型标识
  2'd1 = TYPE1
  2'd2 = TYPE2
  2'd3 = TYPE3
```

#### TYPE1 中断信息

```
IRQ_INFO_O[32:31] = 2'd1
IRQ_INFO_O[30:5+QP_AW]: Reserved
IRQ_INFO_O[4+QP_AW:5]: qpid
IRQ_INFO_O[4:0]: 中断信息
```

| IRQ_INFO_O[3:0] | 说明 | 处理方法 |
|------------------|------|----------|
| `4'd1` | TX packet size 超过 TYPE1_PKT_LEN | 丢掉超长包，继续收后续数据包 (所有 QPID 的包无差别接收) |
| `4'd2` | TX retry times out interrupt | 参见 10.3 节处理流程 |
| `4'd3` | 超 credit 报警 | 丢掉当前包，若后续有 credit 则收后续数据包 (不对丢包的 QPID 做区分) |
| `4'd4` | TX disable QP 接收到报文 | 丢掉当前包，上报中断 |
| `4'd5` | RX disable QP 接收到报文 | 丢掉当前包，上报中断 |

#### TYPE2 中断信息

```
IRQ_INFO_O[32:31] = 2'd2
IRQ_INFO_O[30:7]: Reserved
IRQ_INFO_O[7:5]: 子类型
IRQ_INFO_O[4:0]: 中断信息
```

| IRQ_INFO_O[3:0] | 说明 | 处理方法 |
|------------------|------|----------|
| `4'd1` | TX interrupt (超长包/超流量) | 丢掉超长包和超流量包，继续收后续数据包 |
| `4'd2` | RX interrupt (超长包) | 丢掉超长包，继续收后续数据包 |
| `4'd3` | RSV | - |
| `4'd4` | RX buffer overflow (`IRQ_INFO_O[7:5]=1`: type2) | - |

#### TYPE3 中断信息

```
IRQ_INFO_O[32:31] = 2'd3
IRQ_INFO_O[30:18]: rx md tail
IRQ_INFO_O[17:5]: tx md tail
IRQ_INFO_O[4:0]: 中断信息
```

| IRQ_INFO_O[3:0] | 说明 |
|------------------|------|
| `4'd1` | TYPE3 interrupt，需查询 `0x1150` 寄存器确认中断类型 |

### 12.12.2 RC_INTR_O 电平中断

**[DOC]** `RC_INTR_O` 为电平中断信号，各 bit 定义:

| Bit | 说明 |
|-----|------|
| 0 | TX TYPE1 packet size 超过 TYPE1_PKT_LEN |
| 1 | host 发送的 TYPE1 pkt ost 数量超过 TYPE1_OST_N |
| 2 | Reserved |
| 3 | TYPE1 某一个 QP 的重传次数到达配置阈值 |
| 4 | Reserved |
| 5 | Reserved |
| 6 | TX TYPE2 packet size 超过 TYPE2_PKT_LEN |
| 7 | Reserved |
| 8 | Reserved |
| 9 | host 发送的 TYPE2 pkt ost 数量超过 TYPE2_OST_N |
| 10 | Reserved |
| 11 | Reserved |
| 12 | RX TYPE2 packet size 超过 TYPE2_PKT_LEN |
| 13 | Reserved |
| 14 | TX TYPE1 disable QP 异常 |
| 15 | RX TYPE1 disable QP 异常 |

### 12.12.3 TYPE3 中断接口

**[DOC]** TYPE3 支持独立的电平信号中断接口:

**NONR_INTR_O (事务中断)** - 查询 `0x1150` 寄存器:

| Bit | 说明 |
|-----|------|
| 0 | rx md q empty int |
| 1 | tx md q empty int |
| 2 | rx tail up int (rx tail 前进到设置的值，`0x1190` 寄存器配置) |
| 3 | tx tail up int (tx tail 前进到设置的值，`0x1190` 寄存器配置) |
| 4 | abnormal_status_int (有异常中断) |

**NONR_FAULT_INTR_O (异常中断)** - 查询 `0x1170` 寄存器:

| Bit | 说明 | 处理 |
|-----|------|------|
| 0 | rx fifo overflow | 上报中断 |
| 1 | tx fifo underflow | 该中断不会触发 |
| 2 | md axi wr resp abnormal | 只上报中断 |
| 3 | md axi rd resp abnormal | 只上报中断 |
| 4 | data axi resp abnormal | 只上报中断 |
| 5 | tx get two sop (TX 方向投放的 md 连续两个只有 sop) | 需要复位 non-rdma |
| 6 | tx last eop (TX 方向投放的 md 连续两个只有 eop) | 需要软复位 non-rdma |
| 7 | tx buffer oversize (TX 发送报文长度超过 tx buffer size) | 需要软复位 non-rdma |
| 8 | rx buffer oversize (RX 接收报文长度超过 rx buffer size) | 需要软复位 non-rdma |
| 9 | tx get md size zero (TX 取到的 md 中 size 为 0) | 需要软复位 non-rdma |
| 10 | rx get md size zero (RX 取到的 md 中 size 为 0) | 需要软复位 non-rdma |

**[DOC]** 异常中断时，`0x1150` 寄存器中 bit4 也会置 1 表示当前存在异常中断。

---

## 12.13 配置参数

**[DOC]** 参数列表:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QP_NUM` | 1024 | QP 的数量 |
| `DW` | 512 | 数据总线位宽 |
| `TYPE1_RX_BUF_AW` | 10 | TYPE1 接收缓存地址位宽 |
| `TYPE1_OST_N` | 512 | TYPE1 Outstanding 数量 |
| `TYPE1_PKT_LEN` | 1344 | TYPE1 Payload 最大长度 (字节) |
| `TYPE1_VC_NUM` | 4 | TYPE1 支持 VC 通道数量 |
| `TYPE2_OST_N` | 16 | TYPE2 Outstanding 数量 |
| `TYPE2_PKT_LEN` | 1344 | TYPE2 Payload 最大长度 (字节) |
| `TYPE2_CRD_RST_NUM` | 2048 | TYPE2 credit 初值 |
| `NONR_TX_DEPTH` | 64 | NONR-RoCE 传输方向缓存深度 |
| `NONR_RX_DEPTH` | 64 | NONR-RoCE 接收方向缓存深度 |
| `NONR_DAT_ID_W` | 7 | NONR-RoCE AXI 总线 ID 宽度 |
| `NONR_MD_ID_W` | 7 | NONR-RoCE MD AXI 总线 ID 宽度 |
| `RAM_RD_LATENCY` | 1 | 读 RAM 需要的周期数 |
| `MAC_DW` | 1024 | MAC 接口的数据位宽 |
| `EPSN_W` | 12 | EPSN RAM 数据位宽 |
| `USE_QDMA` | 1 | 1: 使用 QDMA; 0: 不使用 QDMA |

---

## 12.14 操作流程

### 12.14.1 TYPE1 队列初始化

**[DOC]** 配置步骤:

#### 步骤 1: 全局配置初始化

```
// 基本配置
write_reg(RDMA_EN, ...)                                    // 0x0000
write_reg(MAC_ADDR.mac_addr, LOCAL_MAC_ADDR)               // 0x0010 bit[47:0]
write_reg(MAC_ADDR.len_type, LEN_TYPE)                     // 0x0010 bit[63:48]
```

**Standard 模式:**
```
write_reg(AFH_CFG_EN.standard_en, 1)                       // 0x0160 bit0
write_reg(IP_ADDR.ip_addr, LOCAL_IP_ADDR)                  // 0x0020 bit[31:0]
write_reg(IP_ADDR.ip_ID, LOCAL_IP_ID)                      // 0x0020 bit[47:32]
write_reg(IP_ADDR.ip_ttl, LOCAL_IP_TTL)                    // 0x0020 bit[55:48]
write_reg(TRAFFIC_CLASS.type1_req_tc_bank0/1/2/3)          // 0x0220 bit[31:0]
write_reg(TRAFFIC_CLASS.type1_ack_tc)                      // 0x0220 bit[39:32]
```

**AFH_GEN1 模式:**
```
write_reg(AFH_CFG_EN.afh_gen1_en, 1)                      // 0x0160 bit1
// 若支持 TC:
write_reg(AFH_CFG_EN.afh_gen1_tc_en, 1)                   // 0x0160 bit2
write_reg(TRAFFIC_CLASS.type1_req_tc_bank0/1/2/3)          // 0x0220 bit[31:0]
write_reg(TRAFFIC_CLASS.type1_ack_tc)                      // 0x0220 bit[39:32]
// 若不支持 TC 但开启 CBFC，需要强制打开 VLAN:
write_reg(VLAN_CTRL.vlan_en, 1)                            // 0x0028 bit0
```

**AFH_GEN2 模式:**
```
write_reg(AFH_CFG_EN.afh_gen2_en, 1)                      // 0x0160 bit1
write_reg(TRAFFIC_CLASS.type1_req_tc_bank0/1/2/3)          // 0x0220 bit[31:0]
write_reg(TRAFFIC_CLASS.type1_ack_tc)                      // 0x0220 bit[39:32]
```

**VLAN 配置 (可选):**
```
write_reg(VLAN_CTRL.vlan_en, 1)                            // 0x0028 bit0
write_reg(AFH_CFG_EN.tx_vlan_tag_sel_en, 1)                // 0x0160 bit34
write_reg(VLAN_CTRL.cfi, VLAN_CFI)                         // 0x0028 bit1
write_reg(AFH_CFG_EN.type1_req_bank0/1/2/3, VLAN_PRIO)    // 0x0160 bit[23:12]
write_reg(AFH_CFG_EN.type1_ack_cbfc, VLAN_PRIO)            // 0x0160 bit[26:24]
write_reg(VLAN_CTRL.vid, VLAN_VID)                         // 0x0028 bit[15:4]
write_reg(VLAN_CTRL.tpid, VLAN_TPID)                       // 0x0028 bit[31:16]
write_reg(VLAN_CTRL.vlan_flag, VLAN_FLAG)                  // 0x0028 bit[63:31]
```

#### 步骤 2~7: per-QP 配置 (每建立一对 QP 链接都需配置)

```
// 步骤 2: 选择本地队列号
write_reg(QP_ID_SET.qp_id_set, SOURCE_QPID)               // 0x2088

// 步骤 3: 初始化发送 PSN
write_reg(TX_EPSN_MEM.tx_psn, INIT_PSN)                   // 0x2058
write_reg(EXPECT_RESP_PSN_MEM.expect_resp_psn, INIT_PSN)   // 0x2030
write_reg(TX_LAST_PSN_MEM.tx_last_psn, INIT_PSN-1)        // 0x2038
write_reg(TX_MAX_FW_PSN_MEM.max_forward_psn, INIT_PSN-1)  // 0x2080

// 步骤 4: 初始化接收 PSN
write_reg(REQ_CHK_SN_MEM.rx_req_epsn, REMOTE_INIT_PSN)    // 0x2078
write_reg(TX_ACK_SN_MEM.tx_epsn, REMOTE_INIT_PSN)         // 0x2040

// 步骤 5: 初始化队列速率
write_reg(TX_SPEED_MEM.tx_speed_cfg, INIT_SPEED)           // 0x2048
write_reg(TX_CC_BYTE_CNT_MEM.byte_counter, 0)             // 0x2050

// 步骤 6: 初始化报文配置
// Standard 模式下:
write_reg(DEST_IP_MEM.dest_ip_addr, DEST_IP_ADDR)         // 0x2008 bit[31:0]
write_reg(UDP_SRC_PORT_MEM.udp_src_port, UDP_SRC_PORT)    // 0x2018 bit[15:0]
// 通用配置:
write_reg(DEST_IP_MEM.dest_qpid, DEST_QPID)               // 0x2008 bit[55:32]
write_reg(DEST_MAC_MEM.dest_mac_addr, DEST_MAC_ADDR)       // 0x2010 bit[41:0]
write_reg(UDP_SRC_PORT_MEM.qp_p_key, TX_QP_P_KEY)         // 0x2018 bit[23:16]
write_reg(BTH_QP_CFG_MEM.qp_p_key, RX_QP_P_KEY)          // 0x2070 bit[7:0]

// 步骤 7: 初始化重传计数器
write_reg(RETRY_CNT_MEM.retry_counter, 0)                  // 0x2060 bit[3:0]
write_reg(RETRY_CNT_MEM.rnr_retry_counter, 0)             // 0x2060 bit[7:4]

// 步骤 8: 使能 QP
write_reg(QP_CONTROL_STATUS.qp_en, 1)                      // 0x2028 bit0
```

**[DOC]** 注意: 使用多 bank 的 TYPE1 时，`DEST_QPID` 的低位必须与 `SOURCE_QPID` 低位一致 (2 个 bank 时最低 1bit，4 个 bank 时最低 2bit)。

### 12.14.2 TYPE2 配置

**[DOC]** TYPE2 全局配置初始化:

```
// 1. TC 域
write_reg(TRAFFIC_CLASS.type2_req_tc, TYPE2_TRIO)          // 0x0220[39:32]

// 2. DA 组域 (8 组多播地址)
write_reg(TYPE2_DA_0.type2_da_0, TYPE2_DA0)                // 0x0168[47:0]
write_reg(TYPE2_DA_1.type2_da_1, TYPE2_DA1)                // 0x0170[47:0]
write_reg(TYPE2_DA_2.type2_da_2, TYPE2_DA2)                // 0x0178[47:0]
write_reg(TYPE2_DA_3.type2_da_3, TYPE2_DA3)                // 0x0180[47:0]
write_reg(TYPE2_DA_4.type2_da_4, TYPE2_DA4)                // 0x0188[47:0]
write_reg(TYPE2_DA_5.type2_da_5, TYPE2_DA5)                // 0x0190[47:0]
write_reg(TYPE2_DA_6.type2_da_6, TYPE2_DA6)                // 0x0198[47:0]
write_reg(TYPE2_DA_7.type2_da_7, TYPE2_DA7)                // 0x01a0[47:0]

// 3. IP 头域
write_reg(DEST_IP_MEM2.dest_ip_addr2, TYPE2_DEST_IPA)     // 0x0110[31:0]
write_reg(DEST_IP_MEM2.dscp2, TYPE2_DSCP)                  // 0x0110[63:48]

// 4. MAC 头域
write_reg(DEST_MAC_MEM2.dest_mac_addr2, TYPE2_MAC_ADDR)    // 0x0118[47:0]
write_reg(DEST_MAC_MEM2.mac_type2, TYPE2_MAC)              // 0x0118[47:0]

// 5. UDP PORT
write_reg(UDP_PORT_MEM2.udp_src_port2, TYPE2_SRC_UDP_PORT) // 0x0120[15:0]
write_reg(UDP_PORT_MEM2.udp_dest_port2, TYPE2_DEST_UDP_PORT) // 0x0120[31:16]

// 6. CBFC/PFC 配置参考 9.4 节
```

### 12.14.3 TYPE3 收发流程

#### 发送流程

**[DOC]**

1. 软件向 TYPE3 内存空间初始化需要发送的报文
2. 软件向 TYPE3 内存空间初始化 TX MD 信息
3. 初始化 `TX_MDQ_SIZE` 寄存器 (`0x1100`) bit[1:0]
4. 初始化 `TX_MDQ_BA` 寄存器 (`0x1080`) bit[63:0]
5. 上述为静态配置，enable 通过后不可更改 (如需更改需要做 reset 操作)
6. 初始化 `NONR_DMA_CTRL` (`0x1140`)，释放复位使能 TX 通路: `bit[0]` tx_en, `bit[16]` tx_reset
7. 初始化 MD 队列 head 指针 `TX_MD_HEAD_PTR` (`0x1000`) bit[15:0]
8. TYPE3 通过 AXI 接口读取数据
9. 轮询检查 MD 队列 tail 指针 `TX_MD_TAIL_PTR` (`0x1040`) bit[11:0]
10. 系统等待中断或查询尾指针更新 TX MD HEAD PTR
11. 如果使用中断，需写入 `TAIL_UP_CFG` (`0x1190`) bit[15:0] 和 `DMA_INT_MASK` (`0x1178`) bit[3] tx_tail_up
12. 等待中断触发后读取 `DMA_INTR_ST0` (`0x1150`) 和 `DMA_INTR_ST1` (`0x1170`) 寄存器

#### 接收流程

**[DOC]**

1. 软件向 TYPE3 内存空间初始化 RX MD 信息
2. 初始化 `RX_MDQ_SIZE` (`0x1120`) bit[11:0]
3. 初始化 `RX_MDQ_BA` (`0x10C0`) bit[63:0]
4. 上述为静态配置，enable 通过后不可更改
5. 初始化 `NONR_DMA_CTRL` (`0x1140`)，释放复位使能 RX 通路: `bit[8]` rx_en, `bit[31]` rx_reset
6. 初始化 MD 队列 head 指针 `RX_MD_HEAD_PTR` (`0x1020`) bit[15:0]
7. TYPE3 通过 AXI 写入数据
8. 轮询检查 MD 队列 tail 指针 `RX_MD_TAIL_PTR` (`0x1060`) bit[11:0]
9. 系统等待中断或查询尾指针更新 RX MD HEAD PTR
10. 如果使用中断，需写入 `TAIL_UP_CFG` (`0x1190`) bit[31:16] 和 `DMA_INT_MASK` (`0x1178`) bit[4] rx_tail_up
11. 等待中断触发后读取 `DMA_INTR_ST0` (`0x1150`) 和 `DMA_INTR_ST1` (`0x1170`) 寄存器

### 12.14.4 FLOW_CTRL 配置

#### CBFC 配置约束

**[DOC]**
- CBFC 和 PFC 不支持同时使用
- 开启 CBFC 时，TC 域和 IP 报头中的 ToS 域使用寄存器配置 (`TRAFFIC_CLASS 0x0220`)，TC 域高 6 比特体现 VCID 信息
- CBFC 的 VCID 和 PFC 的 PRIORITY 共用寄存器配置
- 使能 CBFC 时，网络中所有节点的 VCID 或 PRIORITY 配置必须保持一致
- 建链双方需为相同 BANK 的 QP，不支持跨 BANK 的 QP 之间建链
- 不支持同一个 VCID 或 PRIORITY 配置给不同的数据流

#### PFC 模式寄存器配置

**[DOC]** 步骤 1: 配置 RX Buffer 水线:
```
write_reg(XON_THR_RDMA.xon_thr_rdma, TYPE1_RX_BUF_DEPTH/2)     // 0x0008[15:0]
write_reg(XON_THR_RDMA.xoff_thr_rdma, TYPE1_RX_BUF_DEPTH*3/4)  // 0x0008[31:16]
write_reg(XON_THR_NON_RDMA.xon_thr_non_rdma, NONR_RX_DEPTH/2)  // 0x0018[15:0]
write_reg(XON_THR_NON_RDMA.xoff_thr_non_rdma, NONR_RX_DEPTH*3/4) // 0x0018[31:16]
write_reg(XON_THR_NON_RDMA.xon_thr_ack_rdma, 0x80)              // 0x0018[47:32]
write_reg(XON_THR_NON_RDMA.xoff_thr_ack_rdma, 0xc0)             // 0x0018[63:48]
```

**[DOC]** 步骤 2: 禁用 CBFC 使能:
```
write_reg(AFH_CFG_EN.stop_cbfc_en, 1)                     // 0x0160[5]
write_reg(AFH_CFG_EN.rx_type1_req_bankX_cbfc_en, 0)       // 0x0160[6:9]
write_reg(AFH_CFG_EN.rx_type1_ack_cbfc_en, 0)             // 0x0160[10]
write_reg(AFH_CFG_EN.rx_type3_cbfc_en, 0)                 // 0x0160[11]
write_reg(AFH_CFG_EN.rx_type2_cbfc_en, 0)                 // 0x0160[12]
```

**[DOC]** 步骤 3: 配置 TX/RX PFC 使能和优先级映射 (参见完整寄存器配置流程)。

#### CBFC 模式寄存器配置

**[DOC]** 核心配置:
```
// 全局使能
write_reg(AFH_CFG_EN.stop_cbfc_en, 0)                     // 0x0160[5]

// 各流量 CBFC 使能
write_reg(AFH_CFG_EN.rx_type1_req_bank0_cbfc_en, 1)       // 0x0160[6]
write_reg(AFH_CFG_EN.rx_type1_req_bank1_cbfc_en, 1)       // 0x0160[7]
write_reg(AFH_CFG_EN.rx_type1_req_bank2_cbfc_en, 1)       // 0x0160[8]
write_reg(AFH_CFG_EN.rx_type1_req_bank3_cbfc_en, 1)       // 0x0160[9]
write_reg(AFH_CFG_EN.rx_type1_ack_cbfc_en, 1)             // 0x0160[10]
write_reg(AFH_CFG_EN.rx_type3_cbfc_en, 1)                 // 0x0160[11]
write_reg(AFH_CFG_EN.rx_type2_cbfc_en, 1)                 // 0x0160[12]

// RX PFC 必须配置为 0
write_reg(XON_THR_RDMA.rx_type1_req_bankX_pfc_en, 0)      // 0x0008[39:42]
write_reg(XON_THR_RDMA.rx_ack_pfc_en, 0)                  // 0x0008[43]
write_reg(XON_THR_RDMA.rx_type2_pfc_en, 0)                // 0x0008[44]
write_reg(XON_THR_RDMA.rx_non_rdma_pfc_en, 0)             // 0x0008[45]

// VC 通道映射
write_reg(AFH_CFG_EN.type1_req_bank0_cbfc, VCID)          // 0x0160[14:12]
write_reg(AFH_CFG_EN.type1_req_bank1_cbfc, VCID)          // 0x0160[17:15]
write_reg(AFH_CFG_EN.type1_req_bank2_cbfc, VCID)          // 0x0160[20:18]
write_reg(AFH_CFG_EN.type1_req_bank3_cbfc, VCID)          // 0x0160[23:21]
write_reg(AFH_CFG_EN.type1_ack_cbfc, VCID)                // 0x0160[26:24]
write_reg(AFH_CFG_EN.type2_cbfc, VCID)                    // 0x0160[29:27]
write_reg(AFH_CFG_EN.type3_cbfc, VCID)                    // 0x0160[32:30]

// Credit 下限 (1~7，禁止配置为 0)
write_reg(AFH_CFG_EN.type1_credit_uf_limit_cbfc, 1)       // 0x0160[38:36]
write_reg(AFH_CFG_EN.type2_credit_uf_limit_cbfc, 1)       // 0x0160[41:39]
write_reg(AFH_CFG_EN.type3_credit_uf_limit_cbfc, 1)       // 0x0160[44:42]
write_reg(AFH_CFG_EN.ack_credit_uf_limit_cbfc, 1)         // 0x0160[47:45]

// 动态下限 (仅 TYPE1 REQ 支持)
write_reg(AFH_CFG_EN.type1_dyn_uf_limit_cbfc_en, 1)       // 0x0160[48]
write_reg(AFH_CFG_EN.type2_dyn_uf_limit_cbfc_en, 0)       // 0x0160[49]
write_reg(AFH_CFG_EN.type3_dyn_uf_limit_cbfc_en, 0)       // 0x0160[50]
write_reg(AFH_CFG_EN.ack_dyn_uf_limit_cbfc_en, 0)         // 0x0160[51]

// 软件流控
write_reg(AFH_CFG_EN.software_ctrl_cbfc_vc_status, 0xFF)  // 0x0160[59:52]

// RNR 使能 (CBFC 模式下建议配置为 0)
write_reg(AFH_CFG_EN.type1_flow_stop_send_rnr_en, 0)      // 0x0160[63:60]
```

---

## 12.15 异常处理

### 12.15.1 TX overcredit 异常

**[DOC]** 当 host 发送的 pkt ost 数量超过 `TYPE1_OST_N` 或 `TYPE2_OST_N` 时，TX 会丢掉当前包并发送中断。

**[DOC]** 只有在调试时可能遇到，建议软件针对异常 QP 重新建链和传输。

### 12.15.2 TX oversize 异常

**[DOC]** 当 host 发送的 packet size 超过 `TYPE1_PKT_LEN` 或 `TYPE2_PKT_LEN` 时，TX 会丢掉当前包并发送中断。

**[DOC]** 只有在调试时可能遇到，建议软件针对异常 QP 重新建链和传输。

### 12.15.3 TX retry times over

**[DOC]** 当 TYPE1 某一个 QP 的重传次数到达配置的阈值 (`RETRY_TIMER_CFG 0x0040 bit[16:13]`) 时，IP 会上报中断 (`INTR_STATUS 0x00A0 bit3`)。

**[DOC]** 处理流程:

1. 配置 `qp_id_set` (`QP_ID_SET 0x2088 bit[31:0]`) 为需要进行 FATAL 处理的 QP
2. 配置相应 QP 的 `qp_en` (`QP_CONTROL_STATUS 0x2028 bit0`) 为 0
3. 配置对应 QP 的 `cc_speed` (`TX_SPEED_MEM 0x2048 bit[21:0]`) 为 0，硬件侦听此行为后产生 `cc_mask` 阻塞该 QP 报文仲裁，并禁止该 QP 的所有 CC 配置行为
4. 等待 1us，读 `XON_THR_RDMA.type1_free_busy` (`0x0008 bit50`)，读出值为 0 时继续
5. 配置对应 QP 的 `free_buf_set` (`QP_CONTROL_STATUS 0x2028 bit15`) 为 1，硬件开始释放该 QP 远端还没有应答的 buffer
6. 定时读 `free_buf_set` (`QP_CONTROL_STATUS 0x2028 bit15`)，直到读出为 0，代表对应 QP 的 buffer 已全部释放
7. 读 `TX_EPSN_MEM` (`0x2058`)，得到对应 QP 在 buffer 中的下一个报文编号，记作 `tx_expect_psn`
8. 按照 TYPE1 初始化流程对相应 QP 做初始化，第一个报文的编号设置为 `tx_expect_psn + 1`

### 12.15.4 下游数据通路异常

#### 短暂断路后恢复

**[DOC]**
- RC_LINK 在断路期间根据 MAC 的流控信息停止发送数据，MAC 恢复后继续工作
- RC_LINK 接收侧处于正常工作状态，待下游数据流量恢复后继续工作

#### 下游复位 (RC_LINK 无需复位)

**[DOC]**
- **发送方向**: 下游复位时 RC_LINK 停止发送报文，并同步复位下游的 credit 流控逻辑；下游复位释放后自动恢复工作
- **接收方向**: 下游复位时 RC_LINK 丢弃当前接口的残留报文，后续报文可以正常接收

#### 长时间断链无法恢复

**[DOC]** RC_LINK TYPE1 的 QP 可能会因为超时进入 fatal 状态，此时需要待链路恢复后，按照 TX retry times over 章节处理，或者复位重新建链。

### 12.15.5 TYPE2 接收缓冲 overflow

**[DOC]** 在接收侧，TYPE2 共享接收缓存，当缓存溢出时，RX 方向会将当前包丢掉，并发送中断，上报 TYPE2 接收缓冲区溢出。

### 12.15.6 TYPE3 异常处理

**[DOC]** TYPE3 上报异常中断后会导致后续行为异常，建议异常中断后对 TYPE3 进行 soft reset。具体参考 `0x1150`、`0x1170` 寄存器以及中断部分描述。

### 12.15.7 上游复位

**[DOC]** 当 RC_LINK 上游客户逻辑发生复位时，需要同时复位 RC_LINK。

---

## 附录: 关键寄存器地址速查

| 寄存器 | 地址 | 关键位域 |
|--------|------|----------|
| RDMA_EN | 0x0000 | 功能使能 |
| XON_THR_RDMA | 0x0008 | [15:0] xon_thr, [31:16] xoff_thr, [32:45] PFC 使能, [49:46] flow_stop_en, [50] free_busy |
| MAC_ADDR | 0x0010 | [47:0] mac_addr, [63:48] len_type |
| XON_THR_NON_RDMA | 0x0018 | [15:0] xon_thr_non_rdma, [31:16] xoff_thr_non_rdma |
| IP_ADDR | 0x0020 | [31:0] ip_addr, [47:32] ip_ID, [55:48] ip_ttl |
| VLAN_CTRL | 0x0028 | [0] vlan_en, [1] cfi, [15:4] vid, [31:16] tpid |
| RETRY_TIMER_CFG | 0x0040 | [16:13] 重传次数阈值 |
| INTR_STATUS | 0x00A0 | [3] retry times over |
| DEST_IP_MEM2 | 0x0110 | [31:0] dest_ip_addr2, [63:48] dscp2 |
| DEST_MAC_MEM2 | 0x0118 | [47:0] dest_mac_addr2 |
| UDP_PORT_MEM2 | 0x0120 | [15:0] udp_src_port2, [31:16] udp_dest_port2 |
| DMA_INTR_ST0 | 0x1150 | [0] rx_md_empty, [1] tx_md_empty, [2] rx_tail_up, [3] tx_tail_up, [4] abnormal |
| DMA_INTR_ST1 | 0x1170 | [10:0] 异常中断类型 |
| DMA_INT_MASK | 0x1178 | [3] tx_tail_up, [4] rx_tail_up |
| NONR_DMA_CTRL | 0x1140 | [0] tx_en, [8] rx_en, [16] tx_reset, [31] rx_reset |
| AFH_CFG_EN | 0x0160 | [0] standard_en, [1] afh_gen1_en, [2] tc_en, [5] stop_cbfc_en, [6:12] cbfc_en, [14:32] vc_mapping, [34] tx_vlan_tag_sel_en, [38:36]~[47:45] credit_uf_limit, [48:51] dyn_uf_limit, [59:52] sw_cbfc_vc_status, [63:60] rnr_en |
| TYPE2_DA_0~7 | 0x0168~0x01A0 | [47:0] 多播目的地址 |
| TAIL_UP_CFG | 0x1190 | [15:0] TX tail up, [31:16] RX tail up |
| TRAFFIC_CLASS | 0x0220 | [31:0] type1_req_tc_bank0/1/2/3, [39:32] type1_ack_tc/type2_req_tc |
| TX_MD_HEAD_PTR | 0x1000 | [15:0] TX MD head pointer |
| RX_MD_HEAD_PTR | 0x1020 | [15:0] RX MD head pointer |
| TX_MD_TAIL_PTR | 0x1040 | [11:0] TX MD tail pointer |
| RX_MD_TAIL_PTR | 0x1060 | [11:0] RX MD tail pointer |
| TX_MDQ_BA | 0x1080 | [63:0] TX MD queue base address |
| TX_MDQ_SIZE | 0x1100 | [1:0] TX MD queue size |
| RX_MDQ_SIZE | 0x1120 | [11:0] RX MD queue size |
| RX_MDQ_BA | 0x10C0 | [63:0] RX MD queue base address |
| DEST_IP_MEM | 0x2008 | [31:0] dest_ip_addr, [55:32] dest_qpid |
| DEST_MAC_MEM | 0x2010 | [41:0] dest_mac_addr |
| UDP_SRC_PORT_MEM | 0x2018 | [15:0] udp_src_port, [23:16] qp_p_key |
| QP_CONTROL_STATUS | 0x2028 | [0] qp_en, [15] free_buf_set |
| EXPECT_RESP_PSN_MEM | 0x2030 | expect_resp_psn |
| TX_LAST_PSN_MEM | 0x2038 | tx_last_psn |
| TX_ACK_SN_MEM | 0x2040 | tx_epsn |
| TX_SPEED_MEM | 0x2048 | [21:0] tx_speed_cfg |
| TX_CC_BYTE_CNT_MEM | 0x2050 | byte_counter |
| TX_EPSN_MEM | 0x2058 | tx_psn |
| RETRY_CNT_MEM | 0x2060 | [3:0] retry_counter, [7:4] rnr_retry_counter |
| BTH_QP_CFG_MEM | 0x2070 | [7:0] qp_p_key |
| REQ_CHK_SN_MEM | 0x2078 | rx_req_epsn |
| TX_MAX_FW_PSN_MEM | 0x2080 | max_forward_psn |
| QP_ID_SET | 0x2088 | [31:0] qp_id_set |
