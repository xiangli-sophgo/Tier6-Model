# 03. Flit编码规则与帧格式

## 3.1 Flit概念

Flit (Flow Control Unit) 是PAXI的基本传输单元。PAXI将AXI总线信号打包成Flit, 再封装到以太网帧中传输。

## 3.2 AXI Sub Flit

**[DOC]** 来自2.1.2.1 AXI sub flit:

> "All valid bits of each flit must be filled to a data boundary of 8 bits."

**大小计算公式** (文档原文):

```
$Round_up(8, DX_flit) = $ceil((9 + DX_WIDTH) / 8) * 8 - (9 + DX_WIDTH)
```

其中:
- `9` = 1 bit flit type 相关开销 + 8 bits valid bits (共9 bits固定头)
- `DX_WIDTH` = 对应AXI通道所有信号的总宽度
- 结果是需要填充的padding bits数量

### Sub Flit类型与大小

**[DOC]** Data Channel Sub Flits (2.1.1.1):

| Flit类型 | Valid Bits = Header + Data | Padding |
|----------|---------------------------|---------|
| NOP | 不固定 | - |
| AW | 9 + DAW_WIDTH | $Round_up(8, DAW_flit) |
| AR | 9 + DAR_WIDTH | $Round_up(8, DAR_flit) |
| WDATA | 9 + DW_WIDTH | $Round_up(8, DW_flit) |
| RDATA | 9 + DR_WIDTH | $Round_up(8, DR_flit) |
| BRESP | 9 + DB_WIDTH | $Round_up(8, DB_flit) |
| WDATA2 | 9 + DW2_WIDTH | $Round_up(8, DW2_flit) |

**[DOC]** Ctrl Channel Sub Flits (2.1.1.2): 结构相同, 前缀为C替代D:

| Flit类型 | Valid Bits = Header + Data |
|----------|---------------------------|
| AW | 9 + CAW_WIDTH |
| AR | 9 + CAR_WIDTH |
| WDATA | 9 + CW_WIDTH |
| RDATA | 9 + CR_WIDTH |
| BRESP | 9 + CB_WIDTH |
| WDATA2 | 9 + CW2_WIDTH |

注意: 每个信号宽度都是可配置的(per-customer), 由cfg_table指定。

## 3.3 AXI Sub Flit字段映射

### Data Channel AW Flit字段

**[DOC]** 2.1.1.1.1 Address Channels:

| 信号 | 宽度 |
|------|------|
| AWID | DAWID_WIDTH |
| AWADDR | DAWADDR_WIDTH |
| AWLEN | DAWLEN_WIDTH |
| AWSIZE | DAWSIZE_WIDTH |
| AWBURST | DAWBURST_WIDTH |
| AWCACHE | DAWCACHE_WIDTH |
| AWPROT | DAWPROT_WIDTH |
| AWLOCK | DAWLOCK_WIDTH |
| AWQOS | DAWQOS_WIDTH |
| AWREGION | DAWREGION_WIDTH |
| AWUSER | DAWUSER_WIDTH |

AR Flit同理(使用DAR*前缀)。

### Data Channel WDATA Flit字段

**[DOC]** 2.1.1.1.2 WDATA Channels:

| 信号 | 宽度 |
|------|------|
| WDATA | DWDATA_WIDTH |
| WSTRB | DWSTRB_WIDTH |
| WLAST | DWLAST_WIDTH |
| WUSER | DWUSER_WIDTH |

### Data Channel WDATA2 Flit字段

**[DOC]** 2.1.1.1.5 WDATA2 Channels:

> "Use WDATA2 flit implicit that all the WSTRB field is all 1."

| 信号 | 宽度 |
|------|------|
| WDATA | DWDATA_WIDTH |
| WLAST | DWLAST_WIDTH |
| WUSER | DWUSER_WIDTH |

**[推导]** WDATA2优化的意义:
- 当写操作的所有字节使能(WSTRB)均为1时, 无需传输WSTRB字段
- 节省带宽 = DWSTRB_WIDTH bits per WDATA beat
- 例如512-bit数据总线: WSTRB = 64 bits, 节省约 64/(512+64) = 11%

### Data Channel RDATA Flit字段

**[DOC]** 2.1.1.1.3:

| 信号 | 宽度 |
|------|------|
| RID | DRID_WIDTH |
| RDATA | DRDATA_WIDTH |
| RRESP | DRRESP_WIDTH |
| RLAST | DRLAST_WIDTH |
| RUSER | DRUSER_WIDTH |

### Data Channel BRESP Flit字段

**[DOC]** 2.1.1.1.4:

| 信号 | 宽度 |
|------|------|
| BID | DBID_WIDTH |
| BRESP | DBRESP_WIDTH |
| BUSER | DBUSER_WIDTH |

## 3.4 AXI Flit组合规则

**[DOC]** 来自2.1.2.2 AXI Flit:

AXI Flit是多个Sub Flit的组合, 有严格的编码规则:

### Flit Type标识

> "Flit type==3'b001 -> data axi flit, Flit type==3'b010 -> ctrl axi flit"

| Flit Type | 含义 |
|-----------|------|
| 3'b001 | Data Channel AXI Flit |
| 3'b010 | Ctrl Channel AXI Flit |
| 3'b100 | APB Access |
| 3'b101 | Message (Linkup等) |

### Flit Header编码

**[DOC]** Header由6-bit位图 + 1-bit WDATA_TYPE组成:

> "WDATA_TYPE determine the flit is wdata with wstrb or not. (WDATA_TYPE==1 -> WDATA2)"

Flit Header位图 (6 bits):

```
bit[5] = AW存在
bit[4] = AR存在
bit[3] = WDATA/WDATA2存在
bit[2] = RDATA存在
bit[1] = ?
bit[0] = BRESP存在
```

**[DOC]** 部分编码示例:

| Header (6b) | WDATA_TYPE | WDATA/RDATA | AW | AR | WDATA2 | BRESP |
|-------------|------------|-------------|----|----|--------|-------|
| 6'b000001 | 0 | 0 | 0 | 0 | 0 | 1 |
| 6'b000010 | 0 | 0 | 0 | 0 | 1(RDATA) | 0 |
| 6'b000100 | 0 | 0 | 0 | 1 | 0 | 0 |
| 6'b001000 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6'b010000 | 0 | 1(WDATA) | 0 | 0 | 0 | 0 |
| 6'b011111 | 0 | 1 | 1 | 1 | 1 | 1 |
| 6'b110000 | 1 | 1(WDATA2) | 0 | 0 | 0 | 0 |
| 6'b111111 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6'b000000 | 0 | 0 | 0 | 0 | 0 | 0 |

### Sub Flit组合顺序

**[DOC]** 严格要求:

> "When composing AXI Flit, the AXI sub order from MSB to LSB must be WDATA/WDATA2 -> RDATA -> AW -> AR -> BRESP."

```
AXI Flit = [Flit Header] [WDATA/WDATA2 sub flit] [RDATA sub flit] [AW sub flit] [AR sub flit] [BRESP sub flit]
                MSB                                                                              LSB
```

### 关键规则

**[DOC]**:

1. > "When the corresponding bit was set, it means the raw data of corresponding sub flit was included in the AXI flit."

2. > "When there is no corresponding transaction data to be sent, the entire AXI flit can be terminated by filling with NOP (Flit head is 6'b000000)."

3. > "There must not have more than 1 Flit head in one transfer."

**[推导]** 关键含义:
- 一个AXI Flit可以聚合多个AXI通道的数据(如同时包含AW+W+AR)
- 这是PAXI实现>99%带宽利用率的关键: 减少Flit header开销
- NOP用于填充, 当没有数据可发送时终止Flit

## 3.5 APB Flit

**[DOC]** 2.1.2.3:

> "Flit type==3'b100 -> apb access. Flit type==3'b101 -> message"

APB Flit用于Management Channel, 字段包括:

| 字段 | 含义 |
|------|------|
| REQ | 1=本地APB请求, 0=远端APB响应 |
| RSVD | 保留字段 |
| Error | 错误指示 |
| Linkup | PCS链路建立指示 |

## 3.6 MAC帧封装

**[DOC]** 来自2.3 DA insertion Handling:

> "PAXI send mac frame as: 6B(DA)+6B(SA)+2B(TYPE)+2B(IDLE)+PAXI_FLIT"

### 帧格式

```
┌──────────────────────────────────────────┐
│ DA (Destination Address)     │ 6 Bytes  │
├──────────────────────────────┤          │
│ SA (Source Address)          │ 6 Bytes  │
├──────────────────────────────┤          │
│ TYPE (Ethernet Type/Length)  │ 2 Bytes  │
├──────────────────────────────┤          │
│ IDLE                         │ 2 Bytes  │
├──────────────────────────────┤          │
│ PAXI_FLIT (payload)         │ Variable │
└──────────────────────────────────────────┘

总帧头开销 = 6 + 6 + 2 + 2 = 16 Bytes
```

**[推导]** 与标准以太网帧对比:

| 字段 | 标准以太网 | PAXI简化帧 |
|------|-----------|-----------|
| Preamble | 7B | 无 |
| SFD | 1B | 无 |
| DA | 6B | 6B |
| SA | 6B | 6B |
| Type/Length | 2B | 2B |
| IDLE | 无 | 2B |
| FCS | 4B | 无 (由L1/L2 Retry替代) |
| IFG | 12B | 最小化 |
| **总计** | **38B** | **16B** |

省去了Preamble/SFD/FCS/IFG, 是实现>99%带宽利用率的关键。

### MAC帧最大长度

**[DOC]** 3.2.5 REG_Ethernet_Flit:

> "Ethernet Maximum Mac frame length(Byte unit). The value should be configure as integer multiples of 128Byte and the max value must not exceeds `UV_PAXI_MAC_FRAME_LEN/8"

- 帧长度必须是128字节的整数倍
- 有设计限定的最大值
- 默认值: 0x0400 = 1024 (即1024字节)
