# 03. Flit编码规则与帧格式

## 3.1 Flit概念

Flit (Flow Control Unit) 是PAXI的基本传输单元。PAXI将AXI总线信号打包成Flit, 再通过RC Link封装到以太网帧中传输。

**[推导]** SUE2.0对Flit编码做了重大重构: 从旧版的"AXI Sub Flit组合"模式改为"REQ/RSP分离"模式, Header从6-bit位图改为10-bit结构化编码。

## 3.2 Flit类型体系

**[DOC]** 来自SUE2.0 2.1.1 Flit Rules:

> "PAXI encapsulates AXI flit into two types: REQ and RSP, corresponding to the AR/AW/W and R/B channels of AXI respectively."

### Flit分类

| Flit Type (2-bit) | Encode (2-bit) | 含义 | Length字段 |
|-------------------|----------------|------|-----------|
| 2'b00 (AXI REQ) | 2'b00 | AXI W Flit (带WSTRB) | burst length |
| 2'b00 (AXI REQ) | 2'b01 | AXI W Flit (不带WSTRB) | burst length |
| 2'b00 (AXI REQ) | 2'b10 | AXI AR Flit | AR number |
| 2'b01 (AXI RSP) | 2'b00 | AXI B Flit | B number |
| 2'b01 (AXI RSP) | 2'b10 | AXI R Flit | burst length |
| 2'b10 (APB) | 2'b00 | APB register read/write Flit | NA |
| 2'b10 (APB) | 2'b01 | APB message Flit | NA |

### 10-bit Header结构

```
[9:8] Flit Type  -- 2-bit, 区分REQ/RSP/APB
[7:6] Encode     -- 2-bit, 区分同类型内的不同编码
[5:0] Length     -- 6-bit, 含义取决于Flit类型
```

## 3.3 帧长计算公式

**[DOC]** 来自SUE2.0 2.1.1:

### AW+W Frame (写请求)

**[DOC]**:

```
AW+W Frame length = $ceil(10-bit header + AW channel + 1, 8)/8
                   + length * $ceil(W channel + 1, 8)/8
```

PAXI将整个写burst打包为单个帧: AW头 + length个W beat。

### R Frame (读响应)

**[DOC]**:

```
R Frame length = $ceil(10-bit header + R channel + 1, 8)/8
               + (length-1) * $ceil(R channel + 1, 8)/8
```

### AR Frame (读请求, 支持打包)

**[DOC]**:

```
单个AR: $ceil(10-bit header + AR channel + 1, 8)/8

多个AR: $ceil(10-bit header + AR channel + 1, 512)/8
       + (AR_NUM-2) * $ceil(AR channel + 1, 512)/8
       + $ceil(AR channel + 1, 8)/8
```

**[DOC]**:

> "Length field indicate how many AR/B transaction in current Flit. The maximum is 16."

AR Frame最多可打包16个AR事务。

### B Frame (写响应, 支持打包)

**[DOC]**:

```
单个B: $ceil(10-bit header + B channel + 1, 8)/8

多个B: $ceil(10-bit header + B channel + 1, 512)/8
      + (length-2) * $ceil(B channel + 1, 512)/8
      + $ceil(B channel + 1, 8)/8
```

B Frame最多可打包16个B事务。

### 最大帧长

**[DOC]** 来自SUE2.0 3.2.5 Ethernet Frame Length Register (0x010):

> "The default reset value UVIP_PAXI_MAX_FRAME_LEN is the larger value between the request and the response, in bytes."

帧长由AW+W Frame和R Frame中较大者决定。用户需配置TX_BUF_WM使AR或B帧长度不超过最大帧长。

## 3.4 WSTRB优化

**[DOC]** Ctrl Register (0x00C) bit6:

> "WSTRB_EN: Indicate whether there are WSTRB bits. 1: present; 0: absent."

当WSTRB_EN=1且所有WSTRB位均为1时, PAXI自动使用Encode=2'b01编码(不带WSTRB的W Flit), 节省带宽。

**[DOC]**:

> "AXI W Flit without wstrb bits (If user has wstrb bits all valid, paxi will not encode to wflit)"

**[推导]** WSTRB优化的带宽收益: 对于512-bit数据总线, WSTRB=64bit, 省略后每beat节省约11%的传输开销。

## 3.5 AXI User Field编码

**[DOC]** 来自SUE2.0 2.1.4 AXI User Field:

> "The PAXI utilizes the higher bit of User field of the AXI interface to transmit DA (Destination Address), VC (Virtual Channel), and broadcast/multicast frame information."

### 编码格式 (MSB -> LSB)

**单播**: `{n-bit DA, 3-bit VC, 1-bit Multicast}`

其中Multicast=0表示单播。

**多播**: `{(n-3)-bit zero, 3-bit multi-grp, 3-bit VC, 1-bit Multicast}`

其中Multicast=1, DA字段低3位作为多播组号。

**[推导]** 这个编码方式是SUE2.0的新设计:
- 旧版通过AXI User Signal映射到128个DA寄存器
- SUE2.0直接在User Field中编码DA、VC和多播信息, 更灵活
- VC信息允许用户在Mode 1下自定义VC分配

## 3.6 APB Flit

**[DOC]** 来自SUE2.0 2.1.1:

> "For APB register flit, REQ bit ==1 indicate an APB request flit to read/write remote APB slave device. REQ bit ==0 indicate an APB response flit return to remote APB master which initiate the APB request."

> "User set message ctrl register to send APB message. When received from remote side, an interrupt is asserted."

APB Flit有两种:
- **APB Register Flit** (Flit Type=2'b10, Encode=2'b00): 远程寄存器读写
- **APB Message Flit** (Flit Type=2'b10, Encode=2'b01): 系统消息(如Linkup)

## 3.7 帧封装与RC Link

**[DOC]** 来自SUE2.0 2.2:

> "PAXI encapsulates the AXI Flit with VC ID/DA and routes it to RC Link to encapsulates MAC Frame."

**[推导]** 与旧版的差异:
- 旧版: PAXI直接生成MAC帧 (6B DA + 6B SA + 2B TYPE + 2B IDLE + PAXI_FLIT)
- SUE2.0: PAXI将Flit送入RC Link, 由RC Link负责MAC帧封装
- RC Link支持多种帧格式: Standard(带IP/UDP头), AFH_GEN1, AFH_GEN2_16b, AFH_Lite

### RC Link报文格式选择

由RC Link寄存器配置决定:
- `tx_standard_en`: 标准格式 (含IP/UDP报头)
- `tx_afh_gen1_en`: AFH_GEN1格式
- `tx_afh_gen2_16b_en`: AFH_GEN2 16-bit压缩格式
- `afh_lite_en`: AFH_Lite简化格式

详见 [12_rclink.md](12_rclink.md) 第12.3节。

## 3.8 与旧版编码的对比

| 特性 | 旧版 (v2R0p6) | SUE2.0 |
|------|---------------|--------|
| Header | 6-bit位图 + 1-bit WDATA_TYPE | 10-bit结构化 (Type+Encode+Length) |
| Flit分类 | 混合式 (一个Flit可含多种Sub Flit) | 分离式 (REQ和RSP独立) |
| AW+W | Sub Flit组合 | 整个burst打包为一帧 |
| AR打包 | 不支持 | 最多16个AR打包 |
| B打包 | 不支持 | 最多16个B打包 |
| 多播编码 | 不支持 | AXI User Field Multicast位 |
| MAC帧生成 | PAXI直接生成 | 委托RC Link |
