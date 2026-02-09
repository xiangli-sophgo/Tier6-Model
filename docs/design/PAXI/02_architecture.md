# 02. PAXI 微架构与协议栈分层

## 2.1 PAXI Core内部分区

**[DOC]** 来自2.1 PAXI Partitions:

> "The PAXI core consists of three main blocks: one that propagates transactions from an AXI Slave interface to the Ethernet Link, one that propagates transactions from the Ethernet Link to an AXI Master interface, and one that enables access to the internal registers from an APB interface."

三大功能块:

```
┌──────────────────────────────────────────────────────┐
│                    PAXI Core                          │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │ Block 1: AXI Slave -> Ethernet TX            │     │
│  │  - 接收本地AXI事务                            │     │
│  │  - 编码为Flit                                 │     │
│  │  - 经VC仲裁和DA插入送MAC                      │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │ Block 2: Ethernet RX -> AXI Master           │     │
│  │  - 接收以太网帧                               │     │
│  │  - 解码Flit                                   │     │
│  │  - 重建AXI事务发到AXI Master                   │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │ Block 3: APB Register Interface              │     │
│  │  - 寄存器配置和状态读取                        │     │
│  │  - 4KB地址空间                                │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

## 2.2 完整协议栈

基于所有文档综合, 合见工软的完整互联方案分为以下层次:

```
┌──────────────────────────────────────────────────────┐
│ Layer 5: Application / NoC                            │
│  - AXI4 Master/Slave (5 channels: AW/W/B/AR/R)      │
│  - APB3 (Register access)                             │
├──────────────────────────────────────────────────────┤
│ Layer 4: RDMA Engine (可选, RoCEv2)                   │
│  [DOC] "RocEv2 RDMA with up to 8192 Queue pair"     │
│  - RC: Send/Receive/Write                             │
│  - UD: Send/Receive                                   │
│  - Go-Back-N重传                                      │
│  - DCQCN拥塞控制                                     │
│  - ECN / PFC                                          │
├──────────────────────────────────────────────────────┤
│ Layer 3: PAXI Core (协议转换层)                       │
│  [DOC] "Point-to-point AXI Wrapper"                  │
│  - AXI Signal -> Flit编码                             │
│  - 3类虚拟通道(Management/Control/Data)               │
│  - WRR仲裁                                           │
│  - 128 DA映射与Round-Robin仲裁                        │
│  - Credit-based流控 (OST 128/256)                    │
│  - PFC水位线触发                                      │
│  - 远程APB访问                                        │
│  - 延迟测量                                           │
├──────────────────────────────────────────────────────┤
│ Layer 2: CESOC (以太网控制器)                          │
│  [DOC] "includes CEMAC_800G, CEPCS_800G, CEFEC_800G" │
│  - CEMAC: MAC层, 简化帧格式                           │
│  - CEPCS: PCS层, 编码/解码                            │
│  - CEFEC: FEC层 (RSFEC), L1 Retry                    │
│  - IF_192B: 用户逻辑接口                              │
│  - MAC L2 Retry                                       │
├──────────────────────────────────────────────────────┤
│ Layer 1: SerDes (物理层)                              │
│  [DOC] "8x112G PAM4 SERDES"                          │
│  - 8条Lane, 每条112 Gbps (PAM4调制)                  │
│  - 8x8任意到任意Lane交换(TX/RX)                       │
│  - 支持6nm/5nm工艺                                    │
└──────────────────────────────────────────────────────┘
```

## 2.3 TX数据路径详细分析

**[DOC]** 基于文档2.2.1 AXI Request Handling:

> "The transfers from AW channel, W channel and AR channel on the AXI slave interface were composed as AXI Flit according to the rule in the previous chapter, and then put it on the axi steam tx interface on the ethernet links."

TX路径的处理阶段:

```
阶段1: AXI接收
  ← AXI Slave接口接收来自NoC的AXI事务
  ← 5个通道: AW (写地址), W (写数据), B (写响应), AR (读地址), R (读数据)
       |
阶段2: Flit编码
  ← 将AXI信号打包为Flit (详见03_flit_encoding.md)
  ← 8-bit对齐填充
  ← WDATA2优化: 全1 WSTRB时省略WSTRB字段
       |
阶段3: 虚拟通道分类
  ← 3类VC: Management / Control / Data
  ← 独立队列缓冲
       |
阶段4: VC仲裁
  ← Management: 最高优先级 (抢占式)
  ← Control/Data: WRR仲裁, 权重1-255
       |
阶段5: DA仲裁与插入
  ← 128 DA, Round-Robin选择
  ← AXI User Signal -> DA映射
       |
阶段6: MAC帧封装
  ← 6B(DA) + 6B(SA) + 2B(TYPE) + 2B(IDLE) + PAXI_FLIT
  ← 简化帧格式 -> >99%带宽利用率
       |
阶段7: Credit检查
  ← Per-DA Outstanding Transaction检查
  ← 超过Credit Threshold则阻塞
       |
  → 送入CESOC (MAC/PCS/FEC)
  → 通过SerDes发送
```

## 2.4 RX数据路径详细分析

**[DOC]** 基于文档2.2.1和2.2.2:

> "On the remote side, the transfers from ethernet links on the axi stream interface were collected and decoded by the rule in the previous chapter, and then put the sub flit to the corresponding channels of AXI master interface."

RX路径的处理阶段:

```
  ← SerDes接收
  ← CESOC解码 (FEC纠错 -> PCS -> MAC)
       |
阶段1: MAC帧解析
  ← 提取DA, 确定目标
  ← 去除帧头(DA/SA/TYPE/IDLE)
       |
阶段2: RX缓冲
  ← 每帧缓冲
  ← [DOC] "RX buffer one mac frame. If this frame is bad
     indicated by mac, drop it. Else process it normally."
       |
阶段3: 水位线监测
  ← 比较High/Low Watermark
  ← 超过High Watermark触发PFC帧发送
       |
阶段4: Flit解码
  ← 从MAC payload提取AXI Sub Flit
  ← 解析Flit header确定类型
       |
阶段5: AXI事务重建
  ← 将Flit映射回AXI信号
  ← 送入对应AXI Master通道
       |
  → AXI Master Interface输出
```

## 2.5 内部缓冲结构

**[DOC]** 来自3.2.7 Overflow_Info寄存器, 可以推断出PAXI内部缓冲的完整结构:

### TX方向缓冲 (Bits[19:10])

| Bit | 缓冲名称       | 说明               |
| --- | -------------- | ------------------ |
| 10  | data B buffer  | Data通道B响应缓冲  |
| 11  | data AR buffer | Data通道读地址缓冲 |
| 12  | data AW buffer | Data通道写地址缓冲 |
| 13  | data R buffer  | Data通道读数据缓冲 |
| 14  | data W buffer  | Data通道写数据缓冲 |
| 15  | ctrl B buffer  | Ctrl通道B响应缓冲  |
| 16  | ctrl AR buffer | Ctrl通道读地址缓冲 |
| 17  | ctrl AW buffer | Ctrl通道写地址缓冲 |
| 18  | ctrl R buffer  | Ctrl通道读数据缓冲 |
| 19  | ctrl W buffer  | Ctrl通道写数据缓冲 |

### RX方向缓冲 (Bits[9:0])

| Bit | 缓冲名称       | 说明               |
| --- | -------------- | ------------------ |
| 0   | data B buffer  | Data通道B响应缓冲  |
| 1   | data AR buffer | Data通道读地址缓冲 |
| 2   | data AW buffer | Data通道写地址缓冲 |
| 3   | data R buffer  | Data通道读数据缓冲 |
| 4   | data W buffer  | Data通道写数据缓冲 |
| 5   | ctrl B buffer  | Ctrl通道B响应缓冲  |
| 6   | ctrl AR buffer | Ctrl通道读地址缓冲 |
| 7   | ctrl AW buffer | Ctrl通道写地址缓冲 |
| 8   | ctrl R buffer  | Ctrl通道读数据缓冲 |
| 9   | ctrl W buffer  | Ctrl通道写数据缓冲 |

**[推导]** 可以看出:

- Data Channel和Ctrl Channel各自有独立的5个缓冲(对应AXI 5通道)
- TX和RX方向各自有完整的缓冲结构
- 总计: 2(通道) x 5(AXI通道) x 2(TX/RX) = **20个独立缓冲**
- Management Channel的缓冲在此寄存器中未体现, 可能有独立机制

## 2.6 外部接口汇总

**[DOC]** 基于第4章Hardware Interface:

### AXI接口 (面向NoC)

| 接口            | 方向   | 说明                             |
| --------------- | ------ | -------------------------------- |
| Data AXI Master | Output | 接收远端数据后驱动本地AXI Master |
| Data AXI Slave  | Input  | 接收本地AXI事务                  |
| Ctrl AXI Master | Output | 控制平面AXI Master               |
| Ctrl AXI Slave  | Input  | 控制平面AXI Slave                |

### MAC接口 (面向以太网)

| 接口        | 方向   | 说明                  |
| ----------- | ------ | --------------------- |
| TX Data Bus | Output | 发送到MAC的AXI Stream |
| RX Data Bus | Input  | 从MAC接收的AXI Stream |

### 管理接口

| 接口       | 方向   | 说明                                |
| ---------- | ------ | ----------------------------------- |
| APB Slave  | Input  | 本地寄存器访问                      |
| APB Master | Output | 远程APB访问(通过Management Channel) |

### 存储接口

**[DOC]** 4.11 Memory Interface定义了PAXI使用的SRAM:

| 存储块         | 用途               |
| -------------- | ------------------ |
| Data AR Memory | Data通道读地址缓冲 |
| Data AW Memory | Data通道写地址缓冲 |
| Data W Memory  | Data通道写数据缓冲 |
| Data B Memory  | Data通道写响应缓冲 |
| Data R Memory  | Data通道读数据缓冲 |
| Ctrl AR Memory | Ctrl通道读地址缓冲 |
| Ctrl AW Memory | Ctrl通道写地址缓冲 |
| Ctrl W Memory  | Ctrl通道写数据缓冲 |
| Ctrl B Memory  | Ctrl通道写响应缓冲 |
| Ctrl R Memory  | Ctrl通道读数据缓冲 |
| F Memory       | Flit相关缓冲       |
| Dec Memory     | 解码相关缓冲       |

## 2.7 时钟与复位

**[DOC]** CESOC文档中定义:

- **SYS_CLK_I**: 系统主时钟, 频率最高1.35GHz, 时钟抖动要求 < +/- 100ppm
- **APB_PCLK_I**: APB寄存器访问时钟
- **SD*_TX_CLK_I** (x8): 8条SerDes Lane的TX时钟
- **SD*_RX_CLK_I** (x8): 8条SerDes Lane的RX时钟
- **SYS_RESET_I**: 异步高有效系统复位
- **APB_PRESTN_I**: 异步APB复位, 低有效

**[DOC]** PAXI支持软复位(3.2.6 REG_Soft_Reset):

> "Write 1 to trigger soft reset. This is a self-clean reset. After reset done, this bit will be clear automatically. User must not write 0 to this field."
