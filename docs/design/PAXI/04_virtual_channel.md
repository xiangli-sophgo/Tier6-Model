# 04. 虚拟通道与仲裁机制

## 4.1 三类虚拟通道

**[DOC]** PAXI Features:

> "Support 3 type virtual channels, inside management/control/data"
> "Independent enablement/reset for 3 virtual channels."

| 虚拟通道 | 用途 | 优先级 | 独立复位 |
|----------|------|--------|---------|
| Management | 系统管理、链路控制 | 最高(抢占) | 支持 |
| Control | 控制平面AXI事务 | 中 (WRR) | 支持 |
| Data | 数据平面AXI事务 | 低 (WRR) | 支持 |

### Management Channel

管理通道承载的消息类型:

1. **Linkup消息**: 链路建立时发送128条消息(DA0-127)
2. **Credit Update Request/Response**: 运行时信用更新协商
3. **远程APB访问**: 访问远端芯片的寄存器

**[DOC]** 2.6 Remote APB ACCESS:

> "paxi generate management flit to MAC"

管理通道通过APB Flit (flit type=3'b100) 和 Message Flit (flit type=3'b101) 传输。

### Control Channel

**[推导]** 控制通道用于承载控制平面的AXI事务, 有独立的AXI Master/Slave接口对:
- Ctrl AXI Master Interface
- Ctrl AXI Slave Interface

控制通道是可选的:

**[DOC]** 3.2.13/3.2.14 CAXI Latency Register:

> "Note: if not define CAXI channel, this CAXI registers type is reserved."

### Data Channel

数据通道承载主要的数据平面AXI事务, 拥有独立的AXI Master/Slave接口对:
- Data AXI Master Interface
- Data AXI Slave Interface

## 4.2 虚拟通道仲裁

**[DOC]** 2.4 Virtual Channel Arbitration (完整原文):

> "The management channel always has the highest priority, Secondly ctrl channel and then data channel. Data and ctrl channel is arbitrated by weight from 1 ~ 255 which is user configured. Paxi will prioritize high priority channels until the configured weight ratio is reached or there is no data to send."

### 仲裁规则

1. **Management Channel**: 绝对最高优先级, 只要有数据就优先发送
2. **Control Channel vs Data Channel**: 使用WRR (Weighted Round-Robin)
   - 权重值范围: 1 ~ 255
   - 高优先级通道优先, 直到达到配置的权重比或无数据可发

### 权重配置

**[DOC]** 3.2.8 WEIGHT CFG Register (地址0x01c):

| 字段 | Bits | 说明 |
|------|------|------|
| DAXI_WEIGHT | [31:16] | Data AXI通道权重配置 |
| CAXI_WEIGHT | [15:0] | Ctrl AXI通道权重配置 |

默认复位值: 0x0001_0100

**[推导]** 默认值分析:
- DAXI_WEIGHT = 0x0001 = 1
- CAXI_WEIGHT = 0x0100 = 256

这意味着默认情况下, Ctrl通道权重远高于Data通道。实际使用时需根据业务调整。

### WRR行为模型

**[推导]** 基于文档描述的WRR行为:

```
每轮调度:
1. 如果Management队列有数据 -> 立即发送, 不消耗其他通道权重
2. 否则进入WRR:
   a. Ctrl通道可发送 CAXI_WEIGHT 个flit
   b. Data通道可发送 DAXI_WEIGHT 个flit
   c. 如果某通道无数据, 其配额转给另一通道
3. 当两个通道的配额都用完, 重新开始一轮
```

## 4.3 DA (Destination Address) 仲裁

**[DOC]** 2.5 DA Arbitration (完整原文):

> "There are 128 DAs for each channel, and the arbiter select the DAs by round robin and read the corresponding channel. So if multi-da exist, same da will not be pick continuously unless the there is not other DA transfer."

### 关键要点

1. **每个虚拟通道**都有独立的128个DA
2. DA选择使用 **Round-Robin** 算法
3. **公平性保证**: 同一DA不会被连续选择(除非只有它有数据)

### DA映射

**[DOC]** 2.3 DA insertion Handling:

> "Can be mapped to 128 different DA."

映射规则:

| AXI User Signal值 | 映射到的DA寄存器 |
|-------------------|----------------|
| 0 | DA00, DA01 |
| 1 | DA10, DA11 |
| 2 | DA20, DA21 |
| ... | ... |
| 127 | DA1270, DA1271 |

每个DA有两个寄存器(DA*0和DA*1), 分别存储:
- DA*0: 目的地址低32位部分 + DA使能等信息
- DA*1: 目的地址高16位部分 + Credit信息

**[DOC]** DA MAP Register (3.2.100~355): 地址范围 0x200 ~ 0x4fc

## 4.4 两级仲裁的整体行为

完整的发送仲裁过程是两级流水:

```
第一级: VC仲裁 (选择哪个虚拟通道)
  Management ─┐
  Control ────┼─→ VC Arbiter ──→ 选中的VC
  Data ───────┘

第二级: DA仲裁 (在选中的VC内选择哪个DA)
  DA[0] ──┐
  DA[1] ──┤
  ...     ├─→ DA Arbiter (Round-Robin) ──→ 选中的DA
  DA[126]─┤
  DA[127]─┘
```

**[推导]** 这种两级结构意味着:
- VC级别保证了管理/控制/数据流量的优先级隔离
- DA级别保证了不同目的芯片之间的公平带宽分配
- 组合效果: 既有优先级保障, 又有公平性保障
