# 11. 操作流程

## 11.1 Bring-up流程

**[DOC]** SUE2.0 2.7 Bring-up Flow:

> "PAXI can start bring-up flow regardless of MAC link status."

### 步骤

**[DOC]**:

1. **配置Multi DA Enable** (如需特定DA使能): 写Multi DA Enable Register (0x380~0x3FC), 默认全部DA使能 (0xFFFF_FFFF)
2. **配置水位线** (可选): 配置RX Buffer水位线寄存器 (0x058/0x05C/0x060)
3. **配置通道权重** (可选): 配置Channel Weight Register (0x070/0x074)
4. **配置CBFC/PFC模式**: 设置Ctrl Register (0x00C) bit7 (CBFC_EN)

**[推导]** 与旧版对比, Bring-up流程大幅简化:
- 移除: DA MAP寄存器配置 (128 DA x 2个寄存器, 共256次寄存器写)
- 移除: Data/Ctrl通道Credit配置
- 移除: 源MAC地址和帧类型配置 (MAC相关配置由CESOC/RC Link承担)
- 保留: 水位线和使能配置
- 新增: Multi DA Enable控制哪些DA参与链路建立

## 11.2 Linkup消息流程

**[DOC]** SUE2.0 2.7:

1. **等待PCS链路**: 读CEPCS核心的PCS link-up寄存器, 等待物理链路建立
2. **触发Linkup**: 配置Message Ctrl Register (0x050) 的link-up字段 (bit1=1)
3. **发送消息**: PAXI自动向Multi DA Enable寄存器中使能的DA发送link-up消息

**[推导]** 与旧版对比:
- 旧版固定发送128条link-up消息 (DA0到DA127)
- SUE2.0只向Multi DA Enable Register中bit=1的DA发送, 减少不必要的消息
- 远端收到后触发Linkup中断 (Int Indicator Register bit5: APB_Linkup_MSG)

## 11.3 远程APB访问流程

**[DOC]** SUE2.0 2.5 Remote APB ACCESS:

### APB写操作

**[DOC]**:

1. 写数据到Remote APB DATA Register (0x048)
2. 设置Remote APB CTRL Register (0x044): 地址/写标志(bit0=1)/目标DA
3. 如需扩展地址空间: 配置Remote APB Higher Address Register (0x078), 设置HADDR_EN=1, 最大支持4G地址空间
4. PAXI自动生成APB Flit发送到RC Link
5. 等待远端APB响应返回

### APB读操作

**[DOC]**:

1. 设置Remote APB CTRL Register (0x044): 地址/读标志(bit0=0)/目标DA
2. PAXI自动生成APB Flit
3. 等待远端APB响应
4. 读Remote APB RDATA Register (0x04C) 获取读数据

### 超时处理

**[DOC]**:

> "If there is no response received for a long time after APB access sends, a SLVERR response will be triggered when time counter reaches the threshold specified by Remote APB Timeout Threshold Register."

超时阈值由Register (0x054) 配置, 默认值0x0010_0000, 单位为us。

## 11.4 延迟测量流程

**[DOC]** SUE2.0 2.8 Latency Measurement:

### 两种测量模式

**[DOC]** Latency Ctrl Register (0x020) bit17 (Mode select):

| 模式 | Mode select | 说明 |
|------|-------------|------|
| AXI ID模式 | 0 | 按AXI事务ID匹配, 测量指定ID的事务延迟 |
| DA模式 | 1 | 按目标DA地址匹配, 测量到指定DA的事务延迟 |

**[推导]** DA模式是SUE2.0新增的, 适用于需要测量特定目标设备延迟而非特定事务ID延迟的场景。

### 基本流程

1. 配置Latency Ctrl Register (0x020):
   - 设置Mode select (bit17): 选择AXI ID或DA模式
   - AXI ID模式: 设置AXI ID (bit[15:0])
   - DA模式: 设置Destination address (bit[20+DAW-1:20])
   - 可选: Latency measure loopback mode (bit18), 需远端先设Remote loopback=1
2. 设置Latency mode enable (bit16) 使能测量
3. 发起AXI读或写事务
4. PAXI自动计算延迟
5. 测量完成后触发中断 (Int Indicator bit3: LM done)
6. 读取延迟结果寄存器:
   - 0x024: 写延迟 (bit31=done, bit[30:0]=cycle数)
   - 0x028: 读延迟
   - 0x02C~0x034: 写事务的匹配ID和地址
   - 0x038~0x040: 读事务的匹配ID和地址

### 超时判断

**[DOC]**: 如果测量超时, 延迟结果低31位全为1, 同时触发LM timeout中断 (Int Indicator bit4)。

### Pattern Generator总延迟测量

**[DOC]** Latency Ctrl (0x020) bit19:

写1使能Pattern Generator总轮次延迟测量。配合Pattern Generator Write/Read Ctrl寄存器中的`WR/RD_Total_Round_Latency_Measure`位 (bit22) 使用, 可测量Pattern Generator完成所有轮次的总延迟。

## 11.5 Pattern Generator流程

**[DOC]** SUE2.0 2.12 Pattern Generator:

PAXI内置Pattern Generator, 用于生成AXI读写模式进行测试/验证。

### 配置参数

**[DOC]**:

| 参数 | 寄存器 | 说明 |
|------|--------|------|
| PAT_EN | Ctrl Register (0x00C) bit5 | Pattern Generator使能 |
| MPS | Pattern Generator Ctrl (0x06C) bit[3:0] | 数据包大小 (128B~4KB) |
| GAP | Pattern Generator Ctrl (0x06C) bit[11:4] | burst间空闲周期数 |
| DA_SEL | Pattern Generator Ctrl (0x06C) bit[15:12] | 目标DA选择 (最多4个) |
| ROUND | Write/Read Ctrl bit[21:1] | 轮次数, 最大 2^21 - 1 |
| PAT_DATA | Write Ctrl (0x064) bit[31:24] | 8-bit写数据填充值 |
| PAT_RDATA | Read Ctrl (0x068) bit[31:24] | 8-bit读数据期望值 |

**[推导]** 与旧版对比增强:
- MPS: 可配置数据包大小 (旧版固定4KB), 支持128B到4KB共6档
- GAP: 可配置burst间空闲周期, 用于模拟不同流量负载
- Max Round: 从2^19-1增加到2^21-1 (约200万轮)
- 新增WR/RD_Total_Round_Latency_Measure: 总轮次延迟测量

### 写操作流程

**[DOC]**:

1. 使能PAT_EN (Ctrl Register bit5)
2. 配置Pattern Generator Ctrl (0x06C): MPS, GAP, DA_SEL
3. 配置Pattern Generator Write Ctrl (0x064): PAT_DATA, ROUND
4. 可选: 设置WR_Total_Round_Latency_Measure (bit22) 使能总延迟测量
5. 写WR_DoorBell (bit0) 触发
6. Pattern Generator自动发送 MPS x ROUND 数据
7. 完成后:
   - PAXI Status (0x01C) PAT_WR_DONE (bit4) 置位
   - 触发中断 (Int Indicator bit7)

### 读操作流程

**[DOC]**:

1. 配置Pattern Generator Read Ctrl (0x068): PAT_RDATA, ROUND
2. 可选: 设置RD_Total_Round_Latency_Measure (bit22)
3. 写RD_DoorBell (bit0) 触发
4. 完成后:
   - PAXI Status (0x01C) PAT_RD_DONE (bit3) 置位
   - 触发中断 (Int Indicator bit8)

### 错误检测

**[DOC]** PAXI Status Register:

- PAT_WR_ERR (bit[12:9]): 每bit对应1个DA设备的写错误
- PAT_RD_ERR (bit[8:5]): 每bit对应1个DA设备的读错误

## 11.6 多播操作流程

**[DOC]** SUE2.0 2.9 Multi-Cast:

### 发送多播写

1. 确认多播功能已使能 (综合宏`UVIP_PAXI_MULTI_CAST_EN`)
2. 在CEMAC Core中配置多播组寄存器 (设备-组映射关系)
3. AXI Master发起写事务, 在AXI USER field中设置:
   - Multicast位 = 1
   - DA低3位 = 多播组号 (0~7)
   - VC = 4 (Mode 0默认)
4. PAXI封装为TYPE2报文通过RC Link发送
5. 等待B响应 (多设备响应合并为一个)

### B响应处理

**[DOC]**:

> "Upon receiving multiple B responses from other multicast group devices, they are merged to one B resp and returned to AXI. If PAXI receives an error response from any devices, it will result in returning a B response error."

### 超时处理

如果在RX Multicast Timeout Register (0x080) 配置的cycle数内未收到全部B响应, 触发MULTI-CAST TIMEOUT中断 (Int Indicator bit12)。

## 11.7 错误恢复流程

**[DOC]** SUE2.0 2.6 Error Handle:

### 单播错误恢复

**[DOC]** 完整步骤:

1. **检测**: DA重传失败时, PAXI上报retry fail中断 (Int Indicator bit1: E2E_RETRY_ERR)
2. **诊断**: 读取以下寄存器确定错误DA和详情:
   - Int Indicator Register (0x008)
   - Retry Error Register (0x300~0x37C)
3. **隔离**: TX侧完成当前burst的AW+W数据传输, 停止向错误DA发送新请求
4. **错误处理**:
   - 自动模式 (Ctrl Register bit13=1): PAXI自动进入错误处理, 丢弃错误DA数据
   - 手动模式 (Ctrl Register bit13=0): 软件写Ctrl Register bit14=1触发错误处理
5. **确认完成**: 读取PAXI Status (0x01C) 的TX_ERR_STAT (bit13) 和RX_ERR_STAT (bit14), 确认均为0
6. **恢复**: 清除Int Indicator Register和Retry Error Register, 恢复AXI传输

### 多播错误恢复

**[DOC]** 步骤:

1. **检测**: 多播重传失败或B响应超时
2. **诊断**: 读取PAXI Status (bit17/18) 和Int Indicator (bit11/12)
3. **TX隔离**: 完成当前burst的AW+W数据传输
4. **数据清除**: 默认TX和RX自动清除多播数据, 也可通过Ctrl Register手动控制
5. **确认完成**: 读取PAXI Status的TX_MULTI_CAST_ERR_STAT (bit15) 和RX_MULTI_CAST_ERR_STAT (bit16), 确认均为0
6. **恢复**: 清除相关中断和状态寄存器, 恢复传输

## 11.8 PFC/CBFC配置流程

### PFC配置

**[DOC]**:

1. 设置Ctrl Register (0x00C) bit7 = 0 (PFC模式)
2. 配置RX Buffer水位线:
   - REQ Water Mark (0x058): 高/低水位线
   - RSP Water Mark (0x05C): 高/低水位线
   - MUL Water Mark (0x060): 高/低水位线
3. 当接收数据量超过高水位线 -> PAXI触发MAC发送PFC帧, 暂停远端对应VC发送
4. 当数据量降到低水位线以下 -> 解除PFC限制

### CBFC配置

**[DOC]**:

1. 设置Ctrl Register (0x00C) bit7 = 1 (CBFC模式)
2. RC Link侧配置各VC的CBFC参数 (credit_size, credit_limit等)
3. PAXI侧通过RX Buffer水位线和RC Link CBFC信号协同控制流量

**[推导]** CBFC和PFC互斥, 不可同时使用。即使使用CBFC模式, RX方向的PFC反压信号仍被复用。

## 11.9 软复位流程

**[DOC]** Soft Reset Register (0x014):

1. 写1到Soft Reset位 (bit0) 触发软复位
2. 复位完成后该位自动清除
3. **禁止写0**: 用户不得向该字段写0
4. 复位后所有寄存器恢复默认值, 需重新执行Bring-up流程
