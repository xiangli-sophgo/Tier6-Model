# 11. 操作流程

## 11.1 Bring-up流程

**[DOC]** 2.11 Bring-up Flow:

> "PAXI can start bring-up flow regardless of MAC link status."

以下为128 DA + Ctrl通道使能的配置示例:

### 步骤

**[DOC]**:

1. **配置DA映射**: 写DA0信息到DA00 MAP和DA01 MAP寄存器, 依次配置DA1~DA127
2. **配置Credit**:
   - Data通道: 写DA0_CREDIT到DA01 MAP寄存器的Credit字段
   - Ctrl通道: 写ctrl_credit_da0到Ctrl Channel Credit Register0
   - 依次配置DA1~DA127
3. **配置源地址**: 写SA到MAC SA Register0 (0x04C) 和MAC SA Register1 (0x050)
4. **配置帧类型**: 写LEN_TYPE到MAC LEN_TYPE Field Register (0x048)
5. **配置仲裁权重**: 写DAXI_WEIGHT和CAXI_WEIGHT到WEIGHT CFG Register (0x01C)

**[推导]** Bring-up不依赖MAC链路状态, 意味着PAXI可以在物理链路建立之前完成软件配置。

## 11.2 Linkup消息流程

**[DOC]** 2.7 Linkup Message Flow:

1. **等待PCS链路**: 读CEPCS核心的PCS link-up寄存器, 等待物理链路建立
2. **配置DA映射**: 配置DA MAP寄存器
3. **触发Linkup**: 配置Message Ctrl Register (0x044) 的link-up字段
4. **发送消息**: PAXI自动发送128条link-up消息 (DA0到DA127)

**[推导]** 128条消息意味着向所有可能的远端DA通告本地链路已就绪。远端收到后会触发Linkup中断 (REG_STATUS bit5)。

## 11.3 运行时Credit变更流程

**[DOC]** 2.12 Runtime Credit change flow:

> "PAXI supports DA and related credit info reconfigure when PAXI is idle."

### 前提条件

PAXI必须处于IDLE状态。

### 步骤

**[DOC]**:

1. **停止新事务**: 停止所有新的AXI和远程APB事务
2. **等待IDLE**: 读PAXI Status Register (0x020) 直到IDLE字段为1
3. **重新配置**: 重复Bring-up流程的步骤1~2 (DA映射和Credit)

**[推导]** 运行时Credit变更需要停止所有事务, 这是一个相对重量级的操作。对于需要频繁调整Credit的场景, 应使用Credit Update协商机制 (通过Message Ctrl), 不需要停止事务。

## 11.4 Credit Update协商流程

**[DOC]** 2.9 Credit Update Flow:

这是不需要停止事务的在线Credit变更方式:

```
发起端(Chip A)                         远端(Chip B)
     |                                      |
1.   |-- 写Message Ctrl (REQ=1) ----------->|
     |   Credit Update Request (Mgmt VC)    |
     |                                      |
2.   |                                      |-- PAXI中断
     |                                      |-- 读Credit Update Status Register
     |                                      |
3.   |<-- 写Message Ctrl (REQ=0, ACK=1) ---|
     |    Credit Update Response            |
     |                                      |
4.   |-- PAXI中断                           |
     |-- 读Credit Update Status             |
     |                                      |
5.   |-- 如ACK: 更新DA MAP + Credit Reg --->|
     |                                      |
```

Message Ctrl关键字段设置:

- REQ=1, INC=1/0, CV=变更量: 发送增/减Credit请求
- REQ=0, ACK=1: 发送确认响应
- REQ=0, ACK=0: 发送拒绝响应

## 11.5 远程APB访问流程

**[DOC]** 2.6 Remote APB ACCESS:

### APB写操作

**[DOC]**:

1. 写数据到Remote APB DATA Register (0x03C)
2. 设置Remote APB CTRL Register (0x038): 地址/写标志/目标DA
3. 设置后PAXI自动生成Management Flit发送到MAC
4. 等待远端APB响应返回后, APB Ready才释放

### APB读操作

**[DOC]**:

1. 设置Remote APB CTRL Register (0x038): 地址/读标志/目标DA
2. 设置后PAXI自动生成Management Flit
3. 等待远端APB响应, APB Ready释放
4. 读Remote APB RDATA Register (0x040) 获取读数据

### 超时处理

**[DOC]**:

> "If there is no response received for a long time after APB access sends, a SLVERR response will be triggered when time counter reaches the threshold specified by Remote APB Timeout Threshold Register. At the same time, paxi will release APB ready forcibly."

超时阈值由Remote APB Timeout Threshold Register (0x054) 配置, 默认值0x0010_0000。

## 11.6 延迟测量流程

**[DOC]** 2.8 Latency Measurement:

### 基本流程

1. 配置Latency Ctrl Register (0x024): 设置AXI ID和使能
2. 发起AXI读或写事务, 使用配置的ID
3. PAXI自动计算该ID事务的延迟
4. 测量完成后触发中断 (REG_STATUS bit3: LM done)
5. 读取延迟结果寄存器 (0x028~0x034)

**[DOC]**: 为精确测量, 建议设置 `length port=0`。

### 超时判断

**[DOC]**: 如果测量超时, 延迟结果寄存器低31位全为1 (0x7FFF_FFFF), 同时触发LM timeout中断 (REG_STATUS bit4)。

### 两种测量模式

**[DOC]**:

#### Mode 0: P2P without NOC

- 测量PAXI到PAXI的纯链路延迟
- 远端PAXI需配置为Remote Loopback模式
- 本端Latency Ctrl的loopback mode位 (bit17) 设为1

#### Mode 1: P2P with NOC

- 测量包含NoC (片上网络) 在内的端到端延迟
- 远端无需特殊配置, 事务正常经过NoC路由

**[DOC]**: 使用前双方必须协商并配置好测量模式。

## 11.7 Pattern Generator流程

**[DOC]** 2.14 Pattern Generator:

PAXI内置Pattern Generator, 用于生成AXI读写模式进行测试/验证。

### 配置参数

**[DOC]** (以写操作为例):

| 参数           | 说明                                                   |
| -------------- | ------------------------------------------------------ |
| PAT_EN         | PAXI Ctrl Register (0x00C) bit5, 使能Pattern Generator |
| DA_MAP         | 目标地址映射, 遵循2.3节DA映射规则, 最多4个DA           |
| Max Round      | 写传输重复次数, 每轮发送4KB数据, 最大2^19 - 1轮        |
| WDATA          | AXI写数据值, 用8位WDATA值拼接填充                      |
| Write_Doorbell | 触发写序列启动                                         |

### 写操作流程

**[DOC]**:

1. 使能PAT_EN
2. 配置DA_MAP (最多4个DA目标)
3. 配置Max Round (最大 2^19 - 1 = 524287)
4. 配置WDATA填充值
5. 写Write_Doorbell触发
6. Pattern Generator自动发送4KB x Max Round数据
7. 所有AW+W请求发完且收到所有B响应后:
   - PAXI Status Register (0x020) 的PAT_WR_DONE (bit4) 置位
   - 触发中断 (REG_STATUS bit7: PAT WR DONE)

### 错误检测

**[DOC]** PAXI Status Register:

- PAT_WR_ERR (bit[12:9]): 每bit对应1个DA设备的写错误
- PAT_RD_ERR (bit[8:5]): 每bit对应1个DA设备的读错误

## 11.8 PFC帧生成流程

**[DOC]** 2.13 PFC Frame Creation flow:

1. **配置水位线**: 设置DAXI Water Mark Register (0x058) 和CAXI Water Mark Register (0x05C) 的高水位线和低水位线
2. **自动触发**: 当接收数据量超过高水位线, PAXI触发MAC发送PFC帧, 限制远端DA数据传输
3. **自动解除**: 当数据量降到低水位线以下, 解除PFC限制

## 11.9 错误恢复流程

**[DOC]** 2.10 Retry Process Flow:

### 触发条件

PAXI在以下情况下触发错误处理并断言中断:

- TX Retry Fatal (TX找不到重传包)
- RX Retry Fail (RX接收重传失败)
- RX Retry Timeout (RX重传超时)
- 收到远端Linkup消息

### 恢复步骤

**[DOC]**:

1. **中断触发**: PAXI断言中断
2. **诊断**: 读取以下寄存器确定错误DA和详情:
   - Int Indicator Register (0x008)
   - Remote PCS Linkup Register (0xA20~0xA2C)
   - TX MAC L2 Retry Field Register (0xA40~0xA4C)
   - RX MAC L2 Retry Field Register (0xA60~0xA6C)
3. **隔离**: 系统停止向错误DA发送新请求
4. **清除**: 写1到上述所有寄存器清除错误状态
5. **恢复**: 确认所有错误位清除后, 恢复发送

**[DOC]** 关键约束:

> "User can send new request only When all the above registers status are clear. Any of error bit assertion cause transaction drop."

## 11.10 软复位流程

**[DOC]** REG_Soft_Reset (0x014):

1. 写1到Soft Reset位 (bit0) 触发软复位
2. 复位完成后该位自动清除
3. **禁止写0**: 用户不得向该字段写0
4. 复位后所有寄存器恢复默认值, 需重新执行Bring-up流程
