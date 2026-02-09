# 03. SG2262 C2C 通信机制

## 3.1 Send/Receive 流程

### 概述

**[DOC]** Send/Receive 的核心优势：
- 不同芯片之间交互的地址可**动态分配**
- 通过配对完成发送方和接收方的**同步**

替代方案：如不支持 Send/Receive，可通过 write + msg 方式替代，但地址需预分配。

### 配对规则

**[DOC]** Send/Receive 的严格约束：

```
Chip A                          Chip B
  CDMA Thread 0 <-- Send -----> CDMA Thread X -- Receive
  CDMA Thread 1 <-- Send -----> CDMA Thread Y -- Receive
  ...

规则:
  - 每个 CDMA Thread 的 Send 只能与一个对端 Thread 配对
  - 每个 CDMA Thread 的 Receive 只能与一个对端 Thread 配对
  - Send 和 Receive 可与不同芯片配对
  - 不支持一对多、多对多、多对一
  - 对应顺序由软件配置
```

### CFS 模式下的 Send/Receive

**[DOC]** Send/Receive under CFS 的流程：

```
时序流程:
  Receive Chip                              Send Chip
  -----------                               ---------
  1. CDMA Thread 收到 Receive 指令
     |
  2. 发送 tcredit ----------------------->  CDMA Thread 收到 tcredit
     (携带 chipid + cdmaid)                   + Send 指令
                                               |
                                            3. 进入 Ready 状态
                                               参与 Datapath 仲裁
                                               |
                                            4. 获得仲裁，执行搬运
                                               (write_send 报文)
                                               |
                                            5. Datapath 退出
                                               Thread 收集 Bresp
                                               |
  6. 收到 Write_done <--------------------  7. 所有 Bresp 收齐
     发送 Bresp 给 Send Chip                   发送 Write_done
     Retire Receive 指令                       Retire Send 指令
```

**[DOC]** CFS 模式限制与设计要点：

| 要点 | 说明 |
|------|------|
| tcredit 队列深度 | 32 per CDMA（防止 tracker 满阻塞网络） |
| tcredit 存储 | 放进 SRAM（多线程共享），chipid+cdmaid 用寄存器做 tracker |
| 一次通信指令数 | 最多 30 个，通信之间可全系统同步 |
| Send/Receive 切换 | 不需要做同步（但可以做） |
| Send descriptor | 需指定 receive chip 和 receive cdma thread |
| Receive descriptor | 需指定 send chip 和 send cdma thread |
| Datapath 退出 | Send 发完所有 write_send 后退出，bresp 收集交还 thread |
| Bresp 计数 | Send cmd 的 bresp 独立计数 |

### CHS 模式下的 Send/Receive

**[DOC]** CHS 模式与 CFS 几乎相同，关键差异：

- write_done 地址放到**保序窗口**
- 保序窗口确保 write_done 在前置 write_send 全部完成后才释放
- 额外限制：write_done 地址需落在保序窗口配置范围内

```
CFS: write_done 通过 fence 保证与 write_send 的顺序
CHS: write_done 通过保序窗口保证与 write_send 的顺序
```

## 3.2 报文格式

### C2C 传输报文

**[DOC]** C2C 使用 AXI user 信号传递关键信息：

| 信号 | 用途 |
|------|------|
| dst_chipid | 目标芯片 ID（MAC 识别 dst macid） |
| src_chipid | 源芯片 ID（在源芯片 MAC 挂载） |
| src_chipen | 标识请求是否由本芯片发起 |

**[DOC]** 转发场景报文流程：

```
Chip A (src) --> Chip B (relay) --> Chip C (dst)

1. Chip A 发出请求：
   - awuser 携带 dst_chipid = C
   - MAC 挂载 src_chipid = A, src_chipen = 1

2. Chip B 收到请求：
   - 识别 dst_chipid != 本芯片
   - 执行转发，保持 src_chipid = A
   - src_chipen = 0（非本芯片发起）

3. Chip C 收到请求：
   - 识别 dst_chipid = 本芯片
   - 使用 src_chipid = A 回复 Bresp/Rresp
```

### Send/Receive 报文

**[DOC]** Send/Receive 涉及的报文类型：

| 报文类型 | 方向 | 关键字段 |
|----------|------|---------|
| tcredit | Receive -> Send | receive 指令内容打包进 wdata + cdmaid |
| write_send | Send -> Receive | awuser 携带 reduce_opcode |
| write_done | Send -> Receive | 写入 send chip 指定的 receive chip cdma write_done 地址 |
| bresp_send | Receive -> Send | 确认 write_done 接收 |

### All Reduce 操作码映射

**[DOC]** C2C All Reduce 的操作码定义：

```
awuser 格式 (k2k):
  [11:8] psum_op    -- Reduce操作类型
  [7:4]  opcode     -- 具体操作码
  [3:0]  dtype      -- 数据类型

C2C 映射 (PCIe Link):
  Reduce_op[4]   --> des_lst (是否最后一个包)
  Reduce_op[3:2] --> psum_op 映射
  Reduce_op[1:0] --> opcode/dtype 映射
```

## 3.3 Datagram 模式

**[DOC]** Datagram 支持通过配置 buffer 触发发送和接收原始以太网帧。C2C 不提供任何可靠性支持，全部由软件完成。

### 发送 Buffer

```
Send Datagram Buffer:
  +-- 32 个元素，每元素 128B（共 4KB 地址空间）
  +-- 配套 size 寄存器 FIFO（深度 32）
  +-- 支持帧长度超过 128B 时跨元素紧凑存储
```

**[DOC]** 发送流程：
1. 软件写入数据到 buffer（以 128B 为单位）
2. 配置 size 寄存器指定帧长度
3. 收到 size 配置后硬件开始发送（允许 MAC 背压）
4. 发送完成后清除长度字段，元素标记为空

### 接收 Buffer

```
Receive Datagram Buffer:
  +-- 32 个元素，每元素 128B（共 4KB，只读）
  +-- 配套 size 寄存器
  +-- 按 MACID + Priority 匹配过滤
```

**[DOC]** 接收流程：
1. PCS 收到帧后，DMAC 与 MACID 比对 + Priority 比对
2. 匹配后写入 buffer 元素，更新长度字段
3. 上报中断通知 CPU 介入
4. CPU 读取长度 -> 读取帧数据（每次完整元素 128B） -> 自动指向下一帧

### Datagram 寄存器

| 寄存器 | 描述 |
|--------|------|
| `send_datagram[31:0]` | 发送帧位置，4KB，128B 写入 |
| `reg_rptr_send_datagram` | 当前待发送首帧的元素指针 |
| `reg_space_send_datagram` | 当前剩余元素空间 |
| `reg_size_send_datagram[15:0]` | 每个帧长度配置 |
| `rcv_datagram[31:0]` | 接收帧位置，只读 |
| `reg_size_rcv_datagram[15:0]` | 接收帧长度 |

### 中断

| 中断类型 | 触发条件 |
|----------|---------|
| 发送帧中断 | 帧发送完成 |
| 接收帧中断 | 帧接收完成 |
| 发送帧溢出中断 | 写入导致 buffer 溢出（帧被丢弃） |
| 接收帧溢出中断 | 无空闲元素（帧被丢弃） |
| 读接收帧溢出中断 | 读取空元素 |

## 3.4 拓扑建立过程

**[DOC]** 支持两种拓扑建立方式：

1. **静态配置**: 预配置所有路由表和 MAC ID 映射
2. **动态建立**: 通过广播发现芯片

**[DOC]** 动态建立注意事项：
- MAC ID 广播阶段需过滤不属于本 chip/cluster 的报文
- 从交换机来的报文一定属于本 chip/cluster
- 防止错误报文造成系统错误

## 3.5 通信性能关键参数汇总

**[推导]** 基于文档的性能关键参数：

| 参数 | 值 | 来源 |
|------|------|------|
| 单芯片 C2C 总带宽 (112G) | 448 GB/s | 8 x4 x 112Gbps |
| 单芯片 C2C 总带宽 (56G) | 224 GB/s | 8 x4 x 56Gbps |
| CDMA 数/Die | 4 | [DOC] |
| CDMA 带宽上限 | 64 GB/s | [DOC] |
| CDMA 总带宽/Die | 256 GB/s | 4 x 64 GB/s |
| CDMA Thread 数/Die | 32 | 4 CDMA x 8 Thread |
| tcredit 队列深度 | 32/CDMA | [DOC] |
| 一次通信最大指令数 | 30 | [DOC] |
| AXI 报文大小 | 256B / 512B | [DOC] |
| MAC MTU | 1.5 KB | [DOC] |
| 保序窗口最大数 | 32 | [DOC] |
| Datagram Buffer 深度 | 32 x 128B = 4KB | [DOC] |
| MAC ID 最大长度 | 10 bit | [DOC] |
| 最大芯片数 | 1024 | [DOC] |
| L1 cluster 最大芯片数 | 32 | [DOC] |
