# 02. SG2262 C2C 微架构详解

## 2.1 MAC ID 映射方案

SG2262 支持两种 MAC ID 映射方案，决定了芯片在网络中的寻址方式。

### MAC ID 结构

**[DOC]** MAC ID 分为三段，总长不超过 **10 bit**（最多寻址 1024 芯片）：

```
MAC ID (最长 10 bit):
  [高位]         [中位]         [低位]
  L2 网络 ID  +  L1 网络 ID  +  Die ID
  (0~10 bit)     (0~10 bit)     (0~10 bit)
```

每段长度均可选择 0~10 bit，但总长 <= 10 bit。

### DMAP (Die-level MAC ID Assignment Plan)

**[DOC]** 以 Die 为单位分配 MAC ID：每个 Die 的所有 C2C MAC 分配一个独立 MAC ID。

```
Chip (多 Die)
  +-- Die 0 --> MAC ID = X    --> 所有 C2C Port 共用
  +-- Die 1 --> MAC ID = X+1  --> 所有 C2C Port 共用
```

**路由流程**: NoC 根据 SAM 表区分发往 dst chip 不同 die。

**限制**:
1. 所有 chip 的 16share/8share/4share/private 分配方案（包括地址区间）要统一
2. 软件需配置所有 die 的 CDMA 才能利用全部 C2C port
3. 支持的最大 chip 数受限
4. 不支持芯片转发（否则 resp 会走错路径）
5. 所有发给 C2C 的流量必须先经过 NoC
6. **仅支持单层 clos 拓扑**

**[推导]** DMAP 的限制源于多 Die 芯片中路径唯一性约束。不同 Die 独立出 port，在非 clos 拓扑中无法保证每个 Die 到目标芯片的路径一致。

### CMAP (Chip-level MAC ID Assignment Plan)

**[DOC]** 以 Chip 为单位分配 MAC ID：同一 Chip 的所有 C2C MAC（跨 Die）分配相同 Chip ID。

```
Chip --> Chip ID = X
  +-- Die 0 --> C2C Port 0~3 --> 均使用 Chip ID = X
  +-- Die 1 --> C2C Port 4~7 --> 均使用 Chip ID = X
```

**路由流程**: 所有 C2C port 视为等价，是否跨 Die 由 NoC 决定。

**优势**: 无额外限制，支持所有六种拓扑。
**潜在问题**: 会出现冗余 D2D 流量，影响 GDMA 性能。

### MAC ID 映射与拓扑兼容性

| 拓扑 | CMAP | DMAP |
|------|------|------|
| clos | Y | Y |
| cube | Y | N |
| all2all + clos | Y | N |
| clos + clos | Y | N |
| 低成本 all2all + clos | Y | N |
| 低成本 clos + clos | Y | N |

**[推导]** CMAP 是更通用的方案，DMAP 仅在单层 clos 下有效，但可避免 D2D 冗余流量。

## 2.2 CLE (Chip Address Lookup Engine) 路由方案

### 路由原则

**[DOC]** 路由遵循以下原则：
1. **先做 L2 路由，再做 L1 路由**
2. L2 仅支持交换机路由
3. L1 既支持交换机路由，也支持点对点互联

### CLE 寄存器

| 寄存器 | 描述 |
|--------|------|
| `reg_chipid` | 本芯片的 Chip ID |
| `reg_l1_chip_num` | L1 网络芯片数量（生成路由表 mask） |
| `reg_l1_port_num` | L1 网络端口数量 |
| `reg_l1_itlv` | L1 网络是否 interleave 模式 |
| `reg_l1_port_list` | L1 网络端口列表 |
| `reg_l2_port_num` | L2 网络端口数量 |
| `reg_l2_port_list` | L2 网络端口列表 |
| `reg_l2_port_exist` | 本芯片是否有 L2 C2C link |
| `reg_l2_sw_cap_port` | 本 cluster 中支持 L2 转发的芯片 |

### CLE 路由查找表

**[DOC]** 查找表规格：
- 每项最多支持 **4 个 port** 做 interleave
- 最多支持 **32 项**查找表
- 输入：10 bit Chip ID --> 输出：4 bit Port ID

### CLE 状态机流程

**[DOC]** 当一笔请求进入 CLE 时的处理流程：

```
输入: dst_chipid (10 bit)
                |
    [1] 比较 L2 Chip ID
         |              |
    相同(同 cluster)  不同(跨 cluster)
         |              |
  [2a] 比较 L1 Chip ID  [2b] 本芯片能做 L2 转发?
    |          |            |            |
  相同       不同         能转发       不能转发
  (本芯片)    |            |            |
    |     [3b] L1模式?   [3c] itlv    [3d] portid =
  portid=0   |    |       粒度选择      l2_sw_cap_port
           查找表  itlv     |
             |     |       选择 L2 port
           [4a]  [4b]
           查表   itlv算法
             |     |
             +--+--+
                |
          [5] 从 port_list 选择最终 portid
```

### Interleave 粒度

**[DOC]** CLE 支持以下 interleave 粒度（由软件配置）：

```
支持粒度: 1, 2, 3, 4, 6, 8, 12, 16
```

**[推导]** interleave 用于将流量均匀分布到多个 C2C port，实现带宽聚合。例如 8 芯片 cube 拓扑中，itlv=8 可将流量平均分配到 8 个 port。

### Port ID 过 NoC 路由

**[DOC]** 地址空间映射（44 bit 地址）：

| bit 范围 | 功能 |
|----------|------|
| [43:40] | portid: 指定访问哪个 C2C port |
| [39:0] | addr offset: 片内物理地址 |

**[DOC]** NoC 支持 3 组 D2D group，每个 non-hash 区间可单独配置属于哪个 D2D group。对于 hash 区间，hash 出的 n bit 值可配置属于哪个 D2D group。

## 2.3 CDMA (Cross-chip DMA) 方案

### CDMA 设计原则

**[DOC]** 关键设计决策：
- CDMA 与 C2C_sys 解耦：任意 CDMA 可访问任意 MAC
- CDMA 支持多线程，线程数 >= TPU_scalar 数
- 同时支持 GDMA/TPU_scalar/AP 访问任意 MAC

### CDMA 规格

**[DOC]** 每个芯片的 CDMA 配置：

| 参数 | 值 | 说明 |
|------|------|------|
| CDMA 数/Die | 4 | 与 C2C MAC 数量对应 |
| Thread 数/CDMA | 8 | 支持异步执行 |
| 总 Thread 数/Die | 32 | 4 CDMA x 8 Thread |
| 带宽上限/CDMA | 64 GB/s | MAC 带宽的 2 倍（缓解同步损失） |
| 执行队列/CDMA | 1 | 所有 Thread 共享一个 datapath |

**[推导]** 对于双 Die 芯片：
- 总 CDMA 数/Chip = 8
- 总 Thread 数/Chip = 64
- CDMA 总带宽/Chip = 512 GB/s（> C2C 总带宽 448 GB/s，有余量）

### 多线程架构

```
CDMA (1 per port)
  +-- cmdq[0] (Thread 0) ---+
  +-- cmdq[1] (Thread 1) ---+--> Thread Arbiter --> CDMA Datapath (唯一)
  +-- cmdq[2] (Thread 2) ---+        |                    |
  +-- ...                   ---+    选择非barrier         执行搬运
  +-- cmdq[7] (Thread 7) ---+     的线程

每个 cmdq:
  - 独立地址空间
  - 遵循 4KB boundary 限制
  - 搬运指令 --> 提交到 datapath
  - sys 指令（含 barrier）--> 线程独立执行
```

**[DOC]** Thread Arbiter 规则：
- 只能选出**非 barrier 指令**的线程给 CDMA datapath
- barrier 指令阻塞本线程，不阻塞其他线程

### CDMA 指令集

**[DOC]** 关键变化（相比前代）：

1. **Transfer 指令替代 Read/Write**: 不再根据指令类型区分片内片间，通过地址区分
2. **fence 指令**: 确保 fence 后的搬运指令在 fence 前的搬运全部完成后才执行
3. **指令切换不保序**: 搬运指令执行过程中指令切换不保证前置指令完成，需 fence 显式保序

## 2.4 Memory Consistency 方案

SG2262 支持两种保序方案：

### CHS (C2C post-write Hardware Sequence)

**[DOC]** 特征：
- C2C 只发送 **post write** 请求
- 硬件沿路保序
- 需保序的请求通过**保序窗口**保证执行顺序

保序链条：
```
CDMA --> MAC 保序 --> 交换系统保序 --> 芯片转发保序 --> 写Memory保序
```

### CFS (C2C Fence Sequence)

**[DOC]** 特征：
- C2C 只发送 **non-post write** 请求
- 通过 **fence 指令**建立保序屏障
- CDMA 搬运指令切换**不保证前置完成**

关键规则：
- fence 指令确保：fence 后的搬运在 fence 前的搬运**全部完成**后才执行
- Send 指令自带 fence 功能：必须收集所有 bresp 后才能 retire

## 2.5 保序窗口

**[DOC]** 保序窗口用于保护中断/MSG 同步/atomic 请求与数据传输之间的顺序。

位置：MAC MST AXI 出口处

### 三种模式

| 模式 | 寄存器值 | 窗口数 | 匹配条件 |
|------|---------|--------|---------|
| 模式 0 | `reg_x4_mod_wr_order_ib_atu = 0` | 8 (order 0~7) | 64 bit 地址全匹配 |
| 模式 1 | `reg_x4_mod_wr_order_ib_atu = 1` | 12 (order 20~31) | chipid + 40 bit 地址匹配 |
| 模式 2 | `reg_x4_mod_wr_order_ib_atu = 2` | 32 (order 0~31) | chipid + fun_num + msi + 40 bit 地址匹配 |

**[DOC]** 保序行为：落在配置地址区域的写请求，需等待前置所有写请求全部完成才释放给下游。

## 2.6 Memory Protect

**[DOC]** 支持 3 种场景的内存保护，每种场景 **16 组**可配置地址空间：

| 场景 | 保护对象 |
|------|---------|
| CDMA -> 本芯片 | CDMA 对本芯片内请求地址做 MP |
| CDMA -> PC | CDMA 对 PC 请求地址做 MP |
| PCIe -> 本芯片 | PCIe 对本芯片内请求地址做 MP |

每组寄存器：
- 40 bit 起始地址 `reg_mpu_start_addr`
- 40 bit 结束地址 `reg_mpu_end_addr`
- 访问属性：0x0(不保护) / 0x1(不可读) / 0x2(不可写) / 0x3(不可读写)

**注意**: C2C 不对转发的请求做 Memory Protect。

## 2.7 RAS 设计

**[DOC]** 核心要求：
- 每个芯片作为算力池中任务分配的最小单元
- 错误不扩散到其他节点
- 支持软件介入恢复和节点下线两种方式

**隔离模式**：
1. 停止发送请求报文 -> 等待接收所有响应 -> 隔离完成
2. 停止接收请求报文 -> 回复 Error 响应给源芯片 -> 源芯片中断获知
3. 隔离完成后可介入恢复或重置

**[推导]** RAS 设计意味着 SG2262 支持运行时热摘除故障节点，这对大规模集群的可用性至关重要。
