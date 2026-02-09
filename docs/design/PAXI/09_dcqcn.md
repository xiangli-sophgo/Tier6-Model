# 09. DCQCN拥塞控制算法

## 9.1 概述

DCQCN (Data Center Quantized Congestion Notification) 是RDMA IP中实现的拥塞控制算法, 用于在以太网环境下进行端到端的速率调节。

**[DOC]** RDMA Reference Guide 6.12:

> "The rate is calculated based on the number of bytes allowed to pass through within the 1024 cycle time window."

速率以1024周期时间窗口内允许通过的字节数来表示。

## 9.2 CNP处理

### 发送CNP

**[DOC]** 6.11.1 Send CNP:

> "When rx get a ECN flagged packet in any RC qp, a CNP packet will generated and send to qp partner to notice congestion happened."

当RX侧收到带有ECN标记的RC QP包时, 自动生成CNP包发送给对端, 通知发生了拥塞。

### 接收CNP

**[DOC]** 6.11.2 Receive CNP:

当收到CNP时:
1. Max outstanding number从16降低到8, 降低发送速度
2. 启动定时器开始计时
3. 如果CNP中断使能, 则触发中断
4. 当定时器计数到达CNP_TIMER_THRD (0x2000a0), max outstanding恢复到16

## 9.3 速率降低 (Rate Decrease)

**[DOC]** 6.12.1 Rate Decrease:

### 触发条件

> "If a CNP is received once or multiple times within dcqcn_rx_cnp_merge_period(0x200220 bit[15:0]) us, the rate is reduced only once."

- 在 `dcqcn_rx_cnp_merge_period` 微秒时间窗口内, 无论收到多少个CNP, 速率只降低一次
- 这是CNP合并机制, 避免因突发CNP导致速率过度降低

### 降低公式

**[DOC]**:

```
current_rate = current_rate * (1 - alpha / 2^(dcqcn_alpha_rate_shift + 10))
```

参数说明:
- `current_rate`: 当前发送速率
- `alpha`: 拥塞因子, 存储在 ALPHA_MEM (0x600010 bit[9:0]), 定点数, 小数部分10位
- `dcqcn_alpha_rate_shift`: 速率调整移位因子

**[推导]** alpha值越大, 每次降速幅度越大。通过 `dcqcn_alpha_rate_shift` 可以进一步控制降速的灵敏度。

## 9.4 速率增加 (Rate Increase)

**[DOC]** 6.12.2 Rate Increase:

### 触发条件

速率增加有两个独立的触发条件 (满足任一即可):

1. **定时器触发**: 经过 `dcqcn_nocnp_timer_th` (0x200200 bit[15:0]) 微秒后
2. **字节计数触发**: 发送了 `dcqcn_byte_cnt_th` (0x200200 bit[50:32]) 字节后

### 三阶段增加

速率增加分为三个阶段, 逐步加速:

#### 阶段1: Fast Recovery (快速恢复)

**[DOC]**:

```
current_rate = (current_rate + target_rate) / 2
```

- 速率快速向target_rate靠拢
- 每次增加量为当前速率与目标速率差值的一半

#### 阶段2: Additive Increase (加性增加)

**[DOC]**:

```
current_rate = (current_rate + target_rate + dcqcn_rate_ai) / 2
```

- 在Fast Recovery基础上额外加上 `dcqcn_rate_ai` (Additive Increase步长)
- 使速率能够超过target_rate

#### 阶段3: Hyper Increase (超级增加)

**[DOC]**:

```
current_rate = (current_rate + target_rate + dcqcn_rate_hai) / 2
```

- 使用更大的步长 `dcqcn_rate_hai` (Hyper Additive Increase)
- 加速恢复到最大速率

**[推导]** 三阶段设计的逻辑:
1. Fast Recovery: 谨慎恢复, 快速接近之前的稳定速率
2. Additive Increase: 温和探测更高速率
3. Hyper Increase: 长时间无拥塞, 积极恢复带宽利用

## 9.5 Alpha更新 (Alpha Update)

**[DOC]** 6.12.3 Alpha Update:

### Alpha存储

- 存储位置: ALPHA_MEM (0x600010 bit[9:0])
- 格式: 定点数, 小数部分10位 (即精度为 1/1024)

### Alpha降低 (无拥塞时)

**[DOC]**:

> "If no CNP is received within dcqcn_alpha_timer_th(0x200218 bit[15:0]) microseconds, alpha is decreased using formula:"

```
alpha = ((1024 - dcqcn_upd_alpha_g) * alpha) / 1024
```

- 如果在 `dcqcn_alpha_timer_th` 微秒内未收到CNP, alpha降低
- `dcqcn_upd_alpha_g` 控制降低速度
- **[推导]** 当 `dcqcn_upd_alpha_g` = 0 时, alpha不变; 值越大, alpha衰减越快

### Alpha增加 (发生拥塞时)

**[DOC]**:

> "When a rate reduction event occurs, alpha is increased using formula:"

```
alpha = ((1024 - dcqcn_upd_alpha_g) * alpha + dcqcn_upd_alpha_g) / 1024
```

- 速率降低事件触发alpha增加
- alpha越高意味着下次降速幅度越大
- **[推导]** 这实现了EWMA (指数加权移动平均) 效果: alpha反映近期拥塞程度

## 9.6 Per-QP状态内存

**[DOC]** 6.12.4 Software Initial:

每个QP需要初始化以下状态内存 (地址 = 基址 + (i-1) x 0x0100):

| 内存 | 基址偏移 | 说明 |
|------|---------|------|
| RATE_MEM | 0x60_0008 | 当前速率 |
| CC_RATE_MEM | 0x60_0028 | CC控制速率 |
| ALPHA_MEM | 0x60_0010 | Alpha拥塞因子 |
| CNP_MERGE_MEM | 0x60_0000 | CNP合并状态 (初始化为0) |
| BYTE_COUNTER_MEM | 0x60_0018 | 字节计数器 (初始化为0) |
| T_BC_MEM | 0x60_0020 | 定时器/字节计数 (初始化为0) |
| CC_BYTE_CNT_MEM | 0x60_0030 | CC字节计数 (初始化为0) |

### 初始化流程

**[DOC]**:

1. 配置全局参数
2. 初始化当前QP的内存:
   a. 配置初始速率到 RATE_MEM
   b. 配置初始速率到 CC_RATE_MEM
   c. 配置初始alpha到 ALPHA_MEM
   d~g. 清零 CNP_MERGE_MEM, BYTE_COUNTER_MEM, T_BC_MEM, CC_BYTE_CNT_MEM
3. 如果初始速率 != `dcqcn_max_rate` (0x200200 bit[31:16]):
   - 软件读 `dcqcn_nocnp_timer_busy` (0x200200 bit51), 若为0
   - 写1到 `dcqcn_nocnp_timer_start` (0x40_0000 + (i-1) x 0x0100 bit5) 启动增速定时器
   - 否则 (初始速率 = max_rate): QP仅在速率降低后才启动定时器

## 9.7 三种拥塞控制方案

**[DOC]** 6.12.6 Congestion control schemes:

IP提供三种典型的拥塞控制方案:

### 方案1: 全硬件DCQCN

| 寄存器 | 值 |
|--------|---|
| dcqcn_hw_en (bit24 of 0x200220) | 1 |
| hw_rate_throttle_en (bit25 of 0x200220) | 1 |
| tx_wqe_ost_limit_en (bit26 of 0x200220) | 1 |
| speed_limiter_en (bit54 of 0x200220) | 1 |

**[DOC]**: IP根据DCQCN算法自动调整发送速率, 考虑收到的CNP包、TX发送流量和定时器。IP根据计算出的速率控制发送流量。固件需要配置DCQCN参数。

### 方案2: 固件配速率, 硬件限速

| 寄存器 | 值 |
|--------|---|
| dcqcn_hw_en (bit24 of 0x200220) | 0 |
| hw_rate_throttle_en (bit25 of 0x200220) | 1 |
| tx_wqe_ost_limit_en (bit26 of 0x200220) | 1 |
| speed_limiter_en (bit54 of 0x200220) | 1 |

**[DOC]**: 用户通过其他CC算法获取TX发送速率, 写入 CC_RATE_MEM (0x60_0028 + (i-1) x 0x0100) 设置当前QP的速率。IP根据该速率控制发送流量。

### 方案3: 纯固件控制

| 寄存器 | 值 |
|--------|---|
| dcqcn_hw_en (bit24 of 0x200220) | 0 |
| hw_rate_throttle_en (bit25 of 0x200220) | 0 |
| tx_wqe_ost_limit_en (bit26 of 0x200220) | 0 |
| speed_limiter_en (bit54 of 0x200220) | 0 |

**[DOC]**: IP不控制TX发送速率。固件通过设置 `sq_arb_mask` (0x40_0000 + (i-1) x 0x0100 bit1) 来阻塞TX WQE的处理。

## 9.8 DFX调试功能

**[DOC]** 6.12.5 DFX:

通过绘制X-时间、Y-速率的折线图可以推断所有DCQCN信息。

### 调试流程

1. 软件为指定QP启动/停止一个TIME-WINDOW, 如64个条目 (X范围 [0, 63])
2. 软件预先读取初始值: Current Rate, Target Rate, alpha, BC和T
3. DFX在TIME-WINDOW内记录连续64个事件
4. 软件读取64次 `dcqcn_dfx_info` 寄存器获取TIME-WINDOW数据

### 事件类型

| 值 | 事件类型 |
|----|---------|
| 3'b000 | Decrease (速率降低) |
| 3'b001 | Bytecounter.FastRecovery |
| 3'b010 | Bytecounter.FastRecovery |
| 3'b011 | Bytecounter.FastRecovery |
| 3'b101 | Noncnp_timer.FastRecovery |
| 3'b110 | Noncnp_timer.HyperIncrease |
| 3'b111 | Noncnp_timer.AdditiveIncrease |
| 3'b100 | Reserved |

每个TIME-WINDOW条目包含: 事件类型、当前速率和时间戳。

软件还可以读取该QP在TIME-WINDOW内的3类事件计数: decrease、bytecounter-increase、nocnp_timer-increase。

## 9.9 DCQCN全局参数汇总

| 参数 | 寄存器地址 | 说明 |
|------|-----------|------|
| dcqcn_rx_cnp_merge_period | 0x200220 bit[15:0] | CNP合并周期 (us) |
| dcqcn_nocnp_timer_th | 0x200200 bit[15:0] | 无CNP增速定时器阈值 (us) |
| dcqcn_byte_cnt_th | 0x200200 bit[50:32] | 字节计数增速阈值 |
| dcqcn_max_rate | 0x200200 bit[31:16] | 最大速率 |
| dcqcn_alpha_timer_th | 0x200218 bit[15:0] | Alpha更新定时器阈值 (us) |
| dcqcn_upd_alpha_g | - | Alpha EWMA权重因子 |
| dcqcn_alpha_rate_shift | - | 速率降低移位因子 |
| dcqcn_rate_ai | - | Additive Increase步长 |
| dcqcn_rate_hai | - | Hyper Additive Increase步长 |
| dcqcn_hw_en | 0x200220 bit24 | 硬件DCQCN使能 |
| hw_rate_throttle_en | 0x200220 bit25 | 硬件限速使能 |
| tx_wqe_ost_limit_en | 0x200220 bit26 | TX WQE OST限制使能 |
| speed_limiter_en | 0x200220 bit54 | 速率限制器使能 |
| CNP_TIMER_THRD | 0x2000a0 | CNP定时器阈值 |
