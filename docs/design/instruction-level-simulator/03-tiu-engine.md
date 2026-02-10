# 03 - TIU 计算引擎

## 1. 对标: TPUPerf TIU 实现

### TPUPerf 中的 TIU

- **文件**: `c_model/src/tpu/tiu.cc` (~311行) + `tiu/tiuImpl.cc` (~5,652行)
- **模型**: SC_METHOD, 每时钟周期执行 3 状态机 (Init -> Compute -> Finish)
- **核心**: 从命令队列取指令, 计算 cycle 数, 倒计时, 完成后更新 tiuSyncId

### 状态机

```
Init:
  - 命令队列为空 -> idle
  - 有命令 -> 检查依赖: cmd.cmd_id_dep <= tdmaSyncId?
    - 依赖不满足 -> 等待 (下一周期再检查)
    - 依赖满足 -> 取出命令
      - SYS_SEND_MSG -> 发消息, 回到 Init
      - SYS_WAIT_MSG -> 检查消息, 不满足则等待
      - 计算指令 -> cycleCount = calCycle(cmd), 进入 Compute

Compute:
  - cycleCount -= 1
  - cycleCount == 0 -> 进入 Finish

Finish:
  - tiuSyncId = cmd.sync_id  (通知 DMA 本指令完成)
  - profiler.record(cmd)
  - 回到 Init
```

## 2. Python TIU Engine 设计

### 2.1 状态机实现

```python
class TIUEngine:
    """TIU 计算引擎 - 对应 TPUPerf 的 Tiu 类"""

    # 状态枚举
    INIT = 0
    COMPUTE = 1
    FINISH = 2

    # 核心属性:
    # - cmd_queue: list[TIUCommand]    # 指令队列
    # - state: int                      # 当前状态
    # - cycle_count: int               # 剩余计算周期
    # - current_cmd: TIUCommand        # 当前执行的指令
    #
    # 同步信号:
    # - tiu_sync_id: Signal            # TIU 完成的最新 cmd_id (写)
    # - tdma_sync_id: Signal           # DMA 完成的最新 cmd_id (读)
    # - idle: Signal                   # 空闲状态
    #
    # 配置:
    # - chip_config: dict              # 芯片参数 (lane_num, cube_m/k/n, ...)
```

### 2.2 同步机制

TIU 和 DMA 之间的同步是 TPUPerf 最核心的机制:

```
每条 TIU 指令有: cmd_id (自身ID), cmd_id_dep (依赖的 DMA 指令 ID)
每条 DMA 指令有: cmd_id (自身ID), cmd_id_dep (依赖的 TIU 指令 ID)

TIU 发射条件: cmd.cmd_id_dep <= tdma_sync_id.read()
DMA 发射条件: cmd.cmd_id_dep <= tiu_sync_id.read()
```

这允许 TIU 和 DMA **并行执行**不相互依赖的指令, 只在有数据依赖时同步。

### 2.3 跨核消息同步

```python
class MessageQueue:
    """跨核/跨引擎消息同步 - 对应 TPUPerf 的 Utility::sendMsg/waitMsg"""

    # 全局消息表: msg_id -> (send_count, remain_wait_count)
    #
    # send_msg(msg_id, wait_cnt):
    #   msg_table[msg_id].send_count += 1
    #   msg_table[msg_id].remain_wait = wait_cnt
    #
    # wait_msg(msg_id, expected_send_cnt) -> bool:
    #   if msg_table[msg_id].send_count == expected_send_cnt:
    #     msg_table[msg_id].remain_wait -= 1
    #     if remain_wait == 0: 清除
    #     return True
    #   return False
```

## 3. TIU 指令延迟计算

这是整个引擎中**代码量最大**的模块, 需要从 `tiuImpl.cc` 逐一翻译。

### 3.1 核心硬件参数

从芯片配置中读取:

```
lane_num           # 并行 lane 数 (SG2262: 16)
cube_m             # Cube 单元 M 维度
cube_k             # Cube 单元 K 维度 (per cycle 输入通道数)
cube_n             # Cube 单元 N 维度
sram_size_kb       # SRAM 大小
eu_num_by_dtype    # 不同精度下的 EU 激活数
                   # INT8: cube_k_8bit, BF16: cube_k_16bit, FP32: cube_k_32bit
```

### 3.2 Conv (卷积) 延迟公式

对标 `tiuImpl.cc` 中的 `sCONV_CMD::getCustomProfile()`:

```
sync_cycle = 23   # R0/R1 流水线同步延迟

# 每次循环处理的通道数 (取决于精度)
ch_per_cyc = cube_k_8bit   if INT8
           = cube_k_16bit  if BF16
           = cube_k_32bit  if FP32

# 内层循环: kernel 遍历 + 后处理
loop_cycle = kh * kw * ceil(C_in / ch_per_cyc) + shift_round_cycle + psum_lat

# 总周期
total_cycle = sync_cycle
            + N * ceil(C_out / lane_num)
              * ceil(H_out * W_out / eu_num)
              * loop_cycle
```

### 3.3 MatMul (矩阵乘) 延迟公式

对标 `sMM_CMD::getCustomProfile()` 和 `sMM2_CMD::getCustomProfile()`:

**MM v1:**
```
sync_cycle = 19
post_process_cycle = 13

loop_cycle = K  (简化情况, 非转置)

total_cycle = sync_cycle
            + ceil(N_out / lane_num)
              * ceil(W_out / eu_num)
              * loop_cycle * M
            + post_process_cycle
```

**MM2 (nn 模式):**
```
init_cycle = 44

total_cycle = ceil(C / lane_num)
            * ceil(W / eu_num)
            * (ceil(K / ch_per_cyc) + bank_conflict_overhead + bias_lat)
            + init_cycle
```

**MM2 (nt 模式):**
```
init_cycle = 44 或更多 (有 discount factor 处理小矩阵)

# nt 模式的 K 维度处理不同, 需要考虑转置开销
```

**MM2 (tt 模式):**
```
init_cycle = 47

total_cycle = ceil(C / lane_num)
            * ceil(W / eu_num)
            * (ceil(K / ch_per_cyc) + bank_conflict_overhead)
            + init_cycle
```

### 3.4 Bank Conflict 计算

```
# 三个操作数的 bank 起始地址
res_bank  = (res_addr  - LMEM_START) >> bank_addr_width
opd0_bank = (opd0_addr - LMEM_START) >> bank_addr_width
opd1_bank = (opd1_addr - LMEM_START) >> bank_addr_width

# 冲突计数
conflict = (res_bank == opd0_bank) + (res_bank == opd1_bank)

# 冲突开销
bank_conflict_overhead = conflict  # 每次冲突增加 1 cycle
```

### 3.5 完整指令类型清单

需要从 `tiuImpl.cc` 翻译的指令类型:

| 指令类型 | tsk_typ | eu_typ | 说明 | 复杂度 |
|---------|---------|--------|------|--------|
| CONV | 0 | - | 卷积 | 高 |
| MM | 2 | 1 | 矩阵乘 v1 | 中 |
| MM2.nn | 2 | 4 | 矩阵乘 v2 (nn) | 高 |
| MM2.nt | 2 | 5 | 矩阵乘 v2 (nt) | 高 |
| MM2.tt | 2 | 6 | 矩阵乘 v2 (tt) | 高 |
| PorD | 1 | - | 池化/反池化 | 中 |
| AR | 3 | - | 算术运算 | 低 |
| CMP | 4 | - | 比较运算 | 低 |
| SG | 5 | - | Scatter/Gather | 中 |
| SFU | 6 | - | 特殊函数 (exp/log/rsqrt) | 中 |
| LIN | 7 | - | 线性运算 | 低 |
| VC | 8 | - | 向量乘 | 低 |
| RQ | 9 | - | 量化 | 低 |
| SYS | 15 | - | 系统命令 (nop/send/wait) | 低 |

**实现优先级**: CONV > MM2 > MM > SFU > AR > LIN > 其他

### 3.6 流水线初始延迟汇总

| 指令类型 | sync 周期 | post 周期 | 说明 |
|---------|----------|----------|------|
| CONV | 23 | 0 | R0/R1 路径同步 |
| MM v1 | 19 | 13 | 前后各有固定延迟 |
| MM2.nn | 44 | 0 | SG2260/SG2262 |
| MM2.nt | 44+ | 0 | 有小矩阵 discount |
| MM2.tt | 47 | 0 | 转置额外开销 |
| SFU | 0 | 13-14 | 按子类型不同 |
| LIN | 0 | 12 | |
| AR | 0 | 10 | |

## 4. 实现策略

### 阶段 1: 核心框架
- TIU 状态机 (Init/Compute/Finish)
- cmd_id 同步机制
- 消息同步 (send_msg/wait_msg)

### 阶段 2: 延迟计算 (最大工作量)
- 从 tiuImpl.cc 逐个翻译指令延迟公式
- 优先: CONV, MM2.nn, MM2.nt, MM2.tt
- 其次: PorD, SFU, AR, LIN
- 最后: SG, CMP, VC, RQ

### 阶段 3: 多引擎支持 (对标 Tiu_MT)
- 多个 TIU 引擎并行执行
- 引擎间依赖网络
- TLM 式异步分发

### 验证方式
- 对同一组 .BD 文件, 对比 Python 引擎和 TPUPerf 的各指令 cycle 数
- 允许 1-2 cycle 的舍入误差
