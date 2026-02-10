# 05 - 内存子系统

## 1. 对标: TPUPerf 内存模型

TPUPerf 建模了完整的内存层级:

```
LMEM (片上本地存储)
  |
  +-- Bank 结构 (16 banks per lane)
  |
  v
GS Cache (可选, 4-way set-associative)
  |
  v
DDR (片外 DRAM)
  |
  +-- Bank Group / Bank / Row 地址映射
  +-- Bank Conflict 检测
  +-- Outstanding 控制
```

## 2. LMEM 模型

### 2.1 对标: `c_model/src/memory/lmem.cpp` (~400行)

**架构**:
- TLM-2.0 target, 接受读写请求
- 按 lane 组织: NPU_NUM 个 lane, 每个 lane PER_LANE_LMEM_SIZE
- 支持 bank conflict 检测

**关键参数** (SG2260):
```
lmem_size = 256KB (每个核)
lane_num = 16 (NPU_NUM)
per_lane_size = lmem_size / lane_num = 16KB
read_latency = 29ns
write_latency = 29ns
bus_width = 128B (LOCAL_MEM_WIDTH)
bank_num = 16 (per lane)
```

### 2.2 Python LMEM 实现

```python
class LMEMModel:
    """Local Memory 模型

    核心行为:
    1. 接收读/写请求 (addr, size)
    2. 计算 lane 映射
    3. 检测 bank conflict
    4. 返回延迟

    不存储实际数据 (只关心时序)
    """

    # 请求处理流程:
    #
    # receive_request(addr, size, type):
    #   1. lane_idx = addr / per_lane_size
    #   2. bank_idx = (addr % per_lane_size) / bank_size
    #   3. 检查 bank conflict (与正在处理的请求比较)
    #   4. 计算延迟:
    #      base_delay = read_latency if READ else write_latency
    #      burst_delay = ceil(size / bus_width) cycles
    #      conflict_delay = bank_conflict_penalty if conflict
    #      total = base_delay + burst_delay + conflict_delay
    #   5. 调度响应事件 (在 total 周期后)
```

### 2.3 Bank Conflict 检测

LMEM 的 bank conflict 发生在 TIU 同时访问多个操作数时:

```
冲突条件: 两个访问的 bank_idx 相同但地址不同

检测方法:
  active_accesses: list  # 当前正在处理的访问

  for access in active_accesses:
    if new_bank == access.bank and new_addr != access.addr:
      conflict = True
      break
```

## 3. DDR 模型

### 3.1 对标: `c_model/src/memory/ddr_wrapper.cpp` (~550行)

**架构**:
- TLM-2.0 target
- 读/写独立的 outstanding FIFO
- 完整的 DRAM 地址映射 (col/bank_group/bank/row)
- Bank conflict 检测

**关键参数** (SG2260):
```
read_latency = 150ns (~300 cycles @ 2GHz)
write_latency = 150ns
bus_width = AXI_BUS_WIDTH (64B)
read_outstanding = 128
write_outstanding = 128
nali_extra_lat = 10 (非对齐额外延迟, cycles)
rw_parallel = false (读写默认串行)
```

### 3.2 DDR 地址映射

TPUPerf 使用精确的 DRAM 地址位映射:

```
地址位分配 (参考 ddr_wrapper.cpp):
  col_bits      = {1,2,3,4,5,6,7, 9,10,11}    # Column
  bank_group_bits = {8, 12}                     # Bank Group
  bank_bits     = {29, 30}                      # Bank
  row_bits      = {13,14,...,28}                 # Row (16 bits)

从地址提取字段:
  col        = extract_bits(addr, col_bits)
  bank_group = extract_bits(addr, bank_group_bits)
  bank       = extract_bits(addr, bank_bits)
  row        = extract_bits(addr, row_bits)
```

### 3.3 DDR Bank Conflict 检测

```
冲突条件:
  新请求与活跃请求命中同一 bank (bank_group + bank 相同)
  但不同 row -> row 切换惩罚

检测流程:
  for active in active_read_cmds + active_write_cmds:
    if (new.bank_group == active.bank_group
        and new.bank == active.bank
        and new.row != active.row):
      bank_conflict_count += 1
```

### 3.4 Python DDR 实现

```python
class DDRModel:
    """DDR DRAM 模型

    核心行为:
    1. Outstanding 控制 (读写独立的信用池)
    2. 地址映射 -> bank_group/bank/row
    3. Bank conflict 检测
    4. 参数化延迟 + 对齐惩罚

    响应延迟计算:
      delay = base_latency
            + burst_cycles
            + alignment_penalty (如果地址不对齐)
            + bank_conflict_penalty (如果命中已打开的 row)
    """

    # DDR 时钟周期计算:
    # bw_per_instance = ddr_freq * ddr_phy_width / 8 / 1000  (GB/s)
    # ddr_clk_period = bus_width / bw_per_instance  (ns)
```

## 4. Cache 模型

### 4.1 对标: `c_model/src/memory/gs_cache.cpp` (~550行)

**架构**:
- 4-way set-associative
- LRU 替换策略
- 5 个 SC_THREAD: 请求处理, 响应处理, 行处理, 读合并, 写合并

**关键参数** (SG2260):
```
cache_line_size = 128B  (SG2260)
cache_ways = 4
cache_entries = 128     (每 way)
hit_latency = 3 cycles
max_burst = 1           (每次合并 1 个 cache line)
max_pending_cmd = 8
out_outstanding = 32    (对外 outstanding 限制)
```

### 4.2 Python Cache 实现

```python
class CacheModel:
    """4-Way Set-Associative Cache 模型

    核心行为:
    1. 请求到达 -> 判断 hit/miss
    2. Hit: 返回 hit_latency
    3. Miss:
       a. 选择替换行 (LRU)
       b. 生成 line-fill 请求 -> 发往 DDR
       c. 等待响应
       d. 更新 cache 内容
       e. 返回总延迟
    4. 请求合并: 对同一 cache line 的多次 miss 只发一次外部请求
    """

    # 数据结构:
    # cache_sets: list[list[CacheLine]]  # [set_idx][way_idx]
    #
    # CacheLine:
    #   tag: int
    #   valid: bool
    #   lru_order: int
    #
    # 去重表:
    # pending_fills: dict[line_addr -> list[waiter]]  # miss 合并
```

### 4.3 Hit/Miss 流程

```
请求到达 (addr, size):
  set_idx = (addr >> line_bits) & (entries - 1)
  tag = addr >> (line_bits + entry_bits)

  # 检查所有 way
  for way in cache_sets[set_idx]:
    if way.valid and way.tag == tag:
      -> HIT: 延迟 = hit_latency, 更新 LRU
      return

  -> MISS:
    # 检查是否有相同 line 的 pending fill
    line_addr = addr & ~(line_size - 1)
    if line_addr in pending_fills:
      # 合并: 只等待已有的 fill 完成
      yield wait_pending(line_addr)
      return

    # 新的 miss: 选择替换行, 发起 fill
    victim = select_lru_victim(set_idx)
    pending_fills[line_addr] = []
    yield send_fill_request(line_addr, line_size)
    # fill 完成后更新 cache 并唤醒所有等待者
```

## 5. 系统地址映射 (SAM)

### 5.1 对标: `c_model/src/memory/system_address_map.cpp` (~570行)

系统地址映射负责将逻辑地址解码为物理目标:

```
SG2262 地址空间:
  LMEM:  0x6800_0000 ~ 0x6804_0000  (per core)
  DDR:   0x0000_0000 ~ 0x1_0000_0000 (4GB)
  MMIO:  0x7000_0000 ~ ...

地址解码:
  if addr in LMEM_range:
    target = LMEM[core_id]
  elif addr in DDR_range:
    target = DDR[addr_to_ddr_port(addr)]
  else:
    error: unmapped address
```

### 5.2 Python SAM 实现

```python
class SystemAddressMap:
    """系统地址映射

    为每种芯片提供地址解码:
    - sg2260: 8核, 每核独立 LMEM + 共享 DDR
    - sg2262: 64核, 每核独立 LMEM + 每核独立 DDR

    地址 -> (target_type, target_id, offset)
    """
```

## 6. 内存层级集成

### 完整数据访问路径

```
DMA 发起请求 (addr, size, type)
    |
    v
地址解码 (SAM)
    |
    +-- LMEM 地址 -> LMEM Model -> 延迟 = 29ns + burst
    |
    +-- GMEM 地址:
        |
        +-- Cache enabled?
        |   |
        |   +-- Yes -> Cache Model
        |   |   |
        |   |   +-- Hit -> 延迟 = 3 cycles
        |   |   |
        |   |   +-- Miss -> DDR Model -> 延迟 = 150ns + burst + conflict
        |   |
        |   +-- No -> DDR Model 直接访问
        |
        v
      总线传输延迟 (距离相关)
```
