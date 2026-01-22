# DS_TPU 部署分析流程详解

本文档详细介绍 DS_TPU_1209 工具的部署分析流程，包括算子计算、Tile 分块策略、并行策略、延迟计算和性能指标等内容。

---

## 目录

1. [整体流程架构](#1-整体流程架构)
2. [模型与算子架构](#2-模型与算子架构)
3. [MatMul 算子详解](#3-matmul-算子详解)
4. [Attention 算子详解](#4-attention-算子详解)
5. [通信算子详解](#5-通信算子详解)
6. [并行策略详解](#6-并行策略详解)
7. [延迟计算模型](#7-延迟计算模型)
8. [性能指标计算公式](#8-性能指标计算公式)
9. [配置参数说明](#9-配置参数说明)
10. [计算示例](#10-计算示例)

---

## 1. 整体流程架构

### 1.1 数据流向

```
输入配置
    ↓
┌─────────────────────────────────┐
│  配置加载层 (config/)            │
│  - 模型配置 (YAML)              │
│  - 部署配置 (TP/DP/EP等)        │
│  - TPU配置                      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  模型构建层 (model/)             │
│  - 构建计算图                    │
│  - 创建算子实例                  │
│  - 分配并行参数                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  硬件抽象层 (tpu/)               │
│  - TPU性能参数                   │
│  - 计算时间估算方法              │
│  - 通信时间估算方法              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  性能分析层 (performance/)       │
│  - 计算延迟评估                  │
│  - 通信延迟评估                  │
│  - 结果缓存与复用                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  结果输出                        │
│  - JSON格式结果                  │
│  - 性能指标汇总                  │
│  - 层级性能分析                  │
└─────────────────────────────────┘
```

### 1.2 核心入口

```
main.py → top/simulator.py → performance/analyzer.py
```

**主要流程步骤：**

1. **配置加载** (`main.py`)
   - 支持 YAML 配置文件或命令行参数
   - 支持多个实验配置同时运行
   - 默认配置合并

2. **模型创建** (`DeepSeekLLM` in `model/networks/deepseek/`)
   - 根据模型配置创建层结构
   - 构建算子实例（MatMul、MHA/MQA、AllReduce等）
   - 生成操作类型映射

3. **TPU实例化** (`tpu/tpu_base.py`)
   - 初始化硬件参数（FLOPS、带宽、SRAM等）
   - 预计算派生参数

4. **性能分析** (`performance/analyzer.py`)
   - 遍历所有算子并按类型分组
   - 使用对应评估器评估每个算子
   - 缓存复用相同配置的算子结果
   - 汇总层级和模型级性能指标

5. **结果输出**
   - 生成JSON格式的性能报告
   - 包含单个算子、层级和全模型的性能数据

---

## 2. 模型与算子架构

### 2.1 支持的模型

| 模型 | 类型 | 层数 | 特点 |
|------|------|------|------|
| **DeepSeek V3** | MLA + MoE | 61层 | 原始版本 |
| **DeepSeek V3.2** | MLA（DSA）+ MoE | 61层 | 带深度稀疏注意力、分解KV投影 |
| **LLaMA-7B** | 标准Transformer | 32层 | 基础参考实现 |

### 2.2 模型层级结构（以 DeepSeek V3.2 为例）

```
DeepSeekLLM (总模型)
├── Embedding (嵌入层)
├── Dense MLP × 3层 (稠密MLP)
├── MLA/MLAv32 × 61层 (多头潜在注意力)
│   ├── q_a_proj (MatMul)
│   ├── q_b_proj (MatMul)
│   ├── kv_a_proj (MatMul)
│   ├── kv_b_proj (MatMul)
│   ├── mha (MHAOperator - Flash Attention 2)
│   ├── o_proj (MatMul)
│   └── allreduce (AllReduceOperator) - 当TP>1时
├── DSA × 61层 (深度稀疏注意力)
│   ├── wq_b (MatMul)
│   ├── wk (MatMul)
│   ├── weights_proj (MatMul)
│   ├── fp8_index (MQAOperator)
│   └── allreduce (AllReduceOperator) - 当TP>1时
├── MoE × 58层 (混合专家)
│   ├── gate (MatMul)
│   ├── shared_experts (MatMul×2)
│   ├── routed_experts (MatMul×2)
│   ├── dispatch (DispatchOperator)
│   ├── combine (CombineOperator)
│   └── allreduce (AllReduceOperator) - 当TP>1时
└── LMHead (输出层)
```

### 2.3 计算算子类型

| 算子类型 | 功能 | 关键参数 |
|---------|------|----------|
| **MatMulOperator** | 矩阵乘法 | G, M, K, N |
| **MHAOperator** | 多头注意力 | B, H, QS, KS, QD, VD |
| **MQAOperator** | 多查询注意力/DSA | B, H, QS, KS, QD, VD |
| **SoftmaxOperator** | Softmax | M, N |
| **RmsNormOperator** | RMSNorm | M, N |

### 2.4 通信算子类型

| 算子类型 | 通信模式 | 功能 | 适用场景 |
|---------|---------|------|---------|
| **AllReduceOperator** | AllReduce | 梯度同步 | TP梯度聚合 |
| **AllGatherOperator** | AllGather | 数据聚集 | 分布式数据收集 |
| **ReduceScatterOperator** | ReduceScatter | 规约分散 | 数据分割与规约 |
| **DispatchOperator** | Dispatch | 数据分发 | MoE专家分发 |
| **CombineOperator** | Combine | 数据合并 | MoE结果合并 |

---

## 3. MatMul 算子详解

### 3.1 FLOPS 计算公式

**基本公式:**

```
FLOPS = 2 × G × M × N × K
```

其中：
- **G**: Batch维度（分布在多个设备上）
- **M**: 输出矩阵行数
- **N**: 输出矩阵列数
- **K**: 输入矩阵的隐藏维度（收缩维度）
- **系数2**: 一次乘加操作计为2个FLOP

**代码实现** (`model/operators/compute/matmul.py`):
```python
self.flops = 2 * self.g * self.m * self.k * self.n
```

### 3.2 Tile 分块策略

#### 什么是 Tile？

**Tile（瓦片/分块）** 是将大矩阵计算分解为小块计算的策略，目的是：
1. 充分利用片上 SRAM（避免频繁访问 DRAM）
2. 最大化计算与内存传输的重叠
3. 提高硬件利用率

#### 分块维度

MatMul 支持 3 维 Tile 划分：`(M_tile, N_tile, K_tile)`

```
矩阵 A [M × K] × 矩阵 B [K × N] = 矩阵 C [M × N]

分块后：
A[m_t × k_t] × B[k_t × n_t] = C[m_t × n_t]（局部计算）
```

#### Legal Tiles 生成策略

```python
def _generate_legal_tiles(m_blk, n_blk, k_blk):
    """生成所有合法的Tile配置"""
    # 约束条件
    # 1. M_tile: 从 align_up(m_blk, cube_m) 向下递减，步长为 cube_m
    # 2. N_tile: 从 align_up(n_blk, cube_n) 向下递减，步长为 cube_n
    # 3. K_tile: 由 SRAM 限制决定

    sram_available = sram_size * 0.45  # 使用45%的SRAM

    # SRAM占用计算
    occupied_sram = (
        align_row(N_tile) * align_col(N_tile, BF16_BYTES) +
        max(align_row(M_tile) + align_row(N_tile)) * FP8_BYTES * K_tile
    )

    # 最大K_tile
    max_K_tile = sram_available / ((align_row(M_tile) + align_row(N_tile)) * FP8_BYTES)
```

#### 对齐要求

```python
align_row(r) = align_up(r, lane_num)           # 行对齐 (默认16)
align_col(c, bytes) = align_up(c * bytes, align_bytes)  # 列对齐 (32字节)

# 示例
align_row(100) = align_up(100, 16) = 112
align_col(64, 2) = align_up(64 * 2, 32) = 128
```

### 3.3 最优 Tile 选择

**策略**: Pareto 最优前沿选择

```python
def _is_pareto_max_tiles(candidates, m_t, n_t, k_t):
    """检查是否为帕累托最优"""
    for m0, n0, k0 in candidates:
        # 如果存在一个候选者在所有维度都≥当前值，则当前不是最优
        if m0 >= m_t and n0 >= n_t and k0 >= k_t:
            return False
    return True
```

**选择标准**: 在所有合法的 Tile 中，选择**最小化 DRAM 流量**的配置

```python
min_traffic = math.inf
best_tile = None
for loop_order in ('mnk', 'nkm', 'mkn'):
    traffic = dram_traffic(loop_order, m_nom, n_nom, k_nom, m_t, n_t, k_t)
    if traffic < min_traffic:
        min_traffic = traffic
        best_tile = (m_t, n_t, k_t)
```

### 3.4 分区（Partition）计算

#### 什么是 Partition？

**Partition（分区）** 是将计算任务分配到多个 TPU 核心的策略。

**分区维度**: `(P_G, P_M, P_N, P_K)` 四个维度

**约束条件**:
```
P_G × P_M × P_N × P_K = tpu_cores
```

#### 有效分区生成

```python
def _valid_partition(tpu_cores):
    """生成所有有效的分区组合"""
    blocks = []
    for P_G in range(1, tpu_cores + 1):
        if tpu_cores % P_G != 0:
            continue  # 必须整除
        rem_m = tpu_cores // P_G
        for P_M in range(1, rem_m + 1):
            if rem_m % P_M != 0:
                continue
            rem_n = rem_m // P_M
            for P_N in range(1, rem_n + 1):
                if rem_n % P_N != 0:
                    continue
                P_K = rem_n // P_N
                blocks.append((P_G, P_M, P_N, P_K))
    return blocks

# 示例：32核心
# 可能的分区: (1,1,1,32), (1,1,2,16), (1,1,4,8), (1,2,2,8), ...
```

#### 每个分区的负载计算

```python
g_nom = ceil_div(G, P_G)    # 每个 P_G 分区处理的样本数
m_nom = ceil_div(M, P_M)    # 每个 P_M 分区处理的行数
n_nom = ceil_div(N, P_N)    # 每个 P_N 分区处理的列数
k_nom = ceil_div(K, P_K)    # 每个 P_K 分区处理的 K 值

# 示例：M=4096, P_M=4
# m_nom = ceil_div(4096, 4) = 1024
```

### 3.5 DRAM 流量计算

#### 三种循环顺序的流量计算

**循环顺序 'mnk'（最常见）:**
```python
tile_num_m = ceil_div(M, M_tile)
tile_num_n = ceil_div(N, N_tile)
tile_num_k = ceil_div(K, K_tile)

dram_traffic = (
    M * K * FP8_BYTES * tile_num_n +   # A 矩阵重复加载 N 次
    N * K * FP8_BYTES * tile_num_m +   # B 矩阵重复加载 M 次
    M * N * BF16_BYTES                  # C 矩阵输出一次
)
```

**循环顺序 'nkm':**
```python
dram_traffic = (
    N * K * FP8_BYTES +                           # B 全局加载一次
    M * K * FP8_BYTES * tile_num_n +             # A 按 N 瓦片加载
    M * N * FP32_BYTES * 2 * (tile_num_k - 1) +  # 中间结果累加
    M * N * BF16_BYTES                            # C 输出
)
```

**循环顺序 'mkn':**
```python
dram_traffic = (
    M * K * FP8_BYTES +                           # A 全局加载一次
    N * K * FP8_BYTES * tile_num_m +             # B 按 M 瓦片加载
    M * N * FP32_BYTES * 2 * (tile_num_k - 1) +  # 中间结果累加
    M * N * BF16_BYTES                            # C 输出
)
```

---

## 4. Attention 算子详解

### 4.1 MHA/MQA 的 FLOPS 计算

**基本公式:**

```
FLOPS = 2 × B × QS × KS × (QD + VD)
```

其中：
- **B**: Batch 大小（或头数）
- **QS**: Query 序列长度
- **KS**: Key/Value 序列长度
- **QD**: Query/Key 的头维度
- **VD**: Value 的头维度
- **系数2**: 乘加操作

**计算分解:**
```
1. QK 乘法 (Query × Key^T):
   FLOPS_QK = 2 × B × QS × KS × QD

2. Softmax (Score → Attention Weight):
   向量操作，计算量较小

3. SV 乘法 (Attention Weight × Value):
   FLOPS_SV = 2 × B × QS × KS × VD

4. 总计:
   FLOPS_total = 2 × B × QS × KS × (QD + VD)
```

**代码实现** (`model/operators/compute/mqa.py`):
```python
self.flops = 2 * self.bs * self.head * self.kv_seq_len * (self.qk_head_dim + self.v_head_dim)
```

### 4.2 Flash Attention 2 实现细节

#### 架构利用率计算

**GEMM 部分理论值:**
```python
gemm_real = QS * KS * (QD + VD)  # 实际计算量

gemm_theo = (
    align_up(QS, cube_m) * align_up(QD, cube_k) * align_up(KS, cube_n) +  # QK
    align_up(QS, cube_m) * align_up(KS, cube_k) * align_up(VD, cube_n)    # SV
)
```

**Softmax 向量操作的理论值:**

FA2 中 Softmax 包含多个操作步骤：

| 步骤 | 操作 | 形状类型 | op_count |
|------|------|----------|----------|
| 1 | add | (QS,1,KS) | 1 |
| 2 | reduce_max | (QS,1,1) | 1 |
| 3 | max | (QS,1,1) | 1 |
| 4-5 | fuse_exp | 两种 | 各35 |
| 6 | reduce_sum | (QS,1,1) | 1 |
| 7-10 | mul/add/copy/convert | 混合 | 1-1 |

```python
def calc_step_theo(shape_type, op_count):
    if shape_type == 0:  # (QS,1,KS)
        return (align_up(QS, lane_num) *
                align_up(KS, eu_num // lane_num // BF16_BYTES) *
                op_count)
    else:  # (QS,1,1)
        return align_up(QS, lane_num) * op_count

vector_theo = sum(calc_step_theo(...) for each step)
```

**总的架构利用率:**
```python
arch_urate = (gemm_real + vector_real) / (gemm_theo + vector_theo)
```

### 4.3 QK、Softmax、SV 各阶段计算

#### 1. QK 阶段（Query-Key 点积）

```python
# 形状: Q(QS, QD) × K(KS, QD)^T → P(QS, KS)
gemm_theo_qk = align_up(QS, cube_m) * align_up(QD, cube_k) * align_up(KS, cube_n)
t_us_qk = gemm_theo_qk * B / (macs_per_cycle * freq) / 1e3

# DRAM 流量
load_q = QS * QD * FP8_BYTES              # Q 加载一次
load_k = KS * QD * FP8_BYTES * tile_num_q # K 按 Q 瓦片重复加载
store_p = QS * KS * BF16_BYTES             # 存储 QK 结果
```

#### 2. Softmax 阶段

```python
# 在序列维度 (QS, KS) 上进行 softmax
# 向量化实现: reduce_max → exp → reduce_sum → div

vector_theo_softmax = (
    align_up(QS, lane_num) * align_up(KS, eu_num//lane_num//BF16_BYTES) *
    (1 + 1 + 1 + 35 + 35 + 1 + 1 + 1 + 1 + 1)  # 所有操作步骤
)
t_us_softmax = vector_theo_softmax * B / (eu_num * freq) / BF16_BYTES / 1e3
```

#### 3. SV 阶段（Value 点积）

```python
# 形状: P(QS, KS) × V(KS, VD) → O(QS, VD)
gemm_theo_sv = align_up(QS, cube_m) * align_up(KS, cube_k) * align_up(VD, cube_n)
t_us_sv = gemm_theo_sv * B / (macs_per_cycle * freq) / 1e3

# DRAM 流量
load_v = KS * VD * FP8_BYTES * tile_num_q  # V 按 Q 瓦片重复加载
load_p = QS * KS * BF16_BYTES              # 加载 softmax 结果
store_o = QS * VD * BF16_BYTES             # 存储最终输出
```

#### 总时间（考虑重叠）

```python
t_total = (
    min(comp_elapse, dma_elapse) * (1 - tpu_gdma_overlap_rate) +
    max(comp_elapse, dma_elapse)
)
# 其中 tpu_gdma_overlap_rate = 0.8 (80%重叠)
```

---

## 5. 通信算子详解

### 5.1 AllReduce 延迟公式

#### 分层 AllReduce（TP=8, 16, 32）

**两层/三层结构（例如 TP=16 → 4个子组，每组4个设备）:**

**第1层（子组内 AllReduce）:**
```python
group_size = 4  # TP=16时为2
num_groups = tp // group_size

comm_size_1 = 2 * (group_size - 1) / group_size * bytes

lat_1 = (comm_size_1 / intra_bw / bw_urate) * 1e6 +
        (group_size - 1) * (start_lat + sync_lat)
```

**第2层（子组间 AllReduce）:**
```python
comm_size_2 = 2 * (num_groups - 1) / num_groups * bytes

lat_2 = (comm_size_2 / inter_bw / bw_urate) * 1e6 +
        (num_groups - 1) * (start_lat + sync_lat + link_delay)
```

**第3层（子组内广播）:**
```python
comm_size_3 = bytes

lat_3 = (comm_size_3 / intra_bw / bw_urate) * 1e6 +
        (group_size - 1) * (start_lat + sync_lat)
```

**总延迟（取三层的最大值）:**
```python
lat_hierarchy = max(lat_1, lat_2, lat_3)
total_comm = comm_size_1 * num_groups + comm_size_2 + comm_size_3 * num_groups
```

#### 平面 AllReduce（其他 TP 值）

```python
# Ring AllReduce 通信量
comm_size = 2 * (tp - 1) / tp * bytes

# 基础延迟
lat_base = (comm_size / intra_bw / bw_urate) * 1e6 +
           (tp - 1) * (start_lat + sync_lat)
```

#### 通信协议变体

| 协议 | 描述 | 延迟公式 |
|------|------|----------|
| 1 | Ring (基础) | `lat_base` |
| 2 | BTT (Binary Tree) | `lat_base + rtt_tp * 2 * (tp - 1)` |
| 3 | HDH (Halving-Doubling) | `lat_base + rtt_tp * min(1, 2 * (tp - 1))` |

其中 `rtt_tp = 0.35` 微秒（往返时间）

### 5.2 AllGather 延迟公式

**分层模式（TP ∈ {8, 16, 32}）:**
```python
# Stage 1 (intra-group)
comm_size_1 = (group_size - 1) * bytes
lat_1 = (comm_size_1 / intra_bw / 0.95) * 1e6 + (group_size - 1) * start_lat

# Stage 2 (inter-group)
comm_size_2 = (num_groups - 1) * bytes
lat_2 = (comm_size_2 / inter_bw / 0.95) * 1e6 + (num_groups - 1) * (start_lat + link_delay)

total_latency = max(lat_1, lat_2)
```

**平面模式:**
```python
comm_size = (tp - 1) * bytes
lat = (comm_size / intra_bw / 0.95) * 1e6 + (tp - 1) * start_lat
```

### 5.3 ReduceScatter 延迟公式

**ReduceScatter = AllReduce 但通信量减半**

**分层模式:**
```python
comm_size_1 = (group_size - 1) / group_size * bytes  # 不是 2×
comm_size_2 = (num_groups - 1) / num_groups * bytes
```

**平面模式:**
```python
comm_size = (tp - 1) / tp * bytes  # Ring 通信量少一半
```

### 5.4 Dispatch/Combine 延迟公式

**用途**: MoE 模型中，将 Token 分发到 Expert，然后合并结果

**Dispatch 延迟:**
```python
# 基础延迟
t_us = (bytes / inter_bw / bw_urate) * 1e6 + start_lat + cpu_fetch_delay

# 协议开销
if comm_protocol == 2:
    if is_prefill:
        t_us += rtt_ep * bs * topk * prefill_factor  # prefill: 8/128
    else:
        t_us += rtt_ep * bs * topk
elif comm_protocol == 3:
    if is_prefill:
        t_us += rtt_ep * min(1, bs * topk * prefill_factor)
    else:
        t_us += rtt_ep * min(1, bs * topk)

# 追加 AllGather 延迟
t_us += allgather_eval.evaluate_raw(moe_tp, bytes)
```

参数说明：
- `rtt_ep = 0.85` 微秒（Expert 之间往返时间）
- `topk = 8`（每个 Token 选择 8 个 Expert）
- `prefill_factor = 8/128 = 0.0625`（Prefill 阶段缩放因子）

### 5.5 启动延迟组成

```python
start_lat = (
    2 * c2c_lat +      # 核心间通信: 0.15 × 2 = 0.30
    ddr_r_lat +         # DDR 读: 0.15
    ddr_w_lat +         # DDR 写: 0.01
    noc_lat +           # NoC 延迟: 0.05
    2 * d2d_lat         # D2D 延迟: 0.04 × 2 = 0.08
)
# = 0.30 + 0.15 + 0.01 + 0.05 + 0.08 = 0.59 微秒
```

---

## 6. 并行策略详解

### 6.1 核心约束条件

```python
assert dp * tp == moe_tp * ep
```

这个约束确保总设备数量在 DP/TP 和 MoE_TP/EP 之间的平衡。

### 6.2 TP（Tensor Parallelism）张量并行

**定义与应用:**
- TP 度数应用于所有 MatMulOperator 的 N 维度（输出维度）
- 用于分割隐藏维度

**具体实现:**

| 层类型 | TP 应用方式 |
|--------|------------|
| Dense MLP | `inter_dim / tp` |
| MoE | `moe_inter_dim / tp` |
| MLA | `n_heads / tp` |
| LMHead | 完整 TP |

**通信开销:**
- 每个 TP 分片后的层需要 AllReduce
- 通信量: `seqs × hidden_dim × BF16` 字节

```python
# MLP 层示例
self.linear_gate = MatMulOperator(
    parallel_params={'G': 1, 'M': seqs, 'K': hidden_dim, 'N': inter_dim // tp}
)
# TP > 1 时触发 AllReduce
if tp > 1:
    self.allreduce = AllReduceOperator(comm_size=seqs * hidden_dim * BF16)
```

### 6.3 DP（Data Parallelism）数据并行

**定义与应用:**
- 全局批大小沿 DP 度数分割
- 每个设备处理 `batch_size / dp` 个样本

**具体实现:**
```python
self.batch_size_local = self.batch_size // self.dp
self.seqs = self.q_seq_len * self.batch_size_local
```

**数据流:**
- 不同 DP 组的梯度需要同步（由 AllReduce 实现）
- Embedding 和 LMHead 通常不应用 DP

### 6.4 EP（Expert Parallelism）专家并行

**定义:**
- 在 MoE 层中，专家分布到 EP 组
- 每个 EP 组处理 `n_routed_experts / ep` 个专家

**具体实现:**
```python
self.expert_per_ep_group = ceil_div(self.n_routed_experts, self.ep)
self.token_per_ep_group = ceil_div(
    self.batch_size * self.q_seq_len * self.n_activated_experts,
    self.ep
)
```

**通信算子:**
- **Dispatch**: 将 Token 路由到正确的 EP 组
- **Combine**: 从各 EP 组收集 Expert 输出

### 6.5 MoE_TP（MoE 张量并行）

**定义:**
- 在 MoE 层内部应用 TP 来分割 Expert 内部的矩阵乘法
- `inter_dim_local = moe_inter_dim / moe_tp`

**实现逻辑:**
```python
# Dispatch 数据大小
dispatch_size = self.token_per_ep_group * self.hidden_dim // self.moe_tp
```

**设计意图:**
- MoE_TP 和 EP 形成二维并行
- EP 管理专家分布，MoE_TP 管理每个专家内的计算

### 6.6 各层对并行度的敏感性

| 层类型 | TP影响 | DP影响 | EP影响 | 总体特点 |
|--------|--------|--------|--------|----------|
| **Embedding** | 弱 (embed_tp=1) | 弱 | - | 1张卡计算可行 |
| **MLA** | 高 (AllReduce) | 弱 | - | TP通信主导 |
| **DSA** | 高 | 弱 | - | TP通信主导 |
| **Dense MLP** | 高 | 弱 | - | TP通信主导 |
| **MoE** | 中 | 弱 | 高 | EP并行高效 |
| **LMHead** | 中 (lmhead_tp=tp) | 弱 | - | TP聚合 |

---

## 7. 延迟计算模型

### 7.1 计算延迟

**基本公式:**
```python
计算延迟 = FLOPS / (有效FLOPS速率 × 计算效率)
         = FLOPS / (TPU_FLOPS × 效率)

其中：
- TPU_FLOPS = macs_per_cycle × frequency × cores × 2
- macs_per_cycle = cube_m × cube_k × cube_n  # TPU 架构参数
- frequency (GHz) 从 FLOPS 反推
```

**MatMul 计算延迟:**
```python
# 理论计算量（对齐后）
theo_ops = align_up(m, cube_m) * align_up(k, cube_k) * align_up(n, cube_n)

# 单核计算时间
t_comp_us = theo_ops * g_blk / (macs_per_cycle * freq) / 1e3

# 架构利用率
arch_urate = (m * k * n) / theo_ops
```

### 7.2 内存延迟

**DMA 传输时间:**
```python
t_dma_us = dram_traffic / dma_bw * 1e6

其中：
- dram_traffic: DRAM 流量（字节）
- dma_bw: 单核 DMA 带宽 = dram_bw / tpu_cores
```

### 7.3 总延迟（考虑重叠）

```python
tpu_gdma_overlap_rate = 0.8  # TPU 计算与 GDMA 传输 80% 重叠

t_total = (
    min(t_comp, t_dma) * (1 - overlap_rate) +
    max(t_comp, t_dma)
)

# 示例：t_comp=100us, t_dma=150us, overlap=0.8
t_total = min(100, 150) * 0.2 + max(100, 150)
        = 100 * 0.2 + 150
        = 20 + 150 = 170us
```

### 7.4 通信延迟模型

参见 [5. 通信算子详解](#5-通信算子详解)

---

## 8. 性能指标计算公式

### 8.1 吞吐量（TPS - Tokens Per Second）

```python
total_tokens = batch_size × q_seq_len
total_elapse_us = total_time(tbo=False)  # 微秒

tps = total_tokens / (total_elapse_us × 1e-6)  # tokens/s

tps_per_batch = tps / batch_size
tps_per_chip = tps / (tp × dp)  # 每芯片吞吐
```

### 8.2 模型 FLOPS 利用率（MFU）

```python
MFU = total_flops / (tpu_flops × (total_elapse_us / 1e6))
    = total_flops / (tpu_flops × total_elapse_seconds)

其中：
- total_flops: 模型总计算量（FLOPs）
- tpu_flops: TPU 峰值性能（来自 TPUConfigBase.flops）
- total_elapse_us: 总执行时间（微秒）
```

**FLOPS 计算（按层加权聚合）:**
```python
total_flops = (
    embedding.flops +
    mla.flops × n_layers +
    dsa.flops × n_layers +
    dense_mlp.flops × dense_layers +
    moe.flops × n_moe_layers +
    lmhead.flops
)
```

### 8.3 计算利用率

**架构利用率（对齐效率）:**
```python
arch_urate = real_computation / theoretical_computation

# MatMul 示例
real = m * n * k
theo = align_up(m, cube_m) * align_up(k, cube_k) * align_up(n, cube_n)
arch_urate = real / theo
```

**实际利用率:**
```python
# 考虑计算与内存重叠
t_comp = theo * g_blk / (macs_per_cycle * freq) / 1e3  # 微秒
t_dma = traffic * 1e6 / dma_bw                         # 微秒

# 实际计算利用率
real_util = t_comp / t_total * arch_urate  # 如果 t_total != 0

# 总体核心利用率
core_urate = total_flops / (max_time × 1e-3 × tpu_cores × macs_per_cycle × freq × 2)
```

### 8.4 内存占用

**DRAM 占用计算:**
```python
total_dram = (
    embedding.dram_occupy +
    mla.dram_occupy × n_layers +
    dsa.dram_occupy × n_layers +
    dense_mlp.dram_occupy × dense_layers +
    moe.dram_occupy × n_moe_layers +
    lmhead.dram_occupy
)
```

**MLA 中的 KV 缓存:**
```python
# MHA 中：KV 缓存用于完整 KV 序列
dram_occupy = (
    batch_size_local × kv_seq_len ×
    (qk_nope_head_dim + qk_rope_head_dim) × BF16
)

# 但受 DSA (Deep Sparse Attention) 限制
effective_kv_len = min(kv_seq_len, topk_index)  # topk_index=2048
```

### 8.5 输出指标汇总

| 指标 | 单位 | 含义 | 计算方法 |
|------|------|------|----------|
| **total_elapse_us** | 微秒 | 整个前向传播时间 | 所有层的计算+通信延迟和 |
| **comm_elapse_us** | 微秒 | 总通信时间 | AllReduce + Dispatch + Combine |
| **flops** | 操作数 | 总浮点操作数 | 所有算子 FLOPS 求和 |
| **dram_occupy** | 字节 | 峰值 DRAM 占用 | 最大权重+激活内存 |
| **dram_traffic** | 字节 | 总 DRAM 数据量 | 所有算子流量求和 |
| **mfu** | 比例(%) | 模型 FLOPS 利用率 | total_flops / (TPU_FLOPS × elapse) |
| **tps** | tokens/s | 总吞吐量 | seq_len × batch_size / elapse |
| **tps_per_chip** | tokens/s | 单芯片吞吐量 | tps / num_chips |

---

## 9. 配置参数说明

### 9.1 DeploymentConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 64 | 全局批大小 |
| `q_seq_len` | int | 4096 | 查询序列长度（prefill时） |
| `kv_seq_len` | int | 4096 | KV序列长度 |
| `tp` | int | 1 | 张量并行度 |
| `dp` | int | 32 | 数据并行度 |
| `moe_tp` | int | 1 | MoE中的张量并行度 |
| `ep` | int | 32 | 专家并行度 |
| `is_prefill` | bool | False | 是否为prefill阶段 |
| `enable_tp_sp` | bool | False | 是否启用TP+SP(序列并行) |
| `embed_tp` | int | 1 | Embedding的TP（固定为1） |
| `lmhead_tp` | int | tp | LMHead的TP（默认等于tp） |
| `comm_protocol` | int | 1 | 通信协议（1/2/3） |
| `kv_cache_rate` | float | 0 | KV缓存比率 |

### 9.2 ModelConfig 参数（DeepSeek V3.2）

#### 基础参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `name` | "DeepSeek-V3.2" | 模型名称 |
| `model_type` | "deepseek" | 模型类型 |
| `hidden_dim` | 7168 | 隐藏层维度 |
| `inter_dim` | 18432 | FFN中间维度 |
| `n_layers` | 61 | 总层数 |
| `n_dense_layers` | 3 | 稠密层数 |
| `n_moe_layers` | 58 | MoE层数 |
| `n_heads` | 128 | 注意力头数 |
| `vocab_size` | 129280 | 词汇表大小 |

#### MoE 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_routed_experts` | 256 | 路由专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `n_activated_experts` | 8 | 激活专家数（每token） |
| `n_expert_groups` | 8 | 专家分组数 |
| `moe_inter_dim` | 2048 | 专家FFN中间维度 |
| `route_scale` | 2.5 | 路由缩放因子 |

#### MLA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `q_lora_rank` | 1536 | 查询LoRA秩 |
| `kv_lora_rank` | 512 | KV LoRA秩 |
| `qk_nope_head_dim` | 128 | QK非绳索头维度 |
| `qk_rope_head_dim` | 64 | QK绳索头维度 |
| `v_head_dim` | 128 | 值头维度 |

#### DSA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_index_heads` | 64 | 索引头数 |
| `index_head_dim` | 128 | 索引头维度 |
| `topk_index` | 2048 | Top-K索引大小 |

### 9.3 TPUConfig 参数

| 参数 | 类型 | 值 | 说明 |
|------|------|-----|------|
| `core` | int | 32 | TPU核心数 |
| `flops` | float | 256e12 | 峰值FLOPs (256 TFLOPS) |
| `dram` | float | 80e9 | DRAM容量 (80GB) |
| `dram_bw` | float | 3.2e12 | DRAM带宽 (3.2TB/s有效) |
| `intra_bw` | float | 500e9 | 芯片内带宽 (500GB/s) |
| `inter_bw` | float | 40e9 | 芯片间带宽 (40GB/s) |
| `cube_m` | int | 16 | 计算单元M维度 |
| `cube_k` | int | 32 | 计算单元K维度 |
| `cube_n` | int | 8 | 计算单元N维度 |
| `sram_size` | int | 2MB | 片上SRAM大小 |

**派生参数计算:**
```python
dma_bw = dram_bw / tpu_cores  # 单核DMA带宽
macs_per_cycle = cube_m × cube_k × cube_n = 16 × 32 × 8 = 4096
freq = flops / (2 × core × macs_per_cycle)
     = 256e12 / (2 × 32 × 4096) ≈ 0.977 GHz
```

---

## 10. 计算示例

### 示例 1: MatMul 性能估计

**参数:**
- G=1, M=4096, N=4096, K=8192
- TPU: 32核心, 256TFLOP/s, cube_m=16, cube_k=32, cube_n=8
- 选择分区: P_G=1, P_M=4, P_N=2, P_K=4

**计算过程:**

1. **每个核心负载:**
   ```
   g_nom = ceil_div(1, 1) = 1
   m_nom = ceil_div(4096, 4) = 1024
   n_nom = ceil_div(4096, 2) = 2048
   k_nom = ceil_div(8192, 4) = 2048
   ```

2. **选择最优 Tile:**
   ```
   例: m_t=64, n_t=128, k_t=32
   ```

3. **DRAM 流量 (mnk 循环):**
   ```
   tile_num_m = ceil_div(1024, 64) = 16
   tile_num_n = ceil_div(2048, 128) = 16
   tile_num_k = ceil_div(2048, 32) = 64

   traffic = 1024×2048×1×16 + 2048×2048×1×16 + 1024×2048×2
           ≈ 134MB
   ```

4. **计算时间:**
   ```
   arch_urate = (1024×2048×2048) / (align(1024,16)×align(2048,32)×align(2048,8))
              ≈ 0.95

   theo_ops = align(1024,16) × align(2048,32) × align(2048,8)
   t_comp = theo_ops × 1 / (4096 × freq) / 1e3 ≈ 150us
   ```

5. **DMA 时间:**
   ```
   dma_bw_per_core = 3200GB/s / 32 = 100GB/s
   t_dma = 134MB / 100GB/s × 1e6 ≈ 1.34us
   ```

6. **总时间:**
   ```
   t_total = min(150, 1.34) × 0.2 + max(150, 1.34) ≈ 150.3us
   ```

### 示例 2: MoE 层性能评估

**配置:**
- batch_size = 64, q_seq_len = 4096 (prefill)
- tp = 8, dp = 4, moe_tp = 1, ep = 32
- n_routed_experts = 256, n_activated_experts = 8
- hidden_dim = 7168, moe_inter_dim = 2048

**计算步骤:**

1. **局部批大小:**
   ```
   batch_size_local = 64 / 4 = 16
   seqs = 4096 × 16 = 65536
   ```

2. **专家分配:**
   ```
   expert_per_ep_group = ceil(256/32) = 8
   token_per_ep_group = ceil(64 × 4096 × 8 / 32) = 65536
   ```

3. **Dispatch 通信:**
   ```
   dispatch_bytes = token_per_ep_group × hidden_dim / moe_tp
                  = 65536 × 7168 / 1 = 470MB

   dispatch_lat_us = (470e6 / 40e9 / 0.95) × 1e6 + 0.59
                   ≈ 12.4ms
   ```

4. **专家计算（每个 EP 组）:**
   ```
   expert_flops = 2 × token_per_ep_group × moe_inter_dim × moe_inter_dim
                = 2 × 65536 × 2048 × 2048
                ≈ 550G FLOPs

   expert_compute_time = 550e9 / (256e12 / 32) ≈ 68.75μs
   ```

5. **Combine 通信（类似 Dispatch）:**
   ```
   combine_lat_us ≈ 12.4ms
   ```

6. **总 MoE 时间:**
   ```
   moe_time_per_layer ≈ 12.4ms + 68.75μs + 12.4ms ≈ 24.87ms
   moe_total = 24.87ms × 58层 ≈ 1.44s
   ```

### 示例 3: Flash Attention 性能估计

**参数:**
- B=32, QS=4096, KS=4096, QD=64, VD=64

**计算:**

1. **FLOPS:**
   ```
   FLOPS = 2 × 32 × 4096 × 4096 × (64 + 64)
         = 2 × 32 × 4096² × 128
         ≈ 137 billion FLOPs
   ```

2. **DRAM 流量:**
   ```
   load_q = 32 × 4096 × 64 × 1 = 8MB
   load_k = 32 × 4096 × 64 × tile_num_q
   load_v = 32 × 4096 × 64 × tile_num_q
   store_o = 32 × 4096 × 64 × 2 = 16MB
   ```

3. **时间估计:**
   ```
   t_comp ≈ 150-200us (包括 Softmax 向量操作)
   t_dma ≈ DRAM_traffic / dma_bw
   t_total ≈ max(t_comp, t_dma) (考虑 80% 重叠)
   ```

---

## 文件位置索引

| 功能 | 文件路径 |
|------|----------|
| 配置定义 | `config/deployment_config.py`, `config/model_config.py` |
| 模型实现 | `model/networks/deepseek/deepseek_v3_2.py` |
| 层实现 | `model/layers/moe.py`, `model/layers/mlp.py`, `model/layers/mla_v3_2.py` |
| 性能分析 | `performance/analyzer.py` |
| 通信评估 | `performance/evaluate/communication/*.py` |
| 计算评估 | `performance/evaluate/compute/matmul/`, `compute/flash_attention/` |
| 配置文件 | `config/model_configs/`, `config/tpu_configs/` |

---

## 总结

DS_TPU 工具提供了一个**完整的 LLM 推理性能模拟框架**，核心特点：

1. **MatMul**: 通过 4 维分区 (P_G, P_M, P_N, P_K) 和 3 维 Tile 划分 (m_t, n_t, k_t) 优化
2. **Attention**: FA2 架构考虑 GEMM 效率和 Softmax 向量操作
3. **通信**: 分层 AllReduce 策略，支持多种协议和拓扑
4. **并行策略**: TP/DP/EP/MoE_TP 组合优化，满足约束 `dp×tp = moe_tp×ep`
5. **性能指标**: MFU、延迟、吞吐、内存占用等全维度分析
6. **优化机制**: 智能缓存、快速评估、批量计算

该框架适用于快速探索不同并行配置下的性能特征，指导实际部署决策。
