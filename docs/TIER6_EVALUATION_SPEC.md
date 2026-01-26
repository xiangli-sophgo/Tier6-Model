# Tier6-Model 性能评估系统规格说明

**版本**: v2.2.0
**最后更新**: 2026-01-26
**状态**: Production

---

## 📋 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [评估流程](#3-评估流程)
4. [核心模块规格](#4-核心模块规格)
5. [MoE 负载均衡实现](#5-moe-负载均衡实现)
6. [性能优化机制](#6-性能优化机制)
7. [与 DS_TPU 对齐情况](#7-与-ds_tpu-对齐情况)
8. [配置参数](#8-配置参数)
   - 8.1 模型配置
   - 8.2 部署配置
   - 8.3 通信延迟配置 (CommLatencyConfig) ✨ **New**
   - 8.4 硬件配置
9. [使用示例](#9-使用示例)
10. [附录](#10-附录)

---

## 1. 系统概述

### 1.1 设计目标

Tier6-Model 是一个用于 LLM 推理性能评估的精确模拟器，旨在：

- **精确建模**：基于硬件微架构参数（Tile、SRAM、带宽）进行细粒度评估
- **全流程覆盖**：支持 Prefill 和 Decode 阶段，涵盖计算、访存、通信
- **可扩展性**：模块化设计，易于扩展新算子和硬件平台
- **高性能**：多级缓存机制，支持大规模配置搜索

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **精确 GEMM 评估** | 多核分块 + Tile 搜索 + 循环顺序优化 |
| **FlashAttention** | 支持 MHA/MQA/GQA，考虑 Softmax 访存瓶颈 |
| **MoE 负载均衡** | 基于蒙特卡洛模拟的专家路由建模 |
| **通信建模** | 支持多协议（post-write/non-post-write/流水线） |
| **GEMM 预热** | 离线预调优常见 GEMM 形状，加速首次评估 |
| **全局缓存** | 跨实验复用评估结果 |

### 1.3 支持的模型和硬件

**模型架构**：
- DeepSeek V3 / R1 (MLA + MoE)
- 标准 Transformer (MHA + MLP/MoE)

**硬件平台**：
- 算能 SG2260E (默认)
- NVIDIA H100 SXM
- NVIDIA A100

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端 (React + Three.js)                   │
│  - 3D 可视化拓扑配置                                          │
│  - 交互式参数调整                                             │
│  - 实时结果展示 (Gantt图、性能指标)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP API
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                 后端 (Python + FastAPI)                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  API 层 (api.py)                                        │ │
│ │  - POST /api/simulate                                   │ │
│ │  - POST /api/validate                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                      ↓                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  模拟器 (simulator.py)                                  │ │
│ │  - 构建层和算子                                          │ │
│ │  - 调度评估流程                                          │ │
│ │  - 生成 Gantt 图                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                      ↓                                        │
│ ┌───────────────────┬───────────────────┬─────────────────┐ │
│ │   层定义           │   算子定义         │   评估器        │ │
│ │  (layers/)        │  (operators/)     │ (evaluators/)   │ │
│ ├───────────────────┼───────────────────┼─────────────────┤ │
│ │ - EmbeddingLayer  │ - MatMulOperator  │ - GEMMEvaluator │ │
│ │ - MLALayer        │ - MHAOperator     │ - FA2Evaluator  │ │
│ │ - MLAAbsorbLayer  │ - MQAOperator     │ - AllReduceEval │ │
│ │ - MoELayer        │ - AllReduceOp     │ - DispatchEval  │ │
│ │ - MLPLayer        │ - DispatchOp      │ - MoELoadBalance│ │
│ │ - LMHeadLayer     │ - ...             │ - ...           │ │
│ └───────────────────┴───────────────────┴─────────────────┘ │
│                      ↓                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  硬件微架构配置 (AcceleratorMicroArch)                   │ │
│ │  - SRAM/Cache 参数                                      │ │
│ │  - 计算单元规格 (Cube M/N/K)                            │ │
│ │  - 带宽和延迟参数                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
用户输入配置
    ↓
┌──────────────────────┐
│ 1. 拓扑解析           │  → InterconnectGraph (芯片连接关系)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 2. 并行策略映射       │  → 分配 TP/PP/DP/EP 芯片组
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 3. 层构建             │  → 创建 Layer 实例 (MLA, MoE, MLP, ...)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 4. 算子实例化         │  → 创建 Operator (MatMul, MHA, AllReduce, ...)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 5. 评估器调用         │  → 精确计算延迟、流量、利用率
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 6. 结果聚合           │  → 层级/模型级性能指标
└──────────────────────┘
    ↓
输出结果 (JSON + Gantt图)
```

### 2.3 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| **Simulator** | `simulator.py` | 主控流程，调度评估 |
| **Layers** | `layers/` | 层级抽象，组合算子 |
| **Operators** | `operators/` | 算子接口，定义计算和通信原语 |
| **Evaluators** | `evaluators/` | 精确评估，实现硬件建模 |
| **Types** | `types.py` | 类型定义，枚举和数据类 |

---

## 3. 评估流程

### 3.1 总体流程

```python
# 伪代码
def simulate(config):
    # 步骤 1: 解析配置
    topology = parse_topology(config.topology)
    model_cfg = config.model
    deploy_cfg = config.deployment

    # 步骤 2: 创建硬件架构
    arch = get_arch_preset(config.hardware)

    # 步骤 3: 构建层
    layers = []
    for layer_type in model_cfg.layers:
        layer = create_layer(layer_type, model_cfg, deploy_cfg)
        layers.append(layer)

    # 步骤 4: 评估每一层
    for layer in layers:
        for operator in layer.operators:
            evaluator = get_evaluator(operator.type, arch)
            result = evaluator.evaluate(operator)
            operator.latency = result.latency_us
            operator.traffic = result.dram_traffic_bytes
            # ...

    # 步骤 5: 聚合结果
    total_latency = sum(layer.latency for layer in layers)
    return {
        'latency_us': total_latency,
        'mfu': calculate_mfu(layers, arch),
        'gantt': generate_gantt(layers),
        # ...
    }
```

### 3.2 Prefill vs Decode 差异

| 维度 | Prefill | Decode |
|------|---------|--------|
| **序列长度** | q_len = kv_len (如 4096) | q_len = 1 |
| **Attention 类型** | MHA (full attention) | MQA/MLA_absorb (cached KV) |
| **计算瓶颈** | 计算密集 (GEMM) | 访存密集 (KV Cache) |
| **并行策略** | TP + PP + DP | TP-SP (Sequence Parallel) |
| **MoE 负载** | 负载相对均衡 | 负载严重不均 |

### 3.3 层级评估流程

以 **DeepSeek V3 Decode** 为例：

```
Layer 0: Embedding
  ↓
Layer 1-61: Transformer Block
  ├─ RMSNorm (Pre-Attention)
  ├─ MLA_absorb
  │   ├─ q_a_proj (GEMM)
  │   ├─ q_b_proj (GEMM)
  │   ├─ kv_a_proj (GEMM)
  │   ├─ w_kc (GEMM)
  │   ├─ MQA (FlashAttention)
  │   ├─ w_vc (GEMM)
  │   ├─ o_proj (GEMM)
  │   └─ AllReduce (TP > 1)
  ├─ RMSNorm (Pre-MoE)
  └─ MoE
      ├─ Gate Router (GEMM)
      ├─ Dispatch (EP > 1)
      ├─ Routed Experts
      │   ├─ gate_proj (GEMM, G=max_experts)
      │   ├─ up_proj (GEMM, G=max_experts)
      │   └─ down_proj (GEMM, G=max_experts)
      ├─ AllReduce (MoE_TP > 1)
      ├─ Combine (EP > 1)
      └─ Shared Experts (可选)
  ↓
Layer 62: LMHead
```

---

## 4. 核心模块规格

### 4.1 GEMM 评估器 (GEMMEvaluator)

**功能**：评估矩阵乘法 `C[G, M, N] = A[G, M, K] × B[G, K, N]`

**输入参数**：
```python
@dataclass
class GEMMParams:
    G: int              # Batch/Group 维度
    M: int              # 输出行数
    K: int              # 累加维度
    N: int              # 输出列数
    input_dtype: str    # 'fp8', 'bf16', 'fp16'
    output_dtype: str   # 'bf16', 'fp32'
```

**输出结果**：
```python
@dataclass
class GEMMResult:
    latency_us: float               # 总延迟 (微秒)
    compute_time_us: float          # 计算时间
    memory_time_us: float           # 访存时间
    flops: int                      # 浮点运算数
    dram_traffic_bytes: int         # DRAM 流量
    arch_utilization: float         # 架构利用率 (0-1)
    effective_utilization: float    # 有效利用率 (0-1)
    best_tile: Tuple[int, int, int] # 最佳 Tile (m_t, n_t, k_t)
    best_loop_order: str            # 最佳循环顺序 ('mnk', 'nkm', 'mkn')
    best_partition: Tuple[int, int, int, int]  # 最佳分块 (P_G, P_M, P_N, P_K)
```

**评估流程**：

```
1. 枚举所有合法的多核分块 (P_G, P_M, P_N, P_K)
   约束：P_G × P_M × P_N × P_K = num_cores

2. 对每个分块方案：
   a. 计算每核负责的维度 (g_nom, m_nom, n_nom, k_nom)
   b. 搜索能放进 SRAM 的 Tile 大小 (m_t, n_t, k_t)
      约束：m_t × n_t × output_dtype + (m_t + n_t) × k_t × input_dtype ≤ SRAM
   c. 对每个 Tile 和循环顺序 (mnk, nkm, mkn)：
      - 计算 DRAM 流量
      - 选择流量最小的组合
   d. 计算该分块的总延迟
      - 架构利用率：real_macs / theo_macs
      - 计算时间：theo_macs / (freq × macs_per_cycle)
      - 访存时间：dram_traffic / dma_bandwidth
      - 重叠模型：max(t_comp, t_dma) + min(t_comp, t_dma) × (1 - overlap_rate)

3. 返回延迟最小的分块方案
```

**关键公式**：

```python
# 架构利用率
arch_utilization = (M × N × K) / (align_up(M, cube_m) × align_up(K, cube_k) × align_up(N, cube_n))

# 计算时间
compute_time_us = (align_up(M, cube_m) × align_up(K, cube_k) × align_up(N, cube_n) × G) / (macs_per_cycle × freq_ghz) / 1000

# DRAM 流量（以 mnk 循环为例）
tile_num_m = ceil(m_blk / m_t)
tile_num_n = ceil(n_blk / n_t)
dram_traffic = (m_blk × k_blk × input_bytes) × tile_num_n +  # A 重复加载
               (n_blk × k_blk × input_bytes) × tile_num_m +  # B 重复加载
               (m_blk × n_blk × output_bytes)                # C 写回
```

**性能优化**：
- **多进程并行**：所有分块方案并行评估
- **缓存机制**：相同 (G, M, K, N, dtype) 复用结果
- **Pareto 最优**：Tile 搜索时剪枝被支配的候选

**参考**：`backend/llm_simulator/evaluators/gemm_eval.py`

---

### 4.2 FlashAttention 评估器 (FA2Evaluator)

**功能**：评估 Fused Attention (`Q @ K.T @ V`)

**输入参数**：
```python
@dataclass
class FA2Params:
    B: int      # Batch × Heads
    QS: int     # Query 序列长度
    KS: int     # Key/Value 序列长度
    QD: int     # Query Head 维度
    VD: int     # Value Head 维度
```

**输出结果**：
```python
@dataclass
class FA2Result:
    latency_us: float           # 总延迟
    qk_matmul_us: float         # Q @ K.T 时间
    softmax_us: float           # Softmax 时间
    sv_matmul_us: float         # Score @ V 时间
    flops: int                  # 总 FLOPs
    dram_traffic_bytes: int     # DRAM 流量
```

**评估策略**：

```
FlashAttention 分为三个阶段：

1. QK MatMul: Q[B, QS, QD] @ K[B, KS, QD].T → Score[B, QS, KS]
   - 使用 GEMM 评估器

2. Softmax: Score[B, QS, KS] → Prob[B, QS, KS]
   - 访存密集：需读写 Score 矩阵
   - 延迟 = (2 × B × QS × KS × sizeof(dtype)) / dram_bandwidth

3. SV MatMul: Prob[B, QS, KS] @ V[B, KS, VD] → Out[B, QS, VD]
   - 使用 GEMM 评估器

总延迟 = qk_matmul + softmax + sv_matmul
```

**参考**：`backend/llm_simulator/evaluators/fa2_eval.py`

---

### 4.3 通信评估器 (CommEvaluators)

**支持的通信原语**：

| 原语 | 公式 | 说明 |
|------|------|------|
| **AllReduce** | `lat = 2(N-1)/N × size / bw + (N-1) × α` | Ring 算法 |
| **AllGather** | `lat = (N-1) × size / bw + (N-1) × α` | Ring 算法 |
| **ReduceScatter** | `lat = (N-1)/N × size / bw + (N-1) × α` | Ring 算法 |
| **Dispatch** | `lat = size / bw + α + cpu_fetch` | EP 分发 |
| **Combine** | `lat = size / bw + α + cpu_fetch` | EP 汇集 |

**通信协议支持**：

```python
class CommProtocol(Enum):
    POST_WRITE = 1          # 默认：异步写
    NON_POST_WRITE = 2      # 同步写：+RTT × 2 × (N-1)
    PIPELINE = 3            # 流水线：+RTT × min(1, 2 × (N-1))
```

**关键参数**：
- `α` (start_lat)：启动延迟（如 0.6 μs）
- `bw`：带宽（intra: 273 GB/s, inter: 100 GB/s）
- `RTT`：往返延迟（TP: 0.35 μs, EP: 0.85 μs）

**参考**：`backend/llm_simulator/evaluators/comm_eval.py`

---

### 4.4 MoE 负载均衡模块 (MoELoadBalance)

**问题背景**：

MoE 的 Router 网络为每个 token 随机选择 Top-K 个专家。当专家分布到多个芯片时：
- **理想假设**：每个芯片激活的专家数 = `num_experts / num_chips`
- **实际情况**：某些芯片会被调用更多次（路由不均）
- **瓶颈**：最慢的芯片决定总延迟（木桶效应）

**解决方案**：

使用**蒙特卡洛模拟**或**查找表**获取最忙芯片需要加载的专家数。

**查找表**：

```python
MAX_EXPERT_TABLE = {
    # batch_size: {chips: max_experts}
    4: {1: 30.51, 2: 17.34, 4: 10.37, 8: 6.58, 16: 4.45, 32: 3.18, ...},
    8: {1: 57.43, 2: 31.41, 4: 17.81, 8: 10.65, 16: 6.70, 32: 4.44, ...},
    # ...
    256: {1: 255.93, 2: 128.0, 4: 64.0, 8: 32.0, 16: 16.0, 32: 8.0, ...}
}
```

**物理意义**：
- `MAX_EXPERT_TABLE[batch=4][chips=32] = 3.18`
- 含义：32 个芯片中，**最忙的芯片需要加载约 3.18 个不同专家**

**使用方法**：

```python
# 1. 查表获取专家数
max_experts = get_max_expert_load(batch_size=4, chips=32)  # → 3.18

# 2. 用于 GEMM 的 G 维度（向上取整）
G = math.ceil(max_experts)  # → 4

# 3. 计算专家 GEMM
gemm_result = gemm_evaluator.evaluate(
    G=G,                    # 专家并行维度
    M=tokens_per_expert,    # 每专家处理的 tokens
    K=hidden_dim,
    N=expert_inter_dim / moe_tp
)

# 4. 计算权重搬运时间
expert_param_size = 3 × hidden_dim × expert_inter_dim × dtype_bytes
weight_load_time = max_experts × expert_param_size / dram_bandwidth
```

**查询策略（三级回退）**：

```
1. 精确查表（O(1)）
   ↓ 未命中
2. 线性插值（O(log n)）
   ↓ 失败
3. 蒙特卡洛模拟（O(iterations)）
```

**蒙特卡洛模拟算法**：

```python
def monte_carlo_max_experts(batch_size, chips, iterations=1000):
    max_experts_list = []
    experts_per_chip = 256 // chips

    for _ in range(iterations):
        chip_experts = [set() for _ in range(chips)]

        # 模拟 batch_size 个 token 的路由
        for _ in range(batch_size):
            selected_experts = random.sample(range(256), 8)  # Top-8
            for expert_id in selected_experts:
                chip_id = expert_id // experts_per_chip
                chip_experts[chip_id].add(expert_id)

        # 统计最忙的芯片
        max_experts = max(len(experts) for experts in chip_experts)
        max_experts_list.append(max_experts)

    return sum(max_experts_list) / len(max_experts_list)
```

**验证结果**：

| Batch | Chips | 表值 | 模拟值 | 误差 |
|-------|-------|------|--------|------|
| 4 | 32 | 3.18 | 3.19 | 0.31% |
| 64 | 32 | 8.00 | 8.00 | 0.00% |
| 256 | 32 | 8.00 | 8.00 | 0.00% |

**影响**：

| 场景 | 理想假设 | 负载均衡 | 改善 |
|------|---------|---------|------|
| Decode (batch=4, EP=32) | 8 专家/芯片 | 3.18 专家 | **-60.2%** 延迟 |
| Prefill (batch=256, EP=32) | 8 专家/芯片 | 8.0 专家 | 无影响 |

**参考**：`backend/llm_simulator/evaluators/moe_load_balance.py`

---

## 5. MoE 负载均衡实现

### 5.1 实现文件

**核心模块**：
```
backend/llm_simulator/evaluators/moe_load_balance.py  # 负载均衡查询
backend/llm_simulator/layers/moe.py                   # MoE 层使用
backend/tests/test_moe_load_balance.py                # 蒙特卡洛验证
backend/tests/test_moe_integration.py                 # 集成测试
```

### 5.2 API 接口

**主查询接口**：

```python
def get_max_expert_load(
    batch_size: int,
    chips: int,
    allow_simulation: bool = True,
    simulation_iterations: int = 1000
) -> float:
    """
    获取最忙芯片需要加载的专家数

    Args:
        batch_size: token 数量
        chips: EP 芯片数
        allow_simulation: 是否允许蒙特卡洛模拟
        simulation_iterations: 模拟迭代次数

    Returns:
        最忙芯片需要加载的专家个数（浮点数）
    """
```

**便捷接口**：

```python
def get_max_expert_load_for_moe_layer(
    batch_size: int,
    ep_parallelism: int,
    num_experts: int = 256,
    topk: int = 8
) -> float:
    """针对 MoE 层的便捷接口，包含参数验证"""

def estimate_moe_expert_load_impact(
    batch_size: int,
    chips: int
) -> Dict[str, float]:
    """返回负载统计（max_experts, avg_experts, load_factor）"""
```

### 5.3 集成到 MoE 层

**修改前**：

```python
# backend/llm_simulator/layers/moe.py (旧代码)
experts_per_ep = num_experts // ep  # 256 // 32 = 8
activated_tokens = tokens * num_activated // ep

routed_gate_op = MatMulOperator(
    ...,
    parallel_params={
        'G': experts_per_ep,      # 假设均匀分布
        'M': activated_tokens,
        ...
    }
)
```

**修改后**：

```python
# backend/llm_simulator/layers/moe.py (新代码)
from ..evaluators import get_max_expert_load_for_moe_layer

# 查表获取最忙芯片的专家数
max_experts_float = get_max_expert_load_for_moe_layer(
    batch_size=tokens,
    ep_parallelism=ep,
    num_experts=num_experts,
    topk=num_activated
)

# GEMM 的 G 维度必须是整数，向上取整
max_experts_per_chip = math.ceil(max_experts_float)

# 每专家平均处理的 tokens
tokens_per_expert = (tokens * num_activated) // num_experts

routed_gate_op = MatMulOperator(
    ...,
    parallel_params={
        'G': max_experts_per_chip,  # 使用负载均衡后的值
        'M': tokens_per_expert,     # 调整 M 维度
        ...
    }
)
```

### 5.4 适用范围

**✅ 适用**：
- DeepSeek V3 / R1 (256 专家，Top-8 路由)
- Decode 阶段（小 batch，负载不均严重）

**❌ 不适用**：
- Mixtral 8×7B (8 专家，Top-2) → 需要重新模拟
- 其他专家数/TopK 配置 → 需要重新生成表

---

## 6. 性能优化机制

### 6.1 多级缓存

**1. GEMM 评估器内部缓存**：
```python
# GEMMEvaluator._cache
cache_key = (G, M, K, N, input_dtype, output_dtype)
if cache_key in self._cache:
    return self._cache[cache_key]  # 命中
```

**2. 全局缓存（跨实验）**：
```python
# simulator.py
global_cache = {}  # 跨多个 simulate 调用复用

analyzer = PerformanceAnalyzer(model, tpu, deploy_cfg, global_cache)
```

**3. GEMM 预热**：
```python
# backend/llm_simulator/gemm_prewarm.py
# 离线预调优常见 GEMM 形状，生成缓存文件
COMMON_SHAPES = [
    (1, 4096, 7168, 18432),   # MLP gate
    (1, 4096, 18432, 7168),   # MLP down
    # ...
]

prewarm_gemm(arch, COMMON_SHAPES, output_file='gemm_cache.json')
```

### 6.2 并行评估

**多进程 GEMM 搜索**：
```python
# gemm_eval.py
with Pool(processes=cpu_count()) as pool:
    results = pool.starmap(evaluate_partition, tasks)
```

**批量算子评估**：
```python
# 对相同类型的算子批量评估，复用缓存
for op_type, operators in grouped_operators.items():
    evaluator = get_evaluator(op_type)
    for op in operators:
        evaluator.evaluate(op)  # 自动复用缓存
```

### 6.3 缓存统计

```python
# 获取缓存统计
gemm_eval = get_gemm_evaluator(arch)
stats = gemm_eval.get_cache_stats()

print(f"缓存命中率: {stats['hit_rate_percent']:.1f}%")
print(f"总搜索时间: {stats['total_search_time_ms']:.2f}ms")
```

---

## 7. 与 DS_TPU 对齐情况

### 7.1 已对齐功能

| 功能模块 | DS_TPU | Tier6 | 对齐状态 |
|---------|--------|-------|---------|
| **GEMM 评估** | ✅ | ✅ | ✅ 完全对齐 |
| **FlashAttention** | ✅ | ✅ | ✅ 完全对齐 |
| **AllReduce/AllGather** | ✅ | ✅ | ✅ 完全对齐 |
| **MLA 变体** | ✅ 4 种 | ✅ 4 种 | ✅ 完全对齐 |
| **MoE 负载均衡** | ✅ | ✅ | ✅ **新对齐** |
| **通信延迟配置** | ✅ | ✅ | ✅ **新对齐** (CommLatencyConfig) |
| **通信协议 1** | ✅ | ✅ | ✅ 完全对齐 |
| **通信协议 2/3** | ✅ | ⚠️ | ⚠️ 需验证 |
| **MoE TBO 重叠** | ✅ | ❌ | ⚠️ 未实现 |

### 7.2 评估精度对比

**GEMM 测试案例**：

| 形状 (G, M, K, N) | DS_TPU (μs) | Tier6 (μs) | 误差 |
|------------------|-------------|------------|------|
| (1, 4096, 7168, 18432) | 125.3 | 125.3 | 0.0% |
| (4, 128, 7168, 2048) | 8.7 | 8.7 | 0.0% |
| (8, 64, 7168, 2048) | 6.2 | 6.2 | 0.0% |

**MoE 负载均衡**：

| Batch | Chips | DS_TPU 专家数 | Tier6 专家数 | 误差 |
|-------|-------|--------------|-------------|------|
| 4 | 32 | 3.18 | 3.18 | 0.0% |
| 64 | 32 | 8.00 | 8.00 | 0.0% |
| 256 | 256 | 1.00 | 1.00 | 0.0% |

### 7.3 待对齐项

**1. MoE Dispatch/Combine 重叠**：
- DS_TPU 显式建模 TBO (Tensor-Bus Overlap)
- Tier6 简化为芯片级重叠
- **影响**：MoE 层延迟误差 10-20%

**2. 通信协议 2/3 验证**：
- 需要验证 RTT 延迟参数是否正确
- 需要测试流水线模式

---

## 8. 配置参数

### 8.1 模型配置

```yaml
model:
  name: "DeepSeek-V3"
  hidden_dim: 7168
  num_layers: 61
  num_heads: 128

  # MLA 参数
  q_lora_rank: 1536
  kv_lora_rank: 512
  qk_nope_head_dim: 128
  qk_rope_head_dim: 64
  v_head_dim: 128

  # MoE 参数
  num_experts: 256
  num_activated_experts: 8
  num_shared_experts: 1
  expert_inter_dim: 2048
```

### 8.2 部署配置

```yaml
deployment:
  batch_size: 64
  q_seq_len: 4096        # Prefill: 4096, Decode: 1
  kv_seq_len: 4096

  # 并行策略
  tp: 1                  # Tensor Parallelism
  dp: 32                 # Data Parallelism
  pp: 1                  # Pipeline Parallelism
  moe_tp: 1              # MoE Tensor Parallelism
  ep: 32                 # Expert Parallelism

  # 通信配置
  comm_protocol: 1       # 1: post-write, 2: non-post-write, 3: pipeline
  enable_tp_sp: false    # TP Sequence Parallelism

  is_prefill: true       # true: Prefill, false: Decode
```

### 8.3 通信延迟配置 (CommLatencyConfig)

**统一配置接口**：前端使用单一 `CommLatencyConfig` 对象配置所有通信延迟参数，通过 API 传递给后端。

```typescript
// frontend/src/utils/llmDeployment/types.ts
interface CommLatencyConfig {
  // === 协议相关 (Protocol) ===
  rtt_tp_us: number;              // TP 往返延迟 (μs)，默认 0.35
  rtt_ep_us: number;              // EP 往返延迟 (μs)，默认 0.85
  bandwidth_utilization: number;  // 带宽利用率 (0-1)，默认 0.95
  sync_latency_us: number;        // 同步延迟 (μs)，默认 0.0

  // === 网络基础设施 (Network Infrastructure) ===
  switch_delay_us: number;        // 交换机延迟 (μs)，默认 1.0
  cable_delay_us: number;         // 线缆延迟 (μs)，默认 0.025

  // === 芯片延迟 (Chip Latency) ===
  chip_to_chip_us: number;        // 芯片间延迟 (μs)，默认 0.2
  memory_read_latency_us: number; // 内存读延迟 (μs)，默认 0.15
  memory_write_latency_us: number;// 内存写延迟 (μs)，默认 0.01
  noc_latency_us: number;         // NoC 延迟 (μs)，默认 0.05
  die_to_die_latency_us: number;  // Die 间延迟 (μs)，默认 0.04
}
```

**数据流**：

```
前端 UI 输入
    ↓
commLatencyConfig (React State)
    ↓
fullTopology.comm_latency_config (API Request)
    ↓
后端 simulator.py 提取
    ↓
内部转换为 ProtocolConfig + NetworkInfraConfig
    ↓
通信评估器 (comm_eval.py) 使用
```

**启动延迟公式**：

| 通信类型 | start_lat 公式 |
|---------|----------------|
| **AllReduce (TP)** | `2×c2c + ddr_r + ddr_w + noc + 2×d2d` |
| **Dispatch/Combine (EP)** | `2×c2c + ddr_r + ddr_w + noc + 2×d2d + 2×switch + 2×cable` |

其中：
- `c2c` = `chip_to_chip_us`
- `ddr_r` = `memory_read_latency_us`
- `ddr_w` = `memory_write_latency_us`
- `noc` = `noc_latency_us`
- `d2d` = `die_to_die_latency_us`
- `switch` = `switch_delay_us`
- `cable` = `cable_delay_us`

**默认值 (与 DS_TPU 对齐)**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rtt_tp_us` | 0.35 | TP 组内 RTT |
| `rtt_ep_us` | 0.85 | EP 组内 RTT |
| `bandwidth_utilization` | 0.95 | 实际/理论带宽 |
| `switch_delay_us` | 1.0 | IB 交换机 |
| `cable_delay_us` | 0.025 | 铜缆/光缆 |
| `chip_to_chip_us` | 0.2 | 板内芯片互联 |
| `memory_read_latency_us` | 0.15 | HBM 读延迟 |
| `memory_write_latency_us` | 0.01 | HBM 写延迟 |
| `noc_latency_us` | 0.05 | 片内 NoC |
| `die_to_die_latency_us` | 0.04 | 多 Die 封装 |

---

### 8.4 硬件配置

```python
# SG2260E 配置 (默认)
SG2260E_ARCH = AcceleratorMicroArch(
    num_cores=8,
    cube_m=32,
    cube_n=32,
    cube_k=64,
    macs_per_cycle=32 * 32,
    freq_ghz=1.6,
    sram_size_kb=2048,
    effective_sram_bytes=int(2048 * 1024 * 0.45),
    dma_bandwidth_per_core=273e9,  # 273 GB/s (HBM3)
    lane_num=32,
    align_bytes=128,
    compute_dma_overlap_rate=0.7,
)
```

---

## 9. 使用示例

### 9.1 基本使用

```python
from llm_simulator import LLMInferenceSimulator

# 创建模拟器
simulator = LLMInferenceSimulator(arch_preset='sg2260e')

# 配置
config = {
    'model': {
        'name': 'DeepSeek-V3',
        'hidden_dim': 7168,
        'num_layers': 61,
        # ...
    },
    'deployment': {
        'batch_size': 4,
        'q_seq_len': 1,      # Decode
        'tp': 1,
        'ep': 32,
        'is_prefill': False,
    }
}

# 运行模拟
result = simulator.simulate(config)

print(f"总延迟: {result['total_latency_us'] / 1000:.2f} ms")
print(f"MFU: {result['mfu'] * 100:.1f}%")
print(f"TPOT: {result['tpot_us']:.2f} μs")
```

### 9.2 MoE 负载均衡查询

```python
from llm_simulator.evaluators import (
    get_max_expert_load,
    estimate_moe_expert_load_impact
)

# 查询最忙芯片的专家数
max_experts = get_max_expert_load(batch_size=4, chips=32)
print(f"最忙芯片加载: {max_experts:.2f} 个专家")  # 3.18

# 获取详细统计
impact = estimate_moe_expert_load_impact(batch_size=4, chips=32)
print(f"负载因子: {impact['load_factor']:.2f}x")  # 3.18x
```

### 9.3 GEMM 预热

```bash
# 预调优常见 GEMM 形状
python -m llm_simulator.gemm_prewarm \
    --arch sg2260e \
    --output cache/gemm_sg2260e.json

# 使用预热缓存
simulator = LLMInferenceSimulator(
    arch_preset='sg2260e',
    gemm_cache_file='cache/gemm_sg2260e.json'
)
```

### 9.4 性能分析

```python
# 获取层级性能分解
for layer in result['layers']:
    print(f"\nLayer {layer['name']}:")
    print(f"  延迟: {layer['latency_us']:.2f} μs")
    print(f"  计算: {layer['compute_us']:.2f} μs")
    print(f"  通信: {layer['comm_us']:.2f} μs")
    print(f"  流量: {layer['dram_traffic_gb']:.2f} GB")
```

---

## 10. 附录

### 10.1 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| **MFU** | Model FLOPs Utilization | 模型 FLOPs 利用率 |
| **MBU** | Model Bandwidth Utilization | 模型带宽利用率 |
| **TTFT** | Time To First Token | 首 token 延迟 (Prefill) |
| **TPOT** | Time Per Output Token | 单 token 延迟 (Decode) |
| **TP** | Tensor Parallelism | 张量并行 |
| **PP** | Pipeline Parallelism | 流水线并行 |
| **DP** | Data Parallelism | 数据并行 |
| **EP** | Expert Parallelism | 专家并行 |
| **TP-SP** | TP Sequence Parallelism | 序列并行 |
| **MLA** | Multi-head Latent Attention | 多头潜在注意力 |
| **MoE** | Mixture of Experts | 混合专家 |
| **TBO** | Tensor-Bus Overlap | 张量总线重叠 |

### 10.2 性能指标公式

**MFU (Model FLOPs Utilization)**：
```python
MFU = total_flops / (peak_flops × total_time_s)
    = total_flops / (num_cores × macs_per_cycle × freq_ghz × 2 × total_time_s × 1e9)
```

**MBU (Model Bandwidth Utilization)**：
```python
MBU = total_dram_traffic / (dram_bandwidth × total_time_s)
```

**TTFT (Time To First Token)**：
```python
TTFT = prefill_latency_ms
```

**TPOT (Time Per Output Token)**：
```python
TPOT = decode_latency_us / batch_size
```

### 10.3 文件结构

```
backend/llm_simulator/
├── evaluators/
│   ├── gemm_eval.py              # GEMM 评估器
│   ├── fa2_eval.py               # FlashAttention 评估器
│   ├── comm_eval.py              # 通信评估器
│   ├── moe_load_balance.py       # MoE 负载均衡 (新)
│   ├── arch_config.py            # 硬件微架构配置
│   └── presets.py                # 预设硬件配置
├── layers/
│   ├── base.py                   # 层基类
│   ├── embedding.py              # Embedding 层
│   ├── attention.py              # MLA/MHA 层
│   ├── moe.py                    # MoE 层 (已修改)
│   ├── ffn.py                    # MLP 层
│   └── lmhead.py                 # LMHead 层
├── operators/
│   ├── base.py                   # 算子基类
│   ├── matmul.py                 # MatMul 算子
│   ├── attention_ops.py          # Attention 算子
│   └── comm_ops.py               # 通信算子
├── simulator.py                  # 主模拟器 (已修改: 接收统一 CommLatencyConfig)
├── gemm_prewarm.py               # GEMM 预热工具
└── types.py                      # 类型定义

backend/tests/
├── test_moe_load_balance.py      # MoE 负载均衡验证 (新)
├── test_moe_integration.py       # MoE 集成测试 (新)
└── test_debug_features.py        # 调试功能测试

frontend/src/
├── utils/
│   ├── llmDeployment/
│   │   └── types.ts              # 类型定义 (已修改: 统一 CommLatencyConfig)
│   └── storage.ts                # 存储模块 (已修改: SavedConfig.comm_latency_config)
└── components/ConfigPanel/DeploymentAnalysis/
    ├── DeploymentAnalysisPanel.tsx         # 部署分析面板 (已修改: 统一配置状态)
    └── components/
        └── ConfigSnapshotDisplay.tsx       # 配置快照展示 (已修改: 统一显示面板)

docs/
├── TIER6_EVALUATION_SPEC.md      # 本文档
└── DS_TPU_Performance_Analysis.md # DS_TPU 分析文档
```

### 10.4 相关资源

**代码仓库**：
- [Tier6-Model](https://github.com/your-org/tier6-model)
- [DS_TPU_1209](c:\Users\DELL\Documents\code\DS_TPU_1209)

**论文参考**：
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [MoE Load Balancing](https://arxiv.org/abs/2408.15664)

---

**最后更新**: 2026-01-26
**维护者**: Tier6-Model Team
**许可**: MIT License
