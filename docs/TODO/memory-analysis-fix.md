# 内存（显存）分析修复方案

## 问题概述

当前内存分析数据不正确，存在三套独立且不一致的内存计算，且实际返回给前端的只是一个粗略的 `dram_occupy` 数字，没有分解信息。

## 现状诊断

### 三套不一致的内存计算

| 位置 | 用途 | 问题 |
|------|------|------|
| `L5_reporting/memory_analysis.py` (MemoryAnalyzer) | 独立分析工具 | **死代码**，从未被调用 |
| `L4_evaluation/engine.py` (_aggregate_metrics) | 实际流水线中的内存统计 | 仅累加 step 的 `local_weight_bytes`，无 KV Cache/激活值 |
| `L0_entry/engine.py` (_calculate_deployment_cost) | 成本计算中的模型大小估算 | 仅用于成本，不返回给前端 |

### 前端期望 vs 后端实际返回

**前端 MemoryPieChart 期望**:
```typescript
interface MemoryAnalysis {
  model_memory_gb: number;           // 模型参数显存
  kv_cache_memory_gb: number;        // KV Cache 显存
  activation_memory_gb: number;      // 激活值显存
  overhead_gb: number;               // 其他开销
  total_per_chip_gb: number;         // 每芯片总显存需求
  is_memory_sufficient: boolean;     // 是否超出显存限制
  memory_utilization: number;        // 显存利用率 (0-1)
}
```

**后端实际返回**: 仅一个 `dram_occupy`（字节数），无分解。

## 真实 LLM 推理显存组成

### 1. 模型权重 (50-80%)

```
Weight_Memory = total_params * bytes_per_param
```

**精度对照**:

| 精度 | bytes/param | 70B 模型 |
|------|-------------|----------|
| FP32 | 4 | 280 GB |
| FP16/BF16 | 2 | 140 GB |
| FP8 | 1 | 70 GB |
| INT8 | 1 | 70 GB |
| INT4 | 0.5 | 35 GB |

**标准 Transformer 每层参数**:
```
attn_params = H * (H + 2 * kv_heads * head_dim) + H * H   # Q + K + V + O (GQA)
ffn_params  = 3 * H * I                                     # SwiGLU: gate + up + down
ln_params   = 2 * H                                         # LayerNorm (可忽略)
embedding   = vocab_size * H                                 # 词表嵌入
```

**MoE 模型关键**:
- 所有专家必须全部加载到显存，不是只加载激活的专家
- 不同 token 路由到不同专家，路由是动态的
- `expert_weights = num_experts * (3 * H * expert_dim)` (SwiGLU)
- `non_expert_weights = attention + shared_experts + embedding + lm_head`

**并行度对权重的影响**:
```
weight_per_gpu = (non_expert_weights / (TP * PP)) + (expert_weights / (TP * PP * EP))
```
- TP: 切分权重矩阵
- PP: 按层分割
- EP: 仅切分专家权重，非专家部分完整复制
- DP: 不减少权重（完整复制）

### 2. KV Cache (10-40%)

**标准 MHA/GQA**:
```
KV_Cache = 2 * batch * seq_len * layers * kv_heads * head_dim * bytes_per_elem
```

**MLA (DeepSeek-V2/V3/R1)**:
```
KV_Cache_MLA = batch * seq_len * layers * (kv_lora_rank + qk_rope_dim) * bytes_per_elem
```
- 没有因子 2（K 和 V 共享压缩向量 c_t）
- DeepSeek-V3: (512 + 64) / (2 * 7168) = 4%, 节省约 96%

**并行度对 KV Cache 的影响**:
- TP: kv_heads / TP (标准模式); MLA 模式下不按 head 分割
- PP: layers / PP

### 3. 激活值 (1-10%)

推理时只需一层的峰值激活:
```
activation_peak = batch * seq_len * max(hidden_size, intermediate_size / TP) * bytes_per_elem
```
- Prefill 阶段: seq_len = prompt_length, 激活值较大
- Decode 阶段: seq_len = 1, 激活值极小

### 4. 开销 (~15-20%)

- CUDA 上下文: 300-800 MB/GPU
- 通信 buffer (NCCL): 256MB-1GB
- 内存分配器碎片: 5-10%
- 实用估算: `overhead = 0.15 * (weights + kv_cache)`

## 现有 MemoryAnalyzer 的具体缺陷

| 缺陷 | 严重性 | 说明 |
|------|--------|------|
| 不支持 MoE 全专家加载 | **严重** | MoE 模型显存差数倍 |
| 不支持 MLA 压缩 KV Cache | **严重** | DeepSeek-V3 KV Cache 差 25 倍 |
| 不支持 PP 层分割 | 中等 | PP>1 时每 GPU 显存计算错误 |
| 不支持 EP 专家分割 | 中等 | EP>1 时专家显存计算错误 |
| KV Cache 未计入实际流水线 | **严重** | 前端完全没有 KV Cache 数据 |
| 激活值未计入实际流水线 | 中等 | 推理时占比较小，但不应缺失 |
| `analyze_from_model()` 使用默认值 | 低 | 违反项目规范 |
| `lm_head_weights` 计算后未使用 | 低 | 死变量 |

## 实现计划

### Step 1: 重写 MemoryAnalyzer

**文件**: `L5_reporting/memory_analysis.py`

重写核心计算方法，使其支持所有模型架构和并行策略:

#### 1.1 权重计算 `calculate_weights_memory()`

输入参数:
```python
# 模型参数
hidden_size: int
num_layers: int
intermediate_size: int
vocab_size: int
num_attention_heads: int
num_kv_heads: int       # GQA
head_dim: int

# MoE 参数 (可选)
moe_enabled: bool
num_experts: int
num_shared_experts: int
expert_intermediate_size: int  # 每个专家的 FFN 中间维度

# 并行度
tp: int
pp: int
ep: int

# 精度
dtype_bytes: int  # FP16=2, FP8=1, INT8=1, INT4=0.5
```

计算逻辑:
```python
# 每层注意力参数 (GQA)
attn_per_layer = H * (H + 2 * kv_heads * head_dim) + H * H  # Q + KV + O

# 每层非专家 FFN (如果 MoE，则为共享专家)
if moe_enabled:
    shared_ffn = num_shared_experts * 3 * H * expert_intermediate_size
    expert_ffn = num_experts * 3 * H * expert_intermediate_size
    router = H * num_experts  # 路由器参数
else:
    shared_ffn = 3 * H * I
    expert_ffn = 0
    router = 0

# LayerNorm
ln_per_layer = 4 * H  # 2 个 LayerNorm, 每个 weight + bias

# 非专家参数 (每层)
non_expert_per_layer = attn_per_layer + shared_ffn + ln_per_layer + router

# 总参数
non_expert_total = num_layers * non_expert_per_layer + vocab_size * H * 2  # embed + lm_head
expert_total = num_layers * expert_ffn  # 仅 MoE 层有专家

# 并行度分割
non_expert_per_gpu = non_expert_total / (tp * pp)
expert_per_gpu = expert_total / (tp * pp * ep) if moe_enabled else 0

weight_bytes_per_gpu = (non_expert_per_gpu + expert_per_gpu) * dtype_bytes
```

#### 1.2 KV Cache 计算 `calculate_kv_cache_memory()`

输入参数:
```python
# 模型参数
num_layers: int
num_kv_heads: int
head_dim: int

# MLA 参数 (可选)
mla_enabled: bool
kv_lora_rank: int
qk_rope_dim: int

# 推理参数
batch_size: int
seq_len: int  # prompt_length + output_length

# 并行度
tp: int
pp: int

# 精度
dtype_bytes: int
```

计算逻辑:
```python
layers_per_gpu = num_layers // pp

if mla_enabled:
    # MLA: K 和 V 共享压缩向量, 无因子 2
    kv_per_token_per_layer = (kv_lora_rank + qk_rope_dim) * dtype_bytes
    # MLA 模式下不按 kv_heads 分割 TP
    kv_cache_per_gpu = batch_size * seq_len * layers_per_gpu * kv_per_token_per_layer
else:
    # 标准 GQA
    kv_heads_per_gpu = num_kv_heads // tp
    kv_per_token_per_layer = 2 * kv_heads_per_gpu * head_dim * dtype_bytes
    kv_cache_per_gpu = batch_size * seq_len * layers_per_gpu * kv_per_token_per_layer
```

#### 1.3 激活值计算 `calculate_activation_memory()`

```python
layers_per_gpu = num_layers // pp

# Prefill 阶段峰值 (处理完整 prompt)
prefill_activation = batch_size * prompt_length * max(
    hidden_size,                      # 隐藏层输出
    intermediate_size // tp,          # FFN 中间层
    num_attention_heads // tp * head_dim  # 注意力输出
) * dtype_bytes

# Decode 阶段峰值 (单 token)
decode_activation = batch_size * 1 * max(hidden_size, intermediate_size // tp) * dtype_bytes

# 取两者中较大的 (通常是 prefill)
activation_per_gpu = max(prefill_activation, decode_activation)
```

#### 1.4 开销估算 `calculate_overhead()`

```python
overhead = 0.15 * (weight_bytes_per_gpu + kv_cache_per_gpu)
# 下限 500MB, 上限 4GB
overhead = max(500 * 1024**2, min(overhead, 4 * 1024**3))
```

### Step 2: 集成到仿真流水线

**修改文件**: `L0_entry/engine.py`

在仿真完成后调用 MemoryAnalyzer:

```python
# 在 run_evaluation_from_request() 中, L4 评估完成后:
memory_analyzer = MemoryAnalyzer()
memory_breakdown = memory_analyzer.analyze(
    # 从 EvalConfig / model_config / parallelism_config 中提取参数
    ...
)
```

将 `memory_breakdown` 加入返回结果:
```python
plan["memory"] = memory_breakdown.to_dict()
# to_dict() 返回:
# {
#   "model_memory_gb": ...,
#   "kv_cache_memory_gb": ...,
#   "activation_memory_gb": ...,
#   "overhead_gb": ...,
#   "total_per_chip_gb": ...,
#   "is_memory_sufficient": ...,  # 对比芯片显存容量
#   "memory_utilization": ...     # total / chip_capacity
# }
```

### Step 3: 更新 MemoryBreakdown.to_dict() 输出格式

确保输出格式与前端 `MemoryPieChart` 的 `MemoryAnalysis` 接口完全匹配:

```python
def to_dict(self, chip_memory_capacity_gb: float) -> dict:
    total_gb = self.total_bytes / (1024**3)
    return {
        "model_memory_gb": self.weights_bytes / (1024**3),
        "kv_cache_memory_gb": self.kv_cache_bytes / (1024**3),
        "activation_memory_gb": self.activations_bytes / (1024**3),
        "overhead_gb": self.overhead_bytes / (1024**3),
        "total_per_chip_gb": total_gb,
        "is_memory_sufficient": total_gb <= chip_memory_capacity_gb,
        "memory_utilization": total_gb / chip_memory_capacity_gb,
    }
```

### Step 4: 清理旧代码

- 删除 `L4_evaluation/engine.py` 中基于 `local_weight_bytes` 的粗略内存累加逻辑（或保留为 `compute_memory_peak` 仅用于调试）
- 修复 `analyze_from_model()` 中的默认值问题（违反项目规范）
- 删除 `lm_head_weights` 死变量

## 验证方法

### 对照计算

用已知模型验证计算结果:

**LLaMA-3 70B (BF16, TP=4, PP=1)**:
- 权重: ~140GB / 4 = 35 GB/GPU
- KV Cache (batch=32, seq=4096): 2 * 32 * 4096 * 80 * 8 * 128 * 2 / 4 = ~10 GB/GPU
- 合计约 48-50 GB/GPU (H100-80GB 可容纳)

**DeepSeek-V3 671B (FP8, TP=8, EP=4, PP=1)**:
- 非专家权重: ~105GB / 8 = ~13 GB/GPU
- 专家权重: ~595GB / 32 = ~18.6 GB/GPU
- KV Cache (MLA, batch=128, seq=4096): 128 * 4096 * 61 * 576 * 1 / 8 = ~2.3 GB/GPU
- 合计约 37-40 GB/GPU

## 涉及文件

| 文件 | 修改类型 |
|------|----------|
| `L5_reporting/memory_analysis.py` | **重写** - 核心计算逻辑 |
| `L0_entry/engine.py` | **修改** - 集成 MemoryAnalyzer，返回分解数据 |
| `L4_evaluation/engine.py` | **清理** - 移除粗略内存累加 |
| `L4_evaluation/metrics.py` | **可选** - Aggregates 增加 memory_breakdown 字段 |
