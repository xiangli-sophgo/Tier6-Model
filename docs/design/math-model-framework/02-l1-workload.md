# L1 Workload -- 负载建模层

## 功能概述

L1 负责将模型定义抽象为硬件无关的 WorkloadIR，包括:
- **模型定义**: 从 ModelConfig 构建层结构 (DeepSeekV3Model, LlamaModel)
- **层与算子**: Layer 包含 Op 列表，Op 携带 FLOPs/字节/shape 信息
- **IR 输出**: WorkloadIR 提供统一的分析接口 (OpsBreakdown, MemoryFootprint)

不在范围: 不涉及硬件参数、并行切分、性能评估。

## 模块清单

| 模块 | 职责 |
|------|------|
| `ir.py` | WorkloadIR 协议 + Model 实现 |
| `layer.py` | Layer / Module dataclass |
| `op.py` | Op dataclass |
| `tensor.py` | TensorDesc (shape, dtype, bytes) |
| `specs.py` | ComputeSpec, MemorySpec, CommSpec |
| `metadata.py` | ModelMetadata |
| `graph.py` | WorkloadGraph (节点 + 边) |
| `breakdown.py` | OpsBreakdown, MemoryFootprint |
| `models/llm/deepseek.py` | DeepSeek V3 模型 |
| `models/llm/llama.py` | LLaMA 模型 |
| `layers/` | 各层实现 |
| `operators/` | 各算子实现 |

## 核心数据结构

### WorkloadIR 协议

```python
class WorkloadIR(Protocol):
    def get_layers(self) -> list[Layer]: ...
    def get_ops_breakdown(self) -> OpsBreakdown: ...
    def get_memory_footprint(self) -> MemoryFootprint: ...
    def get_communication_pattern(self) -> CommPattern: ...
```

### Layer

```python
@dataclass
class Layer:
    name: str                    # "layers.0.mla"
    op_type: str                 # "mla", "ffn", "moe", "embedding"
    inputs: list[TensorDesc]     # 输入张量描述
    outputs: list[TensorDesc]    # 输出张量描述
    ops: list[Op]                # 计算算子列表
    comm: CommSpec | None        # 分布式通信模式提示
    params: dict[str, Any]       # 层参数 (hidden_size, num_heads, ...)
```

### Op

```python
@dataclass
class Op:
    name: str                    # "layers.0.mla.q_proj"
    op_type: str                 # "matmul", "softmax", "rmsnorm"
    compute: ComputeSpec         # cube_ops, vector_ops, scalar_ops
    memory: MemorySpec           # weight_bytes, activation_bytes, read/write
    inputs: list[TensorDesc]
    outputs: list[TensorDesc]
    attrs: dict[str, Any]        # 额外属性 (dtype_bytes, local_weight_bytes)
```

### ComputeSpec / MemorySpec

```python
@dataclass
class ComputeSpec:
    cube_ops: int       # GEMM 运算量 (FLOPs)
    vector_ops: int     # 向量运算量
    scalar_ops: int     # 标量运算量
    hau_ops: int        # HAU 运算量

@dataclass
class MemorySpec:
    weight_bytes: int       # 权重大小
    activation_bytes: int   # 激活大小
    read_bytes: int         # 总读取量
    write_bytes: int        # 总写入量
```

## DeepSeek V3 模型

### 模型结构

```
Embedding
  |
Dense Layers (x num_dense_layers):
  +-- MLA (Multi-head Latent Attention)
  +-- FFN (Dense Feed-Forward)
  |
MoE Layers (x num_moe_layers):
  +-- MLA (Multi-head Latent Attention)
  +-- MoE (Mixture of Experts)
  |
LMHead
```

### from_model_config() classmethod

将结构化 `ModelConfig` 转为 layer 层期望的 flat dict:

```python
@classmethod
def from_model_config(cls, mc: ModelConfig) -> DeepSeekV3Model:
    config = {
        "hidden_size": mc.hidden_size,
        "num_heads": mc.num_attention_heads,   # key rename
        "q_lora_rank": mc.mla.q_lora_rank,     # 从嵌套提取
        "n_routed_experts": mc.moe.num_routed_experts,  # key rename
        # ... 完整映射
    }
    return cls(config)
```

这样 layer 层 (EmbeddingLayer, MLALayer 等) 无需修改。

### MLA (Multi-head Latent Attention)

DeepSeek V3 特有的注意力机制:
- **LoRA 压缩**: Q 用 q_lora_rank=1536 低秩投影，KV 用 kv_lora_rank=512
- **RoPE 混合**: qk_rope_head_dim=64 (旋转) + qk_nope_head_dim=128 (非旋转)
- **KV Cache 高效**: 压缩后 KV Cache 减小 5-8x
- **两种变体**:
  - `MLALayer`: Prefill 阶段 (完整 QKV 投影)
  - `MLAAbsorbLayer`: Decode 阶段 (absorbed 投影)

### MoE (Mixture of Experts)

- **路由**: num_routed_experts=256, num_activated_experts=8 (Top-8)
- **共享专家**: num_shared_experts=1 (始终激活)
- **通信**: EP 并行需要 All2All dispatch/combine
- **FFN 结构**: 每个专家包含独立的 gate_proj, up_proj, down_proj

## 层实现清单

| 层 | 文件 | 包含的算子 |
|------|------|------------|
| EmbeddingLayer | `layers/embedding.py` | matmul (token -> hidden) |
| MLALayer | `layers/mla.py` | Q/KV LoRA proj, QK matmul, softmax, V matmul, output proj |
| MLAAbsorbLayer | `layers/mla_absorb.py` | Absorbed variant for decode |
| FFNLayer | `layers/ffn.py` | gate_proj, up_proj, SiLU, down_proj |
| MoELayer | `layers/moe.py` | router, dispatch, expert FFN x activated, combine |
| LMHeadLayer | `layers/lmhead.py` | matmul (hidden -> vocab) |
| DSALayer | `layers/dsa.py` | Dynamic Sparse Attention |

## Op 计算量估算

### MatMul (GEMM)

```
FLOPs = 2 * M * N * K
Weight bytes = K * N * dtype_bytes
Activation bytes = M * K * dtype_bytes
Output bytes = M * N * dtype_bytes
```

### Softmax

```
Vector ops = 5 * seq_len * seq_len * num_heads
Memory: 2 * seq_len * seq_len * num_heads * dtype_bytes (读+写)
```

### RMSNorm

```
Vector ops = 3 * batch * seq_len * hidden_size
Memory: 2 * batch * seq_len * hidden_size * dtype_bytes
```

## 通信模式

Layer 通过 `CommSpec` 声明分布式通信需求:

```python
@dataclass
class CommSpec:
    pattern: str        # "allreduce", "allgather", "p2p", "all2all"
    tensor_bytes: int   # 通信数据量
    participants: int   # 参与芯片数
    scope: str          # "tp_group", "dp_group", "ep_group"
```

实际通信算子由 L3 ParallelismPlanner 插入。
