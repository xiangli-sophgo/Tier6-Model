# 后端添加详细内存分解计算

## 目标

在后端计算并返回详细的内存分解数据（模型参数、KV Cache、激活值），取代前端的估算逻辑。

## 现状分析

### 当前内存计算机制

**权重参数内存（已实现）**：
- 从算子级别计算：`MatMul.dram_occupy = G × K × N × dtype_bytes`
- 层级聚合：`layer.dram_occupy = sum(op.dram_occupy for op in layer.comp_ops)`
- 模型级聚合：`model.total_dram_occupy = sum(layer.dram_occupy × count)`
- 数据存在于：`analysis['layers'][layer_name]['perf']['dram_occupy']`

**KV Cache（未实现）**：
- 需要根据推理配置计算：`batch_size × seq_len × num_layers × hidden_size × 2 × dtype_bytes`
- MLA 模型需要特殊处理：`batch_size × seq_len × num_layers × kv_lora_rank × dtype_bytes`

**激活值（未实现）**：
- 推理时的中间计算结果内存占用
- 通常是模型参数的 10%（不同于训练阶段需要保存全部激活值）
- 包括：注意力计算的中间矩阵、FFN的中间激活、残差连接的临时缓冲等

### 前端当前的估算逻辑

**位置**：`frontend/src/pages/Results/index.tsx:415-421`

```typescript
const paramsMemoryGB = memoryBreakdown?.model ?? totalMemoryGB * 0.6
const kvCacheMemoryGB = memoryBreakdown?.kv_cache ?? totalMemoryGB * 0.3
const activationMemoryGB = memoryBreakdown?.activation ?? totalMemoryGB * 0.1
```

**问题**：
- 固定比例 60%/30%/10% 不准确
- MoE、MLA 模型的内存分布差异很大
- 不同 batch size 和 seq_len 下 KV Cache 占比差异很大

## 实施方案

### 1. 后端添加内存分解计算

**文件**：`backend/llm_simulator/tasks/deployment.py`

**位置**：`_transform_to_ds_tpu_format` 函数（第 800-964 行）

#### 1.1 添加内存分解计算函数

在 `deployment.py` 中添加新函数（建议放在第 780 行附近）：

```python
def _calculate_memory_breakdown(
    model_config: dict,
    inference_config: dict,
    parallelism: dict,
    dram_occupy: int,
) -> dict:
    """
    计算详细的内存分解

    Args:
        model_config: 模型配置
        inference_config: 推理配置
        parallelism: 并行策略
        dram_occupy: 总内存占用（字节，来自 PerformanceAnalyzer）

    Returns:
        内存分解字典 (GB 单位)
    """
    # 1. 模型参数内存（来自 dram_occupy）
    model_memory_bytes = dram_occupy
    model_memory_gb = model_memory_bytes / (1024 ** 3)

    # 2. KV Cache 内存计算
    kv_cache_gb = _calculate_kv_cache_memory(
        model_config,
        inference_config,
        parallelism
    )

    # 3. 激活值内存估算
    activation_gb = _calculate_activation_memory(
        model_memory_gb
    )

    # 4. 总内存
    total_gb = model_memory_gb + kv_cache_gb + activation_gb

    return {
        'model': model_memory_gb,
        'kv_cache': kv_cache_gb,
        'activation': activation_gb,
        'total': total_gb,
    }
```

#### 1.2 实现 KV Cache 计算

```python
def _calculate_kv_cache_memory(
    model_config: dict,
    inference_config: dict,
    parallelism: dict,
) -> float:
    """
    计算 KV Cache 内存（GB）

    公式：
    - 标准 Attention: batch_size × seq_len × num_layers × hidden_size × 2 × dtype_bytes / TP
    - MLA: batch_size × seq_len × num_layers × kv_lora_rank × dtype_bytes / TP

    解释：
    - batch_size: 推理批次大小
    - seq_len: input_seq_length + output_seq_length（需要缓存的总序列长度）
    - num_layers: 模型层数
    - hidden_size: 隐层维度
    - 2: Key 和 Value 各一份
    - dtype_bytes: 数据类型字节数（FP16/BF16=2字节）
    - TP: Tensor Parallelism 将 KV Cache 分片
    """
    batch_size = inference_config.get('batch_size', 1)
    seq_len = inference_config.get('input_seq_length', 0) + inference_config.get('output_seq_length', 0)
    num_layers = model_config.get('num_layers', 0)
    hidden_size = model_config.get('hidden_size', 0)

    # 数据类型字节数
    weight_dtype = model_config.get('weight_dtype', 'fp16')
    dtype_bytes = 2 if weight_dtype in ['fp16', 'bf16'] else 4

    # TP 并行会分片 KV Cache
    tp = parallelism.get('tp', 1)

    # 检查是否是 MLA
    mla_config = model_config.get('mla_config')
    if mla_config and mla_config.get('enabled'):
        # MLA: 使用压缩后的 kv_lora_rank
        kv_lora_rank = mla_config.get('kv_lora_rank', hidden_size // 4)
        kv_cache_bytes = batch_size * seq_len * num_layers * kv_lora_rank * dtype_bytes
    else:
        # 标准 Attention: K 和 V 各一份
        kv_cache_bytes = batch_size * seq_len * num_layers * hidden_size * 2 * dtype_bytes

    # TP 分片
    kv_cache_bytes_per_chip = kv_cache_bytes / tp

    return kv_cache_bytes_per_chip / (1024 ** 3)
```

#### 1.3 实现激活值内存估算

```python
def _calculate_activation_memory(
    model_memory_gb: float,
) -> float:
    """
    估算激活值内存（GB）

    推理时激活值是什么？
    - 推理过程中前向传播产生的中间计算结果
    - 例如：注意力的 Q×K^T 矩阵、Softmax输出、FFN的Up Projection输出等
    - 这些中间张量在当前层计算完成前必须保留在显存中

    为什么需要激活值开销？
    - 虽然推理不需要反向传播（不像训练需要保存全部激活值）
    - 但计算过程中仍需要临时存储中间结果（否则无法进行计算）
    - 推理时激活值是临时的，层级间可复用内存，开销相对较小

    估算方法：
    - 根据经验数据，推理时激活值约为模型参数的 10%
    - 这基于典型的 Transformer 架构和优化策略
    - 实际值会因为 batch size、sequence length、模型架构而略有变化

    Args:
        model_memory_gb: 模型参数内存（GB）

    Returns:
        激活值内存（GB）
    """
    # 根据经验公式：推理时激活值约为参数内存的 10%
    activation_gb = model_memory_gb * 0.10

    return activation_gb
```

#### 1.4 集成到返回结果

修改 `_transform_to_ds_tpu_format` 函数（第 943-964 行）：

```python
def _transform_to_ds_tpu_format(...) -> dict:
    """将 Tier6-Model 模拟结果转换为 DS_TPU 格式"""

    # ... 现有代码 ...

    # 提取 dram_occupy（第 872 行）
    dram_occupy = analysis.get("dram_occupy", 0)

    # 计算详细的内存分解
    memory_breakdown = _calculate_memory_breakdown(
        model_config=model_config,
        inference_config=inference_config,
        parallelism=parallelism,
        dram_occupy=dram_occupy,
    )

    return {
        "parallelism": parallelism,
        "chips": chips,
        "is_feasible": True,
        # ... 其他字段 ...
        "dram_occupy": dram_occupy,
        "memory_breakdown": memory_breakdown,  # 新增字段
        "stats": stats,
        "gantt_chart": sim_result.get("ganttChart"),
    }
```

### 2. 前端使用后端返回的分解数据

**文件**：`frontend/src/pages/Results/index.tsx`

**位置**：第 418-421 行

**修改前**：
```typescript
const memoryBreakdown = stats.memory_breakdown as Record<string, number> | undefined
const paramsMemoryGB = memoryBreakdown?.model ?? totalMemoryGB * 0.6
const kvCacheMemoryGB = memoryBreakdown?.kv_cache ?? totalMemoryGB * 0.3
const activationMemoryGB = memoryBreakdown?.activation ?? totalMemoryGB * 0.1
```

**修改后**：
```typescript
// 优先使用后端计算的详细分解
const memoryBreakdown = plan.memory_breakdown as Record<string, number> | undefined

if (!memoryBreakdown) {
  console.error('后端未返回 memory_breakdown，数据不完整')
  return []  // 或使用其他错误处理策略
}

const paramsMemoryGB = memoryBreakdown.model
const kvCacheMemoryGB = memoryBreakdown.kv_cache
const activationMemoryGB = memoryBreakdown.activation
const totalMemoryGB = memoryBreakdown.total
```

**同时移除 `is_estimated` 标记**（第 477 行）：
```typescript
// 删除这行，因为现在都是精确计算
// is_estimated: !memoryBreakdown,
```

**前端显示逻辑**（ChartsPanel.tsx 第 229-233 行）：
```typescript
// 删除 "估算值" 标签显示
{(result.memory as any)?.is_estimated && (
  <span className="text-[10px] text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
    估算值
  </span>
)}
```

### 3. 更新类型定义

**文件**：`frontend/src/utils/llmDeployment/types.ts`

**删除 `is_estimated` 字段**（第 377 行附近）：

```typescript
export interface MemoryAnalysis {
  model_memory_gb: number;
  kv_cache_memory_gb: number;
  activation_memory_gb: number;
  overhead_gb: number;
  total_per_chip_gb: number;
  is_memory_sufficient: boolean;
  memory_utilization: number;
  // 删除：is_estimated?: boolean;  // 不再需要
}
```

## 关键文件清单

### 后端文件（必须修改）
1. `backend/llm_simulator/tasks/deployment.py`
   - 添加 `_calculate_memory_breakdown()` 函数（~780 行）
   - 添加 `_calculate_kv_cache_memory()` 函数（~800 行）
   - 添加 `_calculate_activation_memory()` 函数（~840 行）
   - 修改 `_transform_to_ds_tpu_format()` 函数（~943-964 行）

### 前端文件（必须修改）
1. `frontend/src/pages/Results/index.tsx`
   - 修改内存分解数据读取逻辑（~418-421 行）
   - 移除 `is_estimated` 标记（~477 行）

2. `frontend/src/components/ConfigPanel/DeploymentAnalysis/charts/ChartsPanel.tsx`
   - 移除 "估算值" 标签显示（~229-233 行）

3. `frontend/src/utils/llmDeployment/types.ts`
   - 删除 `is_estimated` 字段定义（~377 行）

## 验证方案

### 1. 单元测试（后端）

创建测试文件：`backend/tests/test_memory_breakdown.py`

```python
def test_calculate_kv_cache_memory():
    """测试 KV Cache 内存计算"""
    model_config = {
        'num_layers': 60,
        'hidden_size': 4096,
        'weight_dtype': 'fp16',
    }
    inference_config = {
        'batch_size': 1,
        'input_seq_length': 1024,
        'output_seq_length': 1024,
    }
    parallelism = {'tp': 8}

    kv_cache_gb = _calculate_kv_cache_memory(
        model_config, inference_config, parallelism
    )

    # 预期：1 × 2048 × 60 × 4096 × 2 × 2 / 8 / (1024^3) ≈ 14.6 GB
    assert 14.0 < kv_cache_gb < 15.0

def test_calculate_kv_cache_memory_mla():
    """测试 MLA 模型的 KV Cache 内存计算"""
    model_config = {
        'num_layers': 61,
        'hidden_size': 7168,
        'weight_dtype': 'fp16',
        'mla_config': {
            'enabled': True,
            'kv_lora_rank': 512,
        }
    }
    inference_config = {
        'batch_size': 1,
        'input_seq_length': 4096,
        'output_seq_length': 1024,
    }
    parallelism = {'tp': 8}

    kv_cache_gb = _calculate_kv_cache_memory(
        model_config, inference_config, parallelism
    )

    # 预期：1 × 5120 × 61 × 512 × 2 / 8 / (1024^3) ≈ 3.75 GB
    assert 3.5 < kv_cache_gb < 4.0

def test_calculate_activation_memory():
    """测试激活值内存计算"""
    model_memory_gb = 100.0  # 模型参数 100GB

    activation_gb = _calculate_activation_memory(model_memory_gb)

    # 预期：100 × 10% = 10GB
    assert activation_gb == 10.0
```

### 2. 端到端测试

**步骤**：
1. 启动后端：`cd backend && python -m llm_simulator.main`
2. 运行评估任务（使用 DeepSeek-V3 模型）
3. 检查返回的 `memory_breakdown` 字段：
   ```json
   {
     "memory_breakdown": {
       "model": 65.05,
       "kv_cache": 32.53,
       "activation": 6.51,
       "total": 104.09
     }
   }
   ```
4. 前端显示内存分解，确认：
   - 不再显示 "估算值" 标签
   - 三部分内存数值合理
   - 内存充足/溢出判断正确

### 3. 验证不同场景

| 场景 | 模型 | 配置 | 预期 KV Cache | 预期激活值 |
|------|------|------|--------------|-----------|
| 标准 | LLaMA-70B | bs=1, seq=2K, params=140GB | ~35 GB | ~14 GB |
| 大 batch | LLaMA-70B | bs=8, seq=2K, params=140GB | ~280 GB | ~14 GB |
| MLA 压缩 | DeepSeek-V3 | bs=1, seq=4K, params=65GB | ~3.8 GB | ~6.5 GB |
| MoE 大激活 | Mixtral-8x7B | bs=1, seq=2K, params=45GB | ~15 GB | ~4.5 GB |

## 潜在风险和注意事项

### 1. KV Cache 计算准确性
- **风险**：不同并行策略下的分片逻辑可能不同
- **缓解**：仔细验证 TP/PP/DP 对 KV Cache 的影响

### 2. 激活值估算误差
- **风险**：激活值是基于 10% 经验值的估算，可能与实际有偏差
- **缓解**：10% 是保守估算，实际值通常会小于这个值

### 3. MLA 特殊处理
- **风险**：MLA 的 KV Cache 计算逻辑复杂
- **缓解**：参考 DeepSeek-V3 论文，使用 `kv_lora_rank` 而不是 `hidden_size`

### 4. 向后兼容性
- **风险**：旧的评估结果没有 `memory_breakdown` 字段
- **缓解**：前端保留降级逻辑（检查字段存在性）

## 实施顺序

1. **阶段1**：后端实现内存分解计算
   - 添加 `_calculate_memory_breakdown()`
   - 添加 `_calculate_kv_cache_memory()`
   - 添加 `_calculate_activation_memory()`
   - 修改 `_transform_to_ds_tpu_format()`

2. **阶段2**：后端单元测试
   - 测试标准 Attention KV Cache 计算
   - 测试 MLA KV Cache 计算
   - 测试激活值估算

3. **阶段3**：前端适配
   - 修改 Results/index.tsx 数据读取
   - 移除 ChartsPanel.tsx 估算值标签
   - 更新类型定义

4. **阶段4**：集成测试
   - 运行完整的评估任务
   - 验证前端显示正确
   - 测试不同模型和配置

## 关键概念解释

### 激活值与 KV Cache 的区别

| 方面 | 激活值 | KV Cache |
|------|--------|----------|
| **定义** | 前向传播的中间计算结果 | 注意力机制缓存的 Key/Value 投影 |
| **生命周期** | 临时（层级内计算完成后可复用） | 持久（整个推理过程保留） |
| **大小** | ~10% 的模型参数 | 与 batch_size、seq_len、num_layers 成正比 |
| **用途** | 完成当前计算步骤 | 避免重复计算历史token的注意力 |
| **示例** | Q×K^T 矩阵、FFN中间激活 | 历史token的Key、Value向量 |

### 内存三部分的组成

1. **模型参数**（Model）
   - 权重矩阵和偏置项
   - 从后端 PerformanceAnalyzer 计算得出（dram_occupy）
   - 大小固定，与推理配置无关

2. **KV Cache**
   - 自回归生成过程中缓存的注意力向量
   - 大小与 batch_size、input_seq_length、output_seq_length 成正比
   - 推理过程中持续增长（prefill 到 decode）

3. **激活值**
   - 前向传播过程中的临时中间结果
   - 大小与模型参数成正比（约 10%）
   - 与推理配置关系不大（不同 batch/seq_len 变化不大）
