# DS_TPU 对齐分析报告

## 1. 关键发现

### 1.1 11.28us 的真实来源

**之前的误解：**
我们以为 DS_TPU 的 11.28us 是 MoE Routed Gate GEMM 的总延迟。

**实际情况：**
通过分析 `DS_TPU_1209/results/evaluation.json`，发现 **11.28us 是 DMA 延迟**（内存访问时间），不是总延迟！

```json
{
  "name": "routed_gate_proj",
  "parallel_params": {"G":8, "M":48, "K":7168, "N":2048},
  "elapsed": 16.589714285714287,        // 总延迟
  "comp_elapsed": 14.336,                // 计算延迟
  "dma_elapsed": 11.268571428571429     // DMA 延迟 ← 11.28us!
}
```

### 1.2 DS_TPU 的实际配置

**配置信息：**
- batch_size: 1536 (全局 batch)
- dp: 32
- 本地 batch: 1536 / 32 = 48
- ep: 32
- seq_len: 1 (decode)
- 本地 tokens: 48

**MoE GEMM 参数：**
- G = 8 (最忙芯片专家数)
- M = 48 (每专家处理的 token 数)
- K = 7168 (hidden_dim)
- N = 2048 (expert_inter_dim)

**性能结果：**
- 总延迟: 16.59 us
- 计算延迟: 14.34 us
- DMA 延迟: 11.27 us
- DRAM 流量: 129.24 MB

## 2. Tier6 vs DS_TPU 对比

### 2.1 相同 GEMM 参数对比 (G=8, M=48)

| 指标 | DS_TPU | Tier6 | 差异 |
|------|--------|-------|------|
| 总延迟 | 16.59 us | 24.92 us | **1.50x** |
| DMA 延迟 | 11.27 us | - | - |
| 计算延迟 | 14.34 us | - | - |
| DRAM 流量 | 129.24 MB | 252.18 MB | **1.95x** |

### 2.2 差异分析

#### 延迟差异 (1.5x)
可能的原因：
1. Tile 选择策略不同
2. 内存访问模式优化差异
3. Cache 利用率不同
4. 并行度和负载均衡差异

#### 流量差异 (2x)
**关键问题：DS_TPU 的流量是 Tier6 的一半！**

理论流量计算（bf16）：
- A 张量: G × M × K × 2 = 8 × 48 × 7168 × 2 = 5.5 MB
- B 张量: G × K × N × 2 = 8 × 7168 × 2048 × 2 = 236 MB
- C 张量: G × M × N × 2 = 8 × 48 × 2048 × 2 = 1.57 MB
- **理论总计: ≈ 243 MB**

Tier6 报告 252 MB，接近理论值。
DS_TPU 报告 129 MB，是理论值的一半。

**可能的原因：**
1. DS_TPU 只计算写入流量（不计算读取）
2. DS_TPU 利用 L2 Cache，减少了实际 DRAM 流量
3. DS_TPU 的流量计算模型只计算 C 张量的写回
4. DS_TPU 使用了特殊的内存访问优化（如 tensor core 的寄存器复用）

## 3. 配置 A 的疑问

### 3.1 之前的测试

**Tier6 配置 A：**
- batch_size: 4 (本地)
- dp: 1
- 全局 batch: 4
- G = 4, M = 1

**Tier6 结果：**
- 延迟: 10.75 us
- 流量: 117.69 MB

**我们对比的 DS_TPU "参考值"：**
- 延迟: 11.28 us
- 流量: 118.42 MB

### 3.2 问题

1. **这个 11.28us 是从哪里来的？**
   - 在 DS_TPU 的所有结果文件中，没有找到 batch=4, dp=1 的测试
   - 11.28us 可能是另一个配置的 DMA 延迟
   - 需要确认原始参考值的来源

2. **为什么当时认为对齐？**
   - 如果 11.28us 是 DMA 延迟，不应该与 Tier6 的总延迟对比
   - 可能需要重新验证配置 A 的对齐情况

## 4. DS_TPU 代码分析

### 4.1 MoE 层参数计算

DS_TPU 的 `model/layers/moe.py` (第 76-82, 137, 161 行)：

```python
self.batch_size = deploy_config['batch_size']  # 全局 batch
self.batch_size_local = self.batch_size // self.dp
self.seqs = self.q_seq_len * self.batch_size_local

# 使用全局 batch 计算
self.token_per_ep_group = ceil_div(
    self.batch_size * self.q_seq_len * self.n_activated_experts,  # 全局!
    self.ep
)

self.activated_expert_per_ep_group = max(
    1,
    get_max_expert(self.batch_size, self.ep)  # 全局 batch 查表！
)
```

**结论：DS_TPU 与 Tier6 (修复后) 使用相同的参数计算方式。**

### 4.2 默认配置

DS_TPU `main.py` (第 37-51 行)：

```python
DEFAULT_CONFIG = {
    'tp': 1,
    'dp': 32,
    'moe_tp': 1,
    'ep': 32,
    'batch_size': 48*32,  # = 1536 (全局 batch)
    'q_len': 8192,
    'kv_len': 8192,
    'is_prefill': False,
}
```

## 5. 结论与建议

### 5.1 已验证的一致性

1. ✓ MoE 参数计算方式完全一致（使用全局 batch）
2. ✓ GEMM 参数 (G, M) 的计算逻辑一致
3. ✓ 负载均衡查表方式一致

### 5.2 存在的差异

1. **延迟差异 (1.5x)**
   - Tier6 比 DS_TPU 慢约 50%
   - 需要进一步优化 Tile 选择和内存访问

2. **流量差异 (2x)**
   - **关键问题**：需要理解 DS_TPU 的流量计算模型
   - 可能需要调整 Tier6 的流量计算方式
   - 或者 DS_TPU 利用了 Cache，实际流量更少

### 5.3 下一步行动

**立即行动：**
1. 找到配置 A (batch=4, dp=1) 的原始参考值来源
2. 验证 DS_TPU 的流量计算模型
   - 是否只计算写入流量？
   - 是否考虑了 Cache？
3. 分析 DS_TPU 的 Tile 选择策略
4. 对比 DS_TPU 和 Tier6 的 Cache 模型

**需要外部信息：**
1. DS_TPU 的流量计算公式和假设
2. DS_TPU 的 Cache 层次结构和大小
3. DS_TPU 测试时的实际硬件配置
4. 配置 A 参考值的测试日志

### 5.4 临时结论

**当前状态：**
- Tier6 的 MoE 实现与 DS_TPU 在逻辑上完全一致
- 在相同 GEMM 参数下，Tier6 延迟是 DS_TPU 的 1.5x
- Tier6 流量是 DS_TPU 的 2x，这是需要深入分析的关键问题

**建议的对比方式：**
- 不应该用 11.28us (DMA 延迟) 作为参考值
- 应该用 DS_TPU 的总延迟 (16.59us) 进行对比
- 需要理解 DS_TPU 的流量计算模型后，再对比流量

## 6. 测试数据汇总

### 6.1 DS_TPU 实际测试结果

**文件：** `DS_TPU_1209/results/evaluation.json`

```json
{
  "config": {
    "batch_size": 1536,
    "dp": 32,
    "ep": 32,
    "tp": 1
  },
  "MoE_routed_gate_proj": {
    "GEMM": "G=8, M=48, K=7168, N=2048",
    "elapsed": 16.59,
    "comp_elapsed": 14.34,
    "dma_elapsed": 11.27,
    "dram_traffic": 129.24
  }
}
```

### 6.2 Tier6 对应测试

**配置：**
```python
batch_size = 48  # 本地 (等价于 DS_TPU 的 1536/32)
dp = 32
G = 8, M = 48, K = 7168, N = 2048
```

**结果：**
```
延迟: 24.92 us
流量: 252.18 MB
Tile: (48, 1024, 384)
Partition: (8, 1, 2, 4)
```

**对比：**
- 延迟比: 24.92 / 16.59 = 1.50x
- 流量比: 252.18 / 129.24 = 1.95x
