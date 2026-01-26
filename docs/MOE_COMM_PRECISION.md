# MoE 通信精度说明

## 概述

在 MoE (Mixture of Experts) 层中，Dispatch 和 Combine 通信的精度选择会影响通信数据量和模型精度。本文档说明 Tier6 与 DS_TPU 对齐的精度设置。

## 精度模式对照表

| 精度模式 | Dispatch 精度 | Combine 精度 | Dispatch 字节 | Combine 字节 |
|----------|---------------|--------------|---------------|--------------|
| W8A8     | FP8           | BF16         | 1             | 2            |
| W16A16   | BF16 (待确认) | BF16         | 2             | 2            |

## W8A8 模式 (当前实现)

这是 DS_TPU Benchmark 的默认设置。

### Dispatch (FP8)

```python
dispatch_size = token_per_ep_group * hidden_dim // moe_tp  # 不乘 dtype_bytes
```

- **精度**: FP8 (1 字节)
- **原因**: 发送到专家的激活值可以用低精度传输，减少通信量
- **通信量**: `token_per_ep_group * hidden_dim // moe_tp` 字节

### Combine (BF16)

```python
combine_size = token_per_ep_group * hidden_dim // moe_tp * 2  # 乘 BF16 (2字节)
```

- **精度**: BF16 (2 字节)
- **原因**: 专家计算结果返回时需要高精度，保证模型质量
- **通信量**: `token_per_ep_group * hidden_dim // moe_tp * 2` 字节

## W16A16 模式

> **TODO**: 需要进一步确认 W16A16 模式下 Dispatch 的精度设置。

预期设置：
- Dispatch: BF16 (2 字节) - 待确认
- Combine: BF16 (2 字节)

## 通信量计算公式

### 参数说明

- `token_per_ep_group`: 每个 EP 分组的 token 数
  - 计算: `ceil(global_batch * seq_len * num_activated_experts / ep)`
- `hidden_dim`: 隐藏维度
- `moe_tp`: MoE 张量并行度

### W8A8 模式

```
Dispatch 通信量 = token_per_ep_group * hidden_dim / moe_tp * 1 (FP8)
Combine 通信量  = token_per_ep_group * hidden_dim / moe_tp * 2 (BF16)
总通信量        = Dispatch + Combine
              = token_per_ep_group * hidden_dim / moe_tp * 3
```

### 示例计算

配置:
- `global_batch = 64`
- `seq_len = 1`
- `num_activated_experts = 8`
- `ep = 32`
- `hidden_dim = 7168`
- `moe_tp = 1`

计算:
```
token_per_ep_group = ceil(64 * 1 * 8 / 32) = 16
dispatch_size = 16 * 7168 / 1 = 114688 字节 (FP8)
combine_size  = 16 * 7168 / 1 * 2 = 229376 字节 (BF16)
```

## 对齐 DS_TPU

Tier6 的实现完全对齐 DS_TPU 的 `model/layers/moe.py`:

```python
# DS_TPU 原始代码
dispatch_size = self.token_per_ep_group * self.hidden_dim // self.moe_tp
combine_size = self.token_per_ep_group * self.hidden_dim // self.moe_tp * BF16
```

## 相关文件

- Tier6: `backend/llm_simulator/layers/moe.py`
- DS_TPU: `model/layers/moe.py`

## 待办事项

- [ ] 确认 W16A16 模式下 Dispatch 的精度设置
- [ ] 添加精度模式配置选项 (支持 W8A8/W16A16 切换)
