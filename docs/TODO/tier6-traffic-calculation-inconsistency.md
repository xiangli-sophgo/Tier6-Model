# TODO: Tier6 L3/L4 Traffic 计算不一致问题

## 问题描述

Tier6 框架中 L3 和 L4 的 DRAM traffic 计算逻辑不一致，L4 的实现存在 bug。

## 问题详情

### 1. L4 MatMulPreciseEvaluator 枚举 loop_order 无意义

**文件**: `backend/tier6/L4_evaluation/evaluators/precise.py`

```python
# 第 146-163 行
for order in ["mnk", "nkm", "mkn", "nmk", "kmn", "knm"]:  # 枚举 6 种
    traffic = self._compute_traffic(
        order,  # 传入了 order
        g, tile_m, tile_n, tile_k,
        m_tiles, n_tiles, k_tiles,
        a_bytes, b_bytes, c_bytes, accum_bytes,
    )
    if traffic < best_traffic:
        best_traffic = traffic
        best_order = order
```

但 `_compute_traffic` 没有使用 `order` 参数：

```python
# 第 196-236 行
def _compute_traffic(self, order, ...):  # order 没用！
    a_loads = m_tiles * k_tiles
    b_loads = n_tiles * k_tiles
    c_reads = m_tiles * n_tiles * max(0, k_tiles - 1)
    c_writes = m_tiles * n_tiles * k_tiles

    # 所有 loop_order 计算结果都一样
    return (a_traffic + b_traffic + c_read_traffic + c_write_traffic) * g
```

### 2. L3 和 L4 的 traffic 公式不同

**L3 MatmulTilingEvaluator** (`backend/tier6/L3_mapping/tiling/evaluators.py` 第 142-179 行):

```python
# mnk: A 重复 tiles_n 次, B 重复 tiles_m 次
if order == "mnk":
    return (m * k) * a_bytes * tiles_n
         + (n * k) * b_bytes * tiles_m
         + (m * n) * c_bytes

# nkm: B 只加载一次, 有部分和读写
if order == "nkm":
    return (n * k) * b_bytes
         + (m * k) * a_bytes * tiles_n
         + (m * n) * accum_bytes * 2 * max(tiles_k - 1, 0)
         + (m * n) * c_bytes

# mkn: A 只加载一次, 有部分和读写
return (m * k) * a_bytes
     + (n * k) * b_bytes * tiles_m
     + (m * n) * accum_bytes * 2 * max(tiles_k - 1, 0)
     + (m * n) * c_bytes
```

**L4 MatMulPreciseEvaluator** - 不区分 loop_order，公式完全不同。

### 3. 枚举范围不同

| 枚举项 | L3 MatmulTilingEvaluator | L4 MatMulPreciseEvaluator |
|--------|--------------------------|---------------------------|
| loop_order | 3 种 (mnk, nkm, mkn) | 6 种 (但无意义) |

## 影响

1. **结果不一致**: 如果同时使用 L3 和 L4 评估，traffic 值会不同
2. **L4 枚举浪费**: 枚举 6 种 loop_order 但结果都一样
3. **best_loop_order 错误**: L4 返回的 best_loop_order 没有意义

## 修复方案

### 方案 A: 修复 L4 的 _compute_traffic

让 L4 的 `_compute_traffic` 真正区分 loop_order，参考 L3 的实现：

```python
def _compute_traffic(self, order, g, tile_m, tile_n, tile_k,
                     m_tiles, n_tiles, k_tiles,
                     a_bytes, b_bytes, c_bytes, accum_bytes):
    if order == "mnk":
        # A 重复 n_tiles 次, B 重复 m_tiles 次
        a_traffic = m_tiles * k_tiles * tile_m * tile_k * a_bytes * n_tiles
        b_traffic = n_tiles * k_tiles * tile_k * tile_n * b_bytes * m_tiles
        c_traffic = m_tiles * n_tiles * tile_m * tile_n * c_bytes
        return (a_traffic + b_traffic + c_traffic) * g

    elif order == "nkm":
        # B 只加载一次, 有部分和
        b_traffic = n_tiles * k_tiles * tile_k * tile_n * b_bytes
        a_traffic = m_tiles * k_tiles * tile_m * tile_k * a_bytes * n_tiles
        partial_sum = m_tiles * n_tiles * tile_m * tile_n * accum_bytes * 2 * max(k_tiles - 1, 0)
        c_traffic = m_tiles * n_tiles * tile_m * tile_n * c_bytes
        return (a_traffic + b_traffic + partial_sum + c_traffic) * g

    # ... 其他 order
```

### 方案 B: 统一使用 L3 的逻辑

删除 L4 的 loop_order 枚举，直接使用 L3 传来的 traffic 值。

## 验证方法

1. 单元测试：对比不同 loop_order 下的 traffic 计算结果
2. 回归测试：确保修复后的结果与 llm_simulator 的 `gemm_eval.py` 一致

## 优先级

**中** - 影响评估精度，但不影响基本功能

## 相关文件

- `backend/tier6/L3_mapping/tiling/evaluators.py` - L3 正确实现
- `backend/tier6/L4_evaluation/evaluators/precise.py` - L4 需要修复
- `backend/llm_simulator/evaluators/gemm_eval.py` - 参考实现
