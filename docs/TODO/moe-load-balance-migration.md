# TODO: MoE 负载均衡模块迁移

## 问题描述

Tier6 框架缺少 MoE 负载均衡查表功能，导致对 MoE 模型（如 DeepSeek V3）的计算量估算不准确。

### 当前问题

Tier6 的 `L1_workload/layers/moe.py` 直接使用 `n_activated_experts`（如 8）作为 GEMM 的 G 维度：

```python
# tier6/L1_workload/layers/moe.py 第 235 行
ops.append(
    self._matmul_op(
        f"{self.name}_expert_gate",
        {"G": n_activated, "M": tokens, "K": hidden, "N": moe_inter},  # G=8
    )
)
```

但实际上，由于路由随机性和 EP 并行，每个芯片实际加载的专家数是不同的。

### 示例

| batch_size | chips (EP) | Tier6 假设 G | 实际 G (查表) | 误差 |
|------------|------------|--------------|---------------|------|
| 4 | 32 | 8 | 3.18 | 2.5x 高估 |
| 64 | 32 | 8 | 8.0 | 准确 |
| 4 | 1 | 8 | 30.5 | 3.8x 低估 |

## 解决方案

从 `llm_simulator/evaluators/moe_load_balance.py` 迁移负载均衡模块到 Tier6。

### 源文件

```
backend/llm_simulator/evaluators/moe_load_balance.py (568 行)
```

### 核心功能

1. **硬编码查找表** (`MAX_EXPERT_TABLE`)
   - 预计算的 batch_size × chips 映射表
   - 覆盖 batch=4~256, chips=1~256

2. **线性插值** (`_interpolate_batch`, `_interpolate_chips`)
   - 扩展查表覆盖范围
   - chips 维度使用对数插值

3. **蒙特卡洛模拟** (`monte_carlo_max_experts`)
   - 兜底方案，处理表外配置
   - 10000 次迭代，误差 < 1%

4. **主查询接口** (`get_max_expert_load`)
   - 三级回退：查表 -> 插值 -> 模拟

### 关键数据结构

```python
MAX_EXPERT_TABLE: Dict[int, Dict[int, float]] = {
    # batch_size: {chips: max_experts}
    4: {
        1: 30.5121,
        32: 3.18425,
        256: 1.0
    },
    # ...
}
```

## 迁移计划

### 目标位置

```
backend/tier6/L4_evaluation/cost_models/moe_balance.py
```

### 集成点

1. **L1_workload/layers/moe.py**
   - `_build_ops()` 方法中调用 `get_max_expert_load(batch, ep_chips)`
   - 用返回值替代固定的 `n_activated`

2. **L3_mapping/parallelism/planner.py**
   - EP 并行规划时考虑负载均衡

3. **L4_evaluation/evaluators/compute.py**
   - MoE 层计算评估时使用实际专家数

### 接口设计

```python
# tier6/L4_evaluation/cost_models/moe_balance.py

def get_max_expert_load(
    batch_size: int,
    ep_chips: int,
    num_experts: int = 256,
    topk: int = 8,
    allow_simulation: bool = True,
) -> float:
    """
    获取最忙芯片需要加载的专家数。

    Args:
        batch_size: 当前处理的 token 数量
        ep_chips: EP 并行度（芯片数）
        num_experts: 专家总数（默认 256）
        topk: 每 token 激活专家数（默认 8）
        allow_simulation: 是否允许蒙特卡洛模拟

    Returns:
        最忙芯片需要加载的专家个数（浮点数）
    """
```

## 验证方法

1. 单元测试：对比迁移后的查表结果与原模块
2. 集成测试：DeepSeek V3 模型评估结果对比
3. 边界测试：极端 batch/chips 配置

## 优先级

**高** - 影响所有 MoE 模型的评估准确性

## 相关文件

- 源: `backend/llm_simulator/evaluators/moe_load_balance.py`
- 目标: `backend/tier6/L4_evaluation/cost_models/moe_balance.py`
- 集成: `backend/tier6/L1_workload/layers/moe.py`

## 参考

- DS_TPU_1209/model.py:get_max_expert()
- DeepSeek V3 配置：256 专家，Top-8 路由
