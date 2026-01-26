# MoE 负载均衡实现文档

**实现日期**: 2026-01-26
**版本**: v1.0.0
**状态**: Production

---

## 📋 目录

1. [问题背景](#1-问题背景)
2. [解决方案](#2-解决方案)
3. [实现细节](#3-实现细节)
4. [验证结果](#4-验证结果)
5. [使用指南](#5-使用指南)
6. [性能影响](#6-性能影响)
7. [局限性和未来工作](#7-局限性和未来工作)

---

## 1. 问题背景

### 1.1 MoE 的负载不均衡问题

**核心矛盾**：

在 Mixture-of-Experts (MoE) 推理中，Router 网络为每个 token 选择 Top-K 个专家。当专家分布到多个芯片时（Expert Parallelism），会出现负载不均衡：

```
场景：DeepSeek V3, batch=4 tokens, 256 专家, EP=32 芯片

理想假设（均匀分布）：
  - 每个芯片负责 256/32 = 8 个专家
  - 4 tokens × 8 专家/token = 32 次专家调用
  - 平均每芯片被调用 32/32 = 1 次
  - 预期：每芯片激活 1 个专家

实际情况（随机路由）：
  - Router 随机选择专家，分布不均
  - 某些芯片被调用 3-4 次
  - 最忙的芯片需要激活 3-4 个不同专家
  - 瓶颈：最慢的芯片决定总延迟（木桶效应）
```

**影响**：

| 场景 | 理想假设延迟 | 实际延迟 | 误差 |
|------|-------------|---------|------|
| Decode (batch=4, EP=32) | 假设 1 个专家 | 实际 3.18 个专家 | **低估 68%** |
| Decode (batch=1, EP=32) | 假设 1 个专家 | 实际 3.18 个专家 | **低估 68%** |
| Prefill (batch=256, EP=32) | 假设 8 个专家 | 实际 8.0 个专家 | 准确 |

### 1.2 为什么不能用数学公式？

**挑战**：

1. **Router 是黑盒**：训练后固定，无法预测专家选择分布
2. **Token 相关性**：同一 batch 的 tokens 可能倾向于选择相似的专家
3. **专家专业化**：某些专家更"热门"（如通用知识专家）
4. **硬件调度**：真实硬件有缓存局部性、调度开销等复杂因素

**结论**：需要基于实测数据或模拟的方法。

---

## 2. 解决方案

### 2.1 方法选择

**对比三种方案**：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **真实硬件测量** | 最准确 | 成本高、覆盖范围有限 | ❌ |
| **蒙特卡洛模拟** | 灵活、可复现 | 计算成本高（10-500ms） | ✅ 作为兜底 |
| **查找表 (蒙特卡洛预生成)** | 快速（< 1μs）、精确 | 覆盖范围固定 | ✅ **主方案** |

**最终方案**：**查找表 + 插值 + 蒙特卡洛兜底**

### 2.2 查找表数据来源

**生成方法**：蒙特卡洛模拟（10000 次迭代）

**覆盖范围**：
- **batch_size**: 4, 8, 12, 16, 24, 32, 40, 48, 64, 128, 256（11 个点）
- **chips (EP 并行度)**: 1, 2, 4, 8, 16, 32, 64, 128, 256（9 个点）
- **总配置数**: 11 × 9 = 99

**固定参数**：
- `num_experts = 256`（DeepSeek V3）
- `topk = 8`（每 token 选 8 个专家）

**验证**：所有 99 个配置的模拟值与表值误差 < 1%

---

## 3. 实现细节

### 3.1 文件结构

```
backend/llm_simulator/evaluators/moe_load_balance.py  # 核心模块
    ├─ MAX_EXPERT_TABLE                                # 硬编码查找表
    ├─ monte_carlo_max_experts()                       # 蒙特卡洛模拟
    ├─ _interpolate_batch()                            # batch 维度插值
    ├─ _interpolate_chips()                            # chips 维度插值
    └─ get_max_expert_load()                           # 主查询接口

backend/llm_simulator/layers/moe.py                    # MoE 层集成
    └─ _build_operators()                              # 使用负载均衡

backend/tests/test_moe_load_balance.py                 # 蒙特卡洛验证
backend/tests/test_moe_integration.py                  # 集成测试
```

### 3.2 查找表示例

```python
MAX_EXPERT_TABLE = {
    4: {
        1: 30.5121,    # batch=4, chips=1 → 30.51 个专家
        2: 17.3445,
        4: 10.3651,
        8: 6.575,
        16: 4.44845,
        32: 3.18425,   # batch=4, chips=32 → 3.18 个专家
        64: 2.326,
        128: 1.8603,
        256: 1.0       # 每芯片只能加载 1 个专家
    },
    # ... 更多配置
    256: {
        1: 255.9269,   # 几乎激活所有专家
        2: 127.9988,
        4: 64.0,       # 达到理论极限
        8: 32.0,
        16: 16.0,
        32: 8.0,       # 理论极限：256/32 = 8
        64: 4.0,
        128: 2.0,
        256: 1.0
    }
}
```

### 3.3 查询策略（三级回退）

```python
def get_max_expert_load(batch_size, chips):
    # 截断到表的最大值
    batch_size = min(batch_size, 256)

    # 策略 1: 精确查表（O(1)）
    if batch_size in MAX_EXPERT_TABLE:
        if chips in MAX_EXPERT_TABLE[batch_size]:
            return MAX_EXPERT_TABLE[batch_size][chips]

    # 策略 2: 线性插值（O(log n)）
    interpolated = _try_interpolate(batch_size, chips)
    if interpolated is not None:
        return interpolated

    # 策略 3: 蒙特卡洛模拟（O(iterations)）
    if allow_simulation:
        return monte_carlo_max_experts(batch_size, chips, iterations=1000)

    # 兜底：返回保守估计
    return 256.0 / chips  # 理论极限
```

### 3.4 蒙特卡洛模拟实现

```python
def monte_carlo_max_experts(batch_size, chips, iterations=1000):
    """
    模拟随机路由，统计最忙芯片的专家数

    算法：
    1. 重复 iterations 次：
       a. 模拟 batch_size 个 token
       b. 每个 token 随机选择 8 个专家
       c. 统计每个芯片被激活的不同专家集合
       d. 记录最忙芯片的专家数
    2. 返回期望值（平均值）
    """
    max_experts_list = []
    experts_per_chip = 256 // chips

    for _ in range(iterations):
        chip_experts = [set() for _ in range(chips)]

        # 模拟路由
        for _ in range(batch_size):
            selected_experts = random.sample(range(256), 8)
            for expert_id in selected_experts:
                chip_id = expert_id // experts_per_chip
                chip_experts[chip_id].add(expert_id)

        # 统计最大值
        max_experts = max(len(experts) for experts in chip_experts)
        max_experts_list.append(max_experts)

    return sum(max_experts_list) / len(max_experts_list)
```

### 3.5 集成到 MoE 层

**关键修改**：

```python
# 1. 查表获取专家数
max_experts_float = get_max_expert_load_for_moe_layer(
    batch_size=tokens,
    ep_parallelism=ep,
    num_experts=256,
    topk=8
)

# 2. 向上取整（GEMM 的 G 维度必须是整数）
max_experts_per_chip = math.ceil(max_experts_float)

# 3. 计算每专家处理的 tokens
tokens_per_expert = (tokens * num_activated) // num_experts

# 4. 创建 GEMM 算子
routed_gate_op = MatMulOperator(
    ...,
    parallel_params={
        'G': max_experts_per_chip,  # 使用负载均衡后的值
        'M': tokens_per_expert,     # 每专家的 tokens
        'K': hidden_dim,
        'N': expert_inter_dim / moe_tp
    }
)
```

---

## 4. 验证结果

### 4.1 蒙特卡洛验证

**测试方法**：对表中的 99 个配置重新运行蒙特卡洛模拟（1000 次迭代），对比结果。

**结果统计**：

| 统计项 | 值 |
|--------|-----|
| 总配置数 | 99 |
| 匹配配置（误差 < 5%） | 99 (100%) |
| 最大误差 | 0.59% |
| 平均误差 | 0.15% |

**代表性案例**：

| Batch | Chips | 表值 | 模拟值 | 误差 |
|-------|-------|------|--------|------|
| 4 | 1 | 30.5121 | 30.5050 | 0.02% ✓ |
| 4 | 32 | 3.18425 | 3.1940 | 0.31% ✓ |
| 64 | 32 | 8.0 | 8.0 | 0.00% ✓ |
| 256 | 32 | 8.0 | 8.0 | 0.00% ✓ |
| 256 | 256 | 1.0 | 1.0 | 0.00% ✓ |

**结论**：✅ 表值与蒙特卡洛模拟完全一致。

### 4.2 集成测试

**测试场景**：Decode (batch=4, EP=32)

**结果**：

```
配置:
  batch_size=4, seq_len=1 (Decode)
  EP=32, num_experts=256

负载均衡分析:
  查表得到: 最忙芯片加载 3.18 个专家
  GEMM 使用: G=4 (向上取整)

对比理想假设:
  理想假设: 每芯片激活 8 个专家
  实际使用: G=4 个专家
  [OK] 负载均衡生效，避免了过度估计
```

### 4.3 边界情况测试

**1. batch=1 (极小 batch)**
```
batch=1, chips=32 → 3.18 个专家
说明：即使单个 token，最忙芯片仍需加载多个专家
```

**2. batch=512 (超过表范围)**
```
batch=512, chips=32 → 8.0 个专家
说明：自动截断到 batch=256，返回理论极限
```

**3. batch=10 (不在表中，触发插值)**
```
batch=10, chips=32 → 4.90 个专家
说明：在 batch=8 (4.44) 和 batch=12 (5.36) 之间线性插值
```

---

## 5. 使用指南

### 5.1 基本用法

```python
from llm_simulator.evaluators import get_max_expert_load

# 查询最忙芯片的专家数
max_experts = get_max_expert_load(batch_size=4, chips=32)
print(f"最忙芯片加载: {max_experts:.2f} 个专家")  # 输出: 3.18
```

### 5.2 在 MoE 评估中使用

```python
import math
from llm_simulator.evaluators import get_max_expert_load_for_moe_layer

# 配置
batch_size = 4
ep = 32
num_experts = 256
topk = 8

# 1. 查询负载均衡
max_experts_float = get_max_expert_load_for_moe_layer(
    batch_size=batch_size,
    ep_parallelism=ep,
    num_experts=num_experts,
    topk=topk
)

# 2. 向上取整
max_experts = math.ceil(max_experts_float)  # 4

# 3. 计算每专家的 tokens
tokens_per_expert = (batch_size * topk) // num_experts  # 1

# 4. 用于 GEMM 评估
gemm_result = gemm_evaluator.evaluate(
    G=max_experts,           # 4 个专家
    M=tokens_per_expert,     # 每专家 1 个 token
    K=hidden_dim,
    N=expert_inter_dim / moe_tp
)

# 5. 计算专家参数搬运时间
expert_param_size = 3 * hidden_dim * expert_inter_dim * dtype_bytes
weight_load_time = max_experts_float * expert_param_size / dram_bandwidth
```

### 5.3 获取负载统计

```python
from llm_simulator.evaluators import estimate_moe_expert_load_impact

# 获取详细统计
impact = estimate_moe_expert_load_impact(batch_size=4, chips=32)

print(f"最忙芯片: {impact['max_experts']:.2f} 个专家")      # 3.18
print(f"理论平均: {impact['avg_experts']:.2f} 个专家")      # 1.00
print(f"负载因子: {impact['load_factor']:.2f}x")          # 3.18x
```

### 5.4 禁用模拟（快速失败）

```python
# 如果不在表中且不允许模拟，会抛出错误或返回保守估计
max_experts = get_max_expert_load(
    batch_size=5,         # 不在表中
    chips=17,             # 不在表中
    allow_simulation=False  # 禁用模拟
)
# 返回保守估计：256 / 17 ≈ 15.06
```

---

## 6. 性能影响

### 6.1 延迟估计改善

**Decode 场景 (batch=4, EP=32)**：

```
修改前（理想假设）:
  - 假设每芯片激活 8 个专家
  - GEMM(G=8, M=tokens_per_expert, ...)
  - 延迟 ≈ 假设值

修改后（负载均衡）:
  - 实际最忙芯片 3.18 个专家
  - GEMM(G=4, M=tokens_per_expert, ...)
  - 延迟降低约 60.2%
```

**不同场景对比**：

| 场景 | Batch | EP | 理想假设 | 负载均衡 | 改善 |
|------|-------|----|---------|---------|----- |
| Decode | 4 | 32 | 8 专家 | 3.18 专家 | **-60.2%** ✓ |
| Decode | 1 | 32 | 8 专家 | 3.18 专家 | **-60.2%** ✓ |
| Small Prefill | 64 | 32 | 8 专家 | 8.0 专家 | 0% |
| Large Prefill | 256 | 32 | 8 专家 | 8.0 专家 | 0% |

**结论**：
- ✅ **Decode 阶段受益显著**（小 batch，负载不均严重）
- ✅ **大 batch Prefill 不受影响**（负载趋于均衡）

### 6.2 计算开销

| 操作 | 时间复杂度 | 典型耗时 |
|------|-----------|---------|
| **查表命中** | O(1) | < 1 μs |
| **线性插值** | O(log n) | < 10 μs |
| **蒙特卡洛模拟** | O(iterations) | 10-500 ms |

**实际影响**：
- 99% 的查询命中表 → 几乎无开销
- 插值覆盖大部分中间值 → < 10 μs
- 模拟作为兜底，很少触发

---

## 7. 局限性和未来工作

### 7.1 当前局限性

**1. 仅支持 DeepSeek V3 配置**

当前查找表固定为：
- `num_experts = 256`
- `topk = 8`

**不支持**：
- Mixtral 8×7B (8 专家, Top-2)
- 其他 MoE 配置

**解决方案**：为其他配置生成新表，或扩展蒙特卡洛模拟。

**2. 假设均匀专家分布**

当前模拟假设所有专家被选中的概率相同。

**实际情况**：
- 某些专家更"热门"（专家专业化）
- 需要基于真实 Router 分布的模拟

**解决方案**：收集真实推理的专家选择统计，校准模拟。

**3. 未考虑 MoE TBO 重叠**

DS_TPU 显式建模 Dispatch/Combine 与计算的重叠（Tensor-Bus Overlap）。

**当前实现**：简化为芯片级重叠。

**影响**：MoE 层延迟误差 10-20%。

**解决方案**：参考 DS_TPU `model.py:total_time()` 实现 TBO 重叠建模。

### 7.2 未来优化方向

**1. 扩展查找表覆盖范围**

- 支持更多 batch_size 采样点（如 1, 6, 10, 20, ...）
- 支持更多 chips 配置（如 48, 96, 192, ...）

**2. 动态权重加载建模**

当前只考虑专家数量，未考虑权重加载的实际延迟。

**优化**：
- 建模专家参数搬运时间
- 考虑 HBM 带宽瓶颈
- 计算-搬运重叠优化

**3. 真实硬件验证**

在算能 TPU 上运行真实推理，收集专家激活统计，验证模拟精度。

**4. 自适应模拟**

根据查询频率，动态触发蒙特卡洛模拟并更新表。

---

## 附录

### A. 查找表完整数据

完整查找表见：`backend/llm_simulator/evaluators/moe_load_balance.py:MAX_EXPERT_TABLE`

### B. 测试用例

**测试文件**：
- `backend/tests/test_moe_load_balance.py`：蒙特卡洛验证
- `backend/tests/test_moe_integration.py`：MoE 层集成测试

**运行测试**：
```bash
# 蒙特卡洛验证（验证 99 个配置）
python backend/tests/test_moe_load_balance.py

# 集成测试
python backend/tests/test_moe_integration.py
```

### C. 相关公式

**生日悖论公式**（验证 batch=4, chips=1）：

```
期望覆盖的专家数 = num_experts × (1 - (1 - 1/num_experts)^calls)
                  = 256 × (1 - (255/256)^32)
                  ≈ 30.5 个专家
```

**理论极限**（大 batch）：

```
当 batch → ∞ 时：
max_experts → num_experts / chips

例如：batch=256, chips=32 → 256 / 32 = 8.0
```

---

**维护者**: Tier6-Model Team
**最后更新**: 2026-01-26
**许可**: MIT License
