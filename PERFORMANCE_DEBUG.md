# 性能调试指南

## 已添加的性能日志

我在 `simulator.py` 中添加了详细的性能日志，运行任务时会输出：

### 1. 各阶段墙上时间（Wall Time）

```
⏱️  [H2D] 墙上时间: XX.XXms
⏱️  [Prefill] 墙上时间: XX.XXms, 模拟时间: XX.XXms
⏱️  [Decode] 墙上时间: XX.XXms (XX.XX ms/token), 模拟时间: XX.XXms
    🔹 Token 0/4: 墙上时间 XX.XXms, 遍历了 60 层
    🔹 Token 1/4: 墙上时间 XX.XXms, 遍历了 60 层
    🔹 Token 2/4: 墙上时间 XX.XXms, 遍历了 60 层
    🔹 Token 3/4: 墙上时间 XX.XXms, 遍历了 60 层
      🔸 单层评估: build=XX.XXms, gantt=XX.XXms, ops=XX+XX
⏱️  [D2H] 墙上时间: XX.XXms
⏱️  [Gantt Build] 墙上时间: XX.XXms
⏱️  [Stats] 墙上时间: XX.XXms
⏱️  [Total] 总墙上时间: XX.XXms
```

### 2. 日志字段说明

- **墙上时间（Wall Time）**: 真实运行时间（你等待的时间）
- **模拟时间（Simulation Time）**: 模拟的硬件执行时间（虚拟时间）
- **build**: 构建层算子的时间
- **gantt**: 添加甘特图任务的时间
- **ops**: 计算算子数量+通信算子数量

## 如何查看日志

### 方法1: 查看后端控制台输出

启动后端后，直接在终端看日志输出。

### 方法2: 修改日志级别

在 `backend/llm_simulator/api.py` 或 `backend/main.py` 中设置：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 方法3: 运行性能测试脚本

```bash
cd backend/llm_simulator
python performance_test.py
```

这个脚本会运行一个简化的10层模型，快速定位瓶颈。

## 可能的性能瓶颈

### 1. 精确评估器（Tile搜索）⭐ 最可能

**问题**: `use_precise_evaluator=True` 时，每个GEMM算子会搜索最优的tile配置。

**表现**:
- `build=` 时间非常长（几百毫秒甚至几秒）
- 每个token需要几秒甚至几十秒

**解决方案**:

#### A. 禁用精确评估器（最快）

在 `simulator.py:1353-1358` 修改：

```python
config = SimulationConfig(
    max_simulated_tokens=4,
    use_precise_evaluator=False,  # 🔑 改为 False
    evaluation_granularity="coarse",  # 🔑 改为 coarse
    # ...
)
```

**加速效果**: 可能快 10-100 倍

#### B. 禁用多进程Tile搜索

设置环境变量：

```bash
export GEMM_DISABLE_MULTIPROCESS=1
python main.py
```

**原因**: 多进程有启动开销，对于小规模搜索反而更慢

#### C. 使用粗粒度评估

只修改 `evaluation_granularity`:

```python
config = SimulationConfig(
    use_precise_evaluator=True,
    evaluation_granularity="coarse",  # 不展开每个算子
)
```

### 2. 甘特图构建

**问题**: `gantt_builder.add_compute_task()` 调用太频繁

**表现**:
- `gantt=` 时间很长
- `[Gantt Build]` 时间很长

**解决方案**:

禁用细粒度甘特图：

```python
config = SimulationConfig(
    evaluation_granularity="coarse",  # 聚合算子
)
```

### 3. Decode循环次数

**问题**: 模拟的token数量太多

**表现**:
- 每个token时间正常，但总时间长

**解决方案**:

减少 `max_simulated_tokens`:

```python
config = SimulationConfig(
    max_simulated_tokens=2,  # 从4改为2
)
```

**影响**: TPOT精度略微下降，但对大多数场景足够

## 快速优化建议

### 立即优化（改配置）

在 `backend/llm_simulator/simulator.py:1353` 修改：

```python
config = SimulationConfig(
    max_simulated_tokens=config_dict.get("maxSimulatedTokens", 2),  # 4→2
    use_precise_evaluator=False,  # True→False ⭐ 关键
    evaluation_granularity="coarse",  # fine→coarse
    # ...
)
```

**预期加速**: 手动任务从 5秒 → 0.5秒以内

### 长期优化（改代码）

1. **缓存GEMM评估结果**（已经有缓存，检查是否生效）
2. **并行化层的评估**（你提到的思路）
3. **预计算常见配置的tile**
4. **使用C++重写评估器**（如果需要极致性能）

## Tile优化是什么？

在 GEMM 评估器中，会搜索最优的分块策略：

```
GEMM: C[M, N] = A[M, K] × B[K, N]

Tile配置: (tile_M, tile_N, tile_K)
例如: (128, 128, 64)

需要遍历所有可能的组合，找到最快的配置
→ 这个搜索过程很耗时！
```

**相关代码**: `backend/llm_simulator/evaluators/gemm_eval.py`

## 下一步

1. 运行一个手动任务
2. 查看后端日志，定位哪个阶段最慢
3. 根据日志结果选择优化方案
4. 告诉我结果，我帮你进一步优化

---

**注意**: 所有修改都在后端 Python 代码，不需要重新编译前端。修改后重启后端即可生效。
