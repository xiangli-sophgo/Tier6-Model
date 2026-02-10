# CHIPMathica tpu_simulator vs math_model 差距分析

> 分析日期: 2026-02-10
> 对比对象: `CHIPMathica/tpu_simulator` vs `Tier6-Model/backend/math_model`

---

## 0. 总体结论

- **配置参数**: 已完备。芯片参数在 `configs/chips/`，通信延迟参数在 `configs/topologies/` 的 `interconnect.comm_params`，Python 解析代码均已对接。
- **通信评估器**: `comm_protocol.py` 已对齐 CHIPMathica（AllReduce/AllGather/ReduceScatter/Dispatch/Combine + 分层 Ring + start_lat 建模），差距很小。
- **主要差距集中在计算评估器**: arch_urate、compute-DMA overlap、分核余数感知、softmax/FA2 精度。

---

## 1. 计算评估器差距 (P0 - 核心精度)

### 1.1 arch_urate (架构利用率) - 未实现

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **位置** | `matmul_eval.py` / `fa2_eval.py` | 无对应实现 |
| **公式** | `arch_urate = real_ops / aligned_ops` | 无 |
| **作用** | 量化对齐损耗：当 M/N/K 不是 cube_m/n/k 整数倍时，实际有效计算占比 | 无 |

**CHIPMathica 实现细节**:
```python
# MatMul:
aligned_ops = align_up(M, cube_m) * align_up(K, cube_k) * align_up(N, cube_n) * 2
real_ops = M * K * N * 2
arch_urate = real_ops / aligned_ops

# FA2 (GEMM + Vector):
gemm_theo = align_up(QS, cube_m) * align_up(QD, cube_k) * align_up(KS, cube_n) + ...
gemm_real = QS * KS * (QD + VD) * 2
vector_theo, vector_real = softmax_theoretical_and_real(...)
arch_urate = (gemm_real + vector_real) / (gemm_theo + vector_theo)
```

**math_model 现状**: L4 `MatMulPreciseEvaluator` 只算 `compute_urate = t_compute / max(t_compute, t_memory)`，这是**瓶颈占比**，不是架构利用率。

**影响**: 当 shape 不对齐时（如 M=1000, cube_m=16），实际利用率仅 ~62.5%，不建模会高估计算效率。

---

### 1.2 compute-DMA overlap 公式 - 未使用

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **公式** | `t_total = min(comp, dma) * (1 - overlap_rate) + max(comp, dma)` | `t_total = max(t_compute, t_memory)` |
| **参数** | `tpu_gdma_overlap_rate = 0.8` | `compute_dma_overlap_rate = 0.8` (已配置，未使用) |

**影响**: 当前 `max()` 假设完全 overlap，实际只有 80% overlap。差距约 20% 的 `min(comp, dma)`。

**注意**: `compute_dma_overlap_rate` 参数已经在 `ChipSpecImpl` 中解析存储，但 L4 评估器未读取使用。

---

### 1.3 分核余数感知评估 - 未实现

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **实现** | 遍历每个 core (i_g, i_m, i_n, i_k)，计算余数感知的实际 block 大小 | 只按 `ceil(M/p_m)` 统一估算 |

**CHIPMathica 实现**: 对于 `core (i_m, p_m)`:
```python
m_start = i_m * ceil(M / p_m)
m_end = min(M, (i_m + 1) * ceil(M / p_m))
m_blk = m_end - m_start  # 最后一个核可能更小
```
然后对每个核独立计算 `arch_urate`、`comp_time`、`dma_time`，取**最大值**作为 partition 总时间。

**影响**: 负载不均时（如 M=100, cores=32），边界核的 block 更小、利用率更低，整体延迟由最慢核决定。

---

### 1.4 Softmax 10 步评估 - 缺失

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **实现** | `softmax_eval.py`: 10 步流水线精确建模 | 无独立 softmax 评估器 |

**CHIPMathica 的 10 步**:
1. add (QS x KS)
2. reduce_max (QS x 1)
3. max (QS x KS)
4. fuse_exp variant 1 (QS x KS)
5. fuse_exp variant 2 (QS x KS)
6. reduce_sum (QS x 1)
7. mul (QS x KS)
8. add (QS x KS)
9. copy (QS x KS)
10. data_convert (QS x KS)

每步区分 `theoretical_ops`（对齐到 lane_num/eu_num）和 `real_ops`（实际元素数），用于:
- FA2 评估中的 vector 部分耗时
- 独立 softmax 层评估

**math_model 现状**: FA2 评估只计算 GEMM 的 traffic/compute，softmax 向量开销完全缺失。

---

### 1.5 FA2 zigzag 优化 - 缺失

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **实现** | `enable_zigzag_reorder=True` 时 GEMM FLOPs 减 40% | 无 |
| **条件** | 仅 prefill 阶段 + zigzag 开启时生效 | 无 |

**公式**: `gemm_real = int(gemm_real * 0.6)` (因果注意力的三角矩阵优化)

---

### 1.6 RMSNorm 独立评估 - 精度不足

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **实现** | `rmsnorm_eval.py` 专用评估器 | 归入 `ElementwisePreciseEvaluator`，统一用 `5 ops/element` 估算 |

---

## 2. 通信评估器差距 (已基本对齐)

`math_model/L4_evaluation/cost_models/comm_protocol.py` 已实现:

| 功能 | CHIPMathica | math_model | 状态 |
|------|-------------|------------|------|
| AllReduce (Ring) | `2(tp-1)/tp * bytes` + start_lat | 同 | OK |
| AllReduce (分层) | 无 (只有 Ring) | tp=8/16/32 时自动分层 (intra+inter+intra) | **math_model 更强** |
| AllGather | `(tp-1) * bytes` | 同 + 分层支持 | OK |
| ReduceScatter | `(tp-1)/tp * bytes` | 同 + 分层支持 | OK |
| Dispatch (MoE) | inter_bw scatter + allgather | 同 + comm_protocol 区分 | OK |
| Combine (MoE) | 同 dispatch 反向 | 同 | OK |
| start_lat 建模 | `2*c2c + ddr_r + ddr_w + noc + 2*d2d` | 同 | OK |
| 硬件延迟参数 | 全部在 tpu_config YAML | 全部在 topology comm_params YAML | OK |
| bw_urate | 0.95 | 0.95 | OK |
| switch_delay / cable_delay | 有 | 有 | OK |
| comm_protocol 区分 | 无 (固定公式) | 支持 protocol=2/3 不同 RTT 模式 | **math_model 更强** |

**结论**: 通信部分 math_model 已完整对齐甚至超越 CHIPMathica，无需额外工作。

---

## 3. GDMA 带宽模拟器 - 缺失

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **模块** | `gdma_bw/`: GdmaSimulator, GdmaCmd, ChipConfig, MemoryMap | 无 |
| **功能** | 精确模拟 DMA burst 传输、地址解码、DDR vs LMEM 路由 | 无 |

**影响**: 当前 math_model 的 DMA 耗时用 `traffic / bandwidth` 线性估算，未考虑:
- burst 长度对效率的影响
- 源/目的地址类型 (DDR vs LMEM) 的不同延迟
- pipeline 效应

**优先级**: P2（对整体精度影响较小，线性估算在大多数场景够用）

---

## 4. 模型层定义差距

### 4.1 MoE 负载不均建模 - 缺失

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **实现** | `get_max_expert(batch, chips)` 经验查表 | 假设均匀分布 |
| **作用** | 建模不同 (batch_size, chip_count) 下的最大 expert 激活数 | 无 |

**影响**: MoE 层的实际延迟由负载最重的 expert 决定，均匀假设会低估延迟。

### 4.2 DeepSeek V3.2 变体 - 缺失

CHIPMathica 有 `deepseek_v3_2.py`，math_model 的 `configs/models/` 中有 `DeepSeek-v3.2.yaml` 配置文件但代码层面未见专门处理 V3.2 的差异。

---

## 5. 评估器缓存系统

| 项目 | CHIPMathica | math_model |
|------|-------------|------------|
| **计算评估器缓存** | 每个评估器内置 OrderedDict LRU (max_size=10000) | L3 TilingPlanner 有简单 dict 缓存 |
| **通信评估器缓存** | 同上 | 无 |
| **缓存统计** | `get_cache_stats()`: hit/miss/eviction/fill_rate | 无 |

**影响**: 性能优化问题，不影响正确性。大规模扫参时可能有较大加速效果。

---

## 6. 成本模型微调

| 参数 | CHIPMathica | math_model | 备注 |
|------|-------------|------------|------|
| SG2262 芯片单价 | $2,371.275 | $2,500 | 可配置 |
| RDMA 网卡 | `ceil(chips/4) * $7,500` | 固定 $7,500 power/cooling | 计算方式不同 |
| 杂项成本 | `$1,500 * num_chips` | 无 | 缺少 |
| 互联分段 | 5 档 (1-4/5-8/9-32/33-64/65+) | 6 档 (1-2/8/16/32/64/64+) | 阈值不同 |
| DFOP 指标 | `cost / tps` | `cost_per_million_tokens` | 名称不同，概念类似 |

**优先级**: P2（数值差异，可通过配置调整）

---

## 7. 优先级排序的 TODO 清单

### P0 - 核心精度 (直接影响评估准确性)

| # | 任务 | 涉及文件 | 复杂度 |
|---|------|----------|--------|
| 1 | **实现 arch_urate 计算** - `real_ops / aligned_ops`，用于 MatMul 和 FA2 | L4 `precise.py` 或新增 evaluator | 中 |
| 2 | **实现 compute-DMA overlap 公式** - `min(comp,dma)*(1-overlap) + max(comp,dma)` | L4 `precise.py` + `compute.py` | 低 |
| 3 | **分核余数感知评估** - 遍历每个 core 计算实际 block 大小，取最慢核 | L3 `evaluators.py` 或 L4 新增 | 高 |

### P1 - 重要功能

| # | 任务 | 涉及文件 | 复杂度 |
|---|------|----------|--------|
| 4 | **Softmax 10 步向量评估** - 细粒度 softmax 建模 | 新增 `softmax_eval.py` | 中 |
| 5 | **FA2 softmax 向量开销** - 将 softmax 评估集成到 FA2 | L4 `precise.py` AttentionPreciseEvaluator | 中 |
| 6 | **FA2 zigzag 优化** - prefill 时 causal mask 减 40% GEMM | L4 `precise.py` AttentionPreciseEvaluator | 低 |
| 7 | **MoE 负载不均建模** - expert 激活数查表 | L1 `layers/moe.py` 或 L4 | 中 |

### P2 - 增强项

| # | 任务 | 涉及文件 | 复杂度 |
|---|------|----------|--------|
| 8 | **RMSNorm 独立评估器** | L4 新增或增强 `precise.py` | 低 |
| 9 | **GDMA 带宽模拟器** - DMA burst/pipeline 精确建模 | 新增模块 | 高 |
| 10 | **LRU 缓存 + 命中率统计** | L3/L4 评估器 | 低 |
| 11 | **成本模型微调** - RDMA 网卡、杂项成本对齐 | `cost_evaluator.py` | 低 |

---

## 8. 不需要做的事项 (已完成或 math_model 更强)

- [x] 通信延迟参数配置 - 已在 topology comm_params 中完备
- [x] AllReduce/AllGather/ReduceScatter 建模 - comm_protocol.py 已对齐
- [x] Dispatch/Combine (MoE) 两阶段通信 - 已实现
- [x] start_lat 硬件级建模 - 已实现 (2*c2c + ddr_r + ddr_w + noc + 2*d2d)
- [x] 分层 AllReduce (8/16/32 chips) - math_model 有，CHIPMathica 无
- [x] comm_protocol 多模式支持 - math_model 有 protocol=2/3
- [x] L3 tile 搜索 + L4 精评估回调 - 刚完成 Option B 实现
- [x] 芯片参数解析 (cube m/k/n, sram_utilization, align_bytes, overlap_rate) - 已完成
