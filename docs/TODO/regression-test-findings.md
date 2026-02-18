# DS_TPU 回归测试对比 - 发现与待修复项

对比仓库:
- A: Tier6+Model (`backend/math_model/`)
- B: CHIPMathica (`/Users/lixiang/Documents/工作/code/CHIPMathica/tpu_simulator`)
- 基线: DS_TPU (`DS_TPU/tests/baseline/`)

## 已修复

### 1. SG2262 芯片参数不一致
- cores: 4 -> 32, lanes_per_core: 64 -> 16
- mac_per_lane FP8: 1000 -> 256, BF16: 500 -> 128
- DRAM capacity: 64GB -> 128GB, bandwidth: 273 -> 4096 GB/s
- 统一 `utilization` 命名 (去掉 `bandwidth_utilization`/`sram_utilization` 前缀)
- `chip.get_gmem_bandwidth()` 乘以 utilization (有效带宽 = 4096 * 0.70 = 2867.2)

### 2. Decode 模式 q_seq_len 未设为 1
- `eval_config.py`: `q_seq_len = input_seq_length if is_prefill else 1`
- `utils.py get_seq_len()`: 优先使用 `q_seq_len` 而非 `seq_len`
- `planner.py _get_layer_seq()`: 同上

### 3. Prefill/Decode 统计分类错误
- L4 `_aggregate_metrics` 增加 `is_prefill` 参数, 正确区分 prefill/decode 阶段
- `compat.py convert_to_stats` 用 `is_prefill` 直接判断, 去掉启发式猜测

### 4. FFNLayer 未使用 get_seq_len() 工具函数
- `FFNLayer` (ffn.py) 直接用 `self._config.get("seq_len", default)` 读取 seq_len=4096
- 其他层 (MoE/MLA/Embedding/LMHead) 均使用 `get_seq_len()` 优先读 q_seq_len
- 修复: 重写 FFNLayer 使用 utils.py 的 get_batch/get_seq_len/get_hidden_size/get_intermediate_size
- 同时去掉了违反 CLAUDE.md 的默认值 (default fallback)
- 效果: Total time 6468ms -> 1023ms, TPS 316 -> 2002

### 5. 通信时间 = 0 (TP=1 时虚假 ALLREDUCE + dispatch/combine 延迟缺失 + TBO 粒度错误)
- **根因 1**: TP=1 时 planner 产生 181 个虚假 ALLREDUCE op, 公式 `2*(tp-1)/tp*bytes = 0`
  - `_build_layout_signature`: split_factor<=1 时归一化为 REPLICATE, 消除虚假 signature mismatch
  - `_needs_terminal_collective`: 增加 split_factor 参数, split_factor<=1 返回 False
  - MoE shared_down: 增加 `tp > 1` 条件
  - embedding/lmhead: 增加 effective_tp > 1 条件
- **根因 2**: dispatch/combine 的 start_lat 缺少 switch/cable 延迟 (比 B 少 ~2us)
  - 新增 `dispatch_start_lat` 属性: `start_lat + 2*switch_latency + 2*cable_latency`
  - dispatch() 和 combine() 使用 `dispatch_start_lat` 替代 `start_lat`
- **根因 3**: TBO overlap 用单步 compute time 做预算, 算法与 B 不一致
  - 重写 `_apply_moe_compute_overlap` 为 block-level: 累加 dispatch/combine 前后整个计算块
  - overlap 预算 = `min(前块, 后块)`, 对齐 B 的 `min(mla, shared+routed)`
- 修复文件: `planner.py`, `comm_protocol.py`, `engine.py`

### 6. MoE 负载均衡 G/M 语义错误
- **根因**: `_apply_moe_expert_shard` 的 G/M 语义与 B 不同
  - A(旧): G=n_activated(原始值), M=tokens_per_ep_group_local(含 activated 双重计数)
  - B: G=activated_expert_per_ep(查表), M=ceil(token_per_ep/expert_per_ep)(每专家 token 数)
- **修复**:
  - `_apply_moe_expert_shard` 重写, 对齐 B 的三步计算:
    1. `token_per_ep_group = ceil(batch * seq * activated / ep)` (全局 batch)
    2. `expert_per_ep_group = ceil(n_routed / ep)`, `m_per_group = ceil(tokens/experts)`
    3. `G = min(expert_per_ep, round(get_max_expert_load(batch*seq, ep)))`, `M = m_per_group`
  - 去掉 `_moe_load_factor` 和 `_moe_tokens_per_ep_group_local`(语义错误的间接转换)
  - `_moe_tokens_per_ep_group` 去掉 load_factor(通信量不受负载不均影响)
  - 去掉 G 维度的 moe_tp 放大(B 没有此逻辑)
- 修复文件: `planner.py`

### 7. FlashAttention-2 (FA2) 评估移植 (原 P3)
- **根因**: A 将 attention 分解为 2 个 BatchMatMul (attn_score + attn_out), 走通用 MatMul 评估器; B 使用 FA2Eval (FlashAttention-2 tiling)
- **修复**:
  - 新增 `FA2Op` 算子 (`L1_workload/operators/compute/fa2.py`), shape: `{B, QS, KS, QD, VD}`
  - 4 个 MLA 变体全部移植: `mla.py`, `mla_absorb.py`, `mla_v3_2.py`, `mla_absorb_v3_2.py`
    - 每个文件: 删除 attn_score + attn_out 两个 MatMulOp, 替换为单个 FA2Op
  - `pattern_rules.py`: 新增 `mla_v3_2` 和 `mla_absorb_v3_2` pattern 模板
  - `planner.py`:
    - `_infer_op_role` 增加 `mla_v3_2`, `mla_absorb_v3_2` 支持
    - 新增 `_infer_fa2_shape` 方法, 保留 FA2 shape keys (B/QS/KS/QD/VD) 而非转为 {G,M,N}
    - `_infer_op_shape` 增加 FA2 路由
    - 修复 `_apply_sp_shard` 命名不一致 (`mla_v32` -> `mla_v3_2`)
  - `engine.py`: COMPUTE_OP_TYPES 增加 "fa2"
  - `base.py` (cost_models): estimate_flops/estimate_bytes 增加 FA2 分支
  - `precise.py`: 修复 tile_k/tile_n 映射 (FA2TilingEvaluator 输出 tile_n, AttentionPreciseEvaluator 读 tile_k)
- **效果**: batch=80 decode total_time 88.25ms -> 56.35ms (降 36%), FA2 ops=61, matmul attn ops=0
- **残余差距**: 56ms vs CHIPMathica 19ms, 主因是 decode QS=1 对齐到 cube_m=16 的 16x 浪费 (P4 partition 搜索问题)
- 修复文件: `fa2.py`, `mla.py`, `mla_absorb.py`, `mla_v3_2.py`, `mla_absorb_v3_2.py`, `pattern_rules.py`, `planner.py`, `engine.py`, `base.py`, `precise.py`

### 8. FA2 B*QS 重分布 (原 P4 核心问题)
- **根因**: B 在 FA2 评估时将 `B*QS` 重分布到 `core_count` 个虚拟 batch 上:
  `adjusted_B = core_count, adjusted_QS = ceil(B*QS/core_count)`
  decode 时 QS=1, `align_up(1, cube_m=16) = 16` 产生 16x 对齐浪费。重分布后 QS=256, 无浪费。
  A 直接使用原始 QS=1, 导致 mla_attn 占 73% 总时间。
- **修复**:
  - `FA2TilingEvaluator.select_tile()` (evaluators.py): tile 搜索前做 B*QS 重分布
  - `AttentionPreciseEvaluator.evaluate()` (precise.py): compute time 计算前做 B*QS 重分布
- **效果**:
  - batch=80: 56.35ms -> 22.59ms (降 60%, CHIPMathica 19.25ms, 差距 1.17x)
  - batch=2048: 1915ms -> 132ms (降 93%), MFU 0.054 -> 0.784
  - mla_attn 每层: 0.67ms -> 0.125ms (降 81%)
- 修复文件: `evaluators.py`, `precise.py`

### 9. MFU 峰值基准对齐 (原 P6)
- **根因**: A 硬编码 `get_peak_flops("BF16", "cube")`, weight_dtype=FP8 时应用 FP8 峰值 (768 TFLOPS vs 384 TFLOPS)
- **修复**:
  - `L0_entry/engine.py` L3 Tiling 和 `_build_hardware_spec`: 从 `eval_config.inference.weight_dtype` 动态选择 cube_dtype
  - FP8/INT8 -> 使用对应精度峰值; 其他 -> BF16
- **效果**: compute_time 22.59ms -> 18.85ms (降 17%), 因为 FP8 算力 2x 使 Roofline 计算时间减半
- 修复文件: `engine.py`

### 10. FA2 Traffic Q 加载次数修正
- **根因**: A 的 FA2 traffic 公式中 `q_loads = q_tiles * k_tiles`, 但 Q 在 SRAM 中保留遍历所有 K tiles, 应为 `q_loads = q_tiles`
- **修复**: `precise.py` AttentionPreciseEvaluator: `q_loads = q_tiles` (不再乘 k_tiles)
- **效果**: 减少 FA2 ops 的 traffic 估算, 对 memory-bound ops 降低 t_memory_ms
- 修复文件: `precise.py`

### 11. FA2 Softmax eu_num 粒度错误
- **根因**: A 的 `chip.eu_num` 返回芯片总 EU 数 (eu_per_lane x lane x cores = 32768), 而 B 的 `eu_num` 是单核 EU 数 (512)
  - softmax 对齐: `eu_block = eu_num // lane_num // dtype_bytes`
  - A: eu_block = 32768 // 16 // 1(FP8) = **2048** -> KS=1568 对齐到 2048, 30.5% 浪费
  - B: eu_block = 512 // 16 // 2(BF16) = **16** -> KS=1568 对齐到 1568, 无浪费
- **修复**: `precise.py` AttentionPreciseEvaluator.evaluate(): 使用 `eu_per_core = eu_num_total // core_count` 做 softmax 对齐, 保留 `eu_num_total` 做吞吐量计算
- **效果**: FA2 tiling t_compute: 131.2us -> 122.3us/layer (降 7%)
- 修复文件: `precise.py`

### 12. MoE Combine 通信量 dtype 错误
- **根因**: A 的 `_moe_combine_comm_bytes` 使用 MoE output tensor dtype (FP8=1B), 而 B 使用 BF16=2B
  - dispatch: batch_local * hidden * topk * FP8(1) = 3,670,016 bytes [PASS]
  - combine(旧): 同 dispatch = 3,670,016 bytes -> 11.3us [FAIL, B=19.99us]
  - combine(修复): batch_local * hidden * topk * BF16(2) = 7,340,032 bytes -> 19.91us [PASS]
  - 原因: expert FFN 累积精度为 BF16, combine 传输的是累积后的结果
- **修复**: `planner.py` `_moe_combine_comm_bytes`: `dtype_bytes = max(2, tensor.dtype.bytes)`
- **效果**: raw comm 总量 1.314ms -> 1.812ms, 与 B 的 1.818ms 对齐 (差 0.3%)
- 修复文件: `planner.py`

### 13. FA2 Softmax full-dim + BF16 对齐 (P16+P17)
- **根因 P16**: A 按 (tile_q, tile_k) 分 tile 计算 softmax 再乘以 tile 数, B 在 full (QS, KS) 维度上计算
  - tiled 方式引入额外对齐开销, 总 vector ops 偏多
- **根因 P17**: A 用 `qkv_bytes` (FP8=1) 做 `eu_block` 对齐和向量吞吐量, B 固定用 BF16=2
  - eu_block: A=512//16//1=32, B=512//16//2=16 (对齐粒度不同)
  - 吞吐量公式: A 用 `eu*freq/dtype`, B 用 `eu*freq*dtype` (B 的 EU 吞吐量模型不同)
- **修复**:
  - `precise.py` AttentionPreciseEvaluator: softmax 使用 full (qs, ks) + SOFTMAX_DTYPE_BYTES=2
  - arch_urate 统一用 per-head 值 (gemm 和 vector 一致)
  - 向量吞吐量: `eu_num_total * freq * 1e9 * SOFTMAX_DTYPE_BYTES` (对齐 B)
- **效果**: FA2 per-layer: 137.7us -> 114.5us (降 17%), 与 B 的 112.4us 差 1.9%
  - total_time: 15.415ms -> 13.995ms (降 9.2%)
  - MFU: 0.754 -> 0.831 (提升 10.2%)
- 修复文件: `precise.py`

### 14. TBO (Transport/Bandwidth Overlap) 条件化
- **根因**: A 无条件应用 MoE compute-comm overlap, B 默认 `tbo=False` (顺序累加)
  - tbo=False 时 dispatch/combine 通信时间直接加到总时间, 不与计算重叠
  - A 的 overlap 预算覆盖 100% comm, 导致 comm_time=0, 比 B 快 ~1.8ms
- **修复**:
  - `L4 engine.py`: `_apply_moe_compute_overlap` 仅在 `enable_tbo=True` 时执行
  - `eval_config.py`: DeploymentConfig 新增 `enable_tbo: bool` 字段, 默认 False
  - `L0 engine.py`: 传递 `enable_tbo` 到 deployment_dict
  - **前端**: DeploymentAnalysisPanel.tsx 新增 TBO 开关 (评估选项区域第 4 列)
    - types.ts ParallelismStrategy + math_model.ts ManualParallelism 增加 `enable_tbo` 字段
- **效果**: enable_tbo=False 时 total_time 13.995ms -> 15.806ms, 与 B 的 15.743ms 对齐 (+0.40%)
- 修复文件: `engine.py`(L0+L4), `eval_config.py`, `DeploymentAnalysisPanel.tsx`, `types.ts`, `math_model.ts`

### 15. kv_seq_len 公式修正
- **根因**: A 的 `kv_seq_len = input_seq_length + output_seq_length` (4096+1=4097), B 固定使用 `kv_seq_len = input_seq_length` (4096)
  - B 的 DeploymentConfig: `kv_seq_len: int = 4 * 1024`, 不随 output_seq_length 变化
  - 多出的 1 token 导致 FA2 每层多 ~2us (KS=4097 vs 4096 对齐差异)
- **修复**: `eval_config.py`: `kv_seq_len=input_seq_length` (去掉 `+ output_seq_length`)
- **效果**: total_time 15.806ms -> 15.776ms, 与 B 的 15.743ms 差距从 +0.40% 缩小到 **+0.21%**
- 修复文件: `eval_config.py`

### 16. Tiling 对齐 (c_bytes + FA2 SRAM + SRAM utilization)
- **根因 1 (c_bytes)**: A 的 `output_dtype_bytes` 从 op attrs 读取 FP8=1 byte, B 硬编码 BF16=2
  - SRAM 中的 output buffer 至少需要 BF16 精度 (accumulator FP32 -> BF16 -> 写出)
- **根因 2 (FA2 SRAM)**: A 的 FA2 P buffer 用 FP8=1, B 用 BF16=2; A 无行列对齐, B 有 align_row/align_col
  - A 低估 SRAM 占用, 允许选更大的 tile
- **根因 3 (FA2 traffic)**: A 缺 load_q 项, 多 acc 项; 对齐 B: Q 加载一次 + 无 acc
- **根因 4 (SRAM utilization)**: A `lmem utilization=1.0`, B 硬编码 `sram_size * 0.45`
  - A 的 SRAM budget 是 B 的 2.22 倍 (2048KB vs 921.6KB)
- **修复**:
  - `evaluators.py` `_resolve_dtype_bytes`: `output_bytes = max(output_bytes, 2)`
  - `precise.py`: `c_bytes = max(c_bytes, 2)`
  - `evaluators.py` FA2 `_estimate_lmem`: P buffer 固定 BF16=2, 加 align_row/align_col 对齐
  - `evaluators.py` FA2 `_estimate_traffic`: 加 load_q, 去 acc 项
  - `SG2262.yaml`: `lmem utilization: 0.45`
- **效果**: FA2 tile 完全对齐 B (QS=128, KS=192)
  - Matmul tile 仍有差异 (A 用 elapsed 打分倾向大 K / mkn order, B 用 traffic 打分倾向大 N / mnk order)
  - total_time 15.776ms -> 15.849ms, vs B 15.743ms (+0.67%)
- 修复文件: `evaluators.py`, `precise.py`, `SG2262.yaml`

---

## 待修复 (低优先级)

---

### P5: Embedding 建模差异

**A**: FLOPs = 0 (查表操作), _build_ops() 返回空列表, 不进入 L4 评估
**B**: 有 FLOPs, 且主要时间由 memory_load / dram_bw 决定:
```python
layer.elapse = layer.memory_load / dram_bw * 1e6 + comm_elapse
# memory_load = 2 * batch_local * q_seq_len * hidden_dim * dtype_bytes
```

**影响估算**: decode batch=80 时, memory_load ~1.1MB, time ~0.0004ms, 可忽略

**关键文件**:
- A: `L1_workload/layers/embedding.py`
- B: `model/layers/embedding.py`

---

---

## 回归测试评估结果

### 精确对比 (2026-02-17, 修复 #1-#16)

**测试环境**: Python 3.11, DeepSeek-V3-671B-A37B, SG2262, P1-R1-B4-C32
**配置**: TP=1, DP=32, EP=32, batch=2048, decode, mla_mode=absorb, FP8, enable_tbo=False

#### 总体对比

| 指标 | A (Tier6+Model) | B (CHIPMathica) | 差异 |
|------|:-:|:-:|:-:|
| total_time_ms | **15.849** | **15.743** | **+0.67%** |
| compute_time_ms | 14.038 | ~13.93 | +0.77% |
| comm_time_ms (raw) | 1.812 | 1.818 | **-0.3%** |
| total_flops | 9,142 G | 9,260 G | -1.3% |
| MFU | 0.7334 | 0.748 | -2.0% |
| TPS | 129,217 | ~130,000 | -0.6% |

#### 通信对比 (per MoE layer)

| 指标 | A | B | 差异 |
|------|:-:|:-:|:-:|
| dispatch (raw) | 11.32 us | 11.36 us | -0.4% |
| combine (raw) | 19.91 us | 19.99 us | -0.4% |

#### FA2 (FlashAttention-2) 对比 (per layer)

| 指标 | A (#12) | A (#13) | A (#15) | B | A vs B |
|------|:-:|:-:|:-:|:-:|:-:|
| t_total | 137.7 us | 114.5 us | **~112.6 us** | 112.4 us | **+0.2%** |
| softmax 维度 | tile (64,1568) | full (128,4097) | full (128,4096) | full (128,4096) | 对齐 |
| softmax dtype | FP8 (1) | BF16 (2) | BF16 (2) | BF16 (2) | 对齐 |
| softmax eu_block | 32 | 16 | 16 | 16 | 对齐 |

FA2 per-layer 差距从 22% 缩小到 0.2%, KS 从 4097 修正为 4096 后完全对齐。

#### 剩余差异根因分解

| 来源 | 差异量 | 说明 |
|------|--------|------|
| Matmul tile 打分策略差异 | ~+0.106ms | A 用 elapsed 打分 (倾向大 K / mkn), B 用 traffic 打分 (倾向大 N / mnk) |
| FA2 tile | 0 | 完全对齐 (QS=128, KS=192) |
| FLOPS 差异 (-1.3%) | 内含于 compute | A 少 ~118G flops |
| **总计** | **+0.106ms** | A 慢 0.67% |

#### Tile 选择对比 (layer 0 MLA + LMHead)

| Op | A tile | B tile | 匹配 |
|-----|--------|--------|------|
| q_a | M=64,N=24,K=7168,mkn | M=64,N=384,K=448,mnk | [DIFF] |
| q_b | M=64,N=384,K=1536,mkn | M=64,N=768,K=768,mnk | [DIFF] |
| kv_a | M=32,N=24,K=7168,mkn | M=64,N=144,K=448,mnk | [DIFF] |
| k_compact | M=64,N=512,K=128,mnk | M=64,N=512,K=128,mnk | [PASS] |
| v_compact | M=64,N=512,K=128,mnk | M=64,N=512,K=128,mnk | [PASS] |
| **attn(FA2)** | **QS=128,KS=192** | **QS=128,KS=192** | **[PASS]** |
| out_proj | M=64,N=96,K=5792,mkn | M=64,N=896,K=832,mnk | [DIFF] |
| lm_head | M=64,N=2048,K=320,mnk | M=64,N=4040,K=96,mnk | [DIFF] |

- 差异原因: A 用 `max(t_compute, t_memory)` 打分, B 用 `min(traffic)` 打分
- A 的策略理论上更通用 (考虑 compute-memory 平衡), B 的策略在 memory-bound decode 场景更优

**注**: 经 #14-#16 修复后, A 与 B 对齐到 +0.67%。#16 对齐了 FA2 tile (QS=128, KS=192 完全一致), 但 matmul 打分策略差异 (A 用 elapsed, B 用 traffic) 导致不同 tile 选择。

#### 剩余差异精确诊断 (2026-02-17)

**所有 compute 时间完全一致** (0% 差异), 差异全部来自 DMA traffic (不同 tiling 选择):

| Op (MLA per-layer) | A_comp(us) | B_comp(us) | comp_diff | A_traffic | B_traffic | traffic_diff | elapsed_diff |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| q_a | 1.792 | 1.792 | 0.0% | 11.6MB | 16.0MB | -27.7% | -7.0% |
| q_b | 6.144 | 6.144 | 0.0% | 39.4MB | 47.2MB | -16.5% | -2.5% |
| kv_a | 0.896 | 0.672 | +33.3% | 8.8MB | 7.1MB | +22.5% | +24.0% |
| k_compact | 1.365 | 1.365 | 0.0% | 13.6MB | 17.8MB | -23.5% | -20.8% |
| v_compact | 1.365 | 1.365 | 0.0% | 13.6MB | 17.8MB | -23.5% | -20.8% |
| **attn (FA2)** | **105.475** | **105.475** | **0.0%** | 365MB | 298MB | +22.5% | +1.4% |
| out_proj | 19.115 | 19.115 | 0.0% | 160MB | 133MB | +20.3% | +2.8% |
| **MLA total** | | | | | | | **+0.74%** |

- kv_a comp 差异: A 用 48/64 cores (g*m_tiles*n_tiles=48), B 用 P_N*P_K=4*16=64 cores (含 K 维度并行化)
- MoE 层: comp 完全一致, elapsed -0.64% (shared_* traffic 偏少抵消了 MLA 偏多)
- 净影响: MLA +0.068ms, MoE -0.040ms, Dense +0.006ms, LMHead +0.001ms = **+0.034ms**
- 进一步对齐需逐算子匹配 B 的 tile search 算法, 属于 diminishing returns

### 历史对比 (修复演进)

| 指标 | #1-#6 | #7 FA2 | #8 重分布 | #9+#10 | #11+#12 | #13 | #14+#15 | **#16** | B |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| total_time_ms (bs=80) | 88.25 | 56.35 | 22.59 | 18.85 | - | - | - | - | 19.25 |
| total_time_ms (bs=2048) | 1915 | - | - | ~16.0 | 15.42 | 13.99 | 15.78 | **15.85** | 15.74 |
| raw comm_ms (bs=2048) | 0 | - | - | 1.31 | 1.81 | 1.81 | 1.81 | **1.81** | 1.82 |
| MFU (bs=2048) | 0.054 | - | - | 0.73 | 0.754 | 0.831 | 0.737 | **0.733** | 0.748 |
| FA2 tile | - | - | - | - | - | - | - | **QS=128,KS=192** | QS=128,KS=192 |
| vs B 差异 | - | - | - | - | -2.1% | -11.1% | +0.21% | **+0.67%** | - |

**注**: #13 的 -11.1% 是因为 A 无条件应用 TBO overlap 导致 comm=0。
#14 条件化 TBO + #15 kv_seq_len 修正后对齐到 +0.21%。
#16 对齐 tiling 参数后 FA2 tile 完全对齐, matmul tile 因打分策略差异仍不同, 总体 +0.67%。

### 差异根因分析 (2026-02-16 逐算子诊断)

#### [已验证] SG2262 芯片参数对齐

| 参数 | A (旧) | B (CHIPMathica) | 修正后 A |
|------|:-:|:-:|:-:|
| cores | 32 | 64 | 64 |
| frequency | 1.0 GHz | 1.5 GHz | 1.5 GHz |
| FP8 peak | 262 TFLOPS | 786.4 TFLOPS | 786.4 TFLOPS |
| DRAM BW (effective) | 2867 GB/s | 8601.6 GB/s | 8601.6 GB/s |
| SRAM/core | 0.9 MB | 2 MB | 2 MB |

**已修改**: `configs/chips/SG2262.yaml` 和 `configs/topologies/P1-R1-B4-C32.yaml`

#### [已验证] DP batch 切分正确 ~~[P11]~~

~~A 不做 DP batch 切分~~ -> **经诊断验证, A 的 DP 切分正确!**

`_apply_data_parallel_shard()` (planner.py:1218) 在 L3 对 M 维度除以 DP:
- L1: M=2048 (全局 batch * q_seq_len = 2048 * 1)
- L3 after DP shard: M=ceil(2048/32)=64 -> 与 B 的 `batch_local=2048//32=64` 一致
- FA2: B=batch_local*heads=64*128=8192 -> 也正确

#### [已验证] 投影 Op flops 完全对齐

以第一层 MLA 为例 (M=64, TP=1):

| Op | A flops | B flops | 匹配 |
|----|---------|---------|------|
| q_a (M=64,K=7168,N=1536) | 1,409,286,144 | 1,409,286,144 | [PASS] |
| q_b (M=64,K=1536,N=24576) | 4,831,838,208 | 4,831,838,208 | [PASS] |
| kv_a (M=64,K=7168,N=576) | 528,482,304 | 528,482,304 | [PASS] |
| o_proj (M=64,K=16384,N=7168) | 15,032,385,536 | 15,032,385,536 | [PASS] |

#### [已解决] MLA Absorb 模式 (P14)

**B 在 decode 时也使用 MLA Absorb**, 与 A 一致。之前的对比用了 standard 模式导致虚假差异。

修改 `DeepSeek-V3-671B-A37B.yaml`: `mla_mode: absorb`, 两者 MLA 架构对齐。

#### [已完成] CHIPMathica 精确对比 (P15)

实际运行 B (batch=2048, TP=1, DP=32, EP=32, decode), 获取逐算子数据:

**B 实际结果 (DSV3_decode_bs2048.json):**
- total_elapse=15.743ms, comm=1.818ms (raw), flops=9,260G, MFU=0.748

#### MoE 层对比 (正确)

| Op | A shape | B shape | 匹配 |
|----|---------|---------|------|
| gate | M=64, K=7168, N=256 | M=64, K=7168, N=256 | [PASS] |
| shared_gate | M=64, K=7168, N=2048 | M=64, K=7168, N=2048 | [PASS] |
| expert_gate | G=8, M=64, K=7168, N=2048 | G=8, M=64, K=7168, N=2048 | [PASS] |
| dispatch | ALL2ALL 3.67MB (FP8) | ALL2ALL 3.67MB (FP8) | [PASS] |
| combine | ALL2ALL 7.34MB (BF16) | ALL2ALL 7.34MB (BF16) | [PASS] |

---

## 测试基线参数

### tpu_sg2262_v1 (TP=1, DP=32, EP=32)
```yaml
tp: 1, dp: 32, ep: 32, moe_tp: 1
batch_size: 2048, q_len: 4096, kv_len: 4096
is_prefill: false, enable_tp_sp: true, comm_protocol: 1
weight_dtype: fp8, activation_dtype: fp8
```

### tpu_sg2262_tp2 (TP=2, DP=16, EP=16)
```yaml
tp: 2, dp: 16, ep: 16, moe_tp: 2
batch_size: 2048, q_len: 4096, kv_len: 4096
is_prefill: false
```

### tpu_sg2262_tps153 (TP=16, DP=1, EP=1)
```yaml
tp: 16, dp: 1, ep: 1, moe_tp: 1
batch_size: 2048, q_len: 4096, kv_len: 4096
is_prefill: false
```
