# L3 Mapping -- 映射与调度层

## 功能概述

L3 将 L1 的 WorkloadIR 映射到 L2 的硬件上，分三个阶段:
1. **ParallelismPlanner**: 芯片间切分 (TP/PP/DP/EP) + 插入通信算子
2. **TilingPlanner**: 片内 Tile 映射 + Kernel 选择
3. **Scheduler**: 时序调度 + 依赖排序

输出: ExecPlan (timeline + binding + kernel_config)

不在范围: 不做性能评估 (由 L4)，不生成报告 (由 L5)。

## 模块清单

| 模块 | 职责 |
|------|------|
| `parallelism/planner.py` | ParallelismPlanner, DeploymentSpec, BoardSpec |
| `parallelism/parallel_spec.py` | ParallelSpec, ParallelType |
| `parallelism/pattern_rules.py` | 并行模式规则 (embedding, MLA, FFN, MoE) |
| `tiling/planner.py` | TilingPlanner |
| `tiling/evaluators.py` | Tile 候选评估器 |
| `scheduling/scheduler.py` | Scheduler |
| `plan/distributed_model.py` | DistributedModel, DistributedOp |
| `plan/exec_plan.py` | ExecPlan |

## ParallelismPlanner

### 输入/输出

- **输入**: `DeploymentSpec` + `BoardSpec` + `WorkloadIR`
- **输出**: `DistributedModel`

### DeploymentSpec

```python
@dataclass
class DeploymentSpec:
    tp: int = 1          # Tensor Parallelism
    pp: int = 1          # Pipeline Parallelism
    ep: int = 1          # Expert Parallelism
    moe_tp: int = 1      # MoE 内部 TP
    dp: int = 1          # Data Parallelism
    seq_len: int = 2048
    batch_size: int = 1
    enable_tp_sp: bool = False
    enable_ring_attention: bool = False
    embed_tp: int = 1
    lmhead_tp: int = 1
    comm_protocol: int = 1
    kv_cache_rate: float = 0.0
    is_prefill: bool = False
```

### 计算流程

1. **约束校验**: 验证 `dp * tp == moe_tp * ep` (MoE 层)
2. **PP 分组**: 按 `pp` 将 layers 划分为 stages
3. **Op 切分**: 按 pattern rules 为每个 Op 选择 ParallelSpec
4. **通信插入**: 在切分边界插入通信算子
5. **输出**: DistributedModel (计算 + 通信 Op DAG)

### Pattern Rules

| 层类型 | TP 策略 | 通信 |
|--------|---------|------|
| Embedding | 按 embed_tp 切分 vocab | AllGather (后续需完整 hidden) |
| MLA Q/KV proj | 按 TP 切分 output_dim | - |
| MLA QK matmul | 按 TP 切分 heads | - |
| MLA output proj | 按 TP 切分 input_dim | AllReduce (归约) |
| FFN gate/up | 按 TP 切分 intermediate | - |
| FFN down | 按 TP 切分 input_dim | AllReduce (归约) |
| MoE Router | 不切 | - |
| MoE Expert FFN | 按 moe_tp 切分 | All2All (dispatch/combine) |
| LMHead | 按 lmhead_tp 切分 vocab | AllGather |

### DistributedModel

```python
@dataclass
class DistributedModel:
    ops: dict[str, DistributedOp]    # op_id -> op
    edges: list[tuple[str, str]]     # 依赖边 (src, dst)
    metadata: dict                   # 元数据

@dataclass
class DistributedOp:
    op_id: str
    role: NodeRole          # COMPUTE | COMM
    op_type: str            # "matmul", "allreduce", ...
    local_shape: dict[str, int]  # 切分后的 shape
    parallel_spec: ParallelSpec | None

    # 通信专属字段 (role=COMM)
    comm_type: CommType | None      # ALLREDUCE, ALLGATHER, ALL2ALL, P2P
    comm_bytes: int
    participants: list[int]          # 参与芯片 ID
    topology_path_key: str | None   # "c2c", "b2b", "r2r", "p2p"
    reason: str | None              # "tp_reduce", "moe_dispatch", ...
```

## TilingPlanner

### 输入/输出

- **输入**: `DistributedModel` + `ChipSpecImpl` + `L4TileEvaluator` (可选)
- **输出**: `TilePlan`

### 计算流程

1. **候选生成**: 按 op_type 生成 tile 候选 (M*K*N blocking)
2. **约束剪枝**: SRAM 上限、对齐约束、Cube 整除
3. **快速预筛**: 简化 roofline 预估 + 帕累托剪枝
4. **精评估**: 调用 L4 PreciseTileEvaluator (traffic/urate/loop-order)
5. **选优**: 选择最短执行时间的 tile
6. **缓存**: 对同构 op 做内存缓存 + 可选 SQLite 持久缓存

### TilePlan

```python
@dataclass
class TilePlan:
    tile_configs: dict[str, TileConfig]     # op_id -> tile
    kernel_configs: dict[str, dict]         # op_id -> kernel 元数据
    intra_chip_comms: list[DistributedOp]   # 片内通信 (tiling 引起)
```

### Tile 候选规则 (对齐 CHIPMathica)

- **分区合法**: P_* 必须整除 core_count
- **Tile 对齐**: tile_m/n/k 满足 cube_m/k/n 整除
- **SRAM 上限**: 单 tile 占用 <= 0.45 * SRAM
- **帕累托剪枝**: 各维度均不小于候选时丢弃
- **流量优先**: 选择最小 DRAM traffic 的 tile + loop-order
- **FA2 细化**: Q/K tile 满足 Q+K+V+2P+4O buffer 预算

## Scheduler

### 输入/输出

- **输入**: `DistributedModel` + `TilePlan`
- **输出**: `ExecPlan`

### 计算流程

1. **拓扑排序**: Kahn 算法，确保依赖约束
2. **优先级**: fanout (默认) 或 critical_path 模式
3. **实例展开**: Op/CommOp -> timeline event
4. **资源统计**: 记录 core_slots 与 path_slots
5. **Buffer 估算**: 追踪 buffer 峰值

### ExecPlan

```python
@dataclass
class ExecPlan:
    timeline: list[dict]        # [{op_id, duration, wait_time}, ...]
    binding: dict[str, Any]     # op_id -> chip_id/core_ids
    precedence: list[tuple]     # 依赖边
    kernel_config: dict[str, dict]  # op_id -> kernel 参数
    trace_meta: dict            # 调度元数据
```

### Timeline Event

```python
{
    "op_id": "layers.0.mla.q_proj",
    "start_step": 0,
    "duration": 1,
    "wait_time": 0.0,
    "stage": 0,
}
```

## 通信算子类型

| CommType | 用途 | 典型场景 |
|----------|------|----------|
| ALLREDUCE | 梯度/激活归约 | TP 内 MLA/FFN output_proj |
| ALLGATHER | 参数/激活广播 | Embedding 后, TP-SP |
| REDUCE_SCATTER | 归约 + 分散 | TP-SP MLA |
| ALL2ALL | 全交换 | MoE dispatch/combine |
| P2P | 点对点 | PP stage 边界 |

## 并行策略分配顺序

内到外: TP -> EP -> PP -> DP

- **TP 组**: 优先同 board 芯片 (高带宽 NVLink/c2c)
- **EP 组**: 可跨 board
- **PP 组**: 可跨 rack (P2P 通信)
- **DP 组**: 可跨 pod
