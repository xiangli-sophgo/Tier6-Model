# L0 Entry -- 入口与编排层

## 功能概述

L0 是系统入口层，负责:
- **API 端点**: 30+ REST 端点 + WebSocket 实时推送
- **配置转换**: dict -> EvalConfig 单一转换点
- **管线编排**: 协调 L1-L5 执行完整评估流程
- **任务管理**: 异步队列、进度回调、实验管理
- **数据存储**: SQLite 持久化 (Experiment/Task/Result)

不在范围: 不做切分/评估/报告的具体计算。

## 模块清单

| 模块 | 职责 |
|------|------|
| `api.py` | FastAPI 路由定义 (预设 CRUD, 仿真, 任务, 实验) |
| `engine.py` | `run_evaluation()` / `run_evaluation_from_request()` |
| `eval_config.py` | EvalConfig dataclass + `build_eval_config()` |
| `config_loader.py` | YAML/JSON 预设加载 (`load_chip`, `load_model`) |
| `config_schema.py` | Pydantic 请求模型 + 配置验证 |
| `types.py` | `DataType` / `ParallelMode` 等核心枚举 |
| `tasks.py` | `TaskManager` 任务队列 (ThreadPoolExecutor) |
| `websocket.py` | WebSocket 管理 (`/ws/tasks`) |
| `compat.py` | 前端兼容层 (Gantt/Stats 格式转换) |
| `topology_format.py` | 拓扑格式转换 (grouped_pods 格式) |
| `storage/database.py` | SQLAlchemy ORM (Experiment, EvaluationResult) |

## EvalConfig 详细设计

### 子配置结构

```python
@dataclass
class MLAConfig:
    q_lora_rank: int          # Q 低秩维度 (1536)
    kv_lora_rank: int         # KV 低秩维度 (512)
    qk_nope_head_dim: int     # 非 RoPE head 维度 (128)
    qk_rope_head_dim: int     # RoPE head 维度 (64)
    v_head_dim: int           # V head 维度 (128)

@dataclass
class MoEConfig:
    num_routed_experts: int       # 专家总数 (256)
    num_shared_experts: int       # 共享专家数 (1)
    num_activated_experts: int    # 激活专家数 (8)
    intermediate_size: int        # MoE FFN 中间层 (2048)

@dataclass
class ModelConfig:
    name: str
    hidden_size: int; num_layers: int; num_attention_heads: int
    vocab_size: int; intermediate_size: int
    num_dense_layers: int; num_moe_layers: int
    mla: MLAConfig; moe: MoEConfig
    # 运行时参数
    weight_dtype: str; activation_dtype: str
    seq_len: int; kv_seq_len: int; q_seq_len: int
    batch: int; is_prefill: bool

@dataclass
class TopologyOverrides:
    c2c_bandwidth_gbps: float; c2c_latency_us: float
    b2b_bandwidth_gbps: float; b2b_latency_us: float
    r2r_bandwidth_gbps: float; r2r_latency_us: float
    p2p_bandwidth_gbps: float; p2p_latency_us: float
    switch_latency_us: float; cable_latency_us: float
    memory_read_latency_us: float; memory_write_latency_us: float
    noc_latency_us: float; die_to_die_latency_us: float

@dataclass
class CommOverrides:
    bw_utilization: float     # 带宽利用率 (0-1)
    sync_lat_us: float        # 同步延迟 (us)

@dataclass
class DeploymentConfig:
    tp: int; pp: int; dp: int; ep: int; moe_tp: int
    seq_len: int; batch_size: int
    enable_tp_sp: bool; enable_ring_attention: bool; enable_zigzag: bool
    embed_tp: int; lmhead_tp: int; comm_protocol: int
    kv_cache_rate: float; is_prefill: bool
    q_seq_len: int; kv_seq_len: int

@dataclass
class BoardConfig:
    num_chips: int; chip_memory_gb: int; inter_chip_bw_gbps: float

@dataclass
class InferenceConfig:
    batch_size: int; input_seq_length: int; output_seq_length: int
    weight_dtype: str; activation_dtype: str
```

### build_eval_config() 转换逻辑

```
输入:
  chip_config: dict       <- topology_config.chips 的第一个芯片
  model_config: dict      <- benchmark_config.model (嵌套 YAML 格式)
  topology_config: dict   <- 完整拓扑 (含 interconnect.links + comm_params)
  manual_parallelism: dict <- 前端并行配置
  inference_config: dict  <- benchmark_config.inference

转换:
  1. topology_config.interconnect.links -> TopologyOverrides (c2c/b2b/r2r/p2p bw+lat)
  2. topology_config.interconnect.comm_params -> TopologyOverrides (switch/cable/memory/noc/d2d lat)
  3. topology_config.interconnect.comm_params -> CommOverrides (bw_utilization, sync_lat)
  4. manual_parallelism + inference_config -> DeploymentConfig
  5. model_config(嵌套) + 运行时参数 -> ModelConfig (含 MLAConfig, MoEConfig)
  6. topology_config.pods 结构 -> BoardConfig (num_chips, chip_memory)
  7. inference_config -> InferenceConfig
  8. 所有字段缺失时 raise ValueError

输出:
  EvalConfig (全管线单一 source of truth)
```

## run_evaluation() 编排流程

```python
def run_evaluation(eval_config: EvalConfig, progress_callback=None) -> dict:
    # L1: 构建 WorkloadIR
    model = DeepSeekV3Model.from_model_config(eval_config.model)
    ir = model.to_ir()                                    # 0.05

    # L2: 加载 ChipSpec
    chip = ChipSpecImpl.from_config(name, eval_config.chip_config)  # 0.08

    # L3a: Parallelism Planning
    deployment = DeploymentSpec(...)  # 从 eval_config.deployment
    board = BoardSpec(...)            # 从 eval_config.board
    dist_model = ParallelismPlanner(deployment, board).plan(ir)     # 0.12

    # L3b: Tiling Planning
    tile_plan = TilingPlanner(chip, l4_evaluator).plan(dist_model)  # 0.75

    # L3c: Scheduling
    exec_plan = Scheduler().plan(dist_model, tile_plan)             # 0.78

    # L4: Evaluation
    hardware = _build_hardware_spec(chip, eval_config)  # 使用 topology/comm
    engine_result = EvaluationEngine().evaluate(exec_plan, ...)     # 0.85

    # L5: Reporting
    report = ReportingEngine().run(engine_result, config=run_config) # 0.95

    return result
```

### _build_hardware_spec() -- 核心 Bug 修复

旧版本使用 `TopologySpec()` 默认值 (c2c=400, b2b=200)，用户配置的带宽被覆盖。

新版本直接从 `eval_config.topology` 注入所有 14 个参数:
```python
topology_spec = TopologySpec(
    c2c_bandwidth_gbps=topo.c2c_bandwidth_gbps,  # 来自 YAML, 如 448
    b2b_bandwidth_gbps=topo.b2b_bandwidth_gbps,  # 来自 YAML, 如 450
    ...  # 全部 14 个字段
)
comm_spec = CommProtocolSpec(
    bw_utilization=eval_config.comm.bw_utilization,
    sync_lat_us=eval_config.comm.sync_lat_us,
)
hardware = merge_specs(hardware_spec, topology_spec, comm_spec)
```

## API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/presets/chips` | 列出芯片预设 |
| GET | `/api/presets/models` | 列出模型预设 |
| GET | `/api/topologies` | 列出拓扑配置 |
| GET | `/api/benchmarks` | 列出 Benchmark |
| POST | `/api/simulate` | 同步仿真 (EvaluationRequest) |
| POST | `/api/validate` | 配置验证 |
| POST | `/api/evaluation/submit` | 异步评估任务提交 |
| GET | `/api/evaluation/tasks` | 查询任务状态 |
| GET | `/api/evaluation/experiments` | 查询实验列表 |
| POST | `/api/evaluation/experiments/export` | 导出实验 |
| POST | `/api/evaluation/experiments/check-import` | 导入检查 |
| POST | `/api/evaluation/experiments/execute-import` | 执行导入 |
| WS | `/ws/tasks` | 实时任务状态推送 |

## 数据持久化

```
localStorage (前端)        -- 临时 UI 状态缓存
     |
in-memory TaskQueue       -- 会话级任务队列 (ThreadPoolExecutor)
     |
SQLite (SQLAlchemy)        -- 永久存储
  +-- Experiment           -- 实验元数据 (name, description, timestamps)
  +-- EvaluationTask       -- 任务状态 (status, progress, config_snapshot)
  +-- EvaluationResult     -- 结果数据 (tps, mfu, full_result JSON)
     |
JSON Export                -- 离线快照 (导入/导出)
```

## 配置预设系统

```
backend/math_model/configs/
+-- chip_presets/
|   +-- sg2262.yaml          # SG2262 (4 cores, 768 TFLOPS FP8)
|   +-- sg2260e.yaml         # SG2260E
|   +-- h100.yaml            # NVIDIA H100
+-- model_presets/
|   +-- deepseek-v3.yaml     # 671B MoE, 256E/8A, MLA
|   +-- deepseek-r1.yaml     # DeepSeek R1
|   +-- qwen3-235b.yaml      # Qwen 3 235B MoE
+-- topologies/
|   +-- P1-R1-B1-C8.yaml     # 1 pod, 1 rack, 1 board, 8 chips
|   +-- TOPOLOGY_TEMPLATE.yaml
+-- benchmarks/
    +-- *.json                # 8 个预设测试场景
```
