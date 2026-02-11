# Math Model 总体架构设计

## 概述

math_model 是一个 LLM 推理部署数学建模框架，采用 5 层分层架构 (L0-L5)，
实现从模型定义到性能评估的完整管线。核心目标:

1. **精确建模**: 对齐 CHIPMathica 方法论，支持 Chip/Core/Lane 多精度评估
2. **类型安全**: 使用 `EvalConfig` dataclass 作为单一配置入口，消除无类型 dict 传递
3. **无静默默认值**: 所有必需参数缺失时 raise 异常，杜绝配置错误被掩盖
4. **可扩展性**: Registry 模式支持新增算子、芯片、评估器

## 分层架构

```
     Frontend (React + TypeScript + Three.js)
         |
         | REST API / WebSocket
         v
  +----- L0: Entry & Orchestration -----+
  |  api.py / engine.py / eval_config.py |
  |  config_loader / tasks / websocket   |
  +--------------------------------------+
         |
         | EvalConfig (typed dataclass)
         v
  +----- L1: Workload Modeling ----------+
  |  DeepSeekV3Model -> WorkloadIR       |
  |  Layer/Op/TensorDesc/ComputeSpec     |
  +--------------------------------------+
         |
         | WorkloadIR (Model + Graph)
         v
  +----- L2: Hardware Architecture ------+
  |  ChipSpecImpl / TopologySpec         |
  |  Pod / Rack / Board / Chip / Core    |
  +--------------------------------------+
         |
         | ChipSpec + TopologySpec
         v
  +----- L3: Mapping & Scheduling ------+
  |  ParallelismPlanner -> DistributedModel |
  |  TilingPlanner -> TilePlan              |
  |  Scheduler -> ExecPlan                  |
  +--------------------------------------+
         |
         | ExecPlan + DistributedModel
         v
  +----- L4: Evaluation Engine ----------+
  |  EvaluationEngine -> EngineResult    |
  |  ChipCostModel / CoreCostModel       |
  |  CommProtocolCostModel               |
  +--------------------------------------+
         |
         | EngineResult (StepMetrics + Aggregates)
         v
  +----- L5: Reporting & Analysis ------+
  |  ReportingEngine -> Report           |
  |  CostAnalyzer / Gantt / Roofline    |
  +--------------------------------------+
         |
         v
     Frontend Visualization
```

## 核心数据流

```
1. Frontend JSON Request
   |
2. L0: run_evaluation_from_request(config: dict)
   |  - 提取 benchmark/topology/parallelism
   |  - build_eval_config() -> EvalConfig (单一转换点)
   |
3. L1: DeepSeekV3Model.from_model_config(eval_config.model)
   |  - 构建 Layer 列表 (Embedding, MLA, FFN/MoE, LMHead)
   |  - 生成 WorkloadIR (Model + Graph)
   |
4. L2: ChipSpecImpl.from_config(name, eval_config.chip_config)
   |  - 加载芯片微架构参数 (TFLOPS, 内存, Cube 尺寸)
   |
5. L3a: ParallelismPlanner.plan(ir)
   |  - 按 PP 划分 stage, 按 TP/EP 切分 op
   |  - 插入通信算子 (AllReduce, AllGather, P2P)
   |  -> DistributedModel
   |
6. L3b: TilingPlanner.plan(dist_model)
   |  - 枚举 tile 候选 (M*K*N blocking)
   |  - 用 PreciseTileEvaluator 精评估 (traffic/urate)
   |  -> TilePlan
   |
7. L3c: Scheduler.plan(dist_model, tile_plan)
   |  - 拓扑排序 + 优先级调度
   |  -> ExecPlan (timeline + binding)
   |
8. L4: EvaluationEngine.evaluate(exec_plan, hardware)
   |  - 遍历 timeline, 按 op_type 路由到评估器
   |  - 估算 t_compute, t_comm, t_wait
   |  - 聚合 TPS/TTFT/TPOT/MFU/MBU
   |  -> EngineResult
   |
9. L5: ReportingEngine.run(engine_result)
   |  - 装配性能报告
   |  - 计算部署成本 (CostAnalyzer)
   |  - 生成 Gantt/Roofline/内存分析
   |
10. 返回前端 -> 可视化
```

## 配置管线 (EvalConfig)

### 设计原则

- **单一转换点**: dict -> EvalConfig 只在 `build_eval_config()` 发生一次
- **类型安全**: 全管线传递 EvalConfig dataclass，不再传递 `dict[str, Any]`
- **无默认值**: 所有必需字段缺失时 raise ValueError
- **数据不丢失**: topology/comm 参数直接注入 `_build_hardware_spec()`

### EvalConfig 结构

```python
@dataclass
class EvalConfig:
    model: ModelConfig          # 模型参数 (含 MLA/MoE 嵌套配置)
    chip_config: dict           # 芯片 raw dict (给 ChipSpecImpl)
    topology: TopologyOverrides # 4 级带宽/延迟 + comm 延迟参数
    comm: CommOverrides         # 通信协议参数 (bw_utilization, sync_lat)
    deployment: DeploymentConfig # 并行策略 (TP/PP/DP/EP/MoE_TP)
    board: BoardConfig          # 板卡规格 (num_chips, memory)
    inference: InferenceConfig  # 推理参数 (batch, seq_len, dtype)
    raw_model_config: dict      # 报告/快照用
    raw_topology_config: dict   # 报告/快照用
```

### 与旧管线的对比

| 问题 | 旧管线 | 新管线 (EvalConfig) |
|------|--------|---------------------|
| 数据丢失 | topology_overrides 被提取但未注入 hardware | TopologyOverrides 直接注入 _build_hardware_spec |
| 静默默认值 | `.get(key, 400.0)` 掩盖配置错误 | `hardware["key"]` 缺失即 KeyError |
| 扁平化 | _map_model_config() 嵌套->flat+rename | from_model_config() classmethod 在模型侧转换 |
| 两套 API | SimulateRequest vs EvaluationRequest | 统一使用 EvaluationRequest |
| 无类型 dict | `config: dict[str, Any]` 全管线 | `eval_config: EvalConfig` 全管线 |

## 目录结构

```
backend/math_model/
+-- main.py                    # FastAPI 入口 (port 8003)
+-- L0_entry/                  # 入口层
|   +-- api.py                 # 30+ REST 端点
|   +-- engine.py              # 评估编排器
|   +-- eval_config.py         # EvalConfig + build_eval_config()
|   +-- config_loader.py       # YAML/JSON 预设加载
|   +-- config_schema.py       # Pydantic 验证模型
|   +-- types.py               # DataType, ParallelMode 枚举
|   +-- tasks.py               # 任务队列管理
|   +-- websocket.py           # WebSocket 推送
|   +-- compat.py              # 前端兼容层
|   +-- storage/database.py    # SQLAlchemy ORM
+-- L1_workload/               # 负载建模层
|   +-- ir.py                  # WorkloadIR 协议
|   +-- layer.py               # Layer dataclass
|   +-- op.py                  # Op dataclass
|   +-- models/llm/            # 模型实现
|   +-- layers/                # 层实现 (MLA, FFN, MoE, ...)
|   +-- operators/             # 算子实现
+-- L2_arch/                   # 硬件架构层
|   +-- chip.py                # ChipSpecImpl
|   +-- board.py               # BoardSpecImpl
|   +-- rack.py                # RackSpecImpl
|   +-- pod.py                 # PodSpecImpl
|   +-- topology.py            # TopologySpec
|   +-- core.py                # CoreSpecImpl
|   +-- memory.py              # MemoryHierarchyImpl
|   +-- interconnect.py        # InterconnectSpecImpl
+-- L3_mapping/                # 映射层
|   +-- parallelism/           # 并行策略规划
|   +-- tiling/                # 片内 tile 规划
|   +-- scheduling/            # 时序调度
|   +-- plan/                  # DistributedModel, ExecPlan
+-- L4_evaluation/             # 评估层
|   +-- engine.py              # EvaluationEngine
|   +-- metrics.py             # HardwareSpec, StepMetrics, Aggregates
|   +-- cost_models/           # Chip/Core 级代价模型
|   +-- evaluators/            # 计算/通信评估器
+-- L5_reporting/              # 报告层
|   +-- engine.py              # ReportingEngine
|   +-- cost_analysis.py       # CostAnalyzer
|   +-- gantt.py               # Gantt 图生成
|   +-- roofline.py            # Roofline 分析
+-- configs/                   # 预设配置
    +-- chip_presets/           # 芯片 YAML
    +-- model_presets/          # 模型 YAML
    +-- topologies/             # 拓扑 YAML
    +-- benchmarks/             # 测试场景 JSON
```

## 依赖边界

```
L0 -> L1, L2, L3, L4, L5  (编排全流程)
L1 -> 无外部依赖            (纯负载建模)
L2 -> 无外部依赖            (纯硬件描述)
L3 -> L1 (WorkloadIR), L2 (ChipSpec), L4 (PreciseTileEvaluator)
L4 -> L2 (TopologySpec), L3 (ExecPlan, DistributedModel)
L5 -> L4 (EngineResult)
```

注意: L3 的 TilingPlanner 依赖 L4 的 PreciseTileEvaluator 做精评估，
这是有意的设计 -- TilingPlanner 在候选枚举阶段需要 L4 的精确评估能力。

## 支持的模型架构

| 模型 | 类型 | 特殊架构 | 实现 |
|------|------|----------|------|
| DeepSeek V3 | MoE | MLA + MoE (256E/8A) | DeepSeekV3Model |
| DeepSeek R1 | MoE | MLA + MoE | DeepSeekV3Model (配置区分) |
| Qwen 3-235B | MoE | GQA + MoE | DeepSeekV3Model (适配) |
| LLaMA | Dense | GQA | LlamaModel |

## 支持的并行策略

| 策略 | 缩写 | 切分维度 | 通信模式 |
|------|------|----------|----------|
| Tensor Parallelism | TP | hidden_size | AllReduce / ReduceScatter+AllGather |
| Pipeline Parallelism | PP | num_layers | P2P Send/Recv |
| Data Parallelism | DP | batch_size | AllReduce (梯度) |
| Expert Parallelism | EP | num_experts | All2All (dispatch/combine) |
| Sequence Parallelism | SP | seq_len | AllGather / ReduceScatter |
| MoE TP | MoE_TP | expert 内部 TP | AllReduce |
