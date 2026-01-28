# LLM推理建模平台架构设计参考

> 基于外部架构设计文档整理，为 Tier6-Model 项目提供架构设计参考和改进方向
>
> 创建时间：2026-01-28

## 文档概述

本文档梳理了一份成熟的 LLM 推理数学建模平台的架构设计思想，该设计采用**六层架构（L0-L5）**，强调**粒度分层**、**单一真相来源（SSOT）**和**渐进式细化**。这些设计原则对 Tier6-Model 项目的架构优化具有重要参考价值。

### 核心设计目标

1. **性能评估**：输入部署参数 → 输出可解释的性能指标与瓶颈归因
2. **方案搜索**：给定 SLO 约束 → 自动搜索最优/次优方案列表
3. **参数分析**：参数敏感性分析、硬件升级收益预测
4. **时序可视化**：生成 Gantt 图，展示计算/通信/气泡分布

---

## 一、六层架构设计（L0-L5）

### 整体数据流

```
model.yaml + hardware.yaml
         ↓
    L1: WorkloadIR + L2: HardwareSpec
         ↓
    L0: SweepRunner（参数扫描）
         ↓
    ┌─────────────┬─────────────┬─────────────┐
    ↓             ↓             ↓
L3: Mapping   L3: Mapping   L3: Mapping
    ↓             ↓             ↓
L4: Evaluate  L4: Evaluate  L4: Evaluate
    └─────────────┴─────────────┘
         ↓
    筛选最优 → L5: 输出报告
```

### L0: Entry & Orchestration（入口与编排层）

**职责**：用户入口 + 实验编排 + 配置管理

**核心组件**：
- **ConfigLoader**：加载模型/硬件/部署配置，校验约束（如 TP×PP×DP=N_chips）
- **SweepRunner**：遍历参数组合，调用 L3+L4 评估，筛选最优方案

**设计要点**：
- SweepRunner 可先用 L4 粗粒度评估，筛选 Top-K，再做细粒度建模
- 职责边界：只做编排，不做具体映射和评估

**对应 Tier6-Model 现状**：
- 当前：前端交互 + API 层混合在一起
- 可改进：独立出配置验证和参数扫描逻辑

---

### L1: Workload Representation（工作负载表示层）

**职责**：将模型/算子图统一表示为可分析的 IR

**核心组件**：
- **WorkloadIR**：统一模型表示（Model → Module → Layer → Op）
- **ModelBuilder**：从 YAML/ONNX 构建 WorkloadIR

**IR 最小字段约定**（面向 L3 的通信与分片推导）：

#### 张量（Tensor/Edge）最小字段
- `shape`：全局 shape（未切分视角）
- `dtype`：数据类型（用于 bytes 计算）
- `producer_id` / `consumer_id`：生产者/消费者 op 标识
- `layout_signature`：布局签名（允许为空，L3 填充后用于通信判定）

#### layout_signature 结构
```
{
  parallel_type: TP/PP/DP/EP/SP（或 NONE）
  split_dim: 切分维度（如 heads/hidden/sequence/expert）
  split_factor: 切分因子（与并行度一致）
  replica_group_id: 副本组标识（用于 DP/PP 语义区分）
}
```

**IR 层级示例**：
```
Model (e.g., Llama2-70B)
├── modules: list[Module]
│   ├── EmbeddingModule
│   ├── TransformerBlock[0..79]
│   │   ├── AttentionModule
│   │   │   ├── QKVProjection (Layer)
│   │   │   │   └── MatMul (Op) → MM2_NN (AtomicIns)
│   │   │   ├── Softmax (Layer)
│   │   │   └── OutputProjection (Layer)
│   │   └── FFNModule
│   │       ├── GateProj / UpProj / DownProj
│   │       └── SiLU / Multiply
│   └── LMHeadModule
└── metadata: {hidden_size, num_layers, num_heads, ...}
```

**对应 Tier6-Model 现状**：
- 当前：模型表示较为扁平，缺少结构化 IR
- 可改进：引入 Module/Layer/Op 分层结构，增强可解释性

---

### L2: Architecture Spec（架构规格层）

**职责**：描述各种层级的硬件参数

**主要结构**：

#### 1. ClusterSpec（集群级规格）
- `num_nodes`：节点数量
- `nodes`：NodeSpec 列表
- `inter_node_bandwidth_gbps`：跨节点带宽（IB/RoCE）
- `inter_node_latency_us`：跨节点延迟
- `topology_ref`：指向 TopologySpec 的引用

#### 2. NodeSpec（节点级规格）
- `node_id`：稳定标识（用于 rank 映射与排序）
- `boards`：BoardSpec 列表
- `chips_per_node`：每节点芯片数
- `intra_node_bandwidth_gbps`：节点内互联带宽
- `intra_node_latency_us`：节点内互联延迟

#### 3. BoardSpec（板卡级规格）
- `board_id`：稳定标识
- `chips`：ChipSpec 列表
- `fabric_tag`：互联标签（如 nvlink/pcie/mesh）

#### 4. ChipSpec（芯片级规格）
- `chip_id`：稳定标识
- `compute_tflops`：峰值算力
- `memory_gb`：显存容量
- `memory_bandwidth_gbps`：显存带宽
- `num_cores`：Core 数量
- `intra_chip_fabric_tag`：片内互联标签

#### 5. CoreSpec（核心级规格）
- `core_id`：稳定标识
- `num_lanes`：Lane 数量
- `sram_per_core_kb`：每 Core SRAM 容量
- `cube_mac_per_lane`：Cube 单元 MAC 数
- `vector_eu_per_lane`：Vector 单元 EU 数

#### 6. TopologySpec（通信拓扑规格，一等公民）
- `nodes` / `boards` / `chips`：层级资源清单与从属关系
- `rank_map`：chip_id → rank 的稳定映射
- `link_profiles`：分层链路参数（intra_board / inter_board / inter_node）
- `path_keys`：路径键枚举（作为 L3→L4 的桥接字段）

**设计要点**：
- TopologySpec 是**一等公民**，L3/L4 共享同一份拓扑口径
- 链路参数按层级组织，避免硬编码

**对应 Tier6-Model 现状**：
- 当前：拓扑配置较完整，但缺少 CoreSpec 和细粒度层级
- 可改进：引入 Board/Core/Lane 层级，支持渐进式细化

---

### L3: Mapping & Scheduling（映射与调度层）

**职责**：将 WorkloadIR 映射为分布式执行图，并生成执行计划

**L3 分为两段口径清晰的映射**：

#### L3-A: ParallelismPlanner（跨 Chip 映射）

**职责**：
- 给定一组确定的并行参数，将 WorkloadIR 映射为多 Chip 分布式执行图
- 根据"边的布局不一致（layout mismatch）"自动插入通信算子（**SSOT**）

**输入**：
- WorkloadIR（Layer 列表、Op 列表 + Op 输入输出张量）
- DeploymentSpec（tp, pp, dp, ep, moe_tp, sp）
- TopologySpec（芯片列表、节点/板卡拓扑、链路带宽/延迟参数）

**输出（DistributedModel）**：
1. **graph**：原算子节点 + 新插入的 CommOp 节点
   - 通信节点类型统一为 **CommOp**，字段：
     ```
     scope: inter_chip
     cause: layout_mismatch
     参数: tensor_bytes, participants, src/dst, topology_path_key, algo_hint
     ```
   - 边：必须保留"由哪个原始张量依赖边触发"的可追溯信息（`trigger_edge_id`, `reason`）

2. **chip_assignments**：每个 Op 或 OpInstance 的 chip/rank 列表
   ```
   op_id -> [chip_ids]（TP/DP/EP 组上的芯片集合）
   ```

3. **parallel_groups**：
   - `tp_groups[]` / `dp_groups[]` / `ep_groups[]` / `pp_stages[]`（每个是 chip 列表）

4. **op_parallel_specs**：
   ```
   op_id -> ParallelSpec（包含：parallel_type, split_dim, split_factor, layout_signature）
   ```

**计算流程**：

1. **构建并行组**：
   - 根据 DeploymentSpec（tp=4, pp=2, dp=1, ...）为每组分配芯片
   - 生成 `dp_groups`，每个 dp_group 内划分 `pp_stages`，每个 pp_stage 内划分 `tp_groups`

2. **Layer 到 Stage 的分配**：
   - 按 PP 参数将不同 Layer 分配到 stage
   - PP 切分策略：
     - **均分策略**（MVP 默认）：按 layer 数量均分
     - **代价均衡策略**（可选）：按粗估代价（FLOPs/bytes）均衡
     - **外部指定策略**：允许输入显式 stage 划分

3. **全局合法性检查**：
   - 维度整除：`heads % tp == 0`, `kv_heads % tp == 0`, `experts % ep == 0`, `ffn_dim % tp == 0`
   - 组规模一致性：`tp×pp×dp` 必须不超过可用芯片数

4. **模式匹配**（确定 Layer 内各 Op 的分片维度）：

   | Layer 类型 | Op 位置              | ParallelSpec     | 通信            |
   |-----------|---------------------|------------------|----------------|
   | MLP       | gate_proj / up_proj | TP_COL（列切分）  | 无              |
   | MLP       | down_proj           | TP_ROW（行切分）  | AllReduce      |
   | Attention | qkv_proj            | TP_COL           | 无              |
   | Attention | o_proj              | TP_ROW           | AllReduce      |
   | MoE       | expert_i            | EP（专家切分）    | All-to-All（前后）|

5. **根据 Op 的分片维度计算切分后的 local_shape**

6. **根据"边的布局不一致（layout mismatch）"自动插入通信算子（SSOT）**：
   - PP stage 间插入 P2P
   - 根据切分类型插入 AllReduce/AllGather/All-to-All
   - 自动推导通信算子的参数：
     - 通信量字节数（`tensor_bytes`）与参与者列表（`participants`）
     - `topology_path_key`（链路层级键），供 L4 做带宽/延迟映射
     - `src`、`dst`（P2P）

**并行映射不变量**（建议在 L3 统一校验）：
- **组覆盖与互斥**：
  - 所有可用 chip 必须被 `dp_groups` 完整覆盖且不重叠
  - 每个 `dp_group` 内的 `pp_stages` 覆盖该 group 且不重叠
  - 每个 `pp_stage` 内的 tp/ep 组覆盖该 stage 且不重叠
- **rank 顺序稳定**：`parallel_groups` 中的 chip 顺序必须稳定且可复现
- **形状可逆推导**：每个 Op 的 `global_shape` 与 `local_shape` 必须能通过 `ParallelSpec` 可逆推导
- **通信可解释**：每个通信节点必须能追溯到触发它的原始边（`trigger_edge_id`）与触发原因（`reason=layout mismatch` 摘要）

**设计要点**：
- **边驱动通信**：通信由"producer/consumer 布局不一致"触发，避免硬编码导致的多余通信或漏通信
- **模式复用**：常见 Layer 类型预定义 pattern，减少手动配置
- **通信参数继承**：从 TopologySpec 的链路参数推导，不硬编码

**对应 Tier6-Model 现状**：
- 当前：`map_parallelism()` 实现了基本的并行组映射
- 可改进：
  - 引入 `layout_signature` 机制
  - 统一 CommOp 表示（当前通信插入逻辑分散）
  - 增强可解释性（记录 `trigger_edge_id` 和 `reason`）

---

#### L3-B: TilingPlanner（片内映射层）

**职责定位**：
- 片内映射层（IntraChip/KernelPlanner 口径）
- 对经过 Chip 切分后的 op 或 op 组（如 layer）做片内映射/Kernel 选择
- 并产生片内通信/同步需求

**口径边界（与 SSOT 对齐）**：
- 跨 chip / 跨并行组的通信插入仍由"边的布局不一致（layout mismatch）"唯一决定（SSOT），发生在 ParallelismPlanner（L3 布局阶段）
- TilingPlanner 只产生**片内（intra-chip）通信/同步需求**，并统一表示为 CommOp：
  ```
  scope: intra_chip
  cause: tiling_reduce | tiling_barrier | tiling_relayout
  ```
  例如：
  - 片内归约（multi-core partial sum → reduce）
  - 片内屏障（下一阶段依赖所有 core 完成）
  - 片内搬运/重排（tile 在 core 之间接力或重排）

**输入**：
- `op_spec` / `local_shape`（来自 ParallelismPlanner 的分片结果，Core 视角）
- `ChipSpec`
- `constraints` / `policy`（如 SRAM 预算、对齐约束、可用 core 集合、是否允许 double buffer）

**输出**：
- `TileConfig` / `KernelConfig`（主输出：片内执行配置与关键参数绑定）
- `IntraChipDemand`（片内 CommOp 需求边，`scope=intra_chip`）
- `CostHints`（瓶颈标签与次优候选摘要，用于解释与调参）

**计算流程**：

1. **映射上下文**：
   - 为每个 op 构建片内映射上下文
   - 统一输入口径为 `op_spec/local_shape` + `ChipSpec` + `constraints/policy`
   - 输出为候选空间与约束集合

2. **Core 候选生成**：
   - 生成 Core 分配候选
   - 采用"启发式初值 + 有限候选集合"的策略，避免对 `core_num` 做全组合枚举
   - 现阶段就做全组合枚举

3. **形状推导**：
   - 在每个 Core 分配候选下推导 Core 视角的 `local_shape`
   - 并据此确定可行的并行轴与切分顺序

4. **候选生成**：
   - 基于 `local_shape` 生成 tile / kernel 候选参数
   - 如 tile 大小、是否 double buffer、是否需要片内归约

5. **硬约束剪枝**：
   - 对候选执行硬约束剪枝
   - 至少包含 SRAM 容量约束与对齐/整除约束（向量宽度、Cube 宽度、bank/stride 友好）

6. **快速预筛**：
   - 对剪枝后的候选执行快速代价模型预筛
   - 使用简化 roofline 或 bytes+flops 估算瓶颈类型
   - 并做支配剪枝以压缩调用 L4 的候选数量

7. **精细评估**：
   - 对预筛后的少量候选调用 L4 Core 级评估器
   - 得到执行时间、片内需求量口径与瓶颈归因

8. **片内需求生成**：
   - 在候选需要跨 core 归约/屏障/重排时
   - 生成统一类型的片内 CommOp 需求边（`scope=intra_chip`）
   - 并填充 `cause` 与 `path_key`（如 `intra_noc`）

9. **结果复用**：
   - 对同构 op（相同 `op_type/local_shape/dtype/chip_key/policy_key`）做结果缓存与复用
   - 避免重复评估相同候选族

10. **选优产出**：
    - 默认选择执行时间最小且约束满足的候选作为主输出（`TileConfig` / `KernelConfig`）
    - 并在 `CostHints` 中可选保留少量 Top-K 摘要用于解释与调参

**对应 Tier6-Model 现状**：
- 当前：暂无片内映射层（主要在 Chip 粒度建模）
- 可改进：为支持更细粒度优化，可引入 TilingPlanner（Phase 3 工作）

---

#### L3-C: Scheduler（调度器）

**职责**：
- 消费 DistributedModel 中已存在的 Op/CommOp 节点与 Tile/KernelConfig
- 构建计算与通信的时序安排与资源绑定
- **Scheduler 不再新增通信节点**

**输入**：
- DistributedModel（已包含统一类型的 CommOp，含 `scope/cause/path_key` 等字段）
- TileConfig / KernelConfig

**输出（ExecPlan）**：
- `timeline`：每个可调度单元（Op + CommOp）的 start/end 与阶段标记
- `binding`：资源绑定结果（如 chip_id、core_ids、path_key、buffer_id）
- `precedence`：调度后依赖边（包含原始依赖与调度引入的先后约束）
- `buffer_plan`：buffer 生命周期与峰值占用（而非仅静态分配）
- `overlap`：计算/通信重叠标记与冲突消解结果
- `trace_meta`：可解释性字段透传（如 `scope/cause/path_key`、`blocked_by`、`priority_tag`）

**计算流程**：

1. **DAG 规范化**：
   - 从 `DistributedModel.graph` 取出所有 Op 与 CommOp
   - 并基于切分信息按需展开为实例（如按 chip_id、core_group、path_key、stage_id 展开）
   - 构建 OpInstance 与 CommInstance 的依赖 DAG

2. **调度单元化**：
   - 将实例节点转为可调度单元（OpInstance + CommInstance）
   - 并绑定对应的 Tile/KernelConfig

3. **资源建模**：
   - 建立最小资源视图（core 槽位、链路/path_key 槽位、SRAM/buffer 预算）
   - 作为调度约束

4. **优先级计算**：
   - 基于关键路径长度、slack 或 bytes 权重计算优先级标签
   - 作为放置顺序的稳定依据

5. **放置策略**：
   - 按优先级迭代放置可调度单元
   - 先检查依赖满足性，再检查资源可用性与 buffer 预算

6. **重叠决策**：
   - 在 policy 允许时尝试计算/通信重叠放置
   - 并在冲突时记录冲突原因与回退策略

7. **约束修复**：
   - 对放置后的计划执行一致性检查（资源冲突、buffer 峰值超限、依赖破坏）
   - 必要时做局部重排或降级

8. **计划产出**：
   - 输出 ExecPlan 的 `timeline/binding/precedence/buffer_plan/overlap/trace_meta`
   - 保持 `scope/cause/path_key` 的透传与可追溯性
   - 内部可同时保留 DAG 依赖与线性序列（timeline）

**对应 Tier6-Model 现状**：
- 当前：事件驱动模拟器 + 基于优先级队列的调度
- 可改进：
  - 增强资源冲突检测
  - 显式建模 buffer 生命周期
  - 支持计算/通信重叠决策

---

### L4: Evaluation Engines（评估引擎层）

**设计思想**：基于硬件执行单元层级区分评估精度，而非引擎类型

**精度层级**：

| 硬件层级 | L3 Mapping         | L4 Evaluation      | 精度/速度 |
|---------|--------------------|--------------------|----------|
| Chip    | ParallelismPlanner | Roofline 公式      | 粗/快     |
| Core    | TilingPlanner      | DRAM 流量计算      | 中/中     |
| Lane    | VectorizationPlanner | 流水线模拟        | 细/慢     |

**核心组件**：
- **Evaluator**：统一评估入口
  - 接收 `ExecPlan` + `HardwareSpec/TopologySpec` + `granularity` + `calibration/policy`
  - 输出口径一致的 `EngineResult`
- **CostModelRegistry**：按 `granularity` 选择对应代价模型，实现"同口径、不同精度"的可替换评估器
- **Calibration**：提供可选的校准参数（如有效带宽系数、拥塞系数、启动开销系数），但不改变指标定义与单位

**评估口径边界（与 L3 的职责对齐）**：
- **唯一时延出口**：通信时延、带宽利用率与拥塞影响只在 L4 计算与校准，L3 不产出最终时延
- **需求到时延映射**：L3 提供 `tensor_bytes` + `scope/cause` + `path_key` + `participants`，L4 结合拓扑参数与算法提示映射为时延
- **口径一致性**：Chip/Core/Lane 可采用不同模型，但输出指标的定义、单位与汇总方式必须一致，确保 SweepRunner 可比较
- **范围边界**：L4 不新增通信节点、不改变布局与切分，只对既有计划做估时与聚合

**输入**：
- **ExecPlan**：至少包含 `timeline/binding/precedence/buffer_plan/overlap/trace_meta`，并在节点层面保留 CommOp 的 `scope/cause/path_key/tensor_bytes/participants`
- **HardwareSpec/TopologySpec**：提供不同层级的算力、带宽、延迟与路径键参数，作为需求映射的唯一硬件口径来源
- **granularity**：Chip | Core | Lane，用于选择评估模型与所需字段的最小集合
- **calibration/policy**（可选）：如有效带宽系数、拥塞系数、启动开销、是否启用 overlap 修正等

**输出（EngineResult）**：
- **StepMetrics**：对每个可调度单元（Op/CommOp/阶段）给出 `t_compute/t_comm/t_wait/t_total` 与瓶颈标签（compute-bound/bw-bound/latency-bound）
- **Aggregates**：对 TTFT/TPOT/TPS/MFU/MBU/内存峰值等指标做统一汇总，保证不同 `granularity` 的可比性

**计算流程**：

1. **口径校验**：
   - 对 ExecPlan 做最小字段校验与单位校验
   - 确保 `scope/cause/path_key` 与资源绑定字段完整且自洽

2. **模型选择**：
   - 根据 `granularity` 从 CostModelRegistry 选择评估器
   - 并声明该粒度所依赖的最小字段集合

3. **Step 估时**：
   - 对 timeline 中的每个单元做估时
   - 通信单元按 `scope/path_key/participants/tensor_bytes` 映射链路时延
   - 计算单元按绑定资源与代价模型估时

4. **重叠修正**：
   - 结合 `overlap` 与 `precedence` 关系做重叠与等待时间修正
   - 得到 `t_wait` 与更稳定的 `t_total`

5. **指标聚合**：
   - 按阶段（prefill/decode）、层级（chip/core）与全局口径聚合指标
   - 产出 Aggregates 与瓶颈归因摘要

6. **结果产出**：
   - 输出 EngineResult
   - 并保留 step 级明细以支撑 Gantt、归因分析与后续校准

**对应 Tier6-Model 现状**：
- 当前：主要在 Chip 粒度评估（`latency.py` 中的各种时延计算函数）
- 可改进：
  - 统一评估入口（Evaluator）
  - 引入 `granularity` 参数支持多粒度评估
  - 增强校准参数机制

---

### L5: Metrics & Output（指标与输出层）

**职责**：
- **口径收敛**：以 L4 的 EngineResult 为唯一指标来源，统一指标定义、单位与聚合方式，避免在 L5 重新"估时"
- **指标出具**：产出面向评审与对比的核心指标集合（TTFT/TPOT/TPS/MFU/MBU/内存峰值/瓶颈归因摘要）
- **结果导出**：将结果导出为稳定的数据契约（JSON/CSV/Excel/Gantt JSON/指令串），供外部系统与可视化工具消费

**输入**：
- **EngineResult**：包含 StepMetrics、Aggregates 与必要的 trace/meta 字段，作为 L5 的唯一指标输入
- **ExecPlan**（可选透传）：用于构建更完整的可视化与调试视图（例如 Gantt 的资源泳道与依赖边）
- **输出配置**：导出格式选择、字段白名单/黑名单、是否保留 step 明细、是否输出调试字段等

**输出**：
- **MetricsReport**：面向评审的指标报告对象，包含核心指标、分阶段指标与瓶颈归因摘要，并标注 `resolved_granularity`
- **GanttPayload**：面向时序可视化的结构化数据，至少包含时间轴事件、资源泳道键与必要的 trace 字段透传
- **ExportArtifacts**：导出产物集合（JSON/CSV/Excel/Gantt JSON），字段口径与命名保持稳定且可版本化

**计算流程**：

1. **口径校验**：
   - 校验 EngineResult 的关键字段与单位
   - 确认 `resolved_granularity` 与阶段口径完整且可比较

2. **视图装配**：
   - 将 Aggregates 与 StepMetrics 装配为报告视图（总览、分阶段、分资源、瓶颈归因）

3. **时序整理**：
   - 基于 StepMetrics（必要时结合 ExecPlan）生成 Gantt 所需的事件流与资源泳道键

4. **导出渲染**：
   - 按输出配置渲染为目标格式
   - 并保持字段命名稳定、可追溯到 `scope/cause/path_key`

5. **结果落盘**：
   - 输出 ExportArtifacts
   - 并记录最小元数据（时间戳、配置摘要、口径版本）以支持复现与比对

**对应 Tier6-Model 现状**：
- 当前：前端直接消费 API 返回的结果
- 可改进：
  - 后端统一产出标准化报告
  - 支持多种导出格式
  - 增强可复现性（记录配置摘要）

---

## 二、核心设计原则

### 1. 单一真相来源（SSOT）

**定义**：每一类信息只允许有一个"权威"数据结构/模块负责持有与维护

**应用实例**：

#### 跨 chip / 跨并行组通信
- **SSOT**：是否需要与通信类型由"边的布局不一致（layout mismatch）"唯一决定
- **职责归属**：L3-A ParallelismPlanner
- **避免**：在多个地方重复判定是否需要通信

#### 片内（intra-chip）通信/同步
- **SSOT**：是否需要与通信类型由"core 映射与 tiling 决策"唯一决定
- **职责归属**：L3-B TilingPlanner
- **避免**：在评估层（L4）重新判定片内通信需求

#### 通信时延计算
- **SSOT**：最终通信时延/带宽利用率的口径计算统一放在 L4
- **L3 职责**：布局与分片（sharding/layout）、通信需求量（bytes/participants/path_key）、时序依赖关系
- **L4 职责**：把 L3 产出的"需求量 + 路径键 + 拓扑参数"映射为时延/吞吐/利用率等指标
- **避免**：L3 产出最终时延，L4 又重新计算

**对 Tier6-Model 的启示**：
- 当前代码中可能存在多处计算通信时延的地方，应统一到一个模块
- 通信插入逻辑应该只有一处（基于 layout mismatch）

---

### 2. 粒度分层

**按硬件执行单元层级区分建模粒度**：Chip → Core → Lane

| 硬件层级 | L3 Mapping               | L4 Evaluation      | 精度/速度 | 适用场景         |
|---------|-------------------------|--------------------|----------|-----------------|
| Chip    | ParallelismPlanner      | Roofline 公式      | 粗/快     | 快速筛选 TP/PP/DP 组合 |
| Core    | TilingPlanner           | DRAM 流量计算      | 中/中     | 优化 Tile/Kernel（MVP 主力）|
| Lane    | VectorizationPlanner    | 流水线模拟         | 细/慢     | 微架构验证与瓶颈定位 |

**设计要点**：
- **渐进式细化**：粗粒度结果可作为细粒度的约束或初值，避免重复计算
- **口径一致性**：不同粒度的评估器输出指标定义、单位必须一致
- **MVP 策略**：先实现 Chip + Core 级，Lane 级作为后续扩展

**对 Tier6-Model 的启示**：
- 当前主要在 Chip 粒度建模，可作为快速评估的基础
- 后续可引入 Core 级精细建模（需要增加芯片内部参数）

---

### 3. 渐进式细化

**ExecPlan 采用可伸缩的层级结构，支持按需展开**：

- **Chip 级**：只包含 Op 调度（start/end 时间）
- **Core 级**：展开 Tile/Kernel 调度 + TileConfig/KernelConfig
- **Lane 级**：展开指令序列 + 流水线周期

**优势**：
- 粗粒度结果可作为细粒度的约束或初值
- 避免重复计算
- 支持不同场景的精度需求

**对 Tier6-Model 的启示**：
- 可在当前 Chip 级 Gantt 图基础上，支持 Core 级展开
- 保持数据结构的可扩展性

---

### 4. 跨层数据契约与口径边界

**设计原则**：每一层只产出"下一层决策所必需的信息"，避免跨层重复表达同一语义

**关键边界**：

#### L3（Mapping/Scheduling）口径
- **负责**：布局与分片（sharding/layout）、通信需求量（bytes/participants/path_key）、时序依赖关系
- **不负责**：最终通信时延/带宽利用率的口径计算（这些统一放在 L4）

#### L4（Evaluation）口径
- **负责**：把 L3 产出的"需求量 + 路径键 + 拓扑参数"映射为时延/吞吐/利用率等指标
- **允许**：通过校准参数（如拥塞系数、有效带宽系数）统一修正模型误差

#### 跨层最小字段约定

1. **WorkloadIR 的张量边**至少包含：
   - `shape`, `dtype`, `producer_id`, `consumer_id`
   - `layout_signature`（可为空，待 L3 填充）

2. **DistributedModel 的通信节点**统一为 CommOp，至少包含：
   - `scope`: inter_chip | intra_chip（通信范围）
   - `cause`: layout_mismatch | tiling_reduce | tiling_barrier | tiling_relayout（触发来源）
   - `tensor_bytes`, `participants`, `topology_path_key`
   - `trigger_edge_id`, `reason`（可解释性追溯字段）

3. **ExecPlan** 至少包含：
   - op/comm 的依赖 DAG
   - 资源占用视图（chip/core 级别）
   - 可调度的先后约束

**对 Tier6-Model 的启示**：
- 需要明确各模块的输入输出契约
- 通信相关字段应统一到 CommOp 结构
- 增强可追溯性（记录触发原因）

---

### 5. 并行组与拓扑策略接口

**目标**：把"并行组如何贴合拓扑层级"的经验规则收敛为可替换策略，而不是散落在流程描述里

**策略接口**（概念层面）：
- **输入**：TopologySpec + DeploymentSpec +（可选）代价权重（带宽优先/延迟优先）
- **输出**：parallel_groups（tp/dp/ep/pp 的 chip 列表与稳定 rank 顺序）

**默认策略**（与业界一致的启发式）：
- **TP**：优先同 board / 同 node，最小化 AllReduce 跨层级链路
- **EP**：优先同 board 或同 rack，减少 All-to-All 的跨域代价
- **PP**：允许跨 board，但需要显式记录 stage 间链路层级
- **DP**：允许跨 rack/pod，但需要在评估时考虑低频同步或无同步（推理场景）

**不在范围**：
- 本文档不引入复杂的自动并行搜索算法（如 ILP/SA/RL），仅定义策略插拔点与默认启发式

**对 Tier6-Model 的启示**：
- 当前的 `map_parallelism()` 实现了基本的分组策略
- 可将策略显式化，支持不同拓扑的优化

---

### 6. 约束检查归属

| 约束类型         | 检查位置                                | 示例                                            |
|-----------------|----------------------------------------|------------------------------------------------|
| 配置约束         | L0                                     | TP × PP × DP = N_chips                         |
| 模型约束         | L3                                     | TP ≤ num_heads, PP ≤ num_layers                |
| 显存约束         | L3                                     | 权重 + KV Cache ≤ DRAM                          |
| SLO 约束         | L4                                     | TPOT ≤ target                                  |
| 口径一致性约束    | L4（L3 产出最小字段并做静态校验）        | 指标定义/单位一致，通信时延口径只在 L4           |

**对 Tier6-Model 的启示**：
- 约束检查应该在合适的层级进行
- 避免过早或过晚检查导致的问题

---

## 三、关键概念对照

### Tier6-Model 当前实现 vs 文档设计

| 概念/模块              | Tier6-Model 现状                          | 文档设计                                  | 差异分析                                |
|----------------------|------------------------------------------|------------------------------------------|-----------------------------------------|
| **模型表示**          | 扁平化的模型配置                          | WorkloadIR（Module→Layer→Op）             | 缺少结构化 IR，可引入分层结构             |
| **硬件规格**          | TopologySpec（5层：Die→Chip→Board→Rack→Pod）| ClusterSpec/NodeSpec/BoardSpec/ChipSpec/CoreSpec | 已有较完整拓扑，缺少 Core/Lane 细粒度参数 |
| **并行映射**          | `map_parallelism()` 实现基本分组           | ParallelismPlanner（layout_signature + CommOp）| 可引入 layout_signature，统一 CommOp 表示 |
| **通信插入**          | 分散在模拟器各处                          | 统一 CommOp（scope/cause/path_key）        | 需要统一为 CommOp，增强可解释性           |
| **调度**             | 事件驱动模拟器 + 优先级队列                | Scheduler（ExecPlan + overlap/buffer_plan）| 可增强资源冲突检测和 buffer 建模          |
| **评估**             | Chip 级时延计算（latency.py）             | 多粒度评估器（Chip/Core/Lane）             | 可引入 granularity 参数，支持渐进式细化    |
| **输出**             | 前端直接消费 API 返回                     | 标准化 MetricsReport + ExportArtifacts     | 可增加导出功能和可复现性                  |

---

## 四、可借鉴的设计模式

### 1. CommOp 统一通信表示

**当前问题**：
- 通信插入逻辑分散在多处
- 难以追溯通信产生的原因
- 不同类型通信的表示方式不一致

**设计方案**：
```
CommOp {
  // 通信范围
  scope: "inter_chip" | "intra_chip"

  // 触发原因
  cause: "layout_mismatch" | "tiling_reduce" | "tiling_barrier" | "tiling_relayout"

  // 通信类型
  comm_type: "P2P" | "AllReduce" | "AllGather" | "All2All" | "ReduceScatter"

  // 通信参数
  tensor_bytes: int
  participants: List[chip_id]
  src: chip_id (for P2P)
  dst: chip_id (for P2P)

  // 拓扑路径
  topology_path_key: "intra_board" | "inter_board" | "inter_node" | "intra_noc"

  // 算法提示
  algo_hint: "ring" | "tree" | "halving_doubling" | ...

  // 可追溯性
  trigger_edge_id: edge_id
  reason: string (layout mismatch 摘要)
}
```

**收益**：
- 统一通信建模
- 增强可解释性
- 便于 L4 统一计算时延

---

### 2. layout_signature 布局签名

**当前问题**：
- 缺少显式的张量布局表示
- 难以判定是否需要插入通信

**设计方案**：
```
layout_signature {
  parallel_type: "TP" | "PP" | "DP" | "EP" | "SP" | "NONE"
  split_dim: "heads" | "hidden" | "sequence" | "expert" | null
  split_factor: int (与并行度一致)
  replica_group_id: int (用于 DP/PP 语义区分)
}
```

**通信插入规则**：
```
if producer.layout_signature != consumer.layout_signature:
    insert CommOp with cause="layout_mismatch"
```

**收益**：
- 明确通信触发条件（SSOT）
- 支持可逆的形状推导
- 便于调试和可视化

---

### 3. ExecPlan 可伸缩执行计划

**当前问题**：
- Gantt 图数据结构较为简单
- 难以扩展到更细粒度

**设计方案**：
```
ExecPlan {
  // 时间轴（Chip 级）
  timeline: List[{
    op_id/comm_id: string
    start_time: float
    end_time: float
    phase: "prefill" | "decode"
    chip_id: int
  }]

  // 资源绑定
  binding: Dict[op_id -> {
    chip_ids: List[int]
    core_ids: List[int] (可选，Core 级展开时填充)
    path_key: string (for CommOp)
    buffer_id: int (可选)
  }]

  // 依赖关系
  precedence: List[{
    from_op: op_id
    to_op: op_id
    dep_type: "data" | "resource" | "barrier"
  }]

  // Buffer 计划（可选）
  buffer_plan: Dict[buffer_id -> {
    allocate_time: float
    free_time: float
    size_bytes: int
  }]

  // 重叠标记
  overlap: Dict[op_id -> {
    overlapped_with: List[op_id]
    overlap_type: "compute_comm" | "comm_comm"
  }]

  // 可追溯性
  trace_meta: Dict[op_id -> {
    scope: string
    cause: string
    path_key: string
    blocked_by: List[op_id]
    priority_tag: int
  }]
}
```

**收益**：
- 支持渐进式细化（Chip→Core→Lane）
- 保留完整的可追溯信息
- 便于生成更丰富的可视化

---

### 4. 多粒度评估接口

**当前问题**：
- 评估逻辑主要在 Chip 粒度
- 难以扩展到更细粒度

**设计方案**：
```
Evaluator.evaluate(
  exec_plan: ExecPlan,
  hardware_spec: HardwareSpec,
  topology_spec: TopologySpec,
  granularity: "chip" | "core" | "lane",  // 新增参数
  calibration: CalibrationParams = None
) -> EngineResult
```

**CostModelRegistry**：
```
class CostModelRegistry:
  models = {
    "chip": ChipRooflineModel,
    "core": CoreDRAMFlowModel,
    "lane": LanePipelineModel
  }

  def get_model(granularity: str) -> CostModel:
    return models[granularity]()
```

**EngineResult 统一输出**：
```
EngineResult {
  resolved_granularity: "chip" | "core" | "lane"

  step_metrics: List[{
    op_id: string
    t_compute: float
    t_comm: float
    t_wait: float
    t_total: float
    bottleneck: "compute" | "bw" | "latency"
  }]

  aggregates: {
    TTFT: float
    TPOT: float
    TPS: float
    MFU: float
    MBU: float
    memory_peak: int
    bottleneck_summary: Dict
  }
}
```

**收益**：
- 支持不同精度需求
- 保持输出口径一致
- 便于性能优化和调试

---

### 5. 校准参数机制

**当前问题**：
- 模型计算结果与实测存在偏差
- 缺少统一的误差修正机制

**设计方案**：
```
CalibrationParams {
  // 有效带宽系数
  effective_bw_factor: Dict[path_key -> float] = {
    "intra_board": 0.9,
    "inter_board": 0.85,
    "inter_node": 0.75
  }

  // 拥塞系数（多流竞争时）
  congestion_factor: Dict[path_key -> float] = {
    "intra_board": 1.0,
    "inter_board": 1.2,
    "inter_node": 1.5
  }

  // 启动开销（us）
  startup_overhead: Dict[comm_type -> float] = {
    "AllReduce": 5.0,
    "All2All": 10.0,
    "P2P": 2.0
  }

  // Roofline 修正系数
  roofline_factor: float = 0.95
}
```

**应用**：
```python
# L4 评估时应用校准参数
raw_latency = calc_comm_latency(bytes, bw, lat)
calibrated_latency = raw_latency * calibration.congestion_factor[path_key] \
                     + calibration.startup_overhead[comm_type]
```

**收益**：
- 统一误差修正机制
- 不改变指标定义和单位
- 便于对齐实测数据

---

## 五、后续改进方向

### Phase 1：口径统一与可解释性增强（短期）

**优先级：高**

1. **引入 CommOp 统一通信表示**
   - 统一所有通信节点的数据结构
   - 添加 `scope/cause/path_key/trigger_edge_id/reason` 字段
   - 重构通信插入逻辑（基于 layout mismatch）

2. **增强可追溯性**
   - 记录每个通信节点的触发原因
   - 在 Gantt 图中展示通信来源
   - 支持"点击通信节点→高亮触发边"的交互

3. **口径边界明确化**
   - 明确 L3 只产出通信需求（bytes/participants/path_key）
   - L4 统一计算时延
   - 避免重复计算

4. **配置验证增强**
   - 在 L0 层做完整的约束检查
   - 提前发现不合法配置
   - 提供友好的错误提示

**预期收益**：
- 代码逻辑更清晰
- 调试效率提升
- 为后续扩展打下基础

---

### Phase 2：渐进式细化支持（中期）

**优先级：中**

1. **引入 layout_signature 机制**
   - 为张量添加布局签名
   - 实现基于布局不一致的通信插入（SSOT）
   - 支持形状可逆推导

2. **ExecPlan 可伸缩化**
   - 支持 Chip 级 → Core 级的渐进式展开
   - 保留完整的 trace_meta
   - 增强 buffer_plan 和 overlap 建模

3. **多粒度评估器**
   - 实现 Evaluator 统一入口
   - 引入 `granularity` 参数
   - 保持输出口径一致性

4. **校准参数机制**
   - 引入 CalibrationParams
   - 支持有效带宽系数、拥塞系数等
   - 与实测数据对齐

**预期收益**：
- 支持粗粒度快速筛选 + 细粒度精确评估
- 提升模型准确性
- 更好的性能优化能力

---

### Phase 3：片内建模与微架构细化（长期）

**优先级：低**

1. **引入 Core/Lane 层级参数**
   - 扩展 ChipSpec 增加 CoreSpec
   - 增加 SRAM、NoC 等参数
   - 支持片内拓扑建模

2. **实现 TilingPlanner**
   - Core 分配策略
   - Tile/Kernel 候选生成
   - 片内通信/同步需求生成

3. **Core 级评估器**
   - DRAM 流量计算
   - NoC 带宽建模
   - 片内归约/屏障时延

4. **Lane 级评估器**（可选）
   - 流水线模拟
   - SIMD 执行建模
   - 微架构瓶颈分析

**预期收益**：
- 支持更精细的性能优化
- 更准确的瓶颈归因
- 支持特定硬件的深度优化

---

### Phase 4：自动搜索与优化（研究方向）

**优先级：研究**

1. **参数空间搜索**
   - 实现 SweepRunner 的智能搜索
   - 基于 SLO 约束的剪枝
   - 支持多目标优化

2. **拓扑感知的并行策略**
   - 可插拔的并行组分配策略
   - 基于拓扑的启发式优化
   - 支持异构集群

3. **性能预测模型**
   - 基于历史数据的机器学习模型
   - 快速粗估 + 精确验证
   - 参数敏感性分析

4. **与真实系统对接**
   - 生成可执行的部署配置
   - 与 TPUPerf 等工具集成
   - 闭环验证与校准

**预期收益**：
- 自动化部署方案生成
- 降低人工调优成本
- 持续迭代优化

---

## 六、总结

### 核心价值

这份架构设计文档提供了一个**成熟的、可扩展的**LLM 推理建模平台架构，其核心价值在于：

1. **清晰的层次划分**：L0-L5 六层架构，职责单一、接口清晰
2. **SSOT 原则**：避免重复计算和逻辑不一致
3. **粒度分层**：支持从粗粒度到细粒度的渐进式细化
4. **可解释性**：完整的追溯链路，便于调试和优化
5. **可扩展性**：支持新硬件、新模型、新并行策略的快速接入

### 对 Tier6-Model 的指导意义

**当前项目的优势**：
- 已有较完整的 5 层拓扑建模（Die→Chip→Board→Rack→Pod）
- 实现了基本的并行策略映射（TP/PP/DP/EP）
- 支持 MLA/MoE 等特殊模型
- 有完整的前后端交互系统

**可借鉴的改进点**：
1. **短期**：引入 CommOp 统一通信表示，增强可解释性
2. **中期**：支持多粒度评估，引入校准参数机制
3. **长期**：扩展到 Core/Lane 级建模，支持自动搜索优化

**实施建议**：
- **不需要大规模重构**：当前架构已较合理，主要是**借鉴设计思想**进行局部优化
- **优先改进可解释性**：这对用户体验和调试效率提升最明显
- **渐进式演进**：按 Phase 1→2→3 的顺序逐步实施
- **保持向后兼容**：在引入新设计时保持现有功能的稳定性

---

## 附录：术语对照表

| 术语             | 含义                                    | 对应英文                |
|-----------------|----------------------------------------|------------------------|
| 工作负载表示      | 模型的中间表示                          | Workload IR             |
| 架构规格         | 硬件拓扑和参数描述                       | Architecture Spec       |
| 映射与调度       | 并行策略映射和执行计划生成                | Mapping & Scheduling    |
| 评估引擎         | 性能指标计算                            | Evaluation Engine       |
| 单一真相来源      | 每类信息只有一个权威数据源                | Single Source of Truth  |
| 粒度分层         | 按硬件层级区分建模精度                    | Granularity Layering    |
| 渐进式细化       | 从粗粒度到细粒度逐步展开                  | Progressive Refinement  |
| 布局签名         | 张量的分布式布局描述                      | Layout Signature        |
| 路径键           | 通信路径的层级标识                       | Path Key                |
| 通信算子         | 统一的通信操作表示                       | CommOp                  |
| 执行计划         | 可调度的操作序列与资源绑定                | Execution Plan          |
| 校准参数         | 修正模型误差的系数                       | Calibration Parameters  |

---

**文档版本**：v1.0
**最后更新**：2026-01-28
**维护者**：Tier6-Model 团队
