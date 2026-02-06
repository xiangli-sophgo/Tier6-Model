# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tier6+Model 是一个综合性的 LLM 推理部署分析平台,核心功能包括:

- **3D 拓扑可视化**: 多层级网络拓扑配置 (Die → Chip → Board → Rack → Pod)
- **性能仿真**: 细粒度事件驱动的推理模拟器
- **成本评估**: 完整的部署成本分析 (硬件 + 互联)
- **配置管理**: 芯片、模型、拓扑预设系统
- **实验管理**: 任务队列、实时监控、结果对比、导入导出

**关键特性**: 支持多种并行策略 (TP/PP/DP/EP/SP)、MLA/MoE 架构、性能指标计算 (TTFT/TPOT/TPS/MFU)、成本估算、可视化图表

## Interaction Rules

### 1. Task Analysis Before Implementation

当接收到新的问题或任务时，如果有明确的plan就直接开始执行，如果没有具体的plan，就遵循以下流程：

**阶段一：需求理解与分析**

- 不要立即开始修改代码
- 首先分析并阐述你对任务的理解，包括：
  - 问题的本质和目标
  - 可能的实现思路（逻辑层面，不涉及具体代码）
  - 需要修改的模块/组件范围
  - 潜在的影响和注意事项

**阶段二：确认环节**

- 分析完成后，必须总结你的理解并提出确认问题
- 示例："我的理解是否正确？是否可以开始实现？"
- 等待用户明确确认后才能进入实现阶段

**阶段三：代码实现**

- 必须在获得用户明确许可后开始编写/修改代码
- 实现过程中如发现理解偏差，立即停止并重新确认

**例外情况：**

- 用户明确表示"开始修改"/"可以实现"等指令时，可跳过确认环节

### 3. Project Organization Rules

**文件组织规范：**

- **根目录限制**: 根目录只允许放置以下文件

  - `README.md` - 项目主文档
  - `CLAUDE.md` - AI辅助开发指导
  - `start.sh` / `start.bat` - 启动脚本
  - `.gitignore` / `.env` 等配置文件
- **文档管理**:

  - 技术笔记、调试文档、优化分析等必须放在 `docs/` 目录
  - 禁止在根目录创建任何技术文档（如 `*_ANALYSIS.md`, `*_DEBUG.md`, `*_PLAN.md` 等）
- **测试脚本**:

  - 测试相关脚本必须放在对应模块的 `tests/` 目录
  - 后端测试: `backend/tests/`
  - 前端测试: `frontend/tests/` 或 `frontend/src/__tests__/`
- **临时文件**:

  - 禁止提交任何临时测试脚本、调试脚本到根目录
  - 如需临时测试，应使用 `scripts/` 目录并添加到 `.gitignore`

**创建新文件时的检查清单：**

1. 是否为用户文档？→ 应整合到 `README.md`
2. 是否为技术笔记/分析？→ 放入 `docs/`
3. 是否为测试脚本？→ 放入对应的 `tests/` 目录
4. 是否为工具脚本？→ 放入 `scripts/` 目录
5. 是否为启动脚本？→ 只允许 `start.sh` / `start.bat`

### 4. 配置参数加载规则

**禁止在加载配置文件或从其他地方获取参数时使用默认值 (default/fallback)**。如果必需的参数找不到，必须抛出明确的错误，指出缺失的字段和所在的配置文件。

**原因**: 静默使用默认值会掩盖配置错误，导致难以定位问题。

**实现要求**:
- `dict.get(key)` 不允许提供默认值参数，找不到时应 raise 异常
- 错误信息必须包含: 缺失的字段名、配置文件路径或来源
- 唯一例外: 明确标注为"可选"的字段，且有文档说明其默认行为

**示例**:
```python
# [FAIL] 错误 - 静默使用默认值
core_count = config.get("cores", {}).get("count", 1)

# [PASS] 正确 - 找不到就报错
if "cores" not in config or "count" not in config["cores"]:
    raise ValueError(f"Missing 'cores.count' in chip config: {config_path}")
core_count = config["cores"]["count"]
```

### 5. Windows 编码规则

**禁止在代码输出中使用特殊 Unicode 字符**（如 ✓, ✅, ❌, →, ⚠️ 等），Windows 终端使用 GBK 编码会导致 `UnicodeEncodeError`。

**替代方案**: 使用纯 ASCII 字符

- ✓/✅ → `[OK]` 或 `[PASS]`
- ❌ → `[FAIL]`
- → → `->`
- ⚠️ → `[WARN]`

## Architecture

### Frontend (React + TypeScript + Three.js)

**Entry**: `frontend/src/main.tsx` → `App.tsx`

**核心页面** (`frontend/src/pages/`):

- **DeploymentAnalysis** - 部署分析主页 (配置选择、模型设置、并行策略)
- **Results** - 结果管理页面 (列表展示、批量操作、导入导出)
- **TopologySetup** - 拓扑配置页面 (3D可视化、拓扑编辑)
- **Knowledge/Dashboard** - 知识图谱与主仪表板

**关键组件**:

- `ConfigPanel/` - 配置面板 (DeploymentAnalysisPanel, ParallelismConfigPanel, ConfigSelectors)
- `charts/` - 可视化图表 (KPI面板, Gantt图, Roofline图, 通信/内存分解图)
- `ui/` - UI组件库 (shadcn/ui: form controls, dialogs, tables)
- `Scene3D` - 3D拓扑渲染 (react-three-fiber)

**核心工具模块** (`frontend/src/utils/llmDeployment/`):

- `types.ts` - 核心类型定义 (200+ 行)
- `topologyHardwareExtractor.ts` - 拓扑硬件参数提取
- `backendApi.ts` - 后端 API 封装
- `presets.ts/ganttDataUtils.ts/resultAdapter.ts` - 预设管理、数据处理、结果适配

**状态与通信**:

- `WorkbenchContext` - 全局状态管理 (拓扑、配置、结果)
- WebSocket `/ws/tasks` - 实时任务状态推送

### Backend (Python + FastAPI)

**Entry**: `backend/llm_simulator/main.py` (uvicorn server, port 8001)

**API 接口** (`api.py`, 30+ endpoints):

- **仿真与验证**: `/api/simulate`, `/api/validate`, `/api/model/calculate-params`
- **预设管理**: `/api/presets/{chips,models,runtime}`, `/api/{chip,model}-presets` (CRUD)
- **拓扑配置**: `/api/topologies` (CRUD), `/api/benchmarks` (CRUD)
- **任务管理**: `/api/evaluation/{submit,tasks,config}` (提交、查询、取消、配置)
- **实验管理**: `/api/evaluation/experiments` (列表、详情、批量删除、导入导出)
- **WebSocket**: `/ws/tasks` (实时任务状态推送)

**核心模块**:

- `simulator.py` - 仿真引擎 (计算/内存/通信事件建模)
- `topology.py` - 拓扑解析 (`TopologyParser`, `InterconnectGraph`, `map_parallelism`)
- `latency.py` - 延迟计算 (Attention/FFN/MLA/MoE 计算, AllReduce/P2P/AllToAll 通信)
- `gantt.py` - Gantt 图数据生成
- `types.py` - 类型定义 (40+ enums, 50+ dataclasses)
- `schemas.py` - Pydantic 校验模型

**成本评估** (`evaluators/cost_evaluator.py`):

- **服务器成本**: `(chip_price × chips_per_module + 750) × modules_per_server + 12000 + 7500`
- **互联成本**: 分层定价 (1-2芯片: $1/lane → 64+芯片: $247/lane)
- **芯片定价**: B200 ($6,303), H100 ($4,500), SG2262 ($2,500)

**配置系统** (`configs/`):

- `chip_presets/*.yaml` - 芯片硬件规格
- `model_presets/*.yaml` - 模型定义 (DeepSeek-V3, Qwen3-235B 等)
- `topologies/*.yaml` - 拓扑模板
- `benchmarks/*.json` - 测试场景 (8 个预设)

**数据存储** (`storage/`):

- **SQLite 数据库**: `Experiment`, `EvaluationTask`, `EvaluationResult` (SQLAlchemy ORM)
- **任务队列**: ThreadPoolExecutor (可配置并发数)
- **关键抽象**: `TopologyParser` → `InterconnectGraph`, `map_parallelism()` (并行策略映射)

### Configuration System

**概述**: 集中式预设系统 (v2.1.2+) 管理芯片、模型、拓扑、Benchmark 配置。

**目录结构** (`backend/configs/`):

- `chip_presets/*.yaml` - 芯片硬件规格 (SG2262, SG2260E, H100 等)
- `model_presets/*.yaml` - 模型定义 (deepseek-v3, qwen3-235b 等)
- `topologies/*.yaml` - 拓扑模板 (P1-R1-B1-C8, TOPOLOGY_TEMPLATE)
- `benchmarks/*.json` - 测试场景 (8 个预设)

**加载机制**:

- `load_topology_config(name)` - 加载 YAML 拓扑,自动格式转换
- `load_benchmark_config(id)` - 加载 JSON benchmark
- `_convert_rack_config_to_pods()` - 旧格式迁移

**使用方式**:

- 前端通过 `/api/presets/{chips,models}` 获取预设列表
- `ConfigSelectors.tsx` 提供下拉选择器
- 支持预设引用或内联自定义配置
- 配置快照随实验保存,确保可复现性

### Data Flow

**完整流水线**:

1. **配置加载**: 前端获取预设 → 用户选择/自定义配置 → 加载拓扑和芯片/模型预设
2. **任务提交**: 单次仿真 (`POST /api/simulate`, 同步) 或批量评估 (`POST /api/evaluation/submit`, 异步队列)
3. **仿真执行**:
   - `TopologyParser` 构建互联图 → `map_parallelism()` 分配并行组
   - 运行 prefill/decode 阶段 (计算/通信/内存事件)
   - `CostEvaluator` 计算成本 → 生成 Gantt 数据
4. **结果存储**: 保存 Experiment/EvaluationResult 到 SQLite (含性能指标、成本分解、Gantt数据)
5. **可视化**: Results 页面查询实验列表 → 图表组件渲染 (KPI/Gantt/Roofline/成本/内存)

**持久化层级**: localStorage (临时缓存) → 内存任务队列 (会话) → SQLite (永久) → JSON 快照 (导入导出)

## Common Commands

### Development Setup

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
pnpm install
```

### Running the Application

```bash
# First time: install dependencies and start
./start.sh --setup   # Linux/Mac
start.bat --setup    # Windows

# Subsequent runs
./start.sh           # Linux/Mac
start.bat            # Windows
```

### Build

```bash
cd frontend
pnpm build
```

## Key Configuration Files

**主要配置文件**:

- `backend/requirements.txt`, `backend/llm_simulator/main.py` - Python 依赖与服务器入口
- `backend/configs/` - 预设目录 (chip_presets, model_presets, topologies, benchmarks)
- `frontend/package.json`, `frontend/vite.config.ts` - 前端依赖与构建配置
- `frontend/src/utils/llmDeployment/types.ts` - 核心类型定义 (200+ 行)

**配置文件格式**:

- **芯片预设 (YAML)**: 算力 (compute_tflops_fp8/bf16)、内存 (memory_capacity_gb, bandwidth_gbps)、微架构 (cube_m/k/n, sram_size_kb, lane_num, compute_dma_overlap_rate)
- **模型预设 (YAML)**: 模型结构 (hidden_size, num_layers)、MoE (num_experts, experts_per_token)、MLA (kv_lora_rank, q_lora_rank)
- **拓扑配置 (YAML)**: 层级 (pod_count, racks_per_pod, boards, chips)、硬件参数 (chips dict, interconnect: c2c/b2b/r2r/p2p)、通信配置
- **Benchmark (JSON)**: 模型定义 + 推理参数 (batch_size, prompt_length, output_length)

## Important Implementation Details

### Cost Evaluation System

**Implementation**: `backend/llm_simulator/evaluators/cost_evaluator.py` (v2.0.6+)

**CostEvaluator Class** - Complete deployment cost modeling:

**Cost Formulas**:

```python
# Server cost (per server)
server_cost = (chip_price × chips_per_module + 750) × modules_per_server + 12000 + 7500

# Where:
# - chip_price: per-chip cost in USD
# - chips_per_module: chips per QAM module (default: 1)
# - modules_per_server: QAM modules per server (default: 8)
# - 750: module assembly cost
# - 12000: server chassis and motherboard
# - 7500: power supply and cooling

# Interconnect cost
interconnect_cost = chip_count × modules_per_server × chips_per_module × lanes × lane_cost(chip_count)

# Total cost
total_cost = server_cost + interconnect_cost
```

**Chip Pricing Table** (default USD):

```python
CHIP_PRICES = {
    'B200': 6303,      # NVIDIA Blackwell
    'H100': 4500,      # NVIDIA Hopper
    'H800': 3800,      # NVIDIA Hopper (China export)
    'SG2262': 2500,    # Domestic chip
    'SG2260E': 2500,   # Domestic chip (enhanced)
}
```

**Interconnect Cost Tiers** ($/lane, 112Gbps per lane):

```python
1-2 chips:   $1/lane       # PCIe direct connection
8 chips:     $55/lane      # Ethernet switch
16 chips:    $70/lane      # Switch + DAC cable
32 chips:    $70/lane      # Switch + DAC
64 chips:    $105.247/lane # Switch + AEC (Active Electrical Cable)
64+ chips:   $247/lane     # Full optical solution (AOC + transceiver)
```

**Key Methods**:

- `calculate_server_cost(chip_type, m, n)` - Compute single server cost
- `estimate_interconnect_bandwidth(model_size_gb, tp, tpot_ms)` - Estimate required bandwidth
- `get_lane_cost(chip_count)` - Get per-lane cost based on scale
- `calculate_total_cost()` - Returns detailed cost breakdown:
  ```python
  {
    'server_cost': float,           # Total server hardware cost
    'interconnect_cost': float,     # Total interconnect cost
    'total_cost': float,            # Sum of above
    'cost_per_chip': float,         # Normalized per-chip cost
    'cost_per_million_tokens': float # Operational cost (3-year depreciation)
  }
  ```
- `calculate_cost_per_million_tokens(tps, depreciation_years=3)` - Calculate operational cost considering throughput

**Integration in Simulation Pipeline**:

1. After simulation completes, `CostEvaluator` is invoked with:
   - Chip type and count
   - Parallelism strategy (TP affects interconnect bandwidth)
   - Model size and throughput (for bandwidth estimation)
2. Cost breakdown stored in `EvaluationResult.full_result['cost']`
3. Frontend displays cost metrics in result panels and comparison tables

**Usage Example**:

```python
evaluator = CostEvaluator(
    chip_type='SG2262',
    chip_count=64,
    model_size_gb=1300,  # DeepSeek-V3 671B in FP16
    tp_degree=8,
    tpot_ms=50
)
cost_result = evaluator.calculate_total_cost()
# cost_result = {
#   'server_cost': 180000,
#   'interconnect_cost': 95000,
#   'total_cost': 275000,
#   'cost_per_chip': 4297,
#   'cost_per_million_tokens': 0.42
# }
```

### Parallelism Strategy Mapping

- Assignment order (inner to outer): TP → EP → PP → DP
- TP groups prefer same-board chips (high-bandwidth NVLink)
- PP groups can span boards (P2P communication)
- DP/EP groups can span racks/pods
- Implementation: `topology.py:map_parallelism()`

### Communication Algorithms

**AllReduce Algorithms** (for DP/TP gradient synchronization):

- **Ring**: `O(N-1)` steps, bandwidth-optimal, `2(N-1)/N` bandwidth utilization
  - Best for: Large data sizes, uniform network
- **Double Binary Tree**: `O(log N)` steps, latency-optimal
  - Best for: Small data sizes, latency-sensitive
- **Halving-Doubling**: `O(log N)` steps, efficient for power-of-2 sizes
  - Best for: Power-of-2 node counts, balanced latency/bandwidth
- **Reduce-Broadcast**: `O(2 log N)` steps, simple implementation
  - Best for: Small clusters, debugging

**AllToAll Algorithms** (for MoE expert routing):

- **Pairwise**: `O(N-1)` steps, direct exchange between all pairs
  - Best for: Small expert counts, uniform distribution
- **Ring**: `O(N-1)` steps, pipelined communication
  - Best for: Large data sizes, bandwidth-limited
- **Bruck**: `O(log N)` steps, recursive doubling
  - Best for: Power-of-2 sizes, low latency

**AllGather / ReduceScatter** (for PP/SP):

- Used for pipeline parallelism boundary communication
- Typically uses ring algorithm for bandwidth efficiency

**P2P (Point-to-Point)** (for PP stage transfers):

- Direct send/recv between adjacent pipeline stages
- No algorithm selection, uses shortest path in topology graph

**Algorithm Selection** (`comm_latency_config` in topology YAML):

```yaml
comm_latency_config:
  allreduce_algorithm: "ring"           # ring/double_binary_tree/halving_doubling/reduce_broadcast
  alltoall_algorithm: "pairwise"        # pairwise/ring/bruck
  enable_compute_comm_overlap: true     # Overlap computation with communication
  network_efficiency: 0.85              # Network efficiency factor (0-1)
```

**Latency Calculation** (`latency.py`):

- Functions: `calc_allreduce_latency()`, `calc_alltoall_latency()`, `calc_p2p_latency()`
- Model: α-β model (latency = α + β × message_size)
  - α: base latency (network hop + protocol overhead)
  - β: per-byte transfer time (1 / bandwidth)
- Multi-hop support: Routing through intermediate chips with cumulative latency
- Bandwidth contention: Shared link bandwidth divided by concurrent flows

**Compute-Communication Overlap**:

- When `enable_compute_comm_overlap: true`:
  - Overlap ratio controlled by `compute_dma_overlap_rate` in chip config
  - Effective communication time: `comm_time × (1 - overlap_rate)`
- Applied to: AllReduce in TP groups, P2P in PP stages

### Special Model Types

**Model Presets** (loaded from `backend/configs/model_presets/*.yaml`):

- `deepseek-v3.yaml` - 671B MoE model with MLA
- `deepseek-r1.yaml` - DeepSeek R1 variant
- `qwen2.5-32b.yaml`, `qwen2.5-72b.yaml` - Qwen 2.5 series
- `qwen3-32b.yaml`, `qwen3-235b.yaml` - Qwen 3 series

**MLA (Multi-head Latent Attention)** - DeepSeek V3/R1:

- **Purpose**: Compressed KV cache with LoRA-based projections for memory efficiency
- **Configuration** (in model preset YAML):
  ```yaml
  mla:
    enabled: true
    kv_lora_rank: 512        # KV compression rank
    q_lora_rank: 1536        # Query LoRA rank
    qk_rope_dim: 64          # RoPE dimension for QK
    qk_nope_dim: 128         # Non-RoPE dimension for QK
    v_head_dim: 128          # Value head dimension
  ```
- **Specialized Functions**: `calc_mla_prefill_latency()`, `calc_mla_decode_latency()` in `latency.py`
- **Benefits**: Reduces KV cache memory by ~5-8x, critical for large context lengths
- **Tradeoffs**: Additional LoRA computation overhead in prefill phase

**MoE (Mixture of Experts)**:

- **Configuration** (in model preset YAML):
  ```yaml
  moe:
    enabled: true
    num_experts: 256              # Total expert count
    num_shared_experts: 1         # Always-active shared experts
    experts_per_token: 8          # Top-K experts activated per token
    router_topk_policy: "greedy"  # greedy/sample
  ```
- **Parallelism Integration**:
  - **EP (Expert Parallelism)**: Distributes experts across chips
  - **TP (Tensor Parallelism)**: Each expert can be TP-sharded
  - Combined strategy: `tp × ep = total_parallelism` (for expert layers)
- **Routing Overhead**: Token routing adds AllToAll communication
- **Load Balancing**: Uneven expert activation can cause bubbles
- **Specialized Functions**: `calc_moe_routing_latency()`, `calc_moe_ffn_latency()` in `latency.py`
- **Examples**:
  - DeepSeek-V3: 256 experts, 1 shared, 8 experts/token
  - Typical ratio: ~5% of experts activated per token (8/256)

**Model Type Enum** (types.py):

```python
class ModelType(str, Enum):
    STANDARD = "standard"  # Dense model (GPT, LLaMA)
    MOE = "moe"            # MoE model (DeepSeek-V3, Mixtral)
```

**Architecture-Specific Optimizations**:

- **MLA + MoE**: DeepSeek-V3 combines both for memory efficiency and capacity
- **Shared Experts**: Always-active experts improve model stability
- **Dynamic Routing**: Token-level expert selection based on router scores

### Hardware Abstraction Layers

**Configuration Format (v2.1.0+)**:

**New Multi-Chip Dictionary Structure** (replaces single-chip format):

```yaml
hardware_params:
  # Dictionary indexed by chip name (supports heterogeneous configurations)
  chips:
    SG2262:
      name: "SG2262"
      num_cores: 64
      compute_tflops_fp8: 768
      compute_tflops_bf16: 384
      memory_capacity_gb: 64
      memory_bandwidth_gbps: 12000
      memory_bandwidth_utilization: 0.85

      # Local memory (HBM/GDDR)
      lmem_capacity_mb: 64
      lmem_bandwidth_gbps: 6400

      # Microarchitecture parameters (v2.1.0+ required)
      cube_m: 16                        # Matrix compute unit M dimension
      cube_k: 32                        # K dimension
      cube_n: 8                         # N dimension
      sram_size_kb: 2048                # On-chip SRAM size
      sram_utilization: 0.45            # SRAM utilization ratio
      lane_num: 16                      # Data lane count
      align_bytes: 32                   # Memory alignment
      compute_dma_overlap_rate: 0.8     # Compute-DMA overlap ratio

    H100:  # Second chip type (heterogeneous support)
      name: "H100"
      # ... similar structure

  # Interconnect bandwidth and latency (4 levels)
  interconnect:
    c2c:  # Chip-to-Chip (Die-to-Die)
      bandwidth_gbps: 448
      latency_us: 0.2
    b2b:  # Board-to-Board (within rack, NVLink)
      bandwidth_gbps: 400
      latency_us: 2.0
    r2r:  # Rack-to-Rack (within pod, InfiniBand)
      bandwidth_gbps: 400
      latency_us: 3.0
    p2p:  # Pod-to-Pod (cross-pod, Ethernet)
      bandwidth_gbps: 400
      latency_us: 5.0

  # Communication optimization settings
  comm_latency_config:
    allreduce_algorithm: "ring"           # ring/double_binary_tree/halving_doubling/reduce_broadcast
    alltoall_algorithm: "pairwise"        # pairwise/ring/bruck
    enable_compute_comm_overlap: true
    network_efficiency: 0.85
```

**Hardware Hierarchy** (aligned with topology structure):

- **ChipHardwareConfig**: per-chip specs (TFLOPS, memory, bandwidth, microarchitecture params)
- **BoardConfig**: intra-board connectivity (derived from interconnect.b2b)
- **RackConfig**: intra-rack network (derived from interconnect.r2r, typically NVLink)
- **PodConfig**: inter-pod network (derived from interconnect.p2p, typically InfiniBand/Ethernet)

**Key Parameters**:

- **Compute**: `compute_tflops_fp8`, `compute_tflops_bf16` - peak FLOPS
- **Memory**: `memory_capacity_gb`, `memory_bandwidth_gbps` - global memory specs
- **Cache**: `lmem_capacity_mb`, `sram_size_kb` - on-chip cache hierarchy
- **Microarchitecture**: `cube_m/k/n` (GEMM unit shape), `lane_num`, `align_bytes`
- **Efficiency**: `memory_bandwidth_utilization`, `sram_utilization`, `compute_dma_overlap_rate`

**Format Migration**:

- Old format (pre-v2.1.0): `hardware_params.chip` (single chip)
- New format (v2.1.0+): `hardware_params.chips` (multi-chip dictionary)
- Backend automatically converts old format to new format during loading

### Topology Structure

**5-Level Hierarchy**:

```
Pod (cluster level)
 └─ Rack (cabinet level)
     └─ Board (node/server level)
         └─ Chip (accelerator level)
             └─ Die (optional, for chiplet designs)
```

**Topology Configuration** (YAML format):

```yaml
name: "P1-R1-B1-C8"        # Naming: PodCount-RackCount-BoardCount-ChipCount
pod_count: 1               # Number of pods
racks_per_pod: 1           # Racks per pod

rack_config:
  boards:                  # Board configuration
    - chips:               # Chip configuration per board
        - name: "SG2262"   # Chip type name
          preset_id: "sg2262"  # Optional: reference to chip preset
          count: 8         # Number of chips of this type
      count: 1             # Number of boards with this config

# Hierarchical connectivity (optional, auto-generated if omitted)
connections:
  - from: {pod: 0, rack: 0, board: 0, chip: 0}
    to: {pod: 0, rack: 0, board: 0, chip: 1}
    bandwidth_gbps: 448
    latency_us: 0.2
  # ... more connections
```

**Interconnect Mapping**:

- **c2c (Chip-to-Chip)**: Die-to-die within same chip package (e.g., 448 GB/s, 0.2 μs)
- **b2b (Board-to-Board)**: Cross-board within same rack, typically NVLink (e.g., 400 GB/s, 2 μs)
- **r2r (Rack-to-Rack)**: Cross-rack within same pod, typically InfiniBand (e.g., 400 GB/s, 3 μs)
- **p2p (Pod-to-Pod)**: Cross-pod, typically high-speed Ethernet (e.g., 400 GB/s, 5 μs)

**Heterogeneous Support** (v2.1.0+):

- Multiple chip types within same topology
- Example: Mix SG2262 (high compute) with SG2261 (balanced) on same board
- Each chip type has independent hardware parameters in `hardware_params.chips` dictionary

**Topology Parsing**:

- `TopologyParser` (topology.py) builds `InterconnectGraph`:
  - Nodes: Individual chips with unique IDs
  - Edges: Chip-to-chip connections with bandwidth/latency
- Graph used for:
  - Parallelism group assignment (map_parallelism)
  - Communication routing (shortest path, multi-hop)
  - Bandwidth contention modeling

**3D Visualization** (Frontend):

- `Scene3D` component renders topology using react-three-fiber
- Color-coded layers:
  - Pod level: Blue
  - Rack level: Green
  - Board level: Yellow
  - Chip level: Red
- Interactive controls: rotate, zoom, select, highlight parallelism groups
- Topology graph (2D): Force-directed layout showing connectivity

## Data Storage and Management

**Frontend Storage** (`frontend/src/utils/storage.ts`):

- **localStorage keys**:
  - `tier6_topology_config_cache` - Topology configuration cache
  - `tier6_sider_width_cache` - UI state cache (sidebar width)
- Temporary storage for quick access and UI state persistence

**Backend Database** (`backend/llm_simulator/storage/`):

- **Database**: SQLite (default), configurable to other SQLAlchemy-supported DBs
- **Models** (SQLAlchemy ORM):
  ```python
  class Experiment(Base):
      id: int                    # Primary key
      name: str                  # Experiment name
      description: str           # User description
      created_at: datetime       # Creation timestamp
      updated_at: datetime       # Last modified timestamp
      # Relationships
      tasks: List[EvaluationTask]

  class EvaluationTask(Base):
      id: int                    # Primary key
      task_id: str               # UUID task identifier
      experiment_id: int         # Foreign key to Experiment
      status: str                # pending/running/completed/failed/cancelled
      progress: float            # 0.0 to 1.0
      config_snapshot: JSON      # Full config for reproducibility
      created_at: datetime
      started_at: datetime
      completed_at: datetime
      error_message: str         # If failed
      # Relationships
      result: EvaluationResult   # One-to-one

  class EvaluationResult(Base):
      id: int                    # Primary key
      task_id: str               # Foreign key to EvaluationTask
      # Key metrics (indexed for fast query)
      tps: float                 # Tokens per second
      tpot: float                # Time per output token (ms)
      ttft: float                # Time to first token (ms)
      mfu: float                 # Model FLOPS Utilization
      score: float               # Overall evaluation score
      # Full result data
      full_result: JSON          # Complete result including:
                                 # - Gantt chart data
                                 # - Cost breakdown
                                 # - Bottleneck analysis
                                 # - Communication breakdown
                                 # - Memory allocation
      created_at: datetime
  ```

**Data Persistence Layers**:

1. **Temporary** - localStorage (frontend cache, UI state)
2. **Session** - In-memory task queue state (running tasks)
3. **Permanent** - SQLite database (experiments, tasks, results)
4. **Export** - JSON snapshots for configuration sharing

### Experiment Management

**Task Queue System**:

- **Executor Pool**: Configurable max workers and queue size
  - Default: `max_workers=4`, `max_queued=100`
  - Configure via `PUT /api/evaluation/config`
- **Concurrent Control**: Tasks run in ThreadPoolExecutor with concurrent.futures
- **Task States**: `pending` → `running` → `completed/failed/cancelled`

**Real-Time Updates**:

- **WebSocket**: `WS /ws/tasks` pushes task status updates
- **Queue Subscription**: Frontend subscribes to task updates by task_id
- **Update Events**: status change, progress update, completion, error

**Batch Operations**:

```python
# Batch delete experiments
POST /api/evaluation/experiments/batch-delete
Body: {"experiment_ids": [1, 2, 3]}

# Batch delete results within experiment
POST /api/evaluation/experiments/{experiment_id}/results/batch-delete
Body: {"result_ids": [10, 11, 12]}
```

**Import/Export System**:

**Export** (`GET /api/evaluation/experiments/export?experiment_ids=1,2,3`):

- Returns JSON with complete experiment snapshots:
  ```json
  {
    "experiments": [
      {
        "name": "Experiment 1",
        "description": "...",
        "tasks": [
          {
            "config_snapshot": {...},  // Full reproducible config
            "result": {...}             // Full result data
          }
        ]
      }
    ],
    "export_metadata": {
      "timestamp": "2026-02-02T12:00:00",
      "count": 3
    }
  }
  ```

**Import** (Two-phase process):

1. **Check Phase**: `POST /api/evaluation/experiments/check-import`
   - Upload JSON file, get temporary file ID
   - Backend checks for name conflicts
   - Returns conflict list and import preview
2. **Execute Phase**: `POST /api/evaluation/experiments/execute-import`
   - Specify conflict resolution strategy:
     - `skip` - Skip conflicting experiments
     - `overwrite` - Overwrite existing with same name
     - `rename` - Auto-rename with timestamp suffix
   - Creates experiments and results in database

**Inline Editing** (Results page):

- `PATCH /api/evaluation/experiments/{experiment_id}`
- Update experiment name and description without affecting tasks/results

**Pagination Support**:

- `GET /api/evaluation/experiments?skip=0&limit=20`
- `GET /api/evaluation/tasks?skip=0&limit=50&status=completed`

## Testing Notes

- No automated test framework is currently set up
- **Manual Testing**: Web UI at http://localhost:3100
- **Validation Endpoint**: `POST /api/validate` can verify configs before simulation
- **Benchmark System**: 8 predefined test scenarios in `backend/configs/benchmarks/`
- **Experiment Comparison**: Results page supports side-by-side comparison of multiple runs

## Port Configuration

- Frontend default: 3100
- Backend default: 8001
- CORS is configured to allow all origins in development

## Code Style Notes

**Backend (Python)**:

- **Type Safety**: Extensive use of dataclasses and type hints
- **Enums**: Defined in `types.py` for task types, phases, algorithms (40+ enums)
- **Validation**: Pydantic models in `schemas.py` for API request/response
- **ORM**: SQLAlchemy for database models with relationship definitions
- **Async**: FastAPI with async/await for API endpoints
- **Configuration**: YAML for configs (PyYAML), JSON for benchmarks
- **Comments**: Primarily in Chinese

**Frontend (TypeScript)**:

- **Components**: React functional components with hooks (useState, useEffect, useCallback, useMemo)
- **Context**: WorkbenchContext for global state management
- **Type Definitions**: Comprehensive types in `utils/llmDeployment/types.ts` (200+ lines)
- **UI Components**: shadcn/ui component library with Tailwind CSS
- **3D Rendering**: react-three-fiber (@react-three/fiber, @react-three/drei)
- **Charts**: Custom chart components built on recharts
- **API Calls**: Wrapped in utility modules (`backendApi.ts`)
- **State Management**: Local component state + Context API (no Redux)
- **Comments**: Primarily in Chinese

**Code Organization**:

- **Backend**: Feature-based modules (simulator/, evaluators/, storage/)
- **Frontend**: Component-based structure (pages/, components/, utils/)
- **Configs**: Separate directory for YAML/JSON presets
- **Types**: Centralized type definitions (backend: `types.py`, frontend: `types.ts`)

**Naming Conventions**:

- **Python**: snake_case for functions/variables, PascalCase for classes
- **TypeScript**: camelCase for functions/variables, PascalCase for components/types
- **Files**: kebab-case for multi-word filenames (e.g., `cost-evaluator.py`)
- **Configs**: Descriptive names with version/variant info (e.g., `deepseek-v3.yaml`)
