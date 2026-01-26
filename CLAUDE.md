# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tier6+Model is an interactive 3D multi-level network topology configurator and LLM inference simulator. It provides visualization and simulation capabilities for hierarchical GPU/accelerator cluster topologies (Die → Chip → Board → Server → Pod) and models LLM inference workloads with various parallelism strategies (TP, PP, DP, EP, SP).

## Interaction Rules

### 1. Greeting Protocol

每次回复都必须以"想哥"开头。

### 2. Task Analysis Before Implementation

当接收到新的问题或任务时，遵循以下流程：

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

- 仅在获得用户明确许可后开始编写/修改代码
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

## Architecture

### Frontend (React + TypeScript + Three.js)

- **Entry**: `frontend/src/main.tsx` → `App.tsx`
- **3D Visualization**: `Scene3D` component uses react-three-fiber and @react-three/drei for 3D topology rendering
- **Configuration UI**: `ConfigPanel` provides interactive controls for topology, model, hardware, and parallelism settings
- **Topology Graph**: `TopologyGraph` displays 2D network graph representation
- **Knowledge Graph**: `KnowledgeGraph` component visualizes domain knowledge about distributed computing concepts
- **Context Management**: `WorkbenchContext` manages global application state
- **API Client**: `frontend/src/api/topology.ts` handles backend communication

### Backend (Python + FastAPI)

- **Entry**: `backend/llm_simulator/main.py` (starts uvicorn server)
- **API Server**: `backend/llm_simulator/api.py` exposes REST endpoints:
  - `POST /api/simulate` - runs LLM inference simulation
  - `POST /api/validate` - validates configuration
- **Core Modules**:
  - `simulator.py` - main simulation engine with fine-grained compute/memory/communication modeling
  - `topology.py` - parses hierarchical topology JSON and builds chip interconnect graph
  - `latency.py` - extensive latency calculation functions for compute ops (Attention, FFN, MLA, MoE) and communication (AllReduce, P2P, AllToAll, AllGather, ReduceScatter)
  - `gantt.py` - generates Gantt chart data for visualization
  - `types.py` - comprehensive type definitions using dataclasses
- **Key Abstractions**:
  - `TopologyParser` builds `InterconnectGraph` from `HierarchicalTopology`
  - `map_parallelism()` assigns chips to TP/PP/DP/EP groups based on `ParallelismStrategy`
  - Supports specialized architectures: MLA (Multi-head Latent Attention for DeepSeek V3/R1), MoE (Mixture of Experts)

### Data Flow

1. Frontend sends topology + model + hardware + parallelism configs to `/api/simulate`
2. Backend parses topology, builds chip graph, assigns parallelism groups
3. Simulator runs prefill + decode phases with fine-grained event-based modeling
4. Results include Gantt chart, statistics (MFU, MBU, TTFT, TPOT, bubble ratios)
5. Frontend visualizes results in 3D view, charts, and analysis panels

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

- `backend/requirements.txt` - Python dependencies (FastAPI, uvicorn, pydantic)
- `frontend/package.json` - Node dependencies, uses pnpm as package manager
- `frontend/vite.config.ts` - Vite configuration
- `frontend/tsconfig.json` - TypeScript configuration

## Important Implementation Details

### Parallelism Strategy Mapping

- Assignment order (inner to outer): TP → EP → PP → DP
- TP groups prefer same-board chips (high-bandwidth NVLink)
- PP groups can span boards (P2P communication)
- DP/EP groups can span racks/pods
- Implementation: `topology.py:map_parallelism()`

### Communication Algorithms

- AllReduce: Ring, Double Binary Tree, Halving-Doubling, Reduce-Broadcast
- AllToAll: Pairwise, Ring, Bruck
- Algorithm selection impacts latency calculations in `latency.py`

### Special Model Types

- **MLA (DeepSeek V3/R1)**: Uses compressed KV cache with LoRA-based projections. Has specialized latency functions (`calc_mla_*`)
- **MoE**: Supports standard and shared experts. Combines with EP (Expert Parallelism) and TP. Has routing overhead and load balancing considerations

### Hardware Abstraction Layers

- ChipHardwareConfig: per-chip specs (TFLOPS, memory, bandwidth, cache)
- NodeConfig: intra-node connectivity (NVLink parameters)
- ClusterConfig: inter-node network (InfiniBand/Ethernet)

### Topology Structure

- 5-level hierarchy: Die → Chip → Board (in Rack) → Rack (in Pod) → Pod
- Connections define bandwidth/latency between components
- Frontend visualizes in 3D with color-coded layers and interactive controls

## Testing Notes

- No test framework is currently set up
- Manual testing through the web UI at http://localhost:3100
- Backend validation endpoint `/api/validate` can be used to verify configs before simulation

## Port Configuration

- Frontend default: 3100
- Backend default: 8001
- CORS is configured to allow all origins in development

## Code Style Notes

- Backend uses Python dataclasses extensively for type safety
- Frontend uses TypeScript with React functional components and hooks
- Enums defined in `types.py` for task types, phases, algorithms
- Comments are primarily in Chinese
