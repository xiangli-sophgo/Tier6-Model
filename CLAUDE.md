# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tier6+Model is an interactive 3D multi-level network topology configurator and LLM inference simulator. It provides visualization and simulation capabilities for hierarchical GPU/accelerator cluster topologies (Die → Chip → Board → Server → Pod) and models LLM inference workloads with various parallelism strategies (TP, PP, DP, EP, SP).

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
# Start both frontend and backend (recommended)
./start.sh

# Or start individually:
# Backend (port 8100 - note: code uses 8001 in main.py but start.sh may differ)
cd backend
python main.py

# Frontend (port 3100)
cd frontend
pnpm dev
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
- Backend default: 8001 (main.py) or 8100 (referenced in README)
- CORS is configured to allow all origins in development

## Code Style Notes

- Backend uses Python dataclasses extensively for type safety
- Frontend uses TypeScript with React functional components and hooks
- Enums defined in `types.py` for task types, phases, algorithms
- Comments are primarily in Chinese
