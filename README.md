# Tier6+Model

交互式3D多层级网络拓扑配置器和LLM推理仿真工具。支持层次化GPU集群拓扑可视化（Die → Chip → Board → Server → Pod）和多种并行策略的LLM推理仿真（TP、PP、DP、EP、SP）。

## 快速开始

### 前置依赖

- Python 3.9+
- Node.js 16+
- pnpm（会自动安装）

### 安装与启动

**Windows:**
```cmd
# 首次使用：安装依赖并启动
start.bat --setup

# 后续启动
start.bat
```

**Linux/Mac:**
```bash
# 首次使用：安装依赖并启动
./start.sh --setup

# 后续启动
./start.sh
```

启动后访问: http://localhost:3100

## 功能特性

- **3D可视化**: Three.js实现的可交互3D拓扑渲染
- **多层级拓扑**: 五层结构（Die → Chip → Board → Server → Pod）
- **LLM仿真**: 精细化的推理性能建模（Prefill/Decode、MFU/MBU、TTFT/TPOT）
- **并行策略**: 支持TP、PP、DP、EP、SP及其组合
- **专用架构**: DeepSeek MLA、MoE等特殊模型支持
- **Gantt图**: 可视化执行时间线和气泡分析

## 技术栈

- **后端**: Python + FastAPI + Uvicorn
- **前端**: React + TypeScript + Vite
- **3D渲染**: Three.js + react-three-fiber + drei
- **UI**: Ant Design + Recharts

## 故障排除

**端口被占用:**
```bash
# 修改端口（在start脚本中）
# 后端: --port 8001 → --port 8002
# 前端: vite.config.ts中修改server.port
```

**依赖安装失败（国内用户）:**
```bash
# Python镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# pnpm镜像
pnpm config set registry https://registry.npmmirror.com
```

**Python命令未找到:**
- 确保安装时勾选"Add Python to PATH"
- 或手动添加Python到系统PATH

## 许可证

MIT
