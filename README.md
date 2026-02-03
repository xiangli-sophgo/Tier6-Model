# Tier6+Model

交互式3D多层级网络拓扑配置器和LLM推理仿真工具。支持层次化GPU集群拓扑可视化（Pod → Rack → Board → Chip）和多种并行策略的LLM推理仿真（TP、PP、DP、EP、SP）。

## 快速开始

### 前置依赖

- Python 3.9+
- Node.js 16+
- pnpm（推荐）或 npm

### 🚀 推荐方式（单命令启动）

```bash
# 首次使用：安装依赖
npm run setup

# 启动服务（自动清理旧进程 + 并发启动前后端）
npm run dev

# 停止服务
Ctrl + C
```

**特性:**
- ✅ 单命令启动前后端
- ✅ 自动清理旧进程，防止端口冲突
- ✅ 统一日志输出（带颜色区分）
- ✅ 代码修改自动热重载
- ✅ Ctrl+C 统一停止所有服务

### 传统方式（兼容）

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

**VS Code用户:**
- 按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`
- 选择 `🚀 启动 Tier6+ (推荐)`

启动后访问: http://localhost:3100

### 环境配置

后端需要配置 `.env` 文件：

```bash
cd backend
cp .env.example .env
```

主要配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ALLOWED_ORIGINS` | CORS 允许的来源 | localhost:3100,3000 |
| `DATABASE_URL` | 数据库连接 URL | sqlite:///./llm_evaluations.db |
| `MAX_GLOBAL_WORKERS` | 最大并发 worker 数量 | 4 |

## 功能特性

- **3D可视化**: Three.js实现的可交互3D拓扑渲染
- **多层级拓扑**: 层次化结构（Pod → Rack → Board → Chip）
- **LLM仿真**: 精细化的推理性能建模（Prefill/Decode、MFU/MBU、TTFT/TPOT）
- **并行策略**: 支持TP、PP、DP、EP及其组合
- **专用架构**: DeepSeek MLA、MoE等特殊模型支持
- **Gantt图**: 可视化执行时间线和瓶颈分析

## 技术栈

- **后端**: Python + FastAPI + Uvicorn
- **前端**: React + TypeScript + Vite
- **3D渲染**: Three.js + react-three-fiber + drei
- **UI**: shadcn/ui (Radix UI + Tailwind CSS) + ECharts

## 故障排除

**端口被占用 / 进程残留:**
```bash
# 清理所有相关进程
npm run clean

# 然后重新启动
npm run dev
```

**代码修改不生效:**
```bash
# 清理后端缓存
rm -rf backend/**/__pycache__

# 清理并重启
npm run clean
npm run dev
```

**修改端口配置:**
```bash
# 编辑 .env 文件
VITE_API_PORT=8003  # 修改为你需要的端口
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

## 📚 详细文档

查看 [docs/](./docs/) 目录获取完整技术文档：

- **[用户指南](./docs/guides/)** - UI 使用指南和教程
- **[设计文档](./docs/design/)** - 系统架构和核心模块实现
- **[技术参考](./docs/reference/)** - 规范文档和 API 参考
- **[开发者文档](./docs/development/)** - 项目路线图和贡献指南

推荐从 [docs/README.md](./docs/README.md) 开始浏览。

## 许可证

MIT
