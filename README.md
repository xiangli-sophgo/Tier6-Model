# Tier6+ 3D 拓扑配置器

交互式3D多层级网络拓扑配置和可视化工具。

## 功能特点

- **3D可视化**: 使用 Three.js 实现可旋转、缩放的3D拓扑预览
- **多层级支持**: Die → Chip → Board → Server → Pod 五层结构
- **交互式配置**: 实时调整每层节点数量和拓扑类型
- **多种拓扑**: 支持 Mesh（全连接）、All-to-All（分组全连接）、Ring（环形）
- **导出配置**: 可导出JSON配置文件用于仿真

## 快速开始

### 1. 安装依赖

```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd frontend
pnpm install
```

### 2. 启动服务

```bash
./start.sh
```

或分别启动：

```bash
# 后端 (端口 8100)
cd backend
python main.py

# 前端 (端口 3100)
cd frontend
pnpm dev
```

### 3. 访问

打开浏览器访问: http://localhost:3100

## 项目结构

```
Tier6+model/
├── backend/
│   ├── main.py           # FastAPI后端入口
│   ├── topology.py       # 拓扑生成逻辑
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Scene3D.tsx      # 3D场景
│   │   │   └── ConfigPanel.tsx  # 配置面板
│   │   ├── api/
│   │   │   └── topology.ts
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
├── start.sh
└── README.md
```

## 操作说明

### 3D视图
- **左键拖拽**: 旋转视角
- **滚轮**: 缩放
- **右键拖拽**: 平移

### 配置面板
- **节点数量**: 使用滑块调整每层节点数 (1-16)
- **拓扑类型**: 选择 Mesh/All-to-All/Ring
- **可见性**: 开关控制每层是否显示
- **层间连接**: 开关控制是否显示层级间连接线

## 技术栈

- **后端**: FastAPI + Python
- **前端**: React + TypeScript + Three.js + Ant Design
- **3D渲染**: react-three-fiber + @react-three/drei
