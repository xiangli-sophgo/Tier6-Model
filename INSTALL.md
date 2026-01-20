# Tier6+ 安装指南 (Windows)

## 前置要求

### 1. 安装 Python 3.9+

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载最新的 Python 3.9+ 版本
3. **重要**: 安装时勾选 "Add Python to PATH"
4. 验证安装:
   ```cmd
   python --version
   ```

### 2. 安装 Node.js

1. 访问 [Node.js 官网](https://nodejs.org/)
2. 下载并安装 LTS 版本
3. 验证安装:
   ```cmd
   node --version
   npm --version
   ```

### 3. 安装 pnpm (可选，安装脚本会自动安装)

```cmd
npm install -g pnpm
```

## 快速安装

### 方式 1: 一键安装 (推荐)

双击运行 `setup.bat`，它会自动:
- 检查环境依赖
- 创建 Python 虚拟环境
- 安装所有后端和前端依赖

### 方式 2: 手动安装

#### 后端依赖
```cmd
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### 前端依赖
```cmd
cd frontend
pnpm install
```

## 启动应用

### 方式 1: 使用启动脚本
双击运行 `start.bat`

### 方式 2: 使用 VS Code 任务
1. 在 VS Code 中打开项目
2. 按 `Ctrl+Shift+P`
3. 输入 "Tasks: Run Task"
4. 选择 "启动 Tier6+ 工具"

### 方式 3: 手动启动

**后端** (新建终端):
```cmd
cd backend
venv\Scripts\activate.bat
python -m uvicorn llm_simulator.api:app --port 8001 --reload
```

**前端** (新建终端):
```cmd
cd frontend
pnpm run dev
```

## 访问应用

启动成功后，在浏览器中访问:
- 前端: http://localhost:3100
- 后端 API: http://localhost:8001

## 故障排除

### Python 命令未找到
- 确保安装时勾选了 "Add Python to PATH"
- 或手动将 Python 安装目录添加到系统 PATH
- 重启终端/命令提示符

### pnpm 命令未找到
```cmd
npm install -g pnpm
```
- 重启终端

### 端口被占用
修改 `start.bat` 中的端口号，或手动指定:
```cmd
# 后端使用 8002 端口
python -m uvicorn llm_simulator.api:app --port 8002 --reload
```

### 依赖安装失败
1. 确保网络连接正常
2. 尝试使用国内镜像:
   ```cmd
   # Python pip 镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

   # pnpm 镜像
   pnpm config set registry https://registry.npmmirror.com
   ```
