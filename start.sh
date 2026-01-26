#!/bin/bash

# Tier6+ 启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查是否需要安装依赖
if [ "$1" = "--setup" ] || [ "$1" = "-s" ]; then
    echo "=========================================="
    echo "  Tier6+ 环境配置"
    echo "=========================================="
    echo ""

    # 检查 Python
    echo "[1/3] 检查 Python 环境..."
    if ! command -v python3 &> /dev/null; then
        echo "[错误] Python 未安装"
        echo "请访问 https://www.python.org/downloads/ 安装 Python 3.9+"
        exit 1
    fi
    python3 --version
    echo "Python ✓"

    # 检查 Node.js 和 pnpm
    echo ""
    echo "[2/3] 检查 Node.js 和 pnpm..."
    if ! command -v node &> /dev/null; then
        echo "[错误] Node.js 未安装"
        echo "请访问 https://nodejs.org/ 安装 Node.js"
        exit 1
    fi
    node --version
    echo "Node.js ✓"

    if ! command -v pnpm &> /dev/null; then
        echo "pnpm 未安装，正在安装..."
        npm install -g pnpm
    fi
    pnpm --version
    echo "pnpm ✓"

    # 安装依赖
    echo ""
    echo "[3/3] 安装项目依赖..."
    echo ""
    echo "安装后端依赖..."
    cd "$SCRIPT_DIR/backend"
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[错误] 后端依赖安装失败"
        exit 1
    fi
    echo "后端依赖 ✓"

    echo ""
    echo "安装前端依赖..."
    cd "$SCRIPT_DIR/frontend"
    pnpm install
    if [ $? -ne 0 ]; then
        echo "[错误] 前端依赖安装失败"
        exit 1
    fi
    echo "前端依赖 ✓"

    echo ""
    echo "=========================================="
    echo "  环境配置完成！"
    echo "=========================================="
    echo ""
fi

echo "=========================================="
echo "  Tier6+ 启动中..."
echo "=========================================="

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "[提示] pnpm 未安装，请先运行: ./start.sh --setup"
    exit 1
fi

# 检查前端依赖
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "[提示] 前端依赖未安装，请先运行: ./start.sh --setup"
    exit 1
fi

# 启动后端
echo ""
echo "[1/2] 启动后端服务 (端口 8001)..."
cd "$SCRIPT_DIR/backend"
python3 -m uvicorn llm_simulator.api:app --port 8001 --reload &
BACKEND_PID=$!
echo "后端 PID: $BACKEND_PID"

# 等待后端启动
sleep 2

# 启动前端
echo ""
echo "[2/2] 启动前端服务 (端口 3100)..."
cd "$SCRIPT_DIR/frontend"
pnpm dev &
FRONTEND_PID=$!
echo "前端 PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "  服务已启动"
echo "  前端: http://localhost:3100"
echo "  后端: http://localhost:8001"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待中断
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
