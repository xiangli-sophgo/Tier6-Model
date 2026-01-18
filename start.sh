#!/bin/bash

# Tier6+ 3D拓扑配置工具启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Tier6+ 3D 拓扑配置器"
echo "=========================================="

# 启动后端
echo ""
echo "[1/2] 启动后端服务 (端口 8100)..."
cd "$SCRIPT_DIR/backend"
python main.py &
BACKEND_PID=$!
echo "后端 PID: $BACKEND_PID"

# 等待后端启动
sleep 2

# 启动前端
echo ""
echo "[2/2] 启动前端服务 (端口 3100)..."
cd "$SCRIPT_DIR/frontend"

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    pnpm install
fi

pnpm dev &
FRONTEND_PID=$!
echo "前端 PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "  服务已启动!"
echo "  前端: http://localhost:3100"
echo "  后端: http://localhost:8100"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待中断
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
