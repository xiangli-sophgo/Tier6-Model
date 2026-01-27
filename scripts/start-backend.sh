#!/bin/bash

# 读取 .env 文件中的 API 端口
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

API_PORT=8001
if [ -f "$PROJECT_ROOT/.env" ]; then
    API_PORT=$(grep "^VITE_API_PORT=" "$PROJECT_ROOT/.env" | cut -d'=' -f2 | tr -d ' ')
    API_PORT=${API_PORT:-8001}
fi

# 查找可用的 Python
PYTHON_CMD=""
for cmd in /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[错误] 未找到 Python，请安装 Python 3"
    exit 1
fi

echo "启动后端服务 (端口: $API_PORT)..."
echo "使用 Python: $PYTHON_CMD"
cd "$PROJECT_ROOT/backend"
$PYTHON_CMD -m uvicorn llm_simulator.web.api:app --reload --host 0.0.0.0 --port $API_PORT
