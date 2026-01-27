"""
LLM 推理模拟器启动脚本

使用 uvicorn 启动 FastAPI 服务。
"""

import os
from pathlib import Path
import uvicorn


def load_env():
    """手动加载.env文件（从项目根目录）"""
    # 项目根目录: backend/llm_simulator/main.py -> ../../.. -> Tier6-Model/
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    """启动服务"""
    # 加载.env文件
    load_env()

    # 从环境变量读取端口配置（与前端共用 VITE_API_PORT）
    port = int(os.getenv("VITE_API_PORT", "8001"))

    print(f"启动后端服务，端口: {port}")

    uvicorn.run(
        "llm_simulator.web.api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
