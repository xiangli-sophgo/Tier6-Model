"""math_model 后端入口

独立的 FastAPI 应用，提供 /api 前缀的所有端点。
"""

import os
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from math_model.L0_entry.api import router
from math_model.L0_entry.database import init_db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Math Model API",
    description="LLM 推理部署分析平台 - 数学建模后端",
    version="3.0.0",
)

# 配置 CORS
allowed_origins_str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3100,http://localhost:3000,http://127.0.0.1:3100,http://127.0.0.1:3000",
)
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# 挂载路由
app.include_router(router)  # /api/* 端点


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    import asyncio

    logger.info("Initializing database...")
    init_db()

    logger.info("Initializing WebSocket manager...")
    try:
        from math_model.L0_entry.websocket import get_ws_manager
        ws_manager = get_ws_manager()
        ws_manager.set_event_loop(asyncio.get_running_loop())
        logger.info("WebSocket manager initialized")
    except Exception as e:
        logger.warning(f"WebSocket manager initialization failed: {e}")

    logger.info("Application startup complete")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Math Model API",
        "version": "3.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


def load_env():
    """加载 .env 文件"""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    """启动服务"""
    load_env()
    port_str = os.getenv("VITE_API_PORT")
    if not port_str:
        raise RuntimeError(
            "VITE_API_PORT is not set. "
            "Please create .env file in project root with VITE_API_PORT=<port>"
        )
    port = int(port_str)
    logger.info(f"Starting math_model backend on port: {port}")
    uvicorn.run(
        "math_model.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
