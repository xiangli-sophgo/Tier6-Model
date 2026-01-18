"""
LLM 推理模拟器启动脚本

使用 uvicorn 启动 FastAPI 服务。
"""

import uvicorn


def main():
    """启动服务"""
    uvicorn.run(
        "llm_simulator.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
