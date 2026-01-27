@echo off
chcp 65001 > nul

REM 读取 .env 文件中的 API 端口
set API_PORT=8001
for /f "tokens=2 delims==" %%a in ('findstr /r "^VITE_API_PORT=" .env 2^>nul') do set API_PORT=%%a

echo 启动后端服务 (端口: %API_PORT%)...
cd /d %~dp0..
cd backend
python3 -m uvicorn llm_simulator.web.api:app --reload --host 0.0.0.0 --port %API_PORT%
