@echo off
chcp 65001 > nul

REM 从 .env 文件读取 API 端口
set API_PORT=
for /f "tokens=2 delims==" %%a in ('findstr /r "^VITE_API_PORT=" .env 2^>nul') do set API_PORT=%%a
if "%API_PORT%"=="" (
    echo [ERROR] VITE_API_PORT not found in .env file
    echo Please create .env file with VITE_API_PORT=^<port^>
    pause
    exit /b 1
)

echo 启动后端服务 (端口: %API_PORT%)...
cd /d %~dp0..
cd backend
python3 -m math_model.main
