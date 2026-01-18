@echo off
chcp 65001 > nul
title Tier6+ 3D拓扑配置器

echo ==========================================
echo   Tier6+ 3D 拓扑配置器
echo ==========================================

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: 启动后端
echo 启动后端服务...
cd backend

:: 检查虚拟环境
if not exist "venv" (
    echo 创建Python虚拟环境...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo 安装后端依赖...
pip install -r requirements.txt

:: 启动后端
start "Tier6+ Backend" cmd /k "cd /d %SCRIPT_DIR%backend && call venv\Scripts\activate.bat && python -m uvicorn llm_simulator.api:app --port 8001 --reload"

cd ..

:: 启动前端
echo 启动前端服务...
cd frontend

:: 检查pnpm是否安装
where pnpm >nul 2>nul
if %errorlevel% neq 0 (
    echo pnpm未安装，正在安装...
    call npm install -g pnpm
)

:: 检查node_modules
if not exist "node_modules" (
    echo 安装前端依赖...
    call pnpm install
)

start "Tier6+ Frontend" cmd /k "cd /d %SCRIPT_DIR%frontend && pnpm run dev"

echo.
echo ==========================================
echo 服务已启动:
echo   前端: http://localhost:3100
echo   后端: http://localhost:8001
echo ==========================================
echo 关闭此窗口不会停止服务,请手动关闭后端和前端窗口以停止服务
echo.

:: 等待几秒后自动打开浏览器
timeout /t 3 /nobreak > nul
start http://localhost:3100

pause
