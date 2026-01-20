@echo off
chcp 65001 > nul
title Tier6+ 快速启动

echo ==========================================
echo   Tier6+ 快速启动
echo ==========================================
echo.

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: 启动后端
echo [1/2] 启动后端服务 (端口 8001)...
start "Tier6+ Backend" cmd /k "cd /d %SCRIPT_DIR%backend && python3 -m uvicorn llm_simulator.api:app --port 8001 --reload"

:: 等待2秒
timeout /t 2 /nobreak > nul

:: 启动前端
echo [2/2] 启动前端服务 (端口 3100)...
cd frontend
start "Tier6+ Frontend" cmd /k "cd /d %SCRIPT_DIR%frontend && pnpm run dev"

echo.
echo ==========================================
echo 服务正在启动...
echo   前端: http://localhost:3100
echo   后端: http://localhost:8001
echo ==========================================
echo.
echo 提示:
echo   - 两个新窗口已打开 (后端 + 前端)
echo   - 关闭窗口即可停止对应服务
echo   - 首次启动可能需要等待依赖安装
echo.

:: 等待5秒后打开浏览器
timeout /t 5 /nobreak > nul
start http://localhost:3100

echo 浏览器已打开，可以关闭此窗口
timeout /t 3 /nobreak > nul
