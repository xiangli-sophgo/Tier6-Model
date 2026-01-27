@echo off
chcp 65001 > nul
title Tier6+ 启动

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: 检查是否需要安装依赖
if "%1"=="--setup" goto SETUP
if "%1"=="-s" goto SETUP
goto START

:SETUP
echo ==========================================
echo   Tier6+ 环境配置
echo ==========================================
echo.

:: 检查 Python
echo [1/3] 检查 Python 环境...
python3 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python 未安装或未添加到 PATH
    echo 请访问 https://www.python.org/downloads/ 安装 Python 3.9+
    pause
    exit /b 1
)
python3 --version
echo Python ✓

:: 检查 Node.js 和 pnpm
echo.
echo [2/3] 检查 Node.js 和 pnpm...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Node.js 未安装
    echo 请访问 https://nodejs.org/ 安装 Node.js
    pause
    exit /b 1
)
node --version
echo Node.js ✓

where pnpm >nul 2>&1
if %errorlevel% neq 0 (
    echo pnpm 未安装，正在安装...
    call npm install -g pnpm
)
pnpm --version
echo pnpm ✓

:: 安装依赖
echo.
echo [3/3] 安装项目依赖...
echo.
echo 安装后端依赖...
cd backend
python3 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 后端依赖安装失败
    pause
    exit /b 1
)
echo 后端依赖 ✓

cd ..
echo.
echo 安装前端依赖...
cd frontend
call pnpm install
if %errorlevel% neq 0 (
    echo [错误] 前端依赖安装失败
    pause
    exit /b 1
)
echo 前端依赖 ✓

cd ..
echo.
echo ==========================================
echo   环境配置完成！
echo ==========================================
echo.

:START
echo ==========================================
echo   Tier6+ 启动中...
echo ==========================================
echo.

:: 检查 pnpm
where pnpm >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] pnpm 未安装，请先运行: start.bat --setup
    pause
    exit /b 1
)

:: 检查前端依赖
if not exist "frontend\node_modules" (
    echo [提示] 前端依赖未安装，请先运行: start.bat --setup
    pause
    exit /b 1
)

:: 清理旧进程
echo [0/2] 清理旧进程...

:: 读取 .env 中的端口配置
set API_PORT=8001
for /f "tokens=2 delims==" %%a in ('findstr /r "^VITE_API_PORT=" .env 2^>nul') do set API_PORT=%%a

:: 杀死占用后端端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo 发现旧后端进程 PID: %%a，正在终止...
    taskkill /F /PID %%a >nul 2>&1
)

:: 杀死占用前端端口 3100 的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3100 " ^| findstr "LISTENING" 2^>nul') do (
    echo 发现旧前端进程 PID: %%a，正在终止...
    taskkill /F /PID %%a >nul 2>&1
)

:: 等待进程释放端口
timeout /t 1 /nobreak > nul
echo 旧进程清理完成 ✓
echo.

:: 启动后端
echo [1/2] 启动后端服务 (读取 .env 配置)...
start "Tier6+ Backend" cmd /k "cd /d %SCRIPT_DIR%backend && python3 -m llm_simulator.main"

:: 等待2秒
timeout /t 2 /nobreak > nul

:: 启动前端
echo [2/2] 启动前端服务 (端口 3100)...
start "Tier6+ Frontend" cmd /k "cd /d %SCRIPT_DIR%frontend && pnpm run dev"

echo.
echo ==========================================
echo   服务已启动
echo   前端: http://localhost:3100
echo   后端: http://localhost:8002 (配置在 backend/.env)
echo ==========================================
echo.
echo 提示: 关闭后端/前端窗口可停止对应服务
echo.

:: 等待5秒后打开浏览器
timeout /t 5 /nobreak > nul
start http://localhost:3100

echo 浏览器已打开，可以关闭此窗口
timeout /t 3 /nobreak > nul
