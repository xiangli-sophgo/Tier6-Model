@echo off
chcp 65001 > nul
title Tier6+ 启动
SETLOCAL EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: 设置临时 PID 文件
set PID_FILE=%TEMP%\tier6_pids.tmp

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
echo [1/3] Checking Python environment...
python3 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python3 --version
echo Python OK

:: 检查 Node.js 和 pnpm
echo.
echo [2/3] Checking Node.js and pnpm...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
node --version
echo Node.js OK

where pnpm >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pnpm...
    call npm install -g pnpm
)
pnpm --version
echo pnpm OK

:: 安装依赖
echo.
echo [3/3] Installing project dependencies...
echo.
echo Installing backend dependencies...
cd backend
python3 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Backend dependencies installation failed
    pause
    exit /b 1
)
echo Backend dependencies OK

cd ..
echo.
echo Installing frontend dependencies...
cd frontend
call pnpm install
if %errorlevel% neq 0 (
    echo [ERROR] Frontend dependencies installation failed
    pause
    exit /b 1
)
echo Frontend dependencies OK

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
    echo [NOTICE] pnpm not installed. Please run: start.bat --setup
    pause
    exit /b 1
)

:: 检查前端依赖
if not exist "frontend\node_modules" (
    echo [NOTICE] Frontend dependencies not installed. Please run: start.bat --setup
    pause
    exit /b 1
)

:: 清理旧进程
echo [0/2] Cleaning up old processes...

:: 读取 .env 中的端口配置
set API_PORT=8001
for /f "tokens=2 delims==" %%a in ('findstr /r "^VITE_API_PORT=" .env 2^>nul') do set API_PORT=%%a

:: 清理旧的 PID 记录
if exist "%PID_FILE%" del /f /q "%PID_FILE%" >nul 2>&1

:: 杀死占用后端端口的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo Killing backend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: 杀死占用前端端口 3100 的进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3100 " ^| findstr "LISTENING" 2^>nul') do (
    echo Killing frontend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: 等待进程释放端口
timeout /t 1 /nobreak > nul
echo Cleanup completed
echo.

:: 启动后端（后台运行）
echo.
echo Starting backend service (port %API_PORT%)...
cd /d %SCRIPT_DIR%backend
start /B python3 -m llm_simulator.main
echo Backend started in background

:: 等待后端启动并记录 PID
timeout /t 3 /nobreak > nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo BACKEND_PID=%%a >> "%PID_FILE%"
    echo Backend PID: %%a
    goto :backend_started
)
:backend_started

:: 启动前端（后台运行）
echo.
echo Starting frontend service (port 3100)...
cd /d %SCRIPT_DIR%frontend
start /B pnpm run dev
echo Frontend started in background

:: 等待前端启动并记录 PID
timeout /t 3 /nobreak > nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3100 " ^| findstr "LISTENING" 2^>nul') do (
    echo FRONTEND_PID=%%a >> "%PID_FILE%"
    echo Frontend PID: %%a
    goto :frontend_started
)
:frontend_started

echo.
echo ==========================================
echo   Services started
echo   Frontend: http://localhost:3100
echo   Backend: http://localhost:%API_PORT%
echo ==========================================
echo.
echo Press Ctrl+C or any key to stop all services
echo.

:: 等待3秒后打开浏览器
timeout /t 3 /nobreak > nul
start http://localhost:3100

:: 等待用户停止服务（使用 timeout 循环以便捕获 Ctrl+C）
echo.
:WAIT_LOOP
timeout /t 5 > nul 2>&1
if errorlevel 1 (
    :: Ctrl+C 被按下，清理并退出
    call :CLEANUP
    goto :END
)
goto :WAIT_LOOP

:: ============================================
:: 清理函数 - 停止所有服务
:: ============================================
:CLEANUP
echo.
echo Stopping services...

:: 优先使用保存的 PID
if exist "%PID_FILE%" (
    for /f "tokens=1,2 delims==" %%a in (%PID_FILE%) do (
        set PID_NAME=%%a
        set PID_VALUE=%%b
        echo Stopping !PID_NAME!: !PID_VALUE!
        taskkill /F /PID !PID_VALUE! >nul 2>&1
    )
    del /f /q "%PID_FILE%" >nul 2>&1
)

:: 兜底清理：通过端口查找并杀死进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo Stopping backend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3100 " ^| findstr "LISTENING" 2^>nul') do (
    echo Stopping frontend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All services stopped
timeout /t 1 /nobreak > nul
exit /b 0

:END
ENDLOCAL
exit /b 0
