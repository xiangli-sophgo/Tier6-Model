@echo off
chcp 65001 > nul
title Tier6+ 环境配置

echo ==========================================
echo   Tier6+ 环境配置向导
echo ==========================================
echo.

:: 检查 Python
echo [1/3] 检查 Python 环境...
python3 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python 未安装或未添加到 PATH
    echo.
    echo 请安装 Python 3.9 或更高版本:
    echo   1. 访问: https://www.python.org/downloads/
    echo   2. 下载并安装 Python
    echo   3. 安装时勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
) else (
    python3 --version
    echo Python 已安装 ✓
)

echo.
echo [2/3] 检查 Node.js 和 pnpm...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Node.js 未安装
    echo 请访问 https://nodejs.org/ 下载安装
    pause
    exit /b 1
) else (
    node --version
    echo Node.js 已安装 ✓
)

pnpm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pnpm 未安装，正在安装...
    call npm install -g pnpm
) else (
    pnpm --version
    echo pnpm 已安装 ✓
)

echo.
echo [3/3] 安装项目依赖...
echo.

:: 安装后端依赖
echo 安装后端依赖...
cd backend

python3 -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [错误] 后端依赖安装失败
    pause
    exit /b 1
)
echo 后端依赖安装完成 ✓

cd ..

:: 安装前端依赖
echo.
echo 安装前端依赖...
cd frontend
call pnpm install
if %errorlevel% neq 0 (
    echo [错误] 前端依赖安装失败
    pause
    exit /b 1
)
echo 前端依赖安装完成 ✓

cd ..

echo.
echo ==========================================
echo   环境配置完成!
echo ==========================================
echo.
echo 运行 start.bat 启动应用
echo.
pause
