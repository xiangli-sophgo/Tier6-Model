@echo off
title Tier6+ Startup
SETLOCAL EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

:: Set temporary PID file
set PID_FILE=%TEMP%\tier6_pids.tmp

:: Check if setup is needed
if "%1"=="--setup" goto SETUP
if "%1"=="-s" goto SETUP
goto START

:SETUP
echo ==========================================
echo   Tier6+ Environment Setup
echo ==========================================
echo.

:: Check and create .env file
echo [0/4] Checking environment configuration...
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
        echo .env file created [OK]
    ) else (
        echo [ERROR] .env.example not found
        pause
        exit /b 1
    )
) else (
    echo .env file already exists [OK]
)
echo.

:: Check Python
echo [1/4] Checking Python environment...
python3 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python3 --version
echo Python OK

:: Check Node.js and pnpm
echo.
echo [2/4] Checking Node.js and pnpm...
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

:: Install dependencies
echo.
echo [3/4] Installing project dependencies...
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
echo   Environment setup completed!
echo ==========================================
echo.

:START
echo ==========================================
echo   Starting Tier6+ services...
echo ==========================================
echo.

:: Check pnpm
where pnpm >nul 2>&1
if %errorlevel% neq 0 (
    echo [NOTICE] pnpm not installed. Please run: start.bat --setup
    pause
    exit /b 1
)

:: Check frontend dependencies
if not exist "frontend\node_modules" (
    echo [NOTICE] Frontend dependencies not installed. Please run: start.bat --setup
    pause
    exit /b 1
)

:: Cleanup old processes
echo [0/2] Cleaning up old processes...

:: Read port configuration from .env file
set API_PORT=
for /f "tokens=2 delims==" %%a in ('findstr /r "^VITE_API_PORT=" .env 2^>nul') do set API_PORT=%%a
if "%API_PORT%"=="" (
    echo [ERROR] VITE_API_PORT not found in .env file
    echo Please create .env file with VITE_API_PORT=^<port^>
    pause
    exit /b 1
)

:: Clear old PID records
if exist "%PID_FILE%" del /f /q "%PID_FILE%" >nul 2>&1

:: Kill processes occupying backend port
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo Killing backend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill processes occupying frontend port 3100
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3100 " ^| findstr "LISTENING" 2^>nul') do (
    echo Killing frontend process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: Wait for port release
timeout /t 1 /nobreak > nul
echo Cleanup completed
echo.

:: Start backend (background)
echo.
echo Starting backend service (port %API_PORT%)...
cd /d %SCRIPT_DIR%backend
start "Tier6 Backend (%API_PORT%)" cmd /k "python3 -m perf_model.main"
echo Backend started in a new window

:: Wait for backend startup and record PID
timeout /t 3 /nobreak > nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%API_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo BACKEND_PID=%%a >> "%PID_FILE%"
    echo Backend PID: %%a
    goto :backend_started
)
:backend_started

:: Start frontend (background)
echo.
echo Starting frontend service (port 3100)...
cd /d %SCRIPT_DIR%frontend
start "Tier6 Frontend (3100)" cmd /k "pnpm run dev"
echo Frontend started in a new window

:: Wait for frontend startup and record PID
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
echo Press any key to stop all services
echo.

:: Open browser after 3 seconds
timeout /t 3 /nobreak > nul
start http://localhost:3100

:: Wait for user to stop services (use timeout loop to capture Ctrl+C)
echo.
:WAIT_LOOP
pause > nul
call :CLEANUP
goto :END

:: ============================================
:: Cleanup function - stop all services
:: ============================================
:CLEANUP
echo.
echo Stopping services...

:: Use saved PIDs first
if exist "%PID_FILE%" (
    for /f "tokens=1,2 delims==" %%a in (%PID_FILE%) do (
        set PID_NAME=%%a
        set PID_VALUE=%%b
        echo Stopping !PID_NAME!: !PID_VALUE!
        taskkill /F /PID !PID_VALUE! >nul 2>&1
    )
    del /f /q "%PID_FILE%" >nul 2>&1
)

:: Fallback cleanup: find and kill processes by port
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
