@echo off
cd /d "%~dp0"
title Sentinel-2 Super-Resolution Studio
echo =======================================================================
echo          Starting Sentinel-2 Super-Resolution Studio
echo =======================================================================
echo.

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at %PYTHON_EXE%!
    echo Please ensure the project virtual environment is installed.
    pause
    exit /b 1
)

echo [1/2] Launching browser on http://127.0.0.1:8000...
start "" "http://127.0.0.1:8000"

echo [2/2] Starting FastAPI backend with Uvicorn...
"%PYTHON_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
