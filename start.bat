@echo off
title Sentinel-2 Super-Resolution Studio
echo =======================================================================
echo          Starting Sentinel-2 Super-Resolution Studio
echo =======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please ensure the project virtual environment is installed.
    pause
    exit /b 1
)

echo [1/2] Launching browser on http://127.0.0.1:8000...
start "" "http://127.0.0.1:8000"

echo [2/2] Starting FastAPI backend with Uvicorn...
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
