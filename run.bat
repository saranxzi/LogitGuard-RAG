@echo off
title LogitGuard-RAG Server
cd /d "%~dp0"

echo ============================================================
echo Starting LogitGuard-RAG Server
echo Target: http://127.0.0.1:8000
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create the virtual environment first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Opening web interface in default browser...
start "" "http://127.0.0.1:8000"

echo Starting Uvicorn FastAPI backend on GPU...
echo Press Ctrl+C in this window to stop the server.
echo.

.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause
