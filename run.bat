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
    echo Please create the virtual environment first ^(see README.md, Setup^):
    echo   python -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo The model takes ~20-40 s to load. The browser opens automatically once the server is ready.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 300; $i++) { try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health).StatusCode -eq 200) { Start-Process http://127.0.0.1:8000; break } } catch { }; Start-Sleep -Seconds 1 }"

echo Starting Uvicorn FastAPI backend on GPU...
echo Press Ctrl+C in this window to stop the server.
echo.

.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause
