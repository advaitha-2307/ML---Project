@echo off
title AI Water Intelligence Platform
echo ========================================================
echo Starting AI Water Intelligence Platform
echo ========================================================

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Water Intelligence Backend (FastAPI)" cmd /k "cd /d %~dp0 && set PYTHONPATH=. && py -3.14 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "Water Intelligence Frontend (React)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are launching in separate windows.
echo Dashboard: http://localhost:5173
echo Swagger Docs: http://127.0.0.1:8000/docs
echo ========================================================
timeout /t 3 >nul
start http://localhost:5173
