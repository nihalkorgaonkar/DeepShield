# DeepShield - Start FastAPI Python Backend Server
# Requires Python virtual environment in .venv

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting DeepShield FastAPI Backend (Port 8000)        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "."
& .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
