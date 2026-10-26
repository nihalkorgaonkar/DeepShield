# DeepShield - Dual Launcher Helper
# Launches Backend and Frontend in separate PowerShell processes

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Launching DeepShield Full-Stack Platform                " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Start Backend in new process
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\run_backend.ps1"
Write-Host "[✓] Backend process launched on http://127.0.0.1:8000" -ForegroundColor Green

# Start Frontend in new process
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\run_frontend.ps1"
Write-Host "[✓] Frontend process launched on http://localhost:3000" -ForegroundColor Green

Write-Host "`nOpen your browser at: http://localhost:3000" -ForegroundColor Yellow
