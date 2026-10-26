# DeepShield - Start Next.js Frontend Development Server
# Runs Next.js on Port 3000

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting DeepShield Next.js Dashboard (Port 3000)      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location -Path "$PSScriptRoot\frontend"
cmd.exe /c "npm run dev"
