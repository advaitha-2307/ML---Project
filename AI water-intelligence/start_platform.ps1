Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting AI Water Intelligence Platform" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$root = $PSScriptRoot
if (-not $root) { $root = Get-Location }

Write-Host "`n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; `$env:PYTHONPATH='.'; py -3.14 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

Write-Host "[2/2] Starting React Vite Frontend on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Start-Sleep -Seconds 3
Write-Host "`nOpening application in your default browser..." -ForegroundColor Yellow
Start-Process "http://localhost:5173"

Write-Host "`nPlatform running successfully!" -ForegroundColor Cyan
Write-Host "Frontend:    http://localhost:5173"
Write-Host "Swagger API: http://127.0.0.1:8000/docs`n"
