# Start the JobSpy Dashboard
Write-Host "`n  Starting JobSpy Dashboard..." -ForegroundColor Cyan
Write-Host "  Opening http://localhost:5000`n" -ForegroundColor Green
Start-Process "http://localhost:5000"
& "E:\JOBS\venv\Scripts\python.exe" "E:\JOBS\dashboard\app.py"
