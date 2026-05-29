# run_hourly.ps1
# Windows Task Scheduler wrapper for run_jobspy.py
# Activates the venv, runs the script, logs output with timestamp.
# Prunes log files older than 7 days automatically.

$PythonExe  = "E:\JOBS\venv\Scripts\python.exe"
$ScriptPath = "E:\JOBS\run_jobspy.py"
$LogDir     = "E:\JOBS\logs"
$Timestamp  = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile    = "$LogDir\jobspy_$Timestamp.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

"=== JobSpy run started at $(Get-Date) ===" | Out-File -FilePath $LogFile -Encoding utf8
& $PythonExe $ScriptPath 2>&1 | Out-File -FilePath $LogFile -Encoding utf8 -Append
"=== JobSpy run finished at $(Get-Date) ===" | Out-File -FilePath $LogFile -Encoding utf8 -Append

# Prune logs older than 7 days
Get-ChildItem "$LogDir\jobspy_*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item -Force
