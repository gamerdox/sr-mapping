# PowerShell Launcher for Sentinel-2 Super-Resolution Studio
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Python virtual environment not found at: $PythonExe" -ForegroundColor Red
    Write-Host "Please ensure .venv is installed in $ScriptDir" -ForegroundColor Yellow
    exit 1
}

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         Starting Sentinel-2 Super-Resolution Studio" -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Working Directory : $ScriptDir"
Write-Host "Python Executable : $PythonExe"
Write-Host "Server URL        : http://127.0.0.1:8000"
Write-Host "-----------------------------------------------------------------------"

Start-Process "http://127.0.0.1:8000"

& $PythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
