$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $scriptRoot "runtime\camera_client.pid"

if (-not (Test-Path $pidFile)) {
    Write-Output "No camera client pid file found."
    exit 0
}

$pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue
if ($pidValue) {
    Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Write-Output "Camera client stopped."
