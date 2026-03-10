param(
    [int]$CameraIndex = 0,
    [int]$IntervalMs = 300
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
$runtimeDir = Join-Path $scriptRoot "runtime"
$outLog = Join-Path $runtimeDir "camera_client.log"
$errLog = Join-Path $runtimeDir "camera_client.err.log"
$pidFile = Join-Path $runtimeDir "camera_client.pid"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Path $runtimeDir | Out-Null
}

if (Test-Path $pidFile) {
    $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($existingPid) {
        $existingProc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($existingProc) {
            Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
        }
    }
}

if (Test-Path $outLog) {
    Remove-Item $outLog -Force -ErrorAction SilentlyContinue
}
if (Test-Path $errLog) {
    Remove-Item $errLog -Force -ErrorAction SilentlyContinue
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$scriptPath = Join-Path $repoRoot "deploy\websocket_service\examples\python_camera_client.py"

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList @(
        $scriptPath,
        "--url", "ws://127.0.0.1:8000/ws/infer",
        "--username", "local-tester",
        "--password", "local-secret",
        "--camera", "$CameraIndex",
        "--source-id", "pc-webcam-01",
        "--source-name", "PC-Webcam-Demo",
        "--interval-ms", "$IntervalMs",
        "--return-image",
        "--show",
        "--simulate-raspberry-alert"
    ) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content $pidFile
Write-Output $proc.Id
