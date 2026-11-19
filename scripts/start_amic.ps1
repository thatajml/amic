<#
.SYNOPSIS
    Wake up the whole AMIC stack with one command: infra, Ollama, API, dashboard.

.DESCRIPTION
    Usage (from anywhere):   start-amic.cmd            start everything
                             start-amic.cmd -Stop      stop the API and dashboard
                             start-amic.cmd -NoBrowser start without opening the dashboard

    Steps, each skipped if already up:
      1. Docker Desktop + Redis/Chroma containers, only when the effective config needs them
         (session_backend=redis or chroma_client=http). Use -Infra to start them regardless.
      2. Ollama server, then preload the tutor model so the first /chat is not a cold start.
      3. API   (uvicorn amic.service.api:app) on http://127.0.0.1:8080 (8000 is Chroma's).
      4. Dashboard (streamlit ui/dashboard.py) on http://localhost:8501.
    Logs and PIDs go to data/run/. Settings come from .env / AMIC_* as usual.
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$NoBrowser,
    [switch]$Infra,
    [int]$ApiPort = 8080,
    [int]$UiPort = 8501
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot 'amic_env.ps1')
$RunDir = Join-Path $Root 'data\run'
New-Item -ItemType Directory -Force $RunDir | Out-Null

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Skip([string]$msg) { Write-Host "    $msg" -ForegroundColor DarkGray }

function Test-Url([string]$url) {
    # PowerShell 5.1 tries IPv6 ::1 for "localhost" first; Ollama listens on IPv4 only and the
    # fallback can outlast the timeout, so probe 127.0.0.1 directly.
    $url = $url -replace '//localhost([:/])', '//127.0.0.1$1'
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
    } catch { return $false }
}

function Wait-Until([scriptblock]$check, [int]$seconds, [string]$what, [string]$log = '') {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $check) { return }
        Start-Sleep -Seconds 2
    }
    $hint = ''
    if ($log) { $hint = " See $log" }
    throw "$what did not come up within $seconds s.$hint"
}

function Stop-Tracked([string]$name) {
    $pidFile = Join-Path $RunDir "$name.pid"
    if (-not (Test-Path $pidFile)) { Write-Skip "${name}: not started by this script"; return }
    $procId = (Get-Content $pidFile -Raw).Trim()
    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        # /T: the venv python.exe is a launcher with a child interpreter; kill the whole tree.
        & taskkill.exe /PID $procId /T /F | Out-Null
        Write-Ok "$name stopped (pid $procId)"
    } else {
        Write-Skip "${name}: already stopped"
    }
    Remove-Item $pidFile -Force
}

function Start-Tracked([string]$name, [string[]]$pyArgs) {
    $out = Join-Path $RunDir "$name.log"
    $err = Join-Path $RunDir "$name.err.log"
    $p = Start-Process -FilePath $Py -ArgumentList $pyArgs -WorkingDirectory $Root `
        -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Set-Content -Path (Join-Path $RunDir "$name.pid") -Value $p.Id -Encoding ascii
    return $err
}

if ($Stop) {
    Write-Step 'Stopping AMIC'
    Stop-Tracked 'api'
    Stop-Tracked 'dashboard'
    Write-Skip 'Docker containers and Ollama are left running (shared services).'
    Write-Skip 'Stop them with: docker compose -f docker-compose.dev.yml stop'
    exit 0
}

$Py = Get-AmicPython $Root  # .venv or conda env, per AMIC_ENV (scripts/amic_env.ps1)
Write-Host "    python: $Py" -ForegroundColor DarkGray

# Effective config (.env + AMIC_* env), read the same way the app reads it.
$cfgJson = & $Py -c "import json; from amic.contracts.config import Settings as S; s=S(); print(json.dumps({'session': s.session_backend, 'chroma': s.chroma_client, 'redis': s.redis_url, 'ollama': s.models.ollama_base_url, 'tutor': s.models.tutor}))"
if ($LASTEXITCODE -ne 0) { throw 'Could not load AMIC settings (see the error above).' }
$cfg = $cfgJson | ConvertFrom-Json

# 1. Docker infra -------------------------------------------------------------------------
Write-Step 'Infrastructure (Redis, Chroma)'
$needDocker = $Infra -or $cfg.session -eq 'redis' -or $cfg.chroma -eq 'http'
if (-not $needDocker) {
    Write-Skip "skipped: config uses session_backend=$($cfg.session), chroma_client=$($cfg.chroma) (use -Infra to force)"
} else {
    # Native calls go through cmd: in PowerShell 5.1 any stderr output would throw under 'Stop'.
    $dockerOk = { & cmd.exe /c 'docker info >nul 2>&1'; $LASTEXITCODE -eq 0 }
    if (-not (& $dockerOk)) {
        $dd = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'
        if (-not (Test-Path $dd)) { $dd = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe' }
        if (-not (Test-Path $dd)) { throw 'Docker Desktop not found; install it or set AMIC_SESSION_BACKEND=sqlite.' }
        Write-Skip 'starting Docker Desktop...'
        Start-Process $dd
        Wait-Until $dockerOk 180 'Docker engine'
    }
    & cmd.exe /c 'docker compose -f docker-compose.dev.yml up -d >nul 2>&1'
    if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed; run it by hand to see why.' }
    Wait-Until { (& cmd.exe /c 'docker exec amic-redis-1 redis-cli ping 2>nul') -eq 'PONG' } 60 'Redis'
    Wait-Until { Test-Url 'http://localhost:8000/api/v2/heartbeat' } 60 'Chroma'
    Write-Ok 'Redis :6379 and Chroma :8000 up'
}

# 2. Ollama -------------------------------------------------------------------------------
Write-Step "Ollama ($($cfg.ollama))"
$ollamaOk = { Test-Url "$($cfg.ollama)/api/version" }
if (-not (& $ollamaOk)) {
    Write-Skip 'starting ollama serve...'
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RunDir 'ollama.log') `
        -RedirectStandardError (Join-Path $RunDir 'ollama.err.log')
    Wait-Until $ollamaOk 60 'Ollama' (Join-Path $RunDir 'ollama.err.log')
}
Write-Skip "preloading $($cfg.tutor)..."
$body = @{ model = $cfg.tutor; keep_alive = '30m' } | ConvertTo-Json
try {
    $ollamaApi = $cfg.ollama -replace '//localhost([:/])', '//127.0.0.1$1'
    Invoke-RestMethod -Method Post -Uri "$ollamaApi/api/generate" -Body $body `
        -ContentType 'application/json' -TimeoutSec 180 | Out-Null
    Write-Ok "Ollama up, $($cfg.tutor) loaded"
} catch {
    throw "Could not load $($cfg.tutor) in Ollama (is it pulled? 'ollama pull $($cfg.tutor)'): $_"
}

# 3. API ----------------------------------------------------------------------------------
$apiUrl = "http://127.0.0.1:$ApiPort"
Write-Step "API ($apiUrl)"
if (Test-Url "$apiUrl/health") {
    Write-Skip 'already running'
} else {
    # Shadow baseline feeds the dashboard's savings view; respect an explicit setting.
    if (-not $env:AMIC_TELEMETRY__SHADOW_BASELINE) { $env:AMIC_TELEMETRY__SHADOW_BASELINE = 'true' }
    $log = Start-Tracked 'api' @('-m', 'uvicorn', 'amic.service.api:app', '--host', '127.0.0.1', '--port', "$ApiPort")
    Wait-Until { Test-Url "$apiUrl/health" } 180 'API' $log
    Write-Ok 'API healthy'
}

# 4. Dashboard ----------------------------------------------------------------------------
$uiUrl = "http://localhost:$UiPort"
Write-Step "Dashboard ($uiUrl)"
if (Test-Url "$uiUrl/_stcore/health") {
    Write-Skip 'already running'
} else {
    $log = Start-Tracked 'dashboard' @('-m', 'streamlit', 'run', 'ui/dashboard.py',
        '--server.port', "$UiPort", '--server.headless', 'true', '--browser.gatherUsageStats', 'false')
    Wait-Until { Test-Url "$uiUrl/_stcore/health" } 90 'Dashboard' $log
    Write-Ok 'Dashboard up'
}

Write-Host ''
Write-Host 'AMIC is up.' -ForegroundColor Green
Write-Host "  API        $apiUrl   (docs: $apiUrl/docs)"
Write-Host "  Dashboard  $uiUrl"
Write-Host "  Logs       $RunDir"
Write-Host '  Stop       start-amic.cmd -Stop'
if (-not $NoBrowser) { Start-Process $uiUrl }
