<#
.SYNOPSIS
    Start the AMIC Docker stack and open the chat UI, dashboard and Educator Console together.

.DESCRIPTION
    docker compose cannot open a browser, so this wraps it:
      1. checks Docker is running,
      2. docker compose up -d --wait   (add -Build to rebuild the image),
      3. warns if Ollama is not reachable on the host (the tutor model lives there),
      4. opens http://localhost:8503 (chat), :8501 (dashboard) and :8502 (Educator Console).
    Stop everything with: docker compose down
#>
[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Docker is not running. Start Docker Desktop and run this again.' -ForegroundColor Red
    exit 1
}

$composeArgs = @('compose', 'up', '-d', '--wait')
if ($Build) { $composeArgs += '--build' }
Write-Host "docker $($composeArgs -join ' ')"
& docker @composeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host 'docker compose failed; see the output above (docker compose logs).' -ForegroundColor Red
    exit $LASTEXITCODE
}

try {
    Invoke-WebRequest -Uri 'http://localhost:11434' -UseBasicParsing -TimeoutSec 3 | Out-Null
} catch {
    Write-Host 'Warning: Ollama is not reachable on localhost:11434. Start it (ollama serve) or the tutor cannot answer.' -ForegroundColor Yellow
}

$pages = [ordered]@{ 'Chat UI' = 'http://localhost:8503'; 'Dashboard' = 'http://localhost:8501'; 'Educator Console' = 'http://localhost:8502' }
foreach ($name in $pages.Keys) {
    Write-Host ("{0,-17} {1}" -f $name, $pages[$name])
    if (-not $NoBrowser) { Start-Process $pages[$name] }
}
