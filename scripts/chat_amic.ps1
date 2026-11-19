# Terminal chat with the AMIC tutor (python -m amic.demo) in the configured environment.
# Started by chat-amic.cmd; all arguments are passed to amic.demo.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot 'amic_env.ps1')
$Py = Get-AmicPython $Root

function Test-Ollama {
    try {
        # 127.0.0.1, not localhost: PowerShell 5.1 tries IPv6 ::1 first, which Ollama does not
        # listen on, and the fallback can outlast the timeout.
        Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/version' -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch { return $false }
}

if (-not (Test-Ollama)) {
    Write-Host 'Starting Ollama...'
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Ollama)) {
        if ((Get-Date) -gt $deadline) { throw 'Ollama did not start. Run "ollama serve" by hand to see why.' }
        Start-Sleep -Seconds 1
    }
}

$ErrorActionPreference = 'Continue'  # model-loading warnings on stderr are not failures
& $Py -m amic.demo @args
exit $LASTEXITCODE
