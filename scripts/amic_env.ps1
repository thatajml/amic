<#
    Picks the Python interpreter for AMIC's launchers. Dot-source it, then:
        $Py = Get-AmicPython $Root

    Choice (first match wins):
      AMIC_PYTHON=<path>         an explicit interpreter
      AMIC_ENV=venv              .venv\Scripts\python.exe
      AMIC_ENV=conda             the conda env AMIC_CONDA_ENV (default "amic")
      AMIC_ENV=auto (default)    .venv if it exists, else the conda env
    Each value is read from the process environment first, then from .env.
#>

function Read-AmicDotEnv([string]$Root, [string]$Key) {
    $f = Join-Path $Root '.env'
    if (-not (Test-Path $f)) { return $null }
    foreach ($line in Get-Content $f) {
        if ($line -match "^\s*$Key\s*=\s*(.*?)\s*$") { return $Matches[1].Trim('"', "'") }
    }
    return $null
}

function Get-AmicSetting([string]$Root, [string]$Key, [string]$Default) {
    $v = [Environment]::GetEnvironmentVariable($Key)
    if (-not $v) { $v = Read-AmicDotEnv $Root $Key }
    if (-not $v) { $v = $Default }
    return $v
}

function Get-CondaExe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    $candidates = @(
        "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniforge3",
        "$env:LOCALAPPDATA\miniconda3", "$env:ProgramData\miniconda3", "$env:ProgramData\anaconda3"
    )
    foreach ($base in $candidates) {
        $exe = Join-Path $base 'Scripts\conda.exe'
        if (Test-Path $exe) { return $exe }
    }
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-CondaEnvPrefix([string]$Name) {
    $conda = Get-CondaExe
    if (-not $conda) { return $null }
    $base = Split-Path -Parent (Split-Path -Parent $conda)
    if ($Name -eq 'base') { return $base }
    $prefix = Join-Path $base "envs\$Name"
    if (Test-Path (Join-Path $prefix 'python.exe')) { return $prefix }
    return $null
}

function Enable-CondaPrefix([string]$Prefix) {
    # What `conda activate` puts on PATH, so DLLs from conda packages resolve in child processes.
    $dirs = @($Prefix, "$Prefix\Library\mingw-w64\bin", "$Prefix\Library\usr\bin",
        "$Prefix\Library\bin", "$Prefix\Scripts")
    $env:PATH = ($dirs -join ';') + ';' + $env:PATH
    $env:CONDA_PREFIX = $Prefix
}

function Get-AmicPython([string]$Root) {
    $explicit = Get-AmicSetting $Root 'AMIC_PYTHON' ''
    if ($explicit) {
        if (-not (Test-Path $explicit)) { throw "AMIC_PYTHON points to a missing file: $explicit" }
        return $explicit
    }
    $mode = (Get-AmicSetting $Root 'AMIC_ENV' 'auto').ToLower()
    $envName = Get-AmicSetting $Root 'AMIC_CONDA_ENV' 'amic'
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'

    if ($mode -eq 'venv' -or ($mode -eq 'auto' -and (Test-Path $venvPy))) {
        if (-not (Test-Path $venvPy)) { throw 'AMIC_ENV=venv but there is no .venv. Run: setup-amic.cmd venv' }
        return $venvPy
    }
    if ($mode -eq 'conda' -or $mode -eq 'auto') {
        $prefix = Get-CondaEnvPrefix $envName
        if (-not $prefix) {
            $why = "No conda env '$envName'"
            if ($mode -eq 'auto') { $why += ' and no .venv' }
            throw "$why. Run: setup-amic.cmd conda (or setup-amic.cmd venv)"
        }
        Enable-CondaPrefix $prefix
        return (Join-Path $prefix 'python.exe')
    }
    throw "AMIC_ENV must be venv, conda or auto (got '$mode')."
}
