<#
.SYNOPSIS
    Create or update AMIC's Python environment with conda (Miniconda) or a plain venv.

.DESCRIPTION
    setup-amic.cmd conda              conda env "amic" from environment.yml
    setup-amic.cmd venv               .venv in the repo
    Options: -Name <env>   conda env name (default amic)
             -Cpu          skip the CUDA torch build even on an NVIDIA machine
             -CudaIndex    PyTorch wheel index (default cu134; needs a driver for CUDA 13.4+)
             -NoDefault    do not record the choice as AMIC_ENV in .env

    Safe to re-run: an existing environment is updated in place. On an NVIDIA machine, CUDA
    torch is installed before the project, so pip never fetches PyPI's CPU-only Windows wheel.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('conda', 'venv')][string]$Manager,
    [string]$Name = 'amic',
    [switch]$Cpu,
    [string]$CudaIndex = 'cu134',
    [switch]$NoDefault
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot 'amic_env.ps1')

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

function Invoke-Native([string]$exe, [string[]]$argv) {
    # Native tools print warnings on stderr; judge them by exit code only.
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $exe @argv
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) { throw "Failed ($code): $exe $($argv -join ' ')" }
}

function Set-DotEnvKey([string]$Key, [string]$Value) {
    $f = Join-Path $Root '.env'
    if (-not (Test-Path $f)) { Copy-Item (Join-Path $Root '.env.example') $f }
    $lines = @(Get-Content $f)
    $found = $false
    $out = foreach ($l in $lines) {
        if ($l -match "^\s*#?\s*$Key\s*=") { $found = $true; "$Key=$Value" } else { $l }
    }
    if (-not $found) { $out = @($out) + "$Key=$Value" }
    Set-Content -Path $f -Value $out -Encoding ascii
}

$gpu = $false
if (-not $Cpu) {
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        & cmd.exe /c 'nvidia-smi >nul 2>&1'
        $gpu = ($LASTEXITCODE -eq 0)
    }
}

# 1. The interpreter -------------------------------------------------------------------------
if ($Manager -eq 'conda') {
    $conda = Get-CondaExe
    if (-not $conda) { throw 'conda not found. Install Miniconda, or set CONDA_EXE.' }
    $prefix = Get-CondaEnvPrefix $Name
    if (-not $prefix) {
        $pyver = '3.13'
        $m = Select-String -Path (Join-Path $Root 'environment.yml') -Pattern '^\s*-\s*python=(\S+)'
        if ($m) { $pyver = $m.Matches[0].Groups[1].Value }
        Write-Step "Creating conda env '$Name' (python $pyver, conda-forge)"
        Invoke-Native $conda @('create', '-y', '-n', $Name, '--override-channels', '-c', 'conda-forge',
            "python=$pyver", 'pip')
        $prefix = Get-CondaEnvPrefix $Name
    } else {
        Write-Step "Updating conda env '$Name' ($prefix)"
    }
    $py = Join-Path $prefix 'python.exe'
    Enable-CondaPrefix $prefix
} else {
    $py = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        $base = Get-AmicSetting $Root 'AMIC_BASE_PYTHON' 'python'
        Write-Step "Creating .venv with $base"
        Invoke-Native $base @('-m', 'venv', (Join-Path $Root '.venv'))
    } else {
        Write-Step 'Updating .venv'
    }
}
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]'3.11' -or [version]$ver -ge [version]'3.14') {
    throw "Python $ver is outside the project's range (>=3.11,<3.14)."
}
Invoke-Native $py @('-m', 'pip', 'install', '--upgrade', 'pip', '--quiet')

# 2. Torch -----------------------------------------------------------------------------------
if ($gpu) {
    Write-Step "Installing CUDA torch ($CudaIndex)"
    Invoke-Native $py @('-m', 'pip', 'install', 'torch', '--index-url', "https://download.pytorch.org/whl/$CudaIndex")
} else {
    Write-Step 'No NVIDIA GPU (or -Cpu): torch comes from PyPI with the project'
}

# 3. The project -----------------------------------------------------------------------------
Write-Step 'Installing amic with all extras'
if ($Manager -eq 'conda') {
    # environment.yml is the source of truth for the conda side; its pip section installs -e .[all]
    Invoke-Native $conda @('env', 'update', '-n', $Name, '-f', (Join-Path $Root 'environment.yml'))
} else {
    Invoke-Native $py @('-m', 'pip', 'install', '-e', '.[all]')
}

# 4. Check and record ------------------------------------------------------------------------
Write-Step 'Checking'
Invoke-Native $py @('-c', "import amic, torch; print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())")
Invoke-Native $py @('-m', 'pip', 'check')

if (-not $NoDefault) {
    Set-DotEnvKey 'AMIC_ENV' $Manager
    if ($Manager -eq 'conda' -and $Name -ne 'amic') { Set-DotEnvKey 'AMIC_CONDA_ENV' $Name }
    Write-Host "    .env: AMIC_ENV=$Manager (start-amic.cmd and chat-amic.cmd use it)" -ForegroundColor Green
}
Write-Host ''
if ($Manager -eq 'conda') {
    Write-Host "Done. In a shell: conda activate $Name" -ForegroundColor Green
} else {
    Write-Host 'Done. In a shell: .venv\Scripts\activate' -ForegroundColor Green
}
