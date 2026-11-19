@echo off
rem Create or update AMIC's Python environment: setup-amic.cmd conda  or  setup-amic.cmd venv
rem Options: -Name <env>  -Cpu  -CudaIndex cu134  -NoDefault   (see scripts\setup_env.ps1)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_env.ps1" %*
exit /b %ERRORLEVEL%
