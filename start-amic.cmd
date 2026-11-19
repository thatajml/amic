@echo off
rem Wake up the whole AMIC stack. Pass -Stop to stop it, -NoBrowser to skip opening the dashboard.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_amic.ps1" %*
exit /b %ERRORLEVEL%
