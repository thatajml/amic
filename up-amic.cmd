@echo off
rem Start the whole AMIC stack in Docker and open the chat UI, dashboard and Educator Console.
rem   up-amic.cmd             start, then open the three pages
rem   up-amic.cmd -Build      rebuild the image first (after changing code)
rem   up-amic.cmd -NoBrowser  start only
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\up_amic.ps1" %*
