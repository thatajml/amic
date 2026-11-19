@echo off
rem Chat with the AMIC tutor in this terminal. Each answer is followed by AMIC's decision panel
rem (route, why, memory used, latency, tokens). Type quit to exit.
rem   chat-amic.cmd                     chat as seeded learner edu-001 (new turns are saved to memory)
rem   chat-amic.cmd --learner me        chat as a new learner
rem   chat-amic.cmd --arm always        compare with the always-retrieve baseline
rem   chat-amic.cmd --frozen            do not save new turns
rem   chat-amic.cmd --script            3 canned turns, no typing
rem Uses the environment chosen by AMIC_ENV (venv / conda / auto), see scripts\amic_env.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\chat_amic.ps1" %*
exit /b %ERRORLEVEL%
