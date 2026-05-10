@echo off
setlocal
cmd /c scripts\run_sim.bat %*
exit /b %errorlevel%