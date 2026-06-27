@echo off
REM Double-clickable launcher for the full Kiln stack.
REM Bypasses execution policy for this one script only. Pass-through args:
REM   start-kiln.cmd -Browser      (use Vite/browser UI instead of Electron)
REM   start-kiln.cmd -NoUi         (backend only)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-kiln.ps1" %*
