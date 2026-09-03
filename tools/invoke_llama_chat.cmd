@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\tools\invoke_llama_chat.ps1" %*
