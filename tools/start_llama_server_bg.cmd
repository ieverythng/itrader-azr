@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if not defined LLAMA_CPP_BIN_DIR set "LLAMA_CPP_BIN_DIR=C:\Users\Admin\PROJECTS\llama-b10621-win-cuda133"
if not defined LLAMA_CPP_QWEN35_MODEL_Q3 set "LLAMA_CPP_QWEN35_MODEL_Q3=D:\MODELS\Qwen3.5-27B-Q3_K_M.gguf"
if not defined LLAMA_CPP_QWEN35_MODEL_Q4 set "LLAMA_CPP_QWEN35_MODEL_Q4=D:\MODELS\Qwen3.5-27B.Q4_K_M.gguf"
if not defined LLAMA_CPP_QWEN36_MODEL_Q3 set "LLAMA_CPP_QWEN36_MODEL_Q3=D:\MODELS\Qwen3.6-27B-Q3_K_M.gguf"
if not defined LLAMA_CPP_QWEN36_MODEL_Q4 set "LLAMA_CPP_QWEN36_MODEL_Q4=D:\MODELS\Qwen3.6-27B.Q4_K_M.gguf"

if not defined LLAMA_CPP_MODEL_Q3 set "LLAMA_CPP_MODEL_Q3=%LLAMA_CPP_QWEN35_MODEL_Q3%"
if not defined LLAMA_CPP_MODEL_Q4 set "LLAMA_CPP_MODEL_Q4=%LLAMA_CPP_QWEN35_MODEL_Q4%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\tools\start_llama_server_bg.ps1" %*
