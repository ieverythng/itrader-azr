# Windows llama.cpp for Qwen3.5-27B

This workstation now has a working Windows CUDA `llama.cpp` path for Qwen3.5-27B GGUF models.

## Current Paths

- Binaries: `_tools/llama-bin-win-cuda13-b8604` by default, or `%LLAMA_CPP_BIN_DIR%`
- Models: `D:\MODELS` by default, or `%LLAMA_CPP_MODEL_Q3%` / `%LLAMA_CPP_MODEL_Q4%`

Current model files:

- `D:\MODELS\Qwen3.5-27B-Q3_K_M.gguf`
- `D:\MODELS\Qwen3.5-27B.Q4_K_M.gguf`

## Recommended Starting Point

Use `Q3_K_M` first for day-to-day testing on the RTX 5070 Ti 16 GB.

Why:

- it loaded cleanly with good headroom
- short test generation was much faster than `Q4_K_M`
- `Q4_K_M` fit, but sat on the VRAM limit and was much less responsive

Recommended baseline:

- preset: `q3`
- context: `16384`
- KV cache: `q8_0`
- reasoning: `off`
- thinking: `false` via `--reasoning off`
- flash attention: `on`
- fit: `on`
- temperature: `0.7`
- top_p: `0.8`
- top_k: `20`
- min_p: `0.0`
- presence_penalty: `1.5`
- repeat_penalty: `1.0`

Treat `Q4_K_M` as an experimental preset on this GPU, not the default.

## Verified Contexts

The Q3 launcher path was verified on this workstation at:

- `32768`
- `65536`
- `131072`

Operational notes:

- `32768` is the best speed / reliability baseline for Watson.
- `65536` is a good middle ground if you need more history.
- `131072` loads and serves requests, but it offloads fewer layers to GPU and is slower on large prompts.

## Start the Server

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q3 -Metrics
```

From `cmd.exe` or Conda Prompt:

```bat
cd /d C:\Users\Admin\PROJECTS\iTRADER\itrader-azr
tools\start_llama_server.cmd -Preset q3 -BindHost 0.0.0.0 -Port 8080 -ContextSize 131072 -Parallel 1
```

This foreground `cmd` wrapper is the simplest operational path: keep that terminal open while Watson / Hermes uses the server.

If you explicitly want a detached background launch from `cmd.exe`:

```bat
cd /d C:\Users\Admin\PROJECTS\iTRADER\itrader-azr
tools\start_llama_server_bg.cmd -Preset q3 -BindHost 0.0.0.0 -Port 8080 -ContextSize 131072 -Parallel 1
```

For the Hermes / Watson path, `32768` is the practical default:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q3 -BindHost 0.0.0.0 -ContextSize 32768 -Parallel 1 -Metrics
```

For larger Watson context windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q3 -BindHost 0.0.0.0 -ContextSize 65536 -Parallel 1 -Metrics
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q3 -BindHost 0.0.0.0 -ContextSize 131072 -Parallel 1 -Metrics
```

For the Q4 model:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q4 -Metrics
```

For thinking mode:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\start_llama_server.ps1 -Preset q3 -EnableThinking true -Temperature 1.0 -TopP 0.95
```

## Browser UI and Endpoints

`llama-server` is not CLI-only.

Useful URLs after startup:

- Web UI: `http://127.0.0.1:8080`
- Health: `http://127.0.0.1:8080/health`
- Slots: `http://127.0.0.1:8080/slots`
- Metrics: `http://127.0.0.1:8080/metrics`
- OpenAI-compatible API: `http://127.0.0.1:8080/v1`
- Slot status: `http://127.0.0.1:8080/slots`

The Web UI is the easiest way to experiment interactively.

## Watson Commands

From WSL:

```bash
watson-llama-up -ContextSize 32768
watson-llama-status
watson-llama-test
```

For `128k`:

```bash
watson-llama-down
watson-llama-up -ContextSize 131072
```

## Quick API Test

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\invoke_llama_chat.ps1 -Model qwen35-27b-q3km -Prompt "Summarize the trend in one sentence."
```

From `cmd.exe` or Conda Prompt:

```bat
cd /d C:\Users\Admin\PROJECTS\iTRADER\itrader-azr
tools\invoke_llama_chat.cmd -BaseUrl http://127.0.0.1:8080/v1 -Model qwen35-27b-q3km -Prompt Reply_CMD_OK -Temperature 0 -MaxTokens 16
```

## Parameters You Will Actually Care About

- `-Preset q3|q4`
  - Choose the loaded model quickly.

- `-ContextSize 4096`
  - Main context setting.
  - `16384` is a safe local test default on this machine for Q3.
  - `32768` is the practical Watson target.
  - `65536` and `131072` were both verified to load on this workstation.

- `-CacheTypeK q8_0 -CacheTypeV q8_0`
  - Keeps KV memory under control.
  - This is important for 27B on 16 GB VRAM.

- `-EnableThinking false`
  - Matches Unsloth's recommendation for non-thinking / instruct mode.
  - In this launcher, it maps to `-Reasoning off` by default.

- `-Reasoning off`
  - Explicitly disables template-driven reasoning on recent `llama.cpp` builds.
  - This is the reliable path for Watson.

- `-FlashAttention on`
  - Keep enabled unless a model-specific issue appears.

- `-Fit on`
  - Lets `llama.cpp` auto-adjust unset memory-sensitive settings to fit the GPU.

- `-Temperature`, `-TopP`, `-TopK`, `-MinP`, `-PresencePenalty`, `-RepeatPenalty`
  - These are exposed directly now.
  - Unsloth's non-thinking default for general tasks is `temp=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repeat_penalty=1.0`.
  - For thinking mode, Unsloth suggests `temp=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`.

- `-Parallel 4`
  - Number of server slots.
  - Lower it if you want to minimize background memory pressure.

## itrader Integration

`itrader` already has an OpenAI-compatible proposer backend, so the clean path is:

```powershell
python -m itrader.scripts.gen_task_pool --backend vllm --api_base_url http://127.0.0.1:8080/v1 --api_model qwen35-27b-q3km --batch_size 4 --n_tasks 32 --max_backend_errors 5 --out data/task_pool_qwen35_q3.jsonl
```

The backend name is still `vllm` or `sglang` in the CLI, but the implementation is just "OpenAI-compatible endpoint".

## Current Validation Notes

- `itrader` environment packages were not modified during the CUDA/toolkit work.
- Repo healthcheck passes.
- Repo tests pass in the real environment.
- Latest official Windows CUDA `llama.cpp` bundle can serve Qwen3.5-27B on this machine.

## Local-Only Artifacts

Do not commit these local paths:

- `_downloads/`
- `_tmp/`
- `_tools/`
- `_external/llama.cpp/`

Safe cleanup targets:

- `_downloads/` installer and zip payloads
- `_tmp/` logs, state files, smoke tests
- `_tools/llama-bin-win-cuda13/` old binary bundle
- `_external/llama.cpp/` optional source checkout if you are not editing / building from source

Keep these if you still want the current local launcher path to work without extra flags:

- `_tools/llama-bin-win-cuda13-b8604/`
- model files in `D:\MODELS`

If you want to move the binary bundle outside the repo, set:

```powershell
$env:LLAMA_CPP_BIN_DIR = 'E:\llama-bin-win-cuda13-b8604'
```

If you want to override the default model locations, set:

```powershell
$env:LLAMA_CPP_MODEL_Q3 = 'D:\MODELS\Qwen3.5-27B-Q3_K_M.gguf'
$env:LLAMA_CPP_MODEL_Q4 = 'D:\MODELS\Qwen3.5-27B.Q4_K_M.gguf'
```

## Next Adjustment

Launcher defaults are now aligned with the Unsloth Qwen3.5 guide for non-thinking general tasks.
