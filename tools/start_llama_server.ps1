param(
    [ValidateSet("q3", "q4")]
    [string]$Preset = "q3",

    [ValidateSet("qwen35", "qwen36")]
    [string]$ModelSeries = "qwen35",

    [string]$ModelPath = "",
    [string]$BinDir = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8080,
   [int]$ContextSize = 8192,
    [string]$GpuLayers = "999",

    [ValidateSet("on", "off")]
    [string]$Fit = "on",

    [ValidateSet("on", "off", "auto")]
    [string]$FlashAttention = "on",

    [ValidateSet("true", "false")]
    [string]$EnableThinking = "false",

    [ValidateSet("auto", "on", "off")]
    [string]$Reasoning = "",

    [ValidateSet("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")]
    [string]$CacheTypeK = "q8_0",

    [ValidateSet("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")]
    [string]$CacheTypeV = "q8_0",

    [double]$Temperature = 0.7,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$MinP = 0.0,
    [double]$PresencePenalty = 1.5,
    [double]$RepeatPenalty = 1.0,

     [int]$Parallel = 1,
    [int]$BatchSize = 512,
    [int]$UBatchSize = 256,
    [switch]$ContinuousBatching,
    [string]$Alias = "",
    [string]$ApiKey = "",

    [switch]$Metrics,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $BinDir) {
    if ($env:LLAMA_CPP_BIN_DIR) {
        $BinDir = $env:LLAMA_CPP_BIN_DIR
    }
    else {
        # Share the verified CUDA 13.3 runtime with Watson/Hermes instead of
        # depending on a private, untracked bundle inside this repository.
        $BinDir = "C:\Users\Admin\PROJECTS\llama-b10621-win-cuda133"
    }
}

$BinDir = (Resolve-Path $BinDir).Path
$serverExe = Join-Path $BinDir "llama-server.exe"

if (-not (Test-Path -LiteralPath $serverExe)) {
    throw "llama-server.exe not found at '$serverExe'"
}

if ($BinDir -match '(?i)cuda' -and (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    $probeOut = [IO.Path]::GetTempFileName()
    $probeErr = [IO.Path]::GetTempFileName()
    try {
        $probe = Start-Process -FilePath (Get-Command nvidia-smi.exe).Source `
            -ArgumentList '--query-gpu=name,driver_version', '--format=csv,noheader,nounits' `
            -WindowStyle Hidden -RedirectStandardOutput $probeOut -RedirectStandardError $probeErr -PassThru
        if (-not $probe.WaitForExit(5000)) {
            try { $probe.Kill() } catch { }
            throw 'NVIDIA driver health probe failed or timed out. Reset the GPU/reboot before launching llama.cpp.'
        }
        $probe.Refresh()
        $exitCode = $probe.ExitCode
        $probeOutput = Get-Content -LiteralPath $probeOut -Raw -ErrorAction SilentlyContinue
        if (($null -ne $exitCode -and $exitCode -ne 0) -or [string]::IsNullOrWhiteSpace($probeOutput)) {
            throw 'NVIDIA driver health probe failed or returned no GPU data. Reset the GPU/reboot before launching llama.cpp.'
        }
    }
    finally {
        Remove-Item -LiteralPath $probeOut,$probeErr -Force -ErrorAction SilentlyContinue
    }
}

function Get-FirstDefinedEnvValue([string[]]$Names) {
    foreach ($name in $Names) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) {
            return $value
        }
    }

    return $null
}

$modelCatalog = @{
    qwen35 = @{
        q3 = @{
            env_vars = @("LLAMA_CPP_QWEN35_MODEL_Q3", "LLAMA_CPP_MODEL_Q3")
            default_path = "D:\MODELS\Qwen3.5-27B-Q3_K_M.gguf"
            alias = "qwen35-27b-q3km"
        }
        q4 = @{
            env_vars = @("LLAMA_CPP_QWEN35_MODEL_Q4", "LLAMA_CPP_MODEL_Q4")
            default_path = "D:\MODELS\Qwen3.5-27B.Q4_K_M.gguf"
            alias = "qwen35-27b-q4km"
        }
    }
    qwen36 = @{
        q3 = @{
            env_vars = @("LLAMA_CPP_QWEN36_MODEL_Q3")
            default_path = "D:\MODELS\Qwen3.6-27B-Q3_K_M.gguf"
            alias = "qwen36-27b-q3km"
        }
        q4 = @{
            env_vars = @("LLAMA_CPP_QWEN36_MODEL_Q4")
            default_path = "D:\MODELS\Qwen3.6-27B.Q4_K_M.gguf"
            alias = "qwen36-27b-q4km"
        }
    }
}

$modelConfig = $modelCatalog[$ModelSeries][$Preset]

if (-not $ModelPath) {
    $ModelPath = Get-FirstDefinedEnvValue -Names $modelConfig.env_vars
    if (-not $ModelPath) {
        $ModelPath = $modelConfig.default_path
    }

    if (-not $Alias) {
        $Alias = $modelConfig.alias
    }
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model file not found at '$ModelPath'"
}

if (-not $Alias) {
    $Alias = [IO.Path]::GetFileNameWithoutExtension($ModelPath)
}

if (-not $Reasoning) {
    if ($EnableThinking -eq "true") {
        $Reasoning = "on"
    }
    else {
        $Reasoning = "off"
    }
}

$args = @(
    "-m", $ModelPath,
    "--alias", $Alias,
    "-ngl", $GpuLayers,
    "--fit", $Fit,
    "-c", $ContextSize,
    "-ctk", $CacheTypeK,
    "-ctv", $CacheTypeV,
    "-fa", $FlashAttention,
    "--reasoning", $Reasoning,
    "--temp", $Temperature,
    "--top-p", $TopP,
    "--top-k", $TopK,
    "--min-p", $MinP,
    "--presence-penalty", $PresencePenalty,
    "--repeat-penalty", $RepeatPenalty,
    "--host", $BindHost,
    "--port", $Port,
     "-np", $Parallel,
    "-b", $BatchSize,
    "-ub", $UBatchSize,
    "--slots",
    "--webui"
)

if ($Metrics) {
    $args += "--metrics"
}

if ($ContinuousBatching) {
    $args += "--cont-batching"
}

if ($ApiKey) {
    $args += @("--api-key", $ApiKey)
}

if ($Verbose) {
    $args += "--verbose"
}

$quoted = @($serverExe) + $args | ForEach-Object {
    if ($_ -match "\s") { '"{0}"' -f $_ } else { $_ }
}

Write-Host "Repo root : $repoRoot"
Write-Host "Binary dir: $BinDir"
Write-Host "Series    : $ModelSeries"
Write-Host "Preset    : $Preset"
Write-Host "Model     : $ModelPath"
Write-Host "Alias     : $Alias"
Write-Host "URL       : http://$BindHost`:$Port"
Write-Host "Thinking  : $EnableThinking"
Write-Host "Reasoning : $Reasoning"
Write-Host "Sampling  : temp=$Temperature top_p=$TopP top_k=$TopK min_p=$MinP presence_penalty=$PresencePenalty repeat_penalty=$RepeatPenalty"
Write-Host "Template  : model default template"
Write-Host ""
Write-Host ($quoted -join " ")
Write-Host ""
Write-Host "Open the browser UI at http://$BindHost`:$Port"
Write-Host "Health endpoint: http://$BindHost`:$Port/health"
Write-Host "Slots endpoint : http://$BindHost`:$Port/slots"
if ($Metrics) {
    Write-Host "Metrics endpoint: http://$BindHost`:$Port/metrics"
}
Write-Host ""

Push-Location $BinDir
try {
    & $serverExe @args
}
finally {
    Pop-Location
}
