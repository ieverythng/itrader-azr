param(
    [string]$BaseUrl = "http://127.0.0.1:8080/v1",
    [ValidateSet("q3", "q4")]
    [string]$Preset = "q3",
    [ValidateSet("qwen35", "qwen36")]
    [string]$ModelSeries = "qwen35",
    [string]$Model = "",
    [string]$SystemPrompt = "You are a concise helpful assistant.",
    [string]$Prompt = "Reply with exactly CUDA_OK and nothing else.",
    [double]$Temperature = 0.7,
    [double]$TopP = 0.8,
    [int]$TopK = 20,
    [double]$MinP = 0.0,
    [double]$PresencePenalty = 1.5,
    [double]$RepeatPenalty = 1.0,
    [int]$MaxTokens = 128,
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

if (-not $Model) {
    switch ("{0}/{1}" -f $ModelSeries, $Preset) {
        "qwen35/q3" { $Model = "qwen35-27b-q3km" }
        "qwen35/q4" { $Model = "qwen35-27b-q4km" }
        "qwen36/q3" { $Model = "qwen36-27b-q3km" }
        "qwen36/q4" { $Model = "qwen36-27b-q4km" }
    }
}

$headers = @{
    "Content-Type" = "application/json"
}

if ($ApiKey) {
    $headers["Authorization"] = "Bearer $ApiKey"
}

$body = @{
    model = $Model
    messages = @(
        @{
            role = "system"
            content = $SystemPrompt
        },
        @{
            role = "user"
            content = $Prompt
        }
    )
    temperature = $Temperature
    top_p = $TopP
    top_k = $TopK
    min_p = $MinP
    presence_penalty = $PresencePenalty
    repeat_penalty = $RepeatPenalty
    max_tokens = $MaxTokens
} | ConvertTo-Json -Depth 8

$resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -Headers $headers -Body $body -TimeoutSec 300
$resp | ConvertTo-Json -Depth 10
