param(
    [ValidateSet("q3", "q4")]
    [string]$Preset = "q3",

    [ValidateSet("qwen35", "qwen36")]
    [string]$ModelSeries = "qwen35",

    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8080,
  [int]$ContextSize = 8192,
    [string]$GpuLayers = "999",
    [string]$Alias = "",
    [string]$ModelPath = "",
    [int]$Parallel = 1,
    [switch]$Metrics
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateDir = Join-Path $repoRoot "_tmp\llama-server"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$stateFile = Join-Path $stateDir ("llama-server-{0}-{1}.json" -f $Preset, $Port)
$stdout = Join-Path $stateDir ("llama-server-{0}-{1}.stdout.log" -f $Preset, $Port)
$stderr = Join-Path $stateDir ("llama-server-{0}-{1}.stderr.log" -f $Preset, $Port)

function Test-Health([int]$HealthPort) {
    try {
        $resp = Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/health" -f $HealthPort) -Method Get -TimeoutSec 3
        return $null -ne $resp
    } catch {
        return $false
    }
}

function Get-Models([int]$HealthPort) {
    try {
        return Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/v1/models" -f $HealthPort) -Method Get -TimeoutSec 5
    } catch {
        return $null
    }
}

function Get-ListenerProcessIds([int]$ListenPort) {
    return @(
        Get-NetTCPConnection -State Listen -LocalPort $ListenPort -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -and $_ -ne 0 }
    )
}

function Stop-ProcessTree([int]$ProcessId) {
    if (-not $ProcessId) { return }
    try { & taskkill.exe /PID $ProcessId /T /F *> $null } catch { }
    Start-Sleep -Milliseconds 500
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Get-ExpectedAlias(
    [string]$RequestedSeries,
    [string]$RequestedPreset,
    [string]$RequestedAlias,
    [string]$RequestedModelPath
) {
    if ($RequestedAlias) {
        return $RequestedAlias
    }

    if ($RequestedModelPath) {
        return [IO.Path]::GetFileNameWithoutExtension($RequestedModelPath)
    }

    switch ("{0}/{1}" -f $RequestedSeries, $RequestedPreset) {
        "qwen35/q3" { return "qwen35-27b-q3km" }
        "qwen35/q4" { return "qwen35-27b-q4km" }
        "qwen36/q3" { return "qwen36-27b-q3km" }
        "qwen36/q4" { return "qwen36-27b-q4km" }
        default { return $null }
    }
}

$requestedAlias = Get-ExpectedAlias `
    -RequestedSeries $ModelSeries `
    -RequestedPreset $Preset `
    -RequestedAlias $Alias `
    -RequestedModelPath $ModelPath

if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content $stateFile -Raw | ConvertFrom-Json
        if ($state.port -eq $Port -and (Test-Health -HealthPort $Port)) {
            $models = Get-Models -HealthPort $Port
            $currentModelIds = @()
            if ($models -and $models.data) {
                $currentModelIds = @($models.data | ForEach-Object { $_.id })
            }

            $currentSummary = $null
            if ($currentModelIds.Count -gt 0) {
                $currentSummary = $currentModelIds -join ", "
            }
            elseif ($state.alias) {
                $currentSummary = $state.alias
            }

            if ($requestedAlias -and $currentSummary -and ($currentModelIds.Count -eq 0 -or ($currentModelIds -notcontains $requestedAlias))) {
                Write-Host ("Port {0} is already serving '{1}', not requested '{2}'." -f $Port, $currentSummary, $requestedAlias)
                Write-Host "Stop the existing server or choose a different port."
                exit 1
            }

            Write-Host ("Already running on http://127.0.0.1:{0}" -f $Port)
            if ($currentSummary) {
                Write-Host ("Model(s): {0}" -f $currentSummary)
            }
            exit 0
        }
    } catch {
    }
}

if (Test-Health -HealthPort $Port) {
    Write-Host ("Already running on http://127.0.0.1:{0}" -f $Port)
    exit 0
}

$staleOwners = Get-ListenerProcessIds -ListenPort $Port
foreach ($owner in $staleOwners) {
    Write-Warning ("Port {0} has an unhealthy listener (pid {1}); stopping it before launch." -f $Port, $owner)
    Stop-ProcessTree -ProcessId ([int]$owner)
}
for ($wait = 0; $wait -lt 20; $wait++) {
    if (-not (Get-ListenerProcessIds -ListenPort $Port)) { break }
    Start-Sleep -Milliseconds 250
}
if (Get-ListenerProcessIds -ListenPort $Port) {
    throw "Port $Port is occupied by an unhealthy listener that could not be stopped."
}

if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force }
if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }

$launcher = Join-Path $repoRoot "tools\start_llama_server.ps1"
$argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-Preset", $Preset,
    "-ModelSeries", $ModelSeries,
    "-BindHost", $BindHost,
    "-Port", $Port,
    "-ContextSize", $ContextSize,
    "-GpuLayers", $GpuLayers,
    "-Parallel", $Parallel
)

if ($Alias) {
    $argList += @("-Alias", $Alias)
}

if ($ModelPath) {
    $argList += @("-ModelPath", $ModelPath)
}

if ($Metrics) {
    $argList += "-Metrics"
}

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $argList `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$state = [ordered]@{
    preset = $Preset
    model_series = $ModelSeries
    bind_host = $BindHost
    port = $Port
    pid = $proc.Id
    model_path = $ModelPath
    alias = $Alias
    started_at = (Get-Date).ToString("o")
    stdout_log = $stdout
    stderr_log = $stderr
}
$state | ConvertTo-Json | Set-Content -LiteralPath $stateFile

$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        break
    }
    if (Test-Health -HealthPort $Port) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Write-Host "Server did not become healthy in time."
    Write-Host ("stderr: {0}" -f $stderr)
    if (Test-Path -LiteralPath $stderr) {
        Get-Content -LiteralPath $stderr -Tail 80
    }
    Stop-ProcessTree -ProcessId $proc.Id
    exit 1
}

$models = Get-Models -HealthPort $Port
$modelIds = @()
if ($models -and $models.data) {
    $modelIds = @($models.data | ForEach-Object { $_.id })
}

Write-Host ("Started llama-server on http://127.0.0.1:{0}" -f $Port)
if ($modelIds.Count -gt 0) {
    Write-Host ("Model(s): {0}" -f ($modelIds -join ", "))
}
Write-Host ("PID: {0}" -f $proc.Id)
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
