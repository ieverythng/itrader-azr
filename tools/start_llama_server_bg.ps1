param(
    [ValidateSet("q3", "q4")]
    [string]$Preset = "q3",

    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8080,
    [int]$ContextSize = 16384,
    [string]$GpuLayers = "auto",
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

if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content $stateFile -Raw | ConvertFrom-Json
        if ($state.port -eq $Port -and (Test-Health -HealthPort $Port)) {
            Write-Host ("Already running on http://127.0.0.1:{0}" -f $Port)
            $models = Get-Models -HealthPort $Port
            if ($models -and $models.data) {
                Write-Host ("Model(s): {0}" -f (($models.data | ForEach-Object { $_.id }) -join ", "))
            }
            exit 0
        }
    } catch {
    }
}

if (Test-Path -LiteralPath $stdout) { Remove-Item -LiteralPath $stdout -Force }
if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }

$launcher = Join-Path $repoRoot "tools\start_llama_server.ps1"
$argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-Preset", $Preset,
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
