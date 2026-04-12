param(
    [int]$Port = 8080,
    [string]$Preset = "q3"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateFile = Join-Path $repoRoot "_tmp\llama-server\llama-server-$Preset-$Port.json"

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

function Get-Slots([int]$HealthPort) {
    try {
        return Invoke-RestMethod -Uri ("http://127.0.0.1:{0}/slots" -f $HealthPort) -Method Get -TimeoutSec 5
    } catch {
        return $null
    }
}

$state = $null
if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    } catch {
    }
}

$healthy = Test-Health -HealthPort $Port
Write-Host ("URL: http://127.0.0.1:{0}" -f $Port)
Write-Host ("Healthy: {0}" -f $healthy)

if ($state) {
    Write-Host ("PID: {0}" -f $state.pid)
    Write-Host ("Started: {0}" -f $state.started_at)
    Write-Host ("stdout: {0}" -f $state.stdout_log)
    Write-Host ("stderr: {0}" -f $state.stderr_log)
    if (-not $healthy) {
        Write-Host "State file exists, but no healthy listener was found on this port."
    }
}

if ($healthy) {
    $models = Get-Models -HealthPort $Port
    if ($models -and $models.data) {
        Write-Host ("Model(s): {0}" -f (($models.data | ForEach-Object { $_.id }) -join ", "))
    }

    $slots = Get-Slots -HealthPort $Port
    if ($slots) {
        $slotList = @($slots)
        if ($slots.value) {
            $slotList = @($slots.value)
        }

        if ($slotList.Count -gt 0) {
            $slotSummary = $slotList | ForEach-Object {
                "id={0} n_ctx={1} busy={2}" -f $_.id, $_.n_ctx, $_.is_processing
            }
            Write-Host ("Slot(s): {0}" -f ($slotSummary -join "; "))
        }
    }
}
