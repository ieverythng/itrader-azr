param(
    [int]$Port = 8080,
    [string]$Preset = "q3"
)

$ErrorActionPreference = "SilentlyContinue"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateFile = Join-Path $repoRoot "_tmp\llama-server\llama-server-$Preset-$Port.json"

$pids = New-Object System.Collections.Generic.HashSet[int]

if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        if ($state.pid) {
            [void]$pids.Add([int]$state.pid)
        }
    } catch {
    }
}

try {
    $owner = Get-NetTCPConnection -LocalPort $Port | Where-Object { $_.State -eq "Listen" } | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in @($owner)) {
        if ($procId) {
            [void]$pids.Add([int]$procId)
        }
    }
} catch {
}

foreach ($procId in $pids) {
    try {
        Stop-Process -Id $procId -Force
        Write-Host ("Stopped PID {0}" -f $procId)
    } catch {
    }
}

if (Test-Path -LiteralPath $stateFile) {
    Remove-Item -LiteralPath $stateFile -Force
}

Write-Host ("Stopped server on port {0}" -f $Port)
