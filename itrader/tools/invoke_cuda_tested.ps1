<#
.SYNOPSIS
Run Python arguments through the explicitly selected CUDA test profile.

.DESCRIPTION
This compatibility wrapper does not provision or synchronize environments.
The repository Python runner owns profile path, reparse-point, overlap, and
interpreter-identity validation. Requires PowerShell 7 or newer so native
argument boundaries, including empty and quoted arguments, are preserved.
Windows PowerShell 5.1 is rejected before the wrapper changes caller state; use
a Python caller that constructs a list argv (for example the existing
run_ablation.py) instead.
#>
param(
    [Parameter(Mandatory = $true)]
    [string] $RepoRoot,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArguments
)

$scriptWasDotSourced = $MyInvocation.InvocationName -eq '.'
$exitCode = 1
$failureMessage = $null
$process = $null

if ($PSVersionTable.PSVersion.Major -lt 7) {
    [Console]::Error.WriteLine('PowerShell 7 or newer is required for exact Python argv forwarding. Use a Python caller that constructs a list argv (for example run_ablation.py) on Windows PowerShell 5.1.')
    if ($scriptWasDotSourced) {
        return 1
    }
    exit 1
}

try {
    $resolvedRoot = [IO.Path]::GetFullPath($RepoRoot)
    if (-not (Test-Path -LiteralPath $resolvedRoot -PathType Container)) {
        throw "Repository root does not exist: $resolvedRoot"
    }
    $repoRunner = Join-Path $resolvedRoot 'itrader\tools\run_repo_python.py'
    $bootstrapPython = Join-Path $resolvedRoot '.venv\Scripts\python.exe'
    foreach ($requiredFile in @($bootstrapPython, $repoRunner)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Required repository launcher file is missing: $requiredFile"
        }
    }
    if ($null -eq $PythonArguments -or $PythonArguments.Count -eq 0) {
        throw 'Pass a script, module, or Python option to the repository launcher.'
    }
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $bootstrapPython
    $startInfo.WorkingDirectory = $resolvedRoot
    $startInfo.UseShellExecute = $false
    $startInfo.ArgumentList.Add($repoRunner)
    foreach ($argument in $PythonArguments) {
        $startInfo.ArgumentList.Add($argument)
    }
    $startInfo.Environment['ITRADER_PYTHON_PROFILE'] = 'cuda-tested'

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'Could not start the selected Python process.'
    }
    $process.WaitForExit()
    $exitCode = [int] $process.ExitCode
}
catch {
    $failureMessage = $_.Exception.Message
}
finally {
    if ($null -ne $process) {
        $process.Dispose()
    }
}

if ($null -ne $failureMessage) {
    [Console]::Error.WriteLine($failureMessage)
    $exitCode = 1
}
if ($scriptWasDotSourced) {
    return $exitCode
}
exit $exitCode
