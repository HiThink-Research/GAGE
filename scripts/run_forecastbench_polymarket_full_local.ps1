param(
    [string]$RepoRoot = "E:\Developing\GAGE",
    [string]$DatasetRoot = "E:\Developing\GAGE\.gage_cache\forecastbench-datasets",
    [string]$ApiBase = "",
    [string]$ApiKey = "",
    [string]$Model = "mimo-v2.5",
    [string]$ConfigPath = "config/custom/forecastbench/polymarket_static_full.yaml",
    [string]$OutputRoot = "E:\fb_runs",
    [string]$RunPrefix = "fbm",
    [string]$DatePrefix = "",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $ApiBase) {
    throw "ApiBase is required."
}
if (-not $ApiKey) {
    throw "ApiKey is required."
}

$questionDir = Join-Path $DatasetRoot "datasets\question_sets"
$resolutionDir = Join-Path $DatasetRoot "datasets\resolution_sets"

if (-not (Test-Path $questionDir)) {
    throw "Question set directory not found: $questionDir"
}
if (-not (Test-Path $resolutionDir)) {
    throw "Resolution set directory not found: $resolutionDir"
}

$pairs = @()
Get-ChildItem -Path $questionDir -Filter "*-llm.json" | Sort-Object Name | ForEach-Object {
    if ($_.Name -eq "latest-llm.json") {
        return
    }
    $prefix = $_.BaseName -replace "-llm$", ""
    if ($DatePrefix -and -not $prefix.StartsWith($DatePrefix)) {
        return
    }
    $resolutionPath = Join-Path $resolutionDir "${prefix}_resolution_set.json"
    if (Test-Path $resolutionPath) {
        $pairs += [pscustomobject]@{
            Prefix = $prefix
            QuestionPath = $_.FullName
            ResolutionPath = $resolutionPath
        }
    }
}

if (-not $pairs.Count) {
    throw "No question/resolution pairs found."
}

$pythonExe = "python"
$runRoot = $OutputRoot
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

Push-Location $RepoRoot
try {
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    cmd /c chcp 65001 > $null

    $env:PYTHONPATH = "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    $env:FORECASTBENCH_API_BASE = $ApiBase
    $env:FORECASTBENCH_API_KEY = $ApiKey
    $env:FORECASTBENCH_MODEL = $Model

    foreach ($pair in $pairs) {
        $env:FORECASTBENCH_QUESTION_SET_PATH = $pair.QuestionPath
        $env:FORECASTBENCH_RESOLUTION_SET_PATH = $pair.ResolutionPath
        $runId = "$RunPrefix-$($pair.Prefix)"
        $runDir = Join-Path $runRoot $runId
        $summaryPath = Join-Path $runDir "summary.json"

        if ($Resume -and (Test-Path $summaryPath)) {
            try {
                $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
                $task = $summary.tasks | Select-Object -First 1
                if ($task -and $task.execution.status -eq "completed") {
                    Write-Host "=== $($pair.Prefix) ==="
                    Write-Host "skip completed run: $runId"
                    continue
                }
                Write-Host "=== $($pair.Prefix) ==="
                Write-Host "found incomplete run, removing: $runDir"
                Remove-Item -LiteralPath $runDir -Recurse -Force
            } catch {
                Write-Host "=== $($pair.Prefix) ==="
                Write-Host "found unreadable summary, removing: $runDir"
                Remove-Item -LiteralPath $runDir -Recurse -Force
            }
        }

        $cmd = @(
            $pythonExe, "run.py",
            "--config", $ConfigPath,
            "--output-dir", $runRoot,
            "--run-id", $runId
        )
        Write-Host "=== $($pair.Prefix) ==="
        Write-Host ($cmd -join " ")
        if (-not $DryRun) {
            & $cmd[0] $cmd[1] $cmd[2] $cmd[3] $cmd[4] $cmd[5] $cmd[6] $cmd[7]
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed for $($pair.Prefix) with exit code $LASTEXITCODE"
            }
        }
    }
}
finally {
    Pop-Location
}
