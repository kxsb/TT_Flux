
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m compileall -q ttflux app

$Index = Get-Content .\app\static\index.html -Raw
if ($Index -notmatch "playRaw") { throw "missing playRaw" }
if ($Index -notmatch "stopRaw") { throw "missing stopRaw" }
if ($Index -notmatch "slowerRaw") { throw "missing slowerRaw" }
if ($Index -notmatch "fasterRaw") { throw "missing fasterRaw" }
if ($Index -notmatch "playAnalyzed") { throw "missing playAnalyzed" }

Write-Host "OK playback controls present"
