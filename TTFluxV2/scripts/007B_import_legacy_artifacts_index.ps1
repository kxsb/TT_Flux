$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:TTFLUX_LEGACY_ROOT = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"

& $Python -m compileall -q ttflux app
& $Python -m ttflux.legacy.import_index

Write-Host ""
Write-Host "Rapport : runs\007B_legacy_artifacts_index"
