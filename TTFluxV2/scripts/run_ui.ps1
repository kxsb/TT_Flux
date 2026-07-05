
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LegacyRoot = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"

Write-Host "========================================================================"
Write-Host "TTFlux V2 UI"
Write-Host "========================================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "LegacyRoot  = $LegacyRoot"
Write-Host ""

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Création venv..."
    py -3 -m venv .venv
}

$Python = ".\.venv\Scripts\python.exe"

Write-Host "Upgrade pip..."
& $Python -m pip install --upgrade pip

Write-Host "Install requirements..."
& $Python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Open:"
Write-Host "http://127.0.0.1:8787"
Write-Host ""

# Important : chemin calculé au runtime, pas d'accent écrit en dur dans le script.
$env:TTFLUX_LEGACY_ROOT = $LegacyRoot

& $Python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8787 --reload
