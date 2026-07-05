
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
$env:TTFLUX_LEGACY_ROOT = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m ttflux.viz.openttgames_gt_evolution_video `
    --sample game_1 `
    --split training `
    --seconds 30 `
    --out-fps 30 `
    --width 1920 `
    --height 1080 `
    --min-valid-points 24

if ($LASTEXITCODE -ne 0) {
    throw "007Q video generation failed"
}
