$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONPATH = Join-Path $Root "src"
python -m ttflux smoke
