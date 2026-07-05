$ErrorActionPreference = "Stop"

$Root = Join-Path $env:USERPROFILE `
  "Desktop\développement\ping\TTFlux"

$Py = Join-Path $env:USERPROFILE `
  "Desktop\développement\ping\TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch-master\.venv_ttnet_modern\Scripts\python.exe"

Set-Location $Root

$env:PYTHONPATH = Join-Path $Root "src"

$Config = Join-Path $Root `
  "configs\pingcoach_v62_m2forqbqazc_s01.json"

$Out = Join-Path $Root `
  "runs\validation_001"

New-Item -ItemType Directory -Force `
  -Path $Out | Out-Null

& $Py -m ttflux validate `
  --config $Config `
  --out $Out

notepad (Join-Path $Out "validation_summary.json")

Invoke-Item $Out
