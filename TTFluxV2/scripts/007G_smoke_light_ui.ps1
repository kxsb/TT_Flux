
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LegacyRoot = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"
$env:TTFLUX_LEGACY_ROOT = $LegacyRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m compileall -q ttflux app

& $Python - <<'PY'
from app.backend.main import api_light_raw_videos, api_light_analyzed_videos
raw = api_light_raw_videos(limit=20)
an = api_light_analyzed_videos(limit=20)
print("raw_count=", raw["count"])
print("analyzed_count=", an["count"])
if raw["items"]:
    print("first_raw=", raw["items"][0]["path"])
if an["items"]:
    print("first_analyzed=", an["items"][0]["path"])
PY
