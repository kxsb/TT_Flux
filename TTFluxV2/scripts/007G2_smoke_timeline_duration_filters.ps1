
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

raw = api_light_raw_videos(limit=10)
an = api_light_analyzed_videos(limit=10)

print("raw_count=", raw["count"])
if raw["items"]:
    r = raw["items"][0]
    print("first_raw=", r.get("name"))
    print("duration_sec=", r.get("duration_sec"))
    print("frame_count=", r.get("frame_count"))
    print("fps=", r.get("fps"))

print("analyzed_count=", an["count"])
if an["items"]:
    a = an["items"][0]
    print("first_analyzed=", a.get("name"))
    print("duration_sec=", a.get("duration_sec"))
    print("frame_count=", a.get("frame_count"))
    print("fps=", a.get("fps"))
PY
