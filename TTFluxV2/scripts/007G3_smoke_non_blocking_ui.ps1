
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

raw_fast = api_light_raw_videos(limit=20, with_meta=False)
raw_meta = api_light_raw_videos(limit=5, with_meta=True)
an_fast = api_light_analyzed_videos(limit=20, with_meta=False)

print("raw_fast_count=", raw_fast["count"], "with_meta=", raw_fast.get("with_meta"))
print("raw_meta_count=", raw_meta["count"], "with_meta=", raw_meta.get("with_meta"))
if raw_meta["items"]:
    r = raw_meta["items"][0]
    print("first_raw_meta=", r.get("name"), r.get("duration_sec"), r.get("frame_count"), r.get("fps"))
print("an_fast_count=", an_fast["count"], "with_meta=", an_fast.get("with_meta"))
PY
