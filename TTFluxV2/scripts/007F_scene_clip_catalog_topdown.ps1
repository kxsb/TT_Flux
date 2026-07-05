
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:TTFLUX_LEGACY_ROOT = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"

& $Python -m compileall -q ttflux app

& $Python - <<'PY'
from ttflux.scene.legacy_table_import import list_scene_table_clips, topdown_payload_for_video

clips = list_scene_table_clips(status_filter="METRIC_TABLE_TRUSTED", only_existing=True, limit=10)
print("OK scene clips")
print("trusted_count_probe=", clips["total"])
if clips["clips"]:
    p = clips["clips"][0]["clip_path"]
    top = topdown_payload_for_video(p)
    print("first_clip=", p)
    print("topdown_ok=", top["ok"])
    print("ball_count=", top["ball_count"])
PY
