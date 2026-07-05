
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LegacyRoot = Join-Path (Split-Path -Parent $ProjectRoot) "TTFlux"
$env:TTFLUX_LEGACY_ROOT = $LegacyRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python - <<'PY'
from ttflux.core.paths import legacy_root
from ttflux.scene.legacy_table_import import list_scene_table_clips

print("legacy_root=", legacy_root())
print("legacy_exists=", legacy_root().exists())

for status in ["METRIC_TABLE_TRUSTED", "METRIC_TABLE_REVIEW", "PARTIAL_TABLE_MASK", ""]:
    clips = list_scene_table_clips(status_filter=status, only_existing=True, limit=250)
    print("status=", repr(status), "total=", clips["total"], "counts=", clips["status_counts"])
    if clips["clips"]:
        print("  first=", clips["clips"][0]["review_id"], clips["clips"][0]["clip_path"])
PY
