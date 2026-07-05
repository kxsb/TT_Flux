
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
from ttflux.core.paths import project_root, legacy_root
from ttflux.scene.legacy_table_import import list_scene_table_clips

pr = project_root()
lr = legacy_root()
print("project_root=", pr)
print("project_exists=", pr.exists())
print("legacy_root=", lr)
print("legacy_exists=", lr.exists())
print("scene_csv_exists=", (lr / "runs" / "rally_scene_table_objects_005C9B" / "scene_table_objects_005C9B.csv").exists())

clips = list_scene_table_clips(status_filter="METRIC_TABLE_TRUSTED", only_existing=True, limit=25)
print("trusted_total=", clips["total"])
if clips["clips"]:
    print("first_clip=", clips["clips"][0]["clip_path"])
PY
