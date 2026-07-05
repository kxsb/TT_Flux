
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
& $Python -m ttflux.datasets.external_dataset_registry

& $Python - <<'PY'
from ttflux.datasets.external_dataset_registry import list_external_videos
from ttflux.datasets.transversal_clip_index import list_unified_raw_clips

for dataset_id in ["all", "openttgames_raw_videos", "tt3d_raw_extracted", "ttnet_layout"]:
    p = list_external_videos(dataset_id=dataset_id, limit=500)
    print("external dataset_id=", dataset_id, "count=", p["count"], "total_filtered=", p["total_filtered"], "default_total=", p["total_external_default_videos"])

for mode in ["legacy_transversal", "external", "mixed"]:
    p = list_unified_raw_clips(mode=mode, limit=20)
    print(
        "mode=", mode,
        "count=", p["count"],
        "legacy_count=", p["legacy_count"],
        "external_count=", p["external_count"],
        "legacy_total=", p["legacy_total_available"],
        "external_total=", p["external_total_available"],
    )
PY
