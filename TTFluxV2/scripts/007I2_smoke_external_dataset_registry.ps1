
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
from ttflux.datasets.transversal_clip_index import list_unified_raw_clips

for mode in ["legacy_transversal", "external", "mixed"]:
    payload = list_unified_raw_clips(mode=mode, limit=20)
    print("mode=", mode, "count=", payload["count"], "legacy=", payload["legacy_count"], "external=", payload["external_count"])
    if payload["items"]:
        print("  first=", payload["items"][0].get("dataset_id") or payload["items"][0].get("review_id"), payload["items"][0].get("path"))
PY
