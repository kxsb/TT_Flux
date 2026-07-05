from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.external_dataset_registry import build_external_dataset_registry, list_external_videos
from ttflux.datasets.transversal_clip_index import list_unified_raw_clips

print("PROJECT_ROOT=", PROJECT_ROOT)

build_external_dataset_registry()

for dataset_id in ["all", "openttgames_raw_videos", "tt3d_raw_extracted", "ttnet_layout"]:
    p = list_external_videos(dataset_id=dataset_id, limit=500)
    print(
        "external dataset_id=", dataset_id,
        "count=", p["count"],
        "total_filtered=", p["total_filtered"],
        "default_total=", p["total_external_default_videos"],
        "explicit=", p["explicit_dataset_browse"],
    )
    if p["items"]:
        print("  first=", p["items"][0]["path"])

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

ttnet = list_external_videos(dataset_id="ttnet_layout", limit=500)
if ttnet["count"] <= 0:
    raise SystemExit("ERROR: ttnet_layout explicit browse returned zero videos")

mixed = list_unified_raw_clips(mode="mixed", limit=20)
if mixed["legacy_count"] + mixed["external_count"] != mixed["count"]:
    raise SystemExit("ERROR: mixed count incoherent")

external_all = list_external_videos(dataset_id="all", limit=1000)
if any(item.get("dataset_id") == "ttnet_layout" for item in external_all["items"]):
    raise SystemExit("ERROR: ttnet_layout leaked into external all")

print("OK 007I4D explicit ttnet_layout browse")
