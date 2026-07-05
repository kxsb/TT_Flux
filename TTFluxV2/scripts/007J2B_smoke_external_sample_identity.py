
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.external_sample_index import build_external_sample_index, list_external_samples
from ttflux.datasets.external_dataset_registry import list_external_videos

payload = build_external_sample_index()

print("sample_count=", payload["sample_count"])
print("dataset_counts=", payload["dataset_counts"])
print("split_counts=", payload["split_counts"])
print("annotation_ready_counts=", payload["annotation_ready_counts"])

if payload["dataset_counts"].get("tt3d_raw_extracted", 0) != 358:
    raise SystemExit(f"ERROR: TT3D sample count expected 358, got {payload['dataset_counts'].get('tt3d_raw_extracted', 0)}")

if payload["dataset_counts"].get("openttgames_raw_videos", 0) != 12:
    raise SystemExit("ERROR: OpenTTGames sample count != 12")

if payload["dataset_counts"].get("ttnet_layout", 0) != 12:
    raise SystemExit("ERROR: TTNet layout sample count != 12")

if payload["split_counts"].get("openttgames_raw_videos:training", 0) != 5:
    raise SystemExit("ERROR: OpenTTGames training split != 5")

if payload["split_counts"].get("openttgames_raw_videos:test", 0) != 7:
    raise SystemExit("ERROR: OpenTTGames test split != 7")

if payload["annotation_ready_counts"].get("openttgames_raw_videos", 0) != 12:
    raise SystemExit("ERROR: OpenTTGames annotation_ready != 12")

op = list_external_videos(dataset_id="openttgames_raw_videos", limit=12)
print("openttgames videos=", op["count"])
for item in op["items"][:5]:
    print("  op item=", item.get("sample_id"), item.get("split"), item.get("annotation_exists"), item.get("annotation_file_count"))

if any(item.get("split") == "mixed" for item in op["items"]):
    raise SystemExit("ERROR: OpenTTGames split still mixed in enriched video items")

ttnet = list_external_videos(dataset_id="ttnet_layout", limit=12)
print("ttnet videos=", ttnet["count"])
if ttnet["count"] != 12:
    raise SystemExit("ERROR: TTNet layout explicit count != 12")

tt3d = list_external_samples(dataset_id="tt3d_raw_extracted", limit=500)
print("tt3d samples=", tt3d["count"], "total_filtered=", tt3d["total_filtered"])
print("first tt3d samples:")
for s in tt3d["samples"][:8]:
    print("  ", s.get("sample_id"), s.get("split"), s.get("primary_video"))

ids = [s["sample_id"] for s in tt3d["samples"]]
if len(ids) != len(set(ids)):
    raise SystemExit("ERROR: TT3D duplicate sample_id remains")

print("OK 007J2B external sample identity")
