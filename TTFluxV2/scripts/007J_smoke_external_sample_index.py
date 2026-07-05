
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

if payload["annotation_ready_counts"].get("openttgames_raw_videos", 0) < 12:
    raise SystemExit("ERROR: OpenTTGames annotations not fully linked")

ext = list_external_videos(dataset_id="openttgames_raw_videos", limit=12)
print("openttgames_video_count=", ext["count"])
if ext["count"] != 12:
    raise SystemExit("ERROR: OpenTTGames video count != 12")

first = ext["items"][0]
print("first_enriched=", first.get("sample_id"), first.get("annotation_exists"), first.get("annotation_file_count"))

if not first.get("annotation_exists"):
    raise SystemExit("ERROR: first OpenTTGames video has no annotation link")

ttnet = list_external_videos(dataset_id="ttnet_layout", limit=12)
print("ttnet_count=", ttnet["count"])
if ttnet["count"] != 12:
    raise SystemExit("ERROR: TTNet layout explicit count != 12")

samples = list_external_samples(dataset_id="openttgames_raw_videos", limit=20)
print("samples_openttgames=", samples["count"])

print("OK 007J external sample index")
