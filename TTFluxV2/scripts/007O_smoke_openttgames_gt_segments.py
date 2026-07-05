
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_gt_segments import build_openttgames_gt_segments, segment_points

payload = build_openttgames_gt_segments(gap_threshold=30, min_valid_points=16)

print("sample_count=", payload["sample_count"])
print("segment_count=", payload["segment_count"])
print("usable_segment_count=", payload["usable_segment_count"])
print("point_count=", payload["point_count"])
print("split_counts=", payload["split_counts"])
print("quality_counts=", payload["quality_counts"])
print("json=", payload["outputs"]["json"])
print("csv=", payload["outputs"]["csv"])
print("html=", payload["outputs"]["html"])

if payload["sample_count"] != 12:
    raise SystemExit(f"ERROR: sample_count expected 12, got {payload['sample_count']}")

if payload["point_count"] != 52987:
    raise SystemExit(f"ERROR: point_count expected 52987, got {payload['point_count']}")

if payload["segment_count"] <= 12:
    raise SystemExit(f"ERROR: segment_count too low: {payload['segment_count']}")

if payload["usable_segment_count"] <= 0:
    raise SystemExit("ERROR: no usable segment")

html_path = Path(payload["outputs"]["html"])
csv_path = Path(payload["outputs"]["csv"])

if not html_path.exists() or html_path.stat().st_size < 1000:
    raise SystemExit("ERROR: html report missing or too small")

if not csv_path.exists() or csv_path.stat().st_size < 1000:
    raise SystemExit("ERROR: csv report missing or too small")

first = payload["segments"][0]
pts = segment_points(segment_uid=first["segment_uid"], limit=100000)
print("first_segment=", first)
print("first_segment_points=", pts["point_count"])

if not pts.get("found"):
    raise SystemExit("ERROR: first segment points not found")

for s in payload["sample_summaries"]:
    print(
        s["sample_id"],
        "segments=", s["segment_count"],
        "usable=", s["usable_segment_count"],
        "points=", s["point_count"],
    )

print("OK 007O OpenTTGames GT segments")
