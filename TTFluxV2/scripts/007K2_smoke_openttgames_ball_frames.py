
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_annotation_probe import (
    extract_ball_points_for_sample,
    build_openttgames_ball_points_index,
)

for sid in ["game_1", "test_1"]:
    pts = extract_ball_points_for_sample(sid, max_points=5000)
    print(
        "ball_points",
        sid,
        "found=", pts["found"],
        "candidate_files=", pts.get("candidate_file_count"),
        "point_count=", pts.get("point_count"),
        "frame_min=", pts.get("frame_min"),
        "frame_max=", pts.get("frame_max"),
        "x_min=", pts.get("x_min"),
        "x_max=", pts.get("x_max"),
        "y_min=", pts.get("y_min"),
        "y_max=", pts.get("y_max"),
    )
    print("first_points=", pts.get("points", [])[:5])

    if pts.get("point_count", 0) <= 0:
        raise SystemExit(f"ERROR: no points for {sid}")

    if pts.get("frame_min") is None or pts.get("frame_max") is None:
        raise SystemExit(f"ERROR: frame decode failed for {sid}")

idx = build_openttgames_ball_points_index(max_points_per_sample=200000)
print("index sample_count=", idx["sample_count"])
print("index point_count=", idx["point_count"])
print("index frame_missing_count=", idx["frame_missing_count"])
print("index split_counts=", idx["split_counts"])
print("index csv=", idx["outputs"]["csv"])

if idx["sample_count"] != 12:
    raise SystemExit(f"ERROR: expected 12 samples, got {idx['sample_count']}")

if idx["point_count"] <= 0:
    raise SystemExit("ERROR: empty OpenTTGames ball point index")

if idx["frame_missing_count"] != 0:
    raise SystemExit(f"ERROR: frame_missing_count != 0 : {idx['frame_missing_count']}")

print("OK 007K2 ball markup frame decode")
