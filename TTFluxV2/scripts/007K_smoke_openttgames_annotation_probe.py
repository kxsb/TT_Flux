
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_annotation_probe import (
    probe_openttgames_annotations,
    extract_ball_points_for_sample,
)

probe = probe_openttgames_annotations()
print("probe sample_count=", probe["sample_count"])
print("suffix_total=", probe["suffix_total"])
print("dir_total_top=", dict(list(probe["dir_total"].items())[:20]))
print("json=", probe["outputs"]["json"])

if probe["sample_count"] != 12:
    raise SystemExit(f"ERROR: expected 12 OpenTTGames samples, got {probe['sample_count']}")

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
    )
    print("tried=", pts.get("tried_files", [])[:10])

print("OK 007K annotation probe")
