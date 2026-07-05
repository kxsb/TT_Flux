
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_gt_audit import build_openttgames_gt_audit

payload = build_openttgames_gt_audit(rebuild_ball_index=False, build_sheets=True)

print("sample_count=", payload["sample_count"])
print("point_count=", payload["point_count"])
print("split_counts=", payload["split_counts"])
print("quality_flag_counts=", payload["quality_flag_counts"])
print("json=", payload["outputs"]["json"])
print("csv=", payload["outputs"]["csv"])
print("html=", payload["outputs"]["html"])

if payload["sample_count"] != 12:
    raise SystemExit(f"ERROR: sample_count expected 12, got {payload['sample_count']}")

if payload["point_count"] != 52987:
    raise SystemExit(f"ERROR: point_count expected 52987, got {payload['point_count']}")

if payload["split_counts"].get("training", 0) != 5:
    raise SystemExit("ERROR: training sample count != 5")

if payload["split_counts"].get("test", 0) != 7:
    raise SystemExit("ERROR: test sample count != 7")

bad_video = [s for s in payload["samples"] if not s.get("video_opened")]
if bad_video:
    raise SystemExit(f"ERROR: video_opened false: {[s['sample_id'] for s in bad_video]}")

bad_sheet = [s for s in payload["samples"] if not s.get("sheet_ok")]
if bad_sheet:
    raise SystemExit(f"ERROR: sheet_ok false: {[s['sample_id'] for s in bad_sheet]}")

html_path = Path(payload["outputs"]["html"])
if not html_path.exists() or html_path.stat().st_size < 1000:
    raise SystemExit("ERROR: html report missing or too small")

for s in payload["samples"]:
    print(
        s["sample_id"],
        "points=", s["point_count"],
        "valid=", s["valid_xy_count"],
        "invalid=", s["invalid_xy_count"],
        "out_img=", s["out_of_image_count"],
        "frame=", s["frame_min"], s["frame_max"],
        "max_gap=", s["max_gap"],
        "flags=", "|".join(s["quality_flags"]),
        "sheet=", s["sheet_jpg"],
    )

print("OK 007N OpenTTGames GT audit")
