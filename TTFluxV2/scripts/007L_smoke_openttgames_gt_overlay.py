
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_ball_overlay import (
    load_points_index,
    find_ball_points_for_frame,
    render_external_gt_frame_jpeg,
)

video = PROJECT_ROOT.parent / "TTFlux" / "data_external" / "sources" / "openttgames" / "raw_videos" / "game_1.mp4"

idx = load_points_index()
print("point_count=", idx["point_count"])
print("sample_counts_keys=", sorted(idx["sample_counts"].keys()))

info = find_ball_points_for_frame(str(video), frame=14, window=0)
print("frame_info=", info)

if not info.get("found"):
    raise SystemExit("ERROR: GT point not found for game_1 frame 14")

pts = info.get("points", [])
if not pts:
    raise SystemExit("ERROR: empty point list for game_1 frame 14")

p0 = pts[0]
if int(round(float(p0.get("x")))) != 506 or int(round(float(p0.get("y")))) != 523:
    raise SystemExit(f"ERROR: unexpected first point for game_1 frame 14: {p0}")

jpeg = render_external_gt_frame_jpeg(str(video), frame=14, window=0)
print("jpeg_bytes=", len(jpeg))

if len(jpeg) < 10000:
    raise SystemExit("ERROR: rendered JPEG too small")

out = PROJECT_ROOT / "runs" / "007K_openttgames_annotation_probe" / "gt_overlay_game_1_f14_007L.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(jpeg)
print("overlay=", out)

print("OK 007L OpenTTGames GT frame overlay")
