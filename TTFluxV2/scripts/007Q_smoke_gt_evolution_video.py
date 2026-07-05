
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.viz.openttgames_gt_evolution_video import inspect_video

mp4 = PROJECT_ROOT / "runs" / "007Q_gt_evolution_video" / "openttgames_gt_evolution_game_1_30s_1080p_007Q.mp4"
json_path = PROJECT_ROOT / "runs" / "007Q_gt_evolution_video" / "openttgames_gt_evolution_game_1_30s_1080p_007Q.json"

print("mp4=", mp4)
print("json=", json_path)

if not mp4.exists():
    raise SystemExit("ERROR: mp4 not found")

if not json_path.exists():
    raise SystemExit("ERROR: json not found")

info = inspect_video(mp4)
print("info=", info)

if not info.get("opened"):
    raise SystemExit("ERROR: generated mp4 cannot be opened")

if int(info.get("frame_count") or 0) < 850:
    raise SystemExit(f"ERROR: frame_count too low: {info.get('frame_count')}")

if abs(float(info.get("duration_sec") or 0) - 30.0) > 1.5:
    raise SystemExit(f"ERROR: duration not close to 30s: {info.get('duration_sec')}")

if int(info.get("width") or 0) != 1920 or int(info.get("height") or 0) != 1080:
    raise SystemExit(f"ERROR: expected 1920x1080, got {info.get('width')}x{info.get('height')}")

if int(info.get("size_bytes") or 0) < 1_000_000:
    raise SystemExit("ERROR: mp4 too small")

print("OK 007Q smoke")
