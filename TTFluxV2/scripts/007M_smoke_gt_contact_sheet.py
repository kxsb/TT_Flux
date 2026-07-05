
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_gt_contact_sheet import build_gt_contact_sheet, contact_sheet_jpeg

video = PROJECT_ROOT.parent / "TTFlux" / "data_external" / "sources" / "openttgames" / "raw_videos" / "game_1.mp4"

payload = build_gt_contact_sheet(str(video), max_tiles=24, cols=4)
print("ok=", payload.get("ok"))
print("sample_id=", payload.get("sample_id"))
print("gt_row_count=", payload.get("gt_row_count"))
print("selected_frame_count=", payload.get("selected_frame_count"))
print("jpg=", payload.get("outputs", {}).get("jpg"))
print("json=", payload.get("outputs", {}).get("json"))

if not payload.get("ok"):
    raise SystemExit("ERROR: contact sheet failed")

if payload.get("selected_frame_count", 0) <= 0:
    raise SystemExit("ERROR: no selected frames")

jpg = Path(payload["outputs"]["jpg"])
if not jpg.exists() or jpg.stat().st_size < 20000:
    raise SystemExit("ERROR: contact sheet jpg missing or too small")

raw = contact_sheet_jpeg(str(video), max_tiles=12, cols=4)
print("jpeg_bytes=", len(raw))
if len(raw) < 20000:
    raise SystemExit("ERROR: contact_sheet_jpeg too small")

print("OK 007M GT contact sheet")
