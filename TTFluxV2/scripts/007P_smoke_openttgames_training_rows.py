
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttflux.datasets.openttgames_training_rows import build_openttgames_training_rows

payload = build_openttgames_training_rows(
    rebuild_segments=False,
    gap_threshold=30,
    min_valid_points=16,
    include_unusable_segments=False,
)

print("source_segment_count=", payload["source_segment_count"])
print("segment_count=", payload["segment_count"])
print("row_count=", payload["row_count"])
print("usable_position_row_count=", payload["usable_position_row_count"])
print("visibility_row_count=", payload["visibility_row_count"])
print("train_row_count=", payload["train_row_count"])
print("test_row_count=", payload["test_row_count"])
print("train_usable_position_row_count=", payload["train_usable_position_row_count"])
print("test_usable_position_row_count=", payload["test_usable_position_row_count"])
print("row_role_counts=", payload["row_role_counts"])
print("split_counts=", payload["split_counts"])
print("json=", payload["outputs"]["json"])
print("rows_csv=", payload["outputs"]["training_rows_csv"])
print("segments_csv=", payload["outputs"]["training_segments_csv"])

if payload["source_segment_count"] != 967:
    raise SystemExit(f"ERROR: source_segment_count expected 967, got {payload['source_segment_count']}")

if payload["segment_count"] != 855:
    raise SystemExit(f"ERROR: segment_count expected 855 usable segments, got {payload['segment_count']}")

if payload["row_count"] <= 0:
    raise SystemExit("ERROR: row_count <= 0")

if payload["usable_position_row_count"] <= 0:
    raise SystemExit("ERROR: usable_position_row_count <= 0")

if payload["train_row_count"] <= payload["test_row_count"]:
    raise SystemExit("ERROR: train_row_count should be > test_row_count")

rows_csv = Path(payload["outputs"]["training_rows_csv"])
segments_csv = Path(payload["outputs"]["training_segments_csv"])

if not rows_csv.exists() or rows_csv.stat().st_size < 1000:
    raise SystemExit("ERROR: rows_csv missing or too small")

if not segments_csv.exists() or segments_csv.stat().st_size < 1000:
    raise SystemExit("ERROR: segments_csv missing or too small")

with rows_csv.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    first = next(reader)

print("first_row=", first)

required = [
    "training_uid",
    "sample_id",
    "split",
    "segment_uid",
    "video_path",
    "frame",
    "x",
    "y",
    "visible",
    "valid_xy",
    "usable_for_position",
    "row_role",
]

missing = [c for c in required if c not in first]
if missing:
    raise SystemExit(f"ERROR: missing columns {missing}")

print("OK 007P OpenTTGames training rows")
