from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    r"C:\Users\micka\Desktop\développement\ping\TTFlux"
)

PROBE_ROOT = Path(
    r"C:\Users\micka\pingcoach_probe"
)

TRACK_CSV = (
    PROBE_ROOT
    / "runs"
    / "scene_state_probe_v03"
    / "tracking_rows_v71_motion_segments_enriched.csv"
)

CLIP_DIR = (
    PROBE_ROOT
    / "runs"
    / "dataset2_clips_v62"
    / "clips"
)

OUT_DIR = ROOT / "runs" / "inventory_001E"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TRACK_CSV.exists():
    raise SystemExit(f"CSV introuvable: {TRACK_CSV}")

groups = defaultdict(list)

with TRACK_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []

    required = {"clip_id", "frame", "x", "y"}
    missing = required - set(fieldnames)

    if missing:
        raise SystemExit(
            f"Colonnes manquantes: {missing}"
        )

    for row in reader:
        clip_id = str(row.get("clip_id", "")).strip()

        if not clip_id:
            continue

        try:
            frame = int(float(str(row["frame"]).replace(",", ".")))
            x = float(str(row["x"]).replace(",", "."))
            y = float(str(row["y"]).replace(",", "."))
        except Exception:
            continue

        video_id = str(row.get("video_id", "")).strip()
        segment_id = str(row.get("segment_id", "")).strip()

        groups[clip_id].append(
            {
                "frame": frame,
                "x": x,
                "y": y,
                "video_id": video_id,
                "segment_id": segment_id,
            }
        )

items = []

for clip_id, rows in groups.items():
    rows = sorted(rows, key=lambda r: r["frame"])
    frames = [r["frame"] for r in rows]

    video_path = CLIP_DIR / f"{clip_id}.mp4"

    video_ids = sorted(
        set(r["video_id"] for r in rows if r["video_id"])
    )

    segment_ids = sorted(
        set(r["segment_id"] for r in rows if r["segment_id"])
    )

    items.append(
        {
            "clip_id": clip_id,
            "video_exists": video_path.exists(),
            "video_path": str(video_path),
            "n_points": len(rows),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "span": max(frames) - min(frames) + 1,
            "video_ids": video_ids,
            "segment_ids": segment_ids,
        }
    )

items = sorted(
    items,
    key=lambda x: (
        x["video_exists"],
        x["n_points"],
    ),
    reverse=True,
)

summary = {
    "track_csv": str(TRACK_CSV),
    "clip_dir": str(CLIP_DIR),
    "total_clips": len(items),
    "clips_with_video": sum(
        1 for item in items if item["video_exists"]
    ),
    "items": items,
}

json_path = OUT_DIR / "clip_inventory_001E.json"
csv_path = OUT_DIR / "clip_inventory_001E.csv"

json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "clip_id",
            "video_exists",
            "n_points",
            "first_frame",
            "last_frame",
            "span",
            "video_path",
        ],
    )

    writer.writeheader()

    for item in items:
        writer.writerow(
            {
                "clip_id": item["clip_id"],
                "video_exists": item["video_exists"],
                "n_points": item["n_points"],
                "first_frame": item["first_frame"],
                "last_frame": item["last_frame"],
                "span": item["span"],
                "video_path": item["video_path"],
            }
        )

print("TTFLUX_INVENTORY_001E_OK")
print("total_clips =", summary["total_clips"])
print("clips_with_video =", summary["clips_with_video"])

print("")
print("Top clips:")

for item in items[:12]:
    print(
        item["n_points"],
        item["video_exists"],
        item["clip_id"],
    )
