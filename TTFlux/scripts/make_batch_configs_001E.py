from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    r"C:\Users\micka\Desktop\développement\ping\TTFlux"
)

INVENTORY = (
    ROOT
    / "runs"
    / "inventory_001E"
    / "clip_inventory_001E.json"
)

CONFIG_DIR = ROOT / "configs" / "batch_001E"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

if not INVENTORY.exists():
    raise SystemExit(
        f"Inventaire introuvable: {INVENTORY}"
    )

data = json.loads(
    INVENTORY.read_text(encoding="utf-8-sig")
)

items = [
    item
    for item in data["items"]
    if item["video_exists"]
    and item["n_points"] >= 30
]

items = items[:8]

configs = []

for idx, item in enumerate(items, start=1):
    clip_id = item["clip_id"]

    cfg = {
        "name": f"batch_001E_{idx:02d}_{clip_id}",
        "video_path": item["video_path"],
        "track_csv": data["track_csv"],
        "clip_id": clip_id,
        "fps": 50.0,
        "frame_width": 1280,
        "frame_height": 720,
        "window_span": 80,
        "min_points": 14,
        "top_k": 3,
        "non_overlap_margin": 25,
        "trail_keep_last": 18,
    }

    path = CONFIG_DIR / f"{idx:02d}_{clip_id}.json"

    path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    configs.append(str(path))

batch = {
    "count": len(configs),
    "configs": configs,
}

batch_path = CONFIG_DIR / "batch_001E.json"

batch_path.write_text(
    json.dumps(batch, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("TTFLUX_BATCH_CONFIGS_001E_OK")
print("count =", len(configs))

for path in configs:
    print(path)
