from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


VERSION = "003Q"


def load_used_clip_ids(root: Path, exclude_batch: str) -> set[str]:
    used = set()

    batch_file = root / "configs" / exclude_batch / f"{exclude_batch}.json"
    if not batch_file.is_file():
        return used

    batch = json.loads(batch_file.read_text(encoding="utf-8-sig"))

    for cfg_path_raw in batch.get("configs", []):
        cfg_path = Path(cfg_path_raw)
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path

        if not cfg_path.is_file():
            continue

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        clip_id = str(cfg.get("clip_id", "")).strip()
        if clip_id:
            used.add(clip_id)

    return used


def write_html(path: Path, summary: dict, selected: list[dict]) -> None:
    rows = []
    for item in selected:
        rows.append(
            "<tr>"
            f"<td>{item.get('idx')}</td>"
            f"<td>{item.get('clip_id')}</td>"
            f"<td>{item.get('n_points')}</td>"
            f"<td>{item.get('first_frame')}</td>"
            f"<td>{item.get('last_frame')}</td>"
            f"<td>{item.get('video_path')}</td>"
            f"<td>{item.get('config_path')}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003Q batch config maker</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px}}
th{{background:#20242e}}
</style>
</head>
<body>
<h1>TTFlux · 003Q batch config maker</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Clips sélectionnés</h2>
<table>
<thead>
<tr><th>idx</th><th>clip_id</th><th>n_points</th><th>first</th><th>last</th><th>video</th><th>config</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", default="batch_002A")
    ap.add_argument("--inventory", default="runs/inventory_001E/clip_inventory_001E.json")
    ap.add_argument("--exclude-batch", default="batch_001E")
    ap.add_argument("--offset", type=int, default=0, help="offset apres exclusion des clips deja utilises")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--frame-width", type=int, default=1280)
    ap.add_argument("--frame-height", type=int, default=720)
    ap.add_argument("--window-span", type=int, default=80)
    ap.add_argument("--segment-min-points", type=int, default=14)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--non-overlap-margin", type=int, default=25)
    ap.add_argument("--trail-keep-last", type=int, default=18)
    args = ap.parse_args()

    root = Path.cwd()
    inventory_path = Path(args.inventory)
    if not inventory_path.is_absolute():
        inventory_path = root / inventory_path

    if not inventory_path.is_file():
        raise SystemExit(f"Inventaire introuvable: {inventory_path}")

    data = json.loads(inventory_path.read_text(encoding="utf-8-sig"))

    used_clip_ids = load_used_clip_ids(root, args.exclude_batch)

    eligible = []
    for item in data.get("items", []):
        clip_id = str(item.get("clip_id", "")).strip()

        if not clip_id:
            continue

        if clip_id in used_clip_ids:
            continue

        if not bool(item.get("video_exists", False)):
            continue

        try:
            n_points = int(item.get("n_points", 0))
        except Exception:
            n_points = 0

        if n_points < args.min_points:
            continue

        video_path = str(item.get("video_path", "")).strip()
        if not video_path:
            continue

        eligible.append(item)

    selected = eligible[args.offset: args.offset + args.count]

    if not selected:
        raise SystemExit("Aucun clip sélectionné. Vérifie offset/count/min-points.")

    config_dir = root / "configs" / args.batch_id
    config_dir.mkdir(parents=True, exist_ok=True)

    config_paths = []
    selected_rows = []

    track_csv = data.get("track_csv", "")

    for idx, item in enumerate(selected, start=1):
        clip_id = str(item["clip_id"]).strip()

        cfg = {
            "name": f"{args.batch_id}_{idx:02d}_{clip_id}",
            "video_path": item["video_path"],
            "track_csv": track_csv,
            "clip_id": clip_id,
            "fps": args.fps,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "window_span": args.window_span,
            "min_points": args.segment_min_points,
            "top_k": args.top_k,
            "non_overlap_margin": args.non_overlap_margin,
            "trail_keep_last": args.trail_keep_last,
        }

        cfg_path = config_dir / f"{idx:02d}_{clip_id}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        config_paths.append(str(cfg_path))

        row = {
            "idx": idx,
            "clip_id": clip_id,
            "n_points": item.get("n_points"),
            "first_frame": item.get("first_frame"),
            "last_frame": item.get("last_frame"),
            "span": item.get("span"),
            "video_path": item.get("video_path"),
            "config_path": str(cfg_path),
        }
        selected_rows.append(row)

    batch = {
        "version": VERSION,
        "batch_id": args.batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inventory": str(inventory_path),
        "exclude_batch": args.exclude_batch,
        "used_clip_ids": sorted(used_clip_ids),
        "eligible_after_exclusion": len(eligible),
        "offset": args.offset,
        "count": len(config_paths),
        "configs": config_paths,
    }

    batch_path = config_dir / f"{args.batch_id}.json"
    batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")

    out_dir = root / "runs" / args.batch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"batch_config_selection_{VERSION}.csv"
    json_path = out_dir / f"batch_config_selection_{VERSION}.json"
    html_path = out_dir / f"batch_config_selection_{VERSION}.html"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        **batch,
        "batch_file": str(batch_path),
        "config_dir": str(config_dir),
        "run_dir": str(out_dir),
        "selected_clip_ids": [r["clip_id"] for r in selected_rows],
        "out_csv": str(csv_path),
        "out_json": str(json_path),
        "out_html": str(html_path),
    }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(html_path, summary, selected_rows)

    print("003Q_CONFIGS_OK")
    print("batch_id=", args.batch_id)
    print("batch_file=", batch_path)
    print("run_dir=", out_dir)
    print("selected_count=", len(selected_rows))
    print("selected_clip_ids=" + ",".join(r["clip_id"] for r in selected_rows))
    print("wrote", csv_path)
    print("wrote", json_path)
    print("wrote", html_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
