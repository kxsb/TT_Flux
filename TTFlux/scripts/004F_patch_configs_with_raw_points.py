from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


VERSION = "004F"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_points_csv(obj, points_csv: str):
    if isinstance(obj, dict):
        out = {}

        for k, v in obj.items():
            kl = str(k).lower()

            if (
                kl in {"tracks_csv", "track_csv", "tracking_csv", "points_csv", "candidate_csv", "candidates_csv"}
                or ("csv" in kl and ("track" in kl or "point" in kl or "candidate" in kl or "trajectory" in kl))
            ):
                out[k] = points_csv
            else:
                out[k] = patch_points_csv(v, points_csv)

        # Ajouts top-level explicites.
        out["tracks_csv"] = points_csv
        out["track_csv"] = points_csv
        out["tracking_csv"] = points_csv
        out["points_csv"] = points_csv

        return out

    if isinstance(obj, list):
        return [patch_points_csv(x, points_csv) for x in obj]

    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-batch", default="configs/batch_004D_probe12/batch_004D_probe12.json")
    ap.add_argument("--batch-id", default="batch_004F_probe12")
    ap.add_argument("--points-csv", default="runs/dataset_points_004E_probe12/raw_tracking_points_004E.csv")
    args = ap.parse_args()

    root = Path.cwd()

    source_batch = Path(args.source_batch)
    if not source_batch.is_absolute():
        source_batch = root / source_batch

    points_csv = Path(args.points_csv)
    if not points_csv.is_absolute():
        points_csv = root / points_csv

    if not source_batch.is_file():
        raise SystemExit(f"Source batch absent: {source_batch}")

    if not points_csv.is_file():
        raise SystemExit(f"Points CSV absent: {points_csv}")

    source = load_json(source_batch)

    configs = source.get("configs")
    if not isinstance(configs, list):
        raise SystemExit("source_batch.configs absent ou non-liste")

    out_config_dir = root / "configs" / args.batch_id
    out_run_dir = root / "runs" / args.batch_id
    out_config_dir.mkdir(parents=True, exist_ok=True)
    out_run_dir.mkdir(parents=True, exist_ok=True)

    out_configs = []
    out_items = []

    rel_points = str(points_csv.relative_to(root)).replace("\\", "/")

    for i, cfg_ref in enumerate(configs, start=1):
        cfg_path = Path(str(cfg_ref))
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path

        if not cfg_path.is_file():
            raise SystemExit(f"Config source absente: {cfg_path}")

        cfg = load_json(cfg_path)
        patched = patch_points_csv(copy.deepcopy(cfg), rel_points)

        # On conserve aussi une trace absolue au cas où un vieux loader ne résout pas relatif.
        if isinstance(patched, dict):
            patched["tracks_csv_abs_004F"] = str(points_csv)
            patched["points_csv_abs_004F"] = str(points_csv)
            patched["points_source_004F"] = rel_points

        out_cfg_path = out_config_dir / cfg_path.name
        write_json(out_cfg_path, patched)

        rel_cfg = str(out_cfg_path.relative_to(root)).replace("\\", "/")
        out_configs.append(rel_cfg)
        out_items.append({
            "idx": i,
            "config": rel_cfg,
            "source_config": str(cfg_path),
            "points_csv": rel_points,
        })

    out_batch = {
        "version": VERSION,
        "batch_id": args.batch_id,
        "source_batch": str(source_batch),
        "points_csv": rel_points,
        "count": len(out_configs),
        "run_dir": str(out_run_dir),
        "config_dir": str(out_config_dir),
        "configs": out_configs,
        "items": out_items,
    }

    out_batch_path = out_config_dir / f"{args.batch_id}.json"
    write_json(out_batch_path, out_batch)

    print("004F status=OK")
    print("batch_id=", args.batch_id)
    print("count=", len(out_configs))
    print("points_csv=", rel_points)
    print("batch_file=", out_batch_path)
    print("config_dir=", out_config_dir)
    print("run_dir=", out_run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
