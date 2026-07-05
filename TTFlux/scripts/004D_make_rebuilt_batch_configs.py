from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd


VERSION = "004D"


VIDEO_KEYS = [
    "video",
    "video_path",
    "source_video",
    "input_video",
    "path",
]

ID_KEYS = [
    "clip_id",
    "sequence_key",
    "segment_id",
    "video_id",
    "run_id",
    "name",
]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_batch_items(batch_obj):
    if isinstance(batch_obj, list):
        return batch_obj, None

    if isinstance(batch_obj, dict):
        for key in ["items", "configs", "jobs", "videos", "batch"]:
            v = batch_obj.get(key)
            if isinstance(v, list):
                return v, key

    raise RuntimeError("Format batch JSON non reconnu : impossible de trouver une liste items/configs/jobs/videos/batch.")


def find_config_reference(item: dict) -> str | None:
    for key in ["config", "config_path", "path", "file", "json"]:
        v = item.get(key)
        if isinstance(v, str) and v.lower().endswith(".json"):
            return v
    return None


def find_template_config(template_batch: Path, batch_obj) -> tuple[dict, Path | None, dict]:
    items, _key = find_batch_items(batch_obj)

    if not items:
        raise RuntimeError("Batch template vide")

    first = items[0]

    if isinstance(first, str):
        ref = first
        p = Path(ref)
        if not p.is_absolute():
            p = template_batch.parent / p
        if not p.is_file():
            p = Path.cwd() / ref
        if not p.is_file():
            raise RuntimeError(f"Config template introuvable: {ref}")
        return load_json(p), p, {"mode": "string_ref"}

    if isinstance(first, dict):
        ref = find_config_reference(first)
        if ref:
            p = Path(ref)
            if not p.is_absolute():
                p1 = template_batch.parent / p
                p2 = Path.cwd() / p

                if p1.is_file():
                    p = p1
                elif p2.is_file():
                    p = p2

            if p.is_file():
                return load_json(p), p, {"mode": "dict_ref", "ref_key": ref}

        return copy.deepcopy(first), None, {"mode": "inline_dict"}

    raise RuntimeError(f"Item template non reconnu: {type(first)}")


def recursive_patch(obj, clip_id: str, video_id: str, video_path: str, clip_path: str):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()

            if kl in VIDEO_KEYS or kl.endswith("_video_path") or kl.endswith("_video"):
                out[k] = video_path

            elif kl in ID_KEYS:
                if kl == "video_id":
                    out[k] = video_id
                else:
                    out[k] = clip_id

            elif isinstance(v, str):
                s = v

                # Patch grossier mais utile si anciens IDs ou anciens chemins sont encodés dans des strings.
                if s.lower().endswith(".mp4"):
                    out[k] = video_path
                else:
                    out[k] = s

            else:
                out[k] = recursive_patch(v, clip_id, video_id, video_path, clip_path)

        return out

    if isinstance(obj, list):
        return [recursive_patch(x, clip_id, video_id, video_path, clip_path) for x in obj]

    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-csv", default="runs/dataset_rebuild_004C_balanced/rebuilt_clips_manifest_004C2.csv")
    ap.add_argument("--template-batch", default="configs/batch_002B/batch_002B.json")
    ap.add_argument("--batch-id", default="batch_004D_rebuilt_balanced")
    ap.add_argument("--limit", type=int, default=240)
    args = ap.parse_args()

    root = Path.cwd()

    clips_csv = Path(args.clips_csv)
    if not clips_csv.is_absolute():
        clips_csv = root / clips_csv

    template_batch = Path(args.template_batch)
    if not template_batch.is_absolute():
        template_batch = root / template_batch

    if not clips_csv.is_file():
        raise SystemExit(f"Clips CSV absent: {clips_csv}")

    if not template_batch.is_file():
        raise SystemExit(f"Template batch absent: {template_batch}")

    out_config_dir = root / "configs" / args.batch_id
    out_run_dir = root / "runs" / args.batch_id
    out_config_dir.mkdir(parents=True, exist_ok=True)
    out_run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clips_csv)

    required = {"clip_id", "clip_path", "video_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes dans clips CSV: {sorted(missing)}")

    df = df.copy()
    df["dataset_rank_004C"] = pd.to_numeric(df["dataset_rank_004C"], errors="coerce").fillna(999999).astype(int)
    df = df.sort_values("dataset_rank_004C")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    batch_obj = load_json(template_batch)
    template_config, template_config_path, template_info = find_template_config(template_batch, batch_obj)

    batch_items = []

    config_rows = []

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        old_clip_id = str(r["clip_id"])
        video_id = str(r["video_id"])
        clip_path = Path(str(r["clip_path"]))

        if not clip_path.is_absolute():
            clip_path = root / clip_path

        new_clip_id = f"{args.batch_id}_{i:04d}_{video_id}"

        config = recursive_patch(
            copy.deepcopy(template_config),
            clip_id=new_clip_id,
            video_id=video_id,
            video_path=str(clip_path),
            clip_path=str(clip_path),
        )

        # Ajouts explicites, sans casser les clés existantes.
        if isinstance(config, dict):
            config["clip_id"] = new_clip_id
            config["sequence_key"] = new_clip_id
            config["video_id"] = video_id
            config["video_path"] = str(clip_path)
            config["source_dataset_clip_id_004C2"] = old_clip_id
            config["source_dataset_rank_004C2"] = int(r["dataset_rank_004C"])
            config["source_activity_score_004B"] = float(r.get("activity_score", 0) or 0)
            config["source_start_sec_004B"] = float(r.get("start_sec", 0) or 0)
            config["source_end_sec_004B"] = float(r.get("end_sec", 0) or 0)

        config_name = f"{i:03d}_{new_clip_id}.json"
        config_path = out_config_dir / config_name
        write_json(config_path, config)

        rel_config = str(config_path.relative_to(root)).replace("\\", "/")

        batch_items.append({
            "idx": i,
            "clip_id": new_clip_id,
            "video_id": video_id,
            "config": rel_config,
            "config_path": rel_config,
            "video_path": str(clip_path),
            "source_dataset_clip_id_004C2": old_clip_id,
            "source_dataset_rank_004C2": int(r["dataset_rank_004C"]),
            "source_activity_score_004B": float(r.get("activity_score", 0) or 0),
        })

        config_rows.append(batch_items[-1])

    # On produit un batch simple mais aussi proche des anciens formats.
    # 003Q_run_batch_generic.py attend configs = liste de chemins JSON string.
    # On garde les m?tadonn?es compl?tes dans items.
    out_batch = {
        "version": VERSION,
        "batch_id": args.batch_id,
        "run_dir": str(out_run_dir),
        "config_dir": str(out_config_dir),
        "template_batch": str(template_batch),
        "template_config_path": str(template_config_path) if template_config_path else "",
        "template_info": template_info,
        "count": len(batch_items),
        "items": batch_items,
        "configs": [x["config"] for x in batch_items],
    }

    out_batch_path = out_config_dir / f"{args.batch_id}.json"
    write_json(out_batch_path, out_batch)

    manifest_csv = out_run_dir / "batch_config_manifest_004D.csv"
    pd.DataFrame(config_rows).to_csv(manifest_csv, index=False, encoding="utf-8")

    print("004D status=OK")
    print("batch_id=", args.batch_id)
    print("count=", len(batch_items))
    print("template_batch=", template_batch)
    print("template_config_path=", template_config_path or "INLINE")
    print("batch_file=", out_batch_path)
    print("config_dir=", out_config_dir)
    print("run_dir=", out_run_dir)
    print("manifest_csv=", manifest_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
