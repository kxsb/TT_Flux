from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "004U2"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def norm_path(s: str) -> str:
    return str(s or "").replace("\\", "/").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-clicks", default="runs/ball_goldset_004J/ball_clicks_004J.csv")
    ap.add_argument("--old-manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--old-batch-config-manifest", default="runs/batch_004D_full240/batch_config_manifest_004D.csv")
    ap.add_argument("--old-clips-manifest", default="runs/dataset_rebuild_004C_balanced/rebuilt_clips_manifest_004C2.csv")
    ap.add_argument("--rally-manifest", default="runs/rally_dataset_004T/rally_manifest_004T.csv")
    ap.add_argument("--out-dir", default="runs/rally_goldset_004U2_reused_clicks")
    args = ap.parse_args()

    root = Path.cwd()

    old_clicks_path = root / args.old_clicks
    old_manifest_path = root / args.old_manifest
    old_batch_cfg_path = root / args.old_batch_config_manifest
    old_clips_path = root / args.old_clips_manifest
    rally_manifest_path = root / args.rally_manifest
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [old_clicks_path, old_manifest_path, old_batch_cfg_path, old_clips_path, rally_manifest_path]:
        if not p.is_file():
            raise SystemExit(f"Fichier absent: {p}")

    clicks = pd.read_csv(old_clicks_path).fillna("")
    old_manifest = pd.read_csv(old_manifest_path).fillna("")
    batch_cfg = pd.read_csv(old_batch_cfg_path).fillna("")
    old_clips = pd.read_csv(old_clips_path).fillna("")
    rallies = pd.read_csv(rally_manifest_path).fillna("")

    # On garde ball + not_visible + unsure, mais seuls les clics ball ont x/y.
    clicks["local_frame_num"] = to_num(clicks["local_frame"])
    clicks["source_frame_num"] = to_num(clicks["source_frame"])
    clicks["x_num"] = to_num(clicks["x"])
    clicks["y_num"] = to_num(clicks["y"])

    old_manifest_by_review = {
        str(r["review_id"]): r for _, r in old_manifest.iterrows()
    }

    # Mapping ancien clip batch_004D_full240_xxxx -> clip dataset 004C2.
    batch_cfg_by_clip = {
        str(r["clip_id"]): r for _, r in batch_cfg.iterrows()
    }

    # Mapping clip_id 004C2 -> source vidéo + start_frame absolu.
    old_clips_by_id = {}
    for _, r in old_clips.iterrows():
        cid = str(r.get("clip_id", ""))
        if cid:
            old_clips_by_id[cid] = r

    # Fallback par clip_path.
    old_clips_by_path = {}
    for _, r in old_clips.iterrows():
        p = norm_path(r.get("clip_path", ""))
        if p:
            old_clips_by_path[p] = r

    # Prépare rallys par video_id.
    rallies["start_frame_source_num"] = to_num(rallies["start_frame_source"])
    rallies["end_frame_source_num"] = to_num(rallies["end_frame_source"])
    rallies["fps_num"] = to_num(rallies["fps"]).fillna(50.0)

    rallies_by_video = {
        str(k): g.copy() for k, g in rallies.groupby("video_id", dropna=False)
    }

    out_rows = []
    dropped = []

    for i, c in clicks.iterrows():
        old_review_id = str(c.get("review_id", ""))
        visibility = str(c.get("visibility", ""))

        if old_review_id not in old_manifest_by_review:
            dropped.append({"reason": "old_review_missing", "old_review_id": old_review_id})
            continue

        old_m = old_manifest_by_review[old_review_id]
        old_clip_id = str(old_m.get("clip_id", ""))

        if old_clip_id not in batch_cfg_by_clip:
            dropped.append({"reason": "batch_cfg_clip_missing", "old_review_id": old_review_id, "old_clip_id": old_clip_id})
            continue

        cfg = batch_cfg_by_clip[old_clip_id]
        src_dataset_clip_id = str(cfg.get("source_dataset_clip_id_004C2", ""))

        old_clip_row = None

        if src_dataset_clip_id and src_dataset_clip_id in old_clips_by_id:
            old_clip_row = old_clips_by_id[src_dataset_clip_id]
        else:
            video_path = norm_path(cfg.get("video_path", ""))
            old_clip_row = old_clips_by_path.get(video_path)

        if old_clip_row is None:
            dropped.append({
                "reason": "old_004C2_clip_missing",
                "old_review_id": old_review_id,
                "old_clip_id": old_clip_id,
                "source_dataset_clip_id_004C2": src_dataset_clip_id,
            })
            continue

        video_id = str(old_clip_row.get("video_id", cfg.get("video_id", "")))
        old_source_start_frame = to_num(pd.Series([old_clip_row.get("start_frame", "")])).iloc[0]

        if pd.isna(old_source_start_frame):
            dropped.append({"reason": "old_source_start_frame_missing", "old_review_id": old_review_id})
            continue

        old_local_source_frame = c.get("source_frame_num")
        if pd.isna(old_local_source_frame):
            dropped.append({"reason": "click_source_frame_missing", "old_review_id": old_review_id})
            continue

        absolute_source_frame = int(round(float(old_source_start_frame) + float(old_local_source_frame)))

        if video_id not in rallies_by_video:
            dropped.append({"reason": "no_rally_video_id", "old_review_id": old_review_id, "video_id": video_id})
            continue

        rg = rallies_by_video[video_id].copy()
        candidates = rg[
            (rg["start_frame_source_num"] <= absolute_source_frame) &
            (rg["end_frame_source_num"] >= absolute_source_frame)
        ].copy()

        if candidates.empty:
            dropped.append({
                "reason": "no_rally_overlap",
                "old_review_id": old_review_id,
                "video_id": video_id,
                "absolute_source_frame": absolute_source_frame,
            })
            continue

        # Si plusieurs rallys se recouvrent, choisir celui où la frame est la plus centrale.
        candidates["center_frame"] = (candidates["start_frame_source_num"] + candidates["end_frame_source_num"]) / 2.0
        candidates["center_dist"] = (candidates["center_frame"] - absolute_source_frame).abs()
        candidates = candidates.sort_values(["center_dist", "duration_sec"])

        rly = candidates.iloc[0]

        rally_start = int(round(float(rly["start_frame_source_num"])))
        rally_local_frame = int(absolute_source_frame - rally_start)
        fps = float(rly.get("fps_num", 50.0) or 50.0)

        # Le serveur 004J/les scripts 004M utilisent source_frame comme frame locale du clip analysé.
        # Ici, pour les rally clips, source_frame = local_frame dans le rally.
        out = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "review_id": str(rly.get("review_id", "")),
            "gold_rank_004I": "",
            "video_id": str(rly.get("video_id", "")),
            "clip_id": str(rly.get("clip_id", "")),
            "segment_idx": str(rly.get("segment_idx", 1)),
            "visibility": visibility,
            "local_time_sec": round(rally_local_frame / fps, 4),
            "local_frame": rally_local_frame,
            "source_frame": rally_local_frame,
            "x": "" if pd.isna(c.get("x_num")) else round(float(c.get("x_num")), 3),
            "y": "" if pd.isna(c.get("y_num")) else round(float(c.get("y_num")), 3),
            "video_w": c.get("video_w", ""),
            "video_h": c.get("video_h", ""),
            "comment": str(c.get("comment", "")),
            "user_action": "reused_from_004J_micro_goldset",
            "version": VERSION,

            "old_review_id_004U2": old_review_id,
            "old_clip_id_004U2": old_clip_id,
            "old_local_frame_004U2": c.get("local_frame", ""),
            "old_segment_source_frame_004U2": c.get("source_frame", ""),
            "old_004C2_clip_id_004U2": src_dataset_clip_id,
            "old_source_start_frame_004U2": int(round(float(old_source_start_frame))),
            "absolute_source_frame_004U2": absolute_source_frame,
            "rally_id_004U2": str(rly.get("rally_id", "")),
            "rally_start_frame_source_004U2": rally_start,
            "rally_end_frame_source_004U2": int(round(float(rly["end_frame_source_num"]))),
        }

        out_rows.append(out)

    out_csv = out_dir / "rally_reused_clicks_004U2.csv"
    dropped_csv = out_dir / "rally_reused_clicks_dropped_004U2.csv"
    out_json = out_dir / "rally_reused_clicks_summary_004U2.json"

    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "created_at", "review_id", "video_id", "clip_id", "visibility", "local_frame", "source_frame", "x", "y"
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if dropped:
        dfields = sorted({k for row in dropped for k in row.keys()})
        with dropped_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dfields)
            writer.writeheader()
            writer.writerows(dropped)
    else:
        dropped_csv.write_text("reason\n", encoding="utf-8")

    out_df = pd.DataFrame(out_rows)

    by_visibility = out_df["visibility"].astype(str).value_counts().to_dict() if not out_df.empty else {}
    by_video = out_df["video_id"].astype(str).value_counts().to_dict() if not out_df.empty else {}
    by_review_top = out_df["review_id"].astype(str).value_counts().head(20).to_dict() if not out_df.empty else {}

    old_ball = int((clicks["visibility"].astype(str) == "ball").sum())
    transferred_ball = int((out_df["visibility"].astype(str) == "ball").sum()) if not out_df.empty else 0

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "reuse_old_manual_clicks_by_projecting_to_long_rally_clips",
        "old_clicks": str(old_clicks_path),
        "old_clicks_total": int(len(clicks)),
        "old_ball_clicks": old_ball,
        "transferred_total": int(len(out_rows)),
        "transferred_ball_clicks": transferred_ball,
        "dropped_total": int(len(dropped)),
        "transfer_rate_total": round(len(out_rows) / max(1, len(clicks)), 4),
        "transfer_rate_ball": round(transferred_ball / max(1, old_ball), 4),
        "by_visibility": by_visibility,
        "by_video": by_video,
        "top_reviews": by_review_top,
        "out_csv": str(out_csv),
        "dropped_csv": str(dropped_csv),
        "next": "Run 004M on this converted clicks CSV with rally_manifest_004T and run-dir .",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004U2 status=OK")
    print("old_clicks_total=", summary["old_clicks_total"])
    print("old_ball_clicks=", summary["old_ball_clicks"])
    print("transferred_total=", summary["transferred_total"])
    print("transferred_ball_clicks=", summary["transferred_ball_clicks"])
    print("dropped_total=", summary["dropped_total"])
    print("transfer_rate_ball=", summary["transfer_rate_ball"])
    print("by_visibility=", json.dumps(by_visibility, ensure_ascii=False))
    print("by_video=", json.dumps(by_video, ensure_ascii=False))
    print("top_reviews=", json.dumps(by_review_top, ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", dropped_csv)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
