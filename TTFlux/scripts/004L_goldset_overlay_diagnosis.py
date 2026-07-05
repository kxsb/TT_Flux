from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004L"


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def nearest_raw(raw_clip: pd.DataFrame, frame: int, x: float, y: float, frame_window: int):
    if raw_clip.empty:
        return None

    sub = raw_clip[
        (raw_clip["frame_num"] >= frame - frame_window) &
        (raw_clip["frame_num"] <= frame + frame_window)
    ].copy()

    if sub.empty:
        return None

    sub["dist"] = np.sqrt((sub["x_num"] - x) ** 2 + (sub["y_num"] - y) ** 2)
    sub["frame_delta_abs"] = (sub["frame_num"] - frame).abs()
    sub = sub.sort_values(["dist", "frame_delta_abs"])
    return sub.iloc[0].to_dict()


def draw_cross(img, x, y, color, label):
    x = int(round(float(x)))
    y = int(round(float(y)))

    cv2.circle(img, (x, y), 8, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 14, y), (x + 14, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 14), (x, y + 14), color, 2, cv2.LINE_AA)

    cv2.putText(
        img,
        label,
        (x + 10, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def extract_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def make_contact_sheet(images: list[np.ndarray], cols: int = 4, thumb_w: int = 480):
    if not images:
        return None

    thumbs = []
    for img in images:
        h, w = img.shape[:2]
        scale = thumb_w / max(1, w)
        thumb_h = int(round(h * scale))
        t = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(t)

    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)

    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))

    return np.vstack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", default="runs/ball_goldset_004J/ball_clicks_004J.csv")
    ap.add_argument("--raw-points", default="runs/dataset_points_004E_full240/raw_tracking_points_004E.csv")
    ap.add_argument("--manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/goldset_overlay_004L")
    ap.add_argument("--frame-window", type=int, default=2)
    ap.add_argument("--max-frames-per-review", type=int, default=16)
    ap.add_argument("--max-reviews", type=int, default=20)
    args = ap.parse_args()

    root = Path.cwd()

    clicks_path = Path(args.clicks)
    raw_path = Path(args.raw_points)
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    frames_dir = out_dir / "frames"
    sheets_dir = out_dir / "sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    clicks = pd.read_csv(clicks_path).fillna("")
    clicks = clicks[clicks["visibility"].astype(str).eq("ball")].copy()

    for c in ["x", "y", "source_frame", "local_frame"]:
        clicks[c] = to_num(clicks[c])

    clicks = clicks.dropna(subset=["x", "y", "source_frame", "local_frame"])
    clicks["source_frame"] = clicks["source_frame"].astype(int)
    clicks["local_frame"] = clicks["local_frame"].astype(int)

    raw = pd.read_csv(raw_path)
    raw["clip_id"] = raw["clip_id"].astype(str)
    raw["frame_num"] = to_num(raw["frame"]).astype("Int64")
    raw["x_num"] = to_num(raw["x"])
    raw["y_num"] = to_num(raw["y"])
    raw = raw.dropna(subset=["frame_num", "x_num", "y_num"])
    raw["frame_num"] = raw["frame_num"].astype(int)

    raw_by_clip = {str(k): g.copy() for k, g in raw.groupby("clip_id", dropna=False)}

    manifest = pd.read_csv(manifest_path).fillna("")
    manifest_by_review = {str(r["review_id"]): r for _, r in manifest.iterrows()}

    review_counts = clicks.groupby("review_id").size().sort_values(ascending=False)
    review_ids = review_counts.head(args.max_reviews).index.astype(str).tolist()

    rows = []
    summary_reviews = []

    print("004L reviews=", len(review_ids))

    for review_i, review_id in enumerate(review_ids, start=1):
        g = clicks[clicks["review_id"].astype(str).eq(review_id)].copy()
        g = g.sort_values("local_frame")

        if len(g) > args.max_frames_per_review:
            idxs = np.linspace(0, len(g) - 1, args.max_frames_per_review).round().astype(int)
            g = g.iloc[idxs].copy()

        if review_id not in manifest_by_review:
            continue

        mrow = manifest_by_review[review_id]
        mp4_rel = str(mrow.get("mp4", ""))
        video_path = run_dir / mp4_rel

        clip_id = str(mrow.get("clip_id", ""))
        raw_clip = raw_by_clip.get(clip_id, pd.DataFrame())

        images = []
        distances = []

        review_dir = frames_dir / review_id
        review_dir.mkdir(parents=True, exist_ok=True)

        for j, (_, c) in enumerate(g.iterrows(), start=1):
            local_frame = int(c["local_frame"])
            source_frame = int(c["source_frame"])
            x = float(c["x"])
            y = float(c["y"])

            frame = extract_frame(video_path, local_frame)
            if frame is None:
                continue

            near = nearest_raw(raw_clip, source_frame, x, y, args.frame_window)

            draw_cross(frame, x, y, (0, 255, 0), "HUMAN BALL")

            dist = None

            if near is not None:
                rx = float(near["x_num"])
                ry = float(near["y_num"])
                dist = float(math.hypot(rx - x, ry - y))
                distances.append(dist)

                color = (0, 0, 255) if dist > 50 else (0, 200, 255)
                draw_cross(frame, rx, ry, color, f"004E {dist:.1f}px")

                cv2.line(
                    frame,
                    (int(round(x)), int(round(y))),
                    (int(round(rx)), int(round(ry))),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "004E: no raw point near frame",
                    (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"{review_id} local_f={local_frame} source_f={source_frame}",
                (24, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_img = review_dir / f"{review_id}_{j:03d}_lf{local_frame}_sf{source_frame}.jpg"
            imwrite_unicode(out_img, frame)
            images.append(frame)

            rows.append({
                "review_id": review_id,
                "clip_id": clip_id,
                "video_id": str(mrow.get("video_id", "")),
                "local_frame": local_frame,
                "source_frame": source_frame,
                "human_x": round(x, 3),
                "human_y": round(y, 3),
                "raw_found": near is not None,
                "raw_x": round(float(near["x_num"]), 3) if near is not None else "",
                "raw_y": round(float(near["y_num"]), 3) if near is not None else "",
                "raw_dist": round(dist, 3) if dist is not None else "",
                "image": str(out_img),
            })

        sheet = make_contact_sheet(images, cols=4, thumb_w=480)
        sheet_path = ""

        if sheet is not None:
            sheet_file = sheets_dir / f"{review_i:02d}_{review_id}_sheet.jpg"
            imwrite_unicode(sheet_file, sheet)
            sheet_path = str(sheet_file)

        med = round(float(np.median(distances)), 3) if distances else None
        hit50 = round(float((np.array(distances) <= 50).mean()), 4) if distances else None

        summary_reviews.append({
            "review_id": review_id,
            "clip_id": clip_id,
            "clicks_sampled": int(len(g)),
            "raw_found_frames": int(len(distances)),
            "raw_dist_med": med,
            "raw_hit50": hit50,
            "sheet": sheet_path,
        })

        print(f"  {review_i}/{len(review_ids)} {review_id} sampled={len(g)} med={med} hit50={hit50}")

    out_rows = out_dir / "goldset_overlay_frames_004L.csv"
    out_reviews = out_dir / "goldset_overlay_reviews_004L.csv"
    out_json = out_dir / "goldset_overlay_summary_004L.json"

    pd.DataFrame(rows).to_csv(out_rows, index=False, encoding="utf-8")
    pd.DataFrame(summary_reviews).to_csv(out_reviews, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "manual_gold_click_visual_diagnosis_only",
        "reviews": len(summary_reviews),
        "frames": len(rows),
        "out_frames": str(out_rows),
        "out_reviews": str(out_reviews),
        "sheets_dir": str(sheets_dir),
        "interpretation": {
            "green": "human clicked real ball",
            "red_or_yellow": "nearest 004E raw point",
            "white_line": "distance between human ball and 004E point"
        }
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004L status=OK")
    print("reviews=", len(summary_reviews))
    print("frames=", len(rows))
    print("sheets_dir=", sheets_dir)
    print("wrote", out_rows)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
