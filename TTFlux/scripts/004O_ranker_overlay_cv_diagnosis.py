from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "004O"


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def draw_cross(img, x, y, color, label):
    if x == "" or y == "" or pd.isna(x) or pd.isna(y):
        return

    x = int(round(float(x)))
    y = int(round(float(y)))

    cv2.circle(img, (x, y), 8, color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 14, y), (x + 14, y), color, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 14), (x, y + 14), color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 10, max(22, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


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


def top_candidate(g: pd.DataFrame, score_col: str):
    if g.empty or score_col not in g.columns:
        return None

    gg = g.copy()
    gg[score_col] = to_num(gg[score_col]).fillna(-1e18)
    gg = gg.sort_values(score_col, ascending=False)

    return gg.iloc[0].to_dict()


def dist(a, b, x, y):
    if a is None:
        return None
    return float(math.hypot(float(a["cand_x"]) - x, float(a["cand_y"]) - y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-scores", default="runs/ball_ranker_004N/candidate_ranker_cv_scores_004N.csv")
    ap.add_argument("--manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/ranker_overlay_004O")
    ap.add_argument("--max-reviews", type=int, default=20)
    ap.add_argument("--max-frames-per-review", type=int, default=16)
    args = ap.parse_args()

    root = Path.cwd()

    cv_path = Path(args.cv_scores)
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    if not cv_path.is_absolute():
        cv_path = root / cv_path
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

    if not cv_path.is_file():
        raise SystemExit(f"CV scores absent: {cv_path}")

    cv = pd.read_csv(cv_path).fillna("")
    manifest = pd.read_csv(manifest_path).fillna("")
    manifest_by_review = {str(r["review_id"]): r for _, r in manifest.iterrows()}

    required = {"review_id", "local_frame", "click_x", "click_y", "baseline_score_004N", "model_score_004N"}
    missing = required - set(cv.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes dans CV scores: {sorted(missing)}")

    cv["local_frame_num"] = to_num(cv["local_frame"])
    cv["click_x_num"] = to_num(cv["click_x"])
    cv["click_y_num"] = to_num(cv["click_y"])
    cv = cv.dropna(subset=["local_frame_num", "click_x_num", "click_y_num"]).copy()
    cv["local_frame_num"] = cv["local_frame_num"].astype(int)

    # Un clic = review_id + local_frame + click x/y.
    cv["click_key_004O"] = (
        cv["review_id"].astype(str)
        + "|"
        + cv["local_frame_num"].astype(str)
        + "|"
        + cv["click_x_num"].round(2).astype(str)
        + "|"
        + cv["click_y_num"].round(2).astype(str)
    )

    click_meta = (
        cv.groupby("click_key_004O", dropna=False)
        .agg(
            review_id=("review_id", "first"),
            local_frame=("local_frame_num", "first"),
            click_x=("click_x_num", "first"),
            click_y=("click_y_num", "first"),
        )
        .reset_index()
    )

    review_counts = click_meta.groupby("review_id").size().sort_values(ascending=False)
    review_ids = review_counts.head(args.max_reviews).index.astype(str).tolist()

    rows = []
    review_rows = []

    print("004O reviews=", len(review_ids))

    for review_i, review_id in enumerate(review_ids, start=1):
        if review_id not in manifest_by_review:
            continue

        clicks = click_meta[click_meta["review_id"].astype(str).eq(review_id)].copy()
        clicks = clicks.sort_values("local_frame")

        if len(clicks) > args.max_frames_per_review:
            idxs = np.linspace(0, len(clicks) - 1, args.max_frames_per_review).round().astype(int)
            clicks = clicks.iloc[idxs].copy()

        mrow = manifest_by_review[review_id]
        mp4_rel = str(mrow.get("mp4", ""))
        video_path = run_dir / mp4_rel

        images = []
        baseline_dists = []
        model_dists = []

        review_dir = frames_dir / review_id
        review_dir.mkdir(parents=True, exist_ok=True)

        for j, (_, click) in enumerate(clicks.iterrows(), start=1):
            key = str(click["click_key_004O"])
            local_frame = int(click["local_frame"])
            hx = float(click["click_x"])
            hy = float(click["click_y"])

            frame = extract_frame(video_path, local_frame)
            if frame is None:
                continue

            g = cv[cv["click_key_004O"].astype(str).eq(key)].copy()

            baseline = top_candidate(g, "baseline_score_004N")
            model = top_candidate(g, "model_score_004N")

            bd = dist(baseline, None, hx, hy)
            md = dist(model, None, hx, hy)

            if bd is not None:
                baseline_dists.append(bd)
            if md is not None:
                model_dists.append(md)

            draw_cross(frame, hx, hy, (0, 255, 0), "HUMAN")

            if baseline is not None:
                draw_cross(frame, baseline["cand_x"], baseline["cand_y"], (0, 0, 255), f"BASE {bd:.1f}px")
                cv2.line(
                    frame,
                    (int(round(hx)), int(round(hy))),
                    (int(round(float(baseline["cand_x"]))), int(round(float(baseline["cand_y"])))),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if model is not None:
                draw_cross(frame, model["cand_x"], model["cand_y"], (255, 120, 0), f"MODEL {md:.1f}px")
                cv2.line(
                    frame,
                    (int(round(hx)), int(round(hy))),
                    (int(round(float(model["cand_x"]))), int(round(float(model["cand_y"])))),
                    (255, 120, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"{review_id} f={local_frame} green=human red=baseline blue=model",
                (24, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_img = review_dir / f"{review_id}_{j:03d}_f{local_frame}.jpg"
            imwrite_unicode(out_img, frame)
            images.append(frame)

            rows.append({
                "review_id": review_id,
                "local_frame": local_frame,
                "human_x": round(hx, 3),
                "human_y": round(hy, 3),
                "baseline_x": round(float(baseline["cand_x"]), 3) if baseline is not None else "",
                "baseline_y": round(float(baseline["cand_y"]), 3) if baseline is not None else "",
                "baseline_dist": round(float(bd), 3) if bd is not None else "",
                "model_x": round(float(model["cand_x"]), 3) if model is not None else "",
                "model_y": round(float(model["cand_y"]), 3) if model is not None else "",
                "model_dist": round(float(md), 3) if md is not None else "",
                "image": str(out_img),
            })

        sheet = make_contact_sheet(images, cols=4, thumb_w=480)
        sheet_path = ""

        if sheet is not None:
            sheet_file = sheets_dir / f"{review_i:02d}_{review_id}_ranker_sheet.jpg"
            imwrite_unicode(sheet_file, sheet)
            sheet_path = str(sheet_file)

        def med(vals):
            return round(float(np.median(vals)), 3) if vals else None

        def hit(vals, th):
            return round(float((np.array(vals) <= th).mean()), 4) if vals else None

        review_rows.append({
            "review_id": review_id,
            "frames": int(len(images)),
            "baseline_dist_med": med(baseline_dists),
            "model_dist_med": med(model_dists),
            "baseline_hit50": hit(baseline_dists, 50),
            "model_hit50": hit(model_dists, 50),
            "baseline_hit20": hit(baseline_dists, 20),
            "model_hit20": hit(model_dists, 20),
            "sheet": sheet_path,
        })

        print(
            f"  {review_i}/{len(review_ids)} {review_id} "
            f"base50={hit(baseline_dists,50)} model50={hit(model_dists,50)}"
        )

    out_rows = out_dir / "ranker_overlay_frames_004O.csv"
    out_reviews = out_dir / "ranker_overlay_reviews_004O.csv"
    out_json = out_dir / "ranker_overlay_summary_004O.json"

    pd.DataFrame(rows).to_csv(out_rows, index=False, encoding="utf-8")
    pd.DataFrame(review_rows).to_csv(out_reviews, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "honest_cv_visual_overlay_manual_goldset",
        "reviews": len(review_rows),
        "frames": len(rows),
        "colors": {
            "green": "human clicked ball",
            "red": "baseline top candidate",
            "blue": "004N CV model top candidate"
        },
        "sheets_dir": str(sheets_dir),
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("004O status=OK")
    print("reviews=", len(review_rows))
    print("frames=", len(rows))
    print("sheets_dir=", sheets_dir)
    print("wrote", out_rows)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
