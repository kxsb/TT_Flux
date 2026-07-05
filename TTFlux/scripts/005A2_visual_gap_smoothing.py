from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005A2"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def prep_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["frame_num", "x_num", "y_num"]:
        if c not in df.columns:
            raise SystemExit(f"Colonne absente: {c}")

    if "score_num" not in df.columns:
        df["score_num"] = 1.0

    if "tracklet_id_005A" not in df.columns:
        raise SystemExit("Colonne absente: tracklet_id_005A")

    df["frame_num"] = to_num(df["frame_num"])
    df["x_num"] = to_num(df["x_num"])
    df["y_num"] = to_num(df["y_num"])
    df["score_num"] = to_num(df["score_num"]).fillna(1.0)
    df["tracklet_id_005A"] = to_num(df["tracklet_id_005A"]).fillna(0).astype(int)

    df = df.dropna(subset=["frame_num", "x_num", "y_num"]).copy()
    df["frame_num"] = df["frame_num"].astype(int)

    return df


def make_display_points(
    kept: pd.DataFrame,
    max_interp_gap_frames: int,
    max_interp_jump_px: float,
    min_endpoint_score: float,
) -> pd.DataFrame:
    rows = []

    kept = kept.sort_values(["review_id", "tracklet_id_005A", "frame_num"]).copy()

    for _, r in kept.iterrows():
        d = r.to_dict()
        d["display_source_005A2"] = "orig"
        d["is_interp_005A2"] = 0
        d["interp_gap_005A2"] = 0
        rows.append(d)

    for (review_id, tid), g in kept.groupby(["review_id", "tracklet_id_005A"], dropna=False):
        g = g.sort_values("frame_num").copy()
        recs = g.to_dict(orient="records")

        for a, b in zip(recs[:-1], recs[1:]):
            f1 = int(a["frame_num"])
            f2 = int(b["frame_num"])
            gap = f2 - f1

            if gap <= 1 or gap > max_interp_gap_frames:
                continue

            x1 = float(a["x_num"])
            y1 = float(a["y_num"])
            x2 = float(b["x_num"])
            y2 = float(b["y_num"])
            s1 = float(a["score_num"])
            s2 = float(b["score_num"])

            dist = math.hypot(x2 - x1, y2 - y1)

            if dist > max_interp_jump_px:
                continue

            if min(s1, s2) < min_endpoint_score:
                continue

            for f in range(f1 + 1, f2):
                alpha = (f - f1) / max(1, gap)

                d = a.copy()
                d["frame_num"] = int(f)
                d["x_num"] = round(x1 + alpha * (x2 - x1), 3)
                d["y_num"] = round(y1 + alpha * (y2 - y1), 3)
                d["score_num"] = round(min(s1, s2) * 0.97, 6)
                d["display_source_005A2"] = "interp"
                d["is_interp_005A2"] = 1
                d["interp_gap_005A2"] = int(gap)
                rows.append(d)

    out = pd.DataFrame(rows)

    # En cas de doublon frame/tracklet : priorité aux vrais points.
    out["source_rank_005A2"] = np.where(out["display_source_005A2"].eq("orig"), 0, 1)
    out = out.sort_values(
        ["review_id", "tracklet_id_005A", "frame_num", "source_rank_005A2", "score_num"],
        ascending=[True, True, True, True, False],
    )

    out = out.drop_duplicates(
        subset=["review_id", "tracklet_id_005A", "frame_num"],
        keep="first",
    ).copy()

    out = out.sort_values(["review_id", "frame_num", "tracklet_id_005A"]).copy()

    return out


def draw_point(frame, x, y, score, tid, source):
    x = int(round(float(x)))
    y = int(round(float(y)))

    if source == "interp":
        color = (80, 190, 80)
        radius = 4
        label = f"BALL_FILL {score:.2f}"
    else:
        color = (0, 255, 0)
        radius = 6
        label = f"BALL_OK {score:.2f} T{tid}"

    cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)

    if source != "interp":
        cv2.line(frame, (x - 9, y), (x + 9, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - 9), (x, y + 9), color, 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        label,
        (x + 9, max(20, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        2,
        cv2.LINE_AA,
    )


def make_overlay(
    review_id: str,
    clip_path: Path,
    display_points: pd.DataFrame,
    out_path: Path,
    max_video_frames: int,
    trail_frames: int,
    no_ball_grace_frames: int,
) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("WARN open failed", clip_path)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = n if max_video_frames <= 0 else min(n, max_video_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    by_frame = {}
    for _, r in display_points.iterrows():
        by_frame.setdefault(int(r["frame_num"]), []).append(r.to_dict())

    trails = {}
    last_seen = {}
    last_ball_frame = -999999

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        pts = by_frame.get(fidx, [])

        # Ajoute points courants.
        for p in pts:
            tid = int(p["tracklet_id_005A"])
            x = float(p["x_num"])
            y = float(p["y_num"])
            score = float(p["score_num"])
            src = str(p.get("display_source_005A2", "orig"))

            trails.setdefault(tid, []).append((fidx, x, y, score, src))
            trails[tid] = trails[tid][-trail_frames:]
            last_seen[tid] = fidx
            last_ball_frame = fidx

        # Supprime vieux trails.
        for tid in list(trails.keys()):
            if fidx - last_seen.get(tid, -999999) > trail_frames:
                trails.pop(tid, None)
                last_seen.pop(tid, None)

        # Dessin par tracklet uniquement.
        for tid, tr in trails.items():
            if len(tr) >= 2:
                for a, b in zip(tr[:-1], tr[1:]):
                    fa, ax, ay, _, _ = a
                    fb, bx, by, _, _ = b

                    if fb - fa > trail_frames:
                        continue

                    cv2.line(
                        frame,
                        (int(round(ax)), int(round(ay))),
                        (int(round(bx)), int(round(by))),
                        (0, 210, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if tr:
                _, x, y, score, src = tr[-1]
                draw_point(frame, x, y, score, tid, src)

        # Hystérésis d'affichage : ne pas afficher NO_BALL rouge dès le premier trou.
        if pts:
            state = "BALL_OK"
            color = (0, 255, 0)
        elif fidx - last_ball_frame <= no_ball_grace_frames:
            state = "BALL_HOLD"
            color = (80, 180, 80)
        else:
            state = "NO_BALL"
            color = (90, 90, 180)

        cv2.putText(
            frame,
            f"{review_id} f={fidx} {state}",
            (22, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "005A2 display smoothing: short-gap fill, hysteresis, no cross-tracklet links",
            (22, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept-points", required=True)
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--max-interp-gap-frames", type=int, default=6)
    ap.add_argument("--max-interp-jump-px", type=float, default=70.0)
    ap.add_argument("--min-endpoint-score", type=float, default=0.80)

    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--max-video-frames", type=int, default=3000)
    ap.add_argument("--trail-frames", type=int, default=12)
    ap.add_argument("--no-ball-grace-frames", type=int, default=5)

    args = ap.parse_args()

    root = Path.cwd()

    kept_path = Path(args.kept_points)
    review_path = Path(args.review_summary)
    out_dir = Path(args.out_dir)

    if not kept_path.is_absolute():
        kept_path = root / kept_path
    if not review_path.is_absolute():
        review_path = root / review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    kept = pd.read_csv(kept_path).fillna("")
    reviews = pd.read_csv(review_path).fillna("")

    kept = prep_points(kept)

    display = make_display_points(
        kept,
        max_interp_gap_frames=args.max_interp_gap_frames,
        max_interp_jump_px=args.max_interp_jump_px,
        min_endpoint_score=args.min_endpoint_score,
    )

    review_rows = []

    for review_id, g in display.groupby("review_id", dropna=False):
        review_id = str(review_id)
        orig_n = int((g["display_source_005A2"] == "orig").sum())
        interp_n = int((g["display_source_005A2"] == "interp").sum())
        display_n = int(len(g))

        raw = reviews[reviews["review_id"].astype(str).eq(review_id)]
        base = raw.iloc[0].to_dict() if len(raw) else {}

        review_rows.append({
            "review_id": review_id,
            "video_id": str(base.get("video_id", "")),
            "rally_id": str(base.get("rally_id", "")),
            "orig_points_005A": orig_n,
            "interp_points_005A2": interp_n,
            "display_points_005A2": display_n,
            "interp_ratio_005A2": round(interp_n / max(1, display_n), 4),
            "tracklets_005A": int(g["tracklet_id_005A"].nunique()),
            "first_frame": int(g["frame_num"].min()) if display_n else "",
            "last_frame": int(g["frame_num"].max()) if display_n else "",
            "clip_path": str(base.get("clip_path", "")),
            "overlay_005A2": "",
        })

    review_out = pd.DataFrame(review_rows)

    overlays = []

    if args.make_video:
        viz = review_out.sort_values(
            ["display_points_005A2", "interp_points_005A2"],
            ascending=[False, False],
        ).head(args.max_videos).copy()

        for i, (_, r) in enumerate(viz.iterrows(), start=1):
            review_id = str(r["review_id"])
            clip_path = Path(str(r.get("clip_path", "")))

            if not clip_path.is_absolute():
                clip_path = root / clip_path

            if not clip_path.is_file():
                print("WARN missing clip", review_id, clip_path)
                continue

            pts = display[display["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{i:03d}_{review_id}_005A2_smooth_overlay.mp4"

            ok = make_overlay(
                review_id=review_id,
                clip_path=clip_path,
                display_points=pts,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
                trail_frames=args.trail_frames,
                no_ball_grace_frames=args.no_ball_grace_frames,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})
                review_out.loc[
                    review_out["review_id"].astype(str).eq(review_id),
                    "overlay_005A2",
                ] = str(out_video)

                print(f"overlay {i}/{len(viz)} {review_id} -> {out_video}")

    out_display = out_dir / "display_points_005A2.csv"
    out_reviews = out_dir / "display_review_summary_005A2.csv"
    out_json = out_dir / "display_smoothing_summary_005A2.json"

    display.to_csv(out_display, index=False, encoding="utf-8")
    review_out.to_csv(out_reviews, index=False, encoding="utf-8")

    orig_total = int((display["display_source_005A2"] == "orig").sum())
    interp_total = int((display["display_source_005A2"] == "interp").sum())

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "visual_short_gap_interpolation_after_005A_without_changing_raw_tracking_truth",
        "kept_points": str(kept_path),
        "review_summary": str(review_path),
        "params": {
            "max_interp_gap_frames": args.max_interp_gap_frames,
            "max_interp_jump_px": args.max_interp_jump_px,
            "min_endpoint_score": args.min_endpoint_score,
            "trail_frames": args.trail_frames,
            "no_ball_grace_frames": args.no_ball_grace_frames,
        },
        "reviews": int(review_out["review_id"].nunique()) if len(review_out) else 0,
        "orig_points_005A": orig_total,
        "interp_points_005A2": interp_total,
        "display_points_005A2": int(len(display)),
        "interp_ratio_005A2": round(interp_total / max(1, len(display)), 4),
        "overlays": overlays,
        "outputs": {
            "display_points": str(out_display),
            "review_summary": str(out_reviews),
        },
        "next": "If visual losses are reduced, keep 005A raw + 005A2 display. Then move to 005B camera cut detection."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005A2 status=OK")
    print("reviews=", summary["reviews"])
    print("orig_points_005A=", summary["orig_points_005A"])
    print("interp_points_005A2=", summary["interp_points_005A2"])
    print("display_points_005A2=", summary["display_points_005A2"])
    print("interp_ratio_005A2=", summary["interp_ratio_005A2"])
    print("overlays=", len(overlays))
    print("wrote", out_display)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
