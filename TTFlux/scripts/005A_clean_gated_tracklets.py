from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005A"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def ensure_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "frame_num" not in df.columns:
        if "frame" in df.columns:
            df["frame_num"] = to_num(df["frame"])
        elif "local_frame" in df.columns:
            df["frame_num"] = to_num(df["local_frame"])
        else:
            raise SystemExit("Impossible de trouver frame_num/frame/local_frame dans gated_points.")

    if "x_num" not in df.columns:
        if "x" in df.columns:
            df["x_num"] = to_num(df["x"])
        else:
            raise SystemExit("Impossible de trouver x_num/x dans gated_points.")

    if "y_num" not in df.columns:
        if "y" in df.columns:
            df["y_num"] = to_num(df["y"])
        else:
            raise SystemExit("Impossible de trouver y_num/y dans gated_points.")

    if "score_num" not in df.columns:
        if "score" in df.columns:
            df["score_num"] = to_num(df["score"])
        elif "path_score" in df.columns:
            df["score_num"] = to_num(df["path_score"])
        else:
            df["score_num"] = 1.0

    df["frame_num"] = to_num(df["frame_num"])
    df["x_num"] = to_num(df["x_num"])
    df["y_num"] = to_num(df["y_num"])
    df["score_num"] = to_num(df["score_num"]).fillna(1.0)

    df = df.dropna(subset=["frame_num", "x_num", "y_num"]).copy()
    df["frame_num"] = df["frame_num"].astype(int)

    return df


def split_tracklets(
    g: pd.DataFrame,
    max_gap_frames: int,
    max_jump_px: float,
    max_speed_px_frame: float,
    angle_split_deg: float,
    min_speed_for_angle: float,
) -> pd.DataFrame:
    g = g.sort_values("frame_num").copy()

    clean_id = 0
    ids = []
    reasons = []

    prev = None
    prev_v = None

    for _, r in g.iterrows():
        f = int(r["frame_num"])
        x = float(r["x_num"])
        y = float(r["y_num"])

        reason = "continue"

        if prev is None:
            clean_id += 1
            reason = "start"
            prev_v = None
        else:
            pf, px, py = prev
            dt = max(1, f - pf)
            dx = x - px
            dy = y - py
            dist = math.hypot(dx, dy)
            speed = dist / dt

            split = False

            if f <= pf:
                split = True
                reason = "non_monotonic_frame"
            elif dt > max_gap_frames:
                split = True
                reason = f"gap>{max_gap_frames}"
            elif dist > max_jump_px:
                split = True
                reason = f"jump>{max_jump_px}"
            elif speed > max_speed_px_frame:
                split = True
                reason = f"speed>{max_speed_px_frame}"
            elif prev_v is not None:
                pvx, pvy = prev_v
                pnorm = math.hypot(pvx, pvy)
                cnorm = math.hypot(dx, dy)

                if pnorm >= min_speed_for_angle and cnorm >= min_speed_for_angle:
                    dot = (pvx * dx + pvy * dy) / max(1e-9, pnorm * cnorm)
                    dot = max(-1.0, min(1.0, dot))
                    angle = math.degrees(math.acos(dot))

                    # On reste prudent : les rebonds peuvent vraiment changer la direction.
                    if angle >= angle_split_deg and dist > max_jump_px * 0.45:
                        split = True
                        reason = f"angle>{angle_split_deg}"

            if split:
                clean_id += 1
                prev_v = None
            else:
                prev_v = (dx, dy)

        ids.append(clean_id)
        reasons.append(reason)
        prev = (f, x, y)

    g["tracklet_id_005A_raw"] = ids
    g["split_reason_005A"] = reasons

    return g


def remove_weak_tracklets(
    g: pd.DataFrame,
    min_tracklet_points: int,
    min_tracklet_span: int,
    keep_score_min: float,
) -> pd.DataFrame:
    if g.empty:
        return g

    rows = []

    stats = []

    for tid, t in g.groupby("tracklet_id_005A_raw", dropna=False):
        frames = t["frame_num"].astype(int)
        span = int(frames.max() - frames.min()) if len(frames) else 0
        n = int(len(t))
        score_max = float(to_num(t["score_num"]).max()) if len(t) else 0.0
        score_med = float(to_num(t["score_num"]).median()) if len(t) else 0.0

        keep = (
            n >= min_tracklet_points
            and span >= min_tracklet_span
        ) or (
            n >= 2 and score_max >= keep_score_min
        )

        reason = "keep" if keep else f"drop_weak:n={n},span={span},score_max={score_max:.3f}"

        tmp = t.copy()
        tmp["tracklet_keep_005A"] = int(keep)
        tmp["tracklet_drop_reason_005A"] = reason
        tmp["tracklet_points_005A"] = n
        tmp["tracklet_span_005A"] = span
        tmp["tracklet_score_med_005A"] = round(score_med, 6)
        tmp["tracklet_score_max_005A"] = round(score_max, 6)
        rows.append(tmp)

        stats.append({
            "raw_tracklet": int(tid),
            "n": n,
            "span": span,
            "score_med": round(score_med, 6),
            "score_max": round(score_max, 6),
            "keep": int(keep),
        })

    out = pd.concat(rows, ignore_index=True) if rows else g.copy()

    kept_ids = []
    remap = {}
    next_id = 0

    for tid in out["tracklet_id_005A_raw"].tolist():
        if tid not in remap:
            if int(out.loc[out["tracklet_id_005A_raw"].eq(tid), "tracklet_keep_005A"].iloc[0]) == 1:
                next_id += 1
                remap[tid] = next_id
            else:
                remap[tid] = 0

        kept_ids.append(remap[tid])

    out["tracklet_id_005A"] = kept_ids
    out["state_005A"] = np.where(out["tracklet_keep_005A"].astype(int).eq(1), "BALL_OK", "DROP")

    return out


def clean_all(
    gated: pd.DataFrame,
    max_gap_frames: int,
    max_jump_px: float,
    max_speed_px_frame: float,
    angle_split_deg: float,
    min_speed_for_angle: float,
    min_tracklet_points: int,
    min_tracklet_span: int,
    keep_score_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []

    for review_id, g in gated.groupby("review_id", dropna=False):
        g = split_tracklets(
            g,
            max_gap_frames=max_gap_frames,
            max_jump_px=max_jump_px,
            max_speed_px_frame=max_speed_px_frame,
            angle_split_deg=angle_split_deg,
            min_speed_for_angle=min_speed_for_angle,
        )

        g = remove_weak_tracklets(
            g,
            min_tracklet_points=min_tracklet_points,
            min_tracklet_span=min_tracklet_span,
            keep_score_min=keep_score_min,
        )

        parts.append(g)

    cleaned_all = pd.concat(parts, ignore_index=True) if parts else gated.copy()
    kept = cleaned_all[cleaned_all["state_005A"].eq("BALL_OK")].copy()

    review_rows = []

    all_reviews = sorted(cleaned_all["review_id"].astype(str).unique().tolist())

    for review_id in all_reviews:
        g_all = cleaned_all[cleaned_all["review_id"].astype(str).eq(review_id)].copy()
        g_keep = kept[kept["review_id"].astype(str).eq(review_id)].copy()

        raw_points = int(len(g_all))
        kept_points = int(len(g_keep))
        dropped_points = raw_points - kept_points

        if kept_points:
            tracklets = int(g_keep["tracklet_id_005A"].nunique())
            longest = int(g_keep.groupby("tracklet_id_005A").size().max())
            first_frame = int(g_keep["frame_num"].min())
            last_frame = int(g_keep["frame_num"].max())
            score_med = round(float(g_keep["score_num"].median()), 6)
            score_p10 = round(float(g_keep["score_num"].quantile(0.10)), 6)
        else:
            tracklets = 0
            longest = 0
            first_frame = ""
            last_frame = ""
            score_med = ""
            score_p10 = ""

        sample = g_all.iloc[0].to_dict() if len(g_all) else {}

        review_rows.append({
            "review_id": review_id,
            "video_id": str(sample.get("video_id", "")),
            "rally_id": str(sample.get("rally_id", "")),
            "raw_points_004Y": raw_points,
            "kept_points_005A": kept_points,
            "dropped_points_005A": dropped_points,
            "keep_ratio_005A": round(kept_points / max(1, raw_points), 4),
            "tracklets_005A": tracklets,
            "longest_tracklet_points_005A": longest,
            "first_frame_005A": first_frame,
            "last_frame_005A": last_frame,
            "score_med_005A": score_med,
            "score_p10_005A": score_p10,
        })

    return cleaned_all, pd.DataFrame(review_rows)


def draw_marker(frame, x, y, score, tracklet_id):
    x = int(round(float(x)))
    y = int(round(float(y)))

    cv2.circle(frame, (x, y), 6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"BALL_OK {score:.2f} T{tracklet_id}",
        (x + 10, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def make_clean_overlay(
    review_id: str,
    clip_path: Path,
    points: pd.DataFrame,
    out_path: Path,
    max_video_frames: int,
    fade_frames: int,
) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("WARN open failed", clip_path)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = frame_count if max_video_frames <= 0 else min(frame_count, max_video_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    by_frame = {}
    for _, r in points.iterrows():
        by_frame.setdefault(int(r["frame_num"]), []).append(r.to_dict())

    trail_by_tracklet = {}
    last_seen_frame = {}

    fidx = 0

    while fidx < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        current_points = by_frame.get(fidx, [])

        # Nettoyage des trails trop vieux.
        for tid in list(trail_by_tracklet.keys()):
            if fidx - last_seen_frame.get(tid, -999999) > fade_frames:
                trail_by_tracklet.pop(tid, None)
                last_seen_frame.pop(tid, None)

        # Ajout points courants.
        for p in current_points:
            tid = int(p["tracklet_id_005A"])
            x = float(p["x_num"])
            y = float(p["y_num"])
            score = float(p["score_num"])

            trail_by_tracklet.setdefault(tid, []).append((fidx, x, y, score))
            trail_by_tracklet[tid] = trail_by_tracklet[tid][-fade_frames:]
            last_seen_frame[tid] = fidx

        # Dessin trails, sans jamais relier deux tracklets différents.
        for tid, trail in trail_by_tracklet.items():
            if len(trail) >= 2:
                for a, b in zip(trail[:-1], trail[1:]):
                    fa, ax, ay, _ = a
                    fb, bx, by, _ = b

                    # Pas de ligne si trou visuel important.
                    if fb - fa > fade_frames:
                        continue

                    cv2.line(
                        frame,
                        (int(round(ax)), int(round(ay))),
                        (int(round(bx)), int(round(by))),
                        (0, 210, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if trail:
                _, x, y, score = trail[-1]
                draw_marker(frame, x, y, score, tid)

        if current_points:
            state = "BALL_OK"
            state_color = (0, 255, 0)
        else:
            # NO_BALL discret : on ne stroboscope plus en énorme.
            state = "NO_BALL"
            state_color = (80, 80, 220)

        cv2.putText(
            frame,
            f"{review_id} f={fidx} {state}",
            (22, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            state_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "005A clean: split jumps/gaps, no cross-tracklet links",
            (22, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated-points", required=True)
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--max-gap-frames", type=int, default=8)
    ap.add_argument("--max-jump-px", type=float, default=95.0)
    ap.add_argument("--max-speed-px-frame", type=float, default=38.0)
    ap.add_argument("--angle-split-deg", type=float, default=155.0)
    ap.add_argument("--min-speed-for-angle", type=float, default=18.0)
    ap.add_argument("--min-tracklet-points", type=int, default=3)
    ap.add_argument("--min-tracklet-span", type=int, default=2)
    ap.add_argument("--keep-score-min", type=float, default=0.92)

    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--max-video-frames", type=int, default=3000)
    ap.add_argument("--fade-frames", type=int, default=10)

    args = ap.parse_args()

    root = Path.cwd()

    gated_path = Path(args.gated_points)
    review_path = Path(args.review_summary)
    out_dir = Path(args.out_dir)

    if not gated_path.is_absolute():
        gated_path = root / gated_path
    if not review_path.is_absolute():
        review_path = root / review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    gated = pd.read_csv(gated_path).fillna("")
    reviews = pd.read_csv(review_path).fillna("")

    gated = ensure_num_cols(gated)

    cleaned_all, clean_summary = clean_all(
        gated,
        max_gap_frames=args.max_gap_frames,
        max_jump_px=args.max_jump_px,
        max_speed_px_frame=args.max_speed_px_frame,
        angle_split_deg=args.angle_split_deg,
        min_speed_for_angle=args.min_speed_for_angle,
        min_tracklet_points=args.min_tracklet_points,
        min_tracklet_span=args.min_tracklet_span,
        keep_score_min=args.keep_score_min,
    )

    kept = cleaned_all[cleaned_all["state_005A"].eq("BALL_OK")].copy()

    # Ajoute chemins clip depuis review summary si possible.
    if "review_id" in reviews.columns:
        extra_cols = [c for c in ["review_id", "clip_path", "overlay", "video_id", "rally_id"] if c in reviews.columns]
        if "clip_path" in extra_cols:
            clean_summary = clean_summary.merge(
                reviews[extra_cols].drop_duplicates("review_id"),
                on="review_id",
                how="left",
                suffixes=("", "_review"),
            )

    overlays = []

    if args.make_video:
        viz = clean_summary.sort_values(
            ["kept_points_005A", "longest_tracklet_points_005A"],
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

            pts = kept[kept["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{i:03d}_{review_id}_005A_clean_overlay.mp4"

            ok = make_clean_overlay(
                review_id=review_id,
                clip_path=clip_path,
                points=pts,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
                fade_frames=args.fade_frames,
            )

            if ok:
                overlays.append({
                    "review_id": review_id,
                    "overlay": str(out_video),
                })
                clean_summary.loc[
                    clean_summary["review_id"].astype(str).eq(review_id),
                    "overlay_005A",
                ] = str(out_video)

                print(f"overlay {i}/{len(viz)} {review_id} -> {out_video}")

    out_all = out_dir / "tracklets_clean_all_points_005A.csv"
    out_kept = out_dir / "tracklets_clean_kept_points_005A.csv"
    out_reviews = out_dir / "tracklets_clean_review_summary_005A.csv"
    out_json = out_dir / "tracklets_clean_summary_005A.json"

    cleaned_all.to_csv(out_all, index=False, encoding="utf-8")
    kept.to_csv(out_kept, index=False, encoding="utf-8")
    clean_summary.to_csv(out_reviews, index=False, encoding="utf-8")

    raw_points = int(len(cleaned_all))
    kept_points = int(len(kept))

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "clean_gated_ball_tracklets_without_rerunning_ranker",
        "gated_points": str(gated_path),
        "review_summary": str(review_path),
        "params": {
            "max_gap_frames": args.max_gap_frames,
            "max_jump_px": args.max_jump_px,
            "max_speed_px_frame": args.max_speed_px_frame,
            "angle_split_deg": args.angle_split_deg,
            "min_speed_for_angle": args.min_speed_for_angle,
            "min_tracklet_points": args.min_tracklet_points,
            "min_tracklet_span": args.min_tracklet_span,
            "keep_score_min": args.keep_score_min,
            "fade_frames": args.fade_frames,
        },
        "reviews": int(clean_summary["review_id"].nunique()) if len(clean_summary) else 0,
        "raw_points_004Y": raw_points,
        "kept_points_005A": kept_points,
        "dropped_points_005A": raw_points - kept_points,
        "keep_ratio_005A": round(kept_points / max(1, raw_points), 4),
        "reviews_with_kept": int(clean_summary[clean_summary["kept_points_005A"] > 0]["review_id"].nunique()) if len(clean_summary) else 0,
        "total_tracklets_005A": int(kept.groupby(["review_id", "tracklet_id_005A"]).ngroups) if len(kept) else 0,
        "overlays": overlays,
        "outputs": {
            "all_points": str(out_all),
            "kept_points": str(out_kept),
            "review_summary": str(out_reviews),
        },
        "next": "Review clean overlays. If zigzags are reduced, freeze 005A and move to 005B camera cut detection."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005A status=OK")
    print("reviews=", summary["reviews"])
    print("raw_points_004Y=", summary["raw_points_004Y"])
    print("kept_points_005A=", summary["kept_points_005A"])
    print("dropped_points_005A=", summary["dropped_points_005A"])
    print("keep_ratio_005A=", summary["keep_ratio_005A"])
    print("reviews_with_kept=", summary["reviews_with_kept"])
    print("total_tracklets_005A=", summary["total_tracklets_005A"])
    print("overlays=", len(overlays))
    print("wrote", out_all)
    print("wrote", out_kept)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
