from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005A3"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["review_id", "frame_num", "x_num", "y_num", "score_num", "tracklet_id_005A"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Colonne absente: {c}")

    df["frame_num"] = to_num(df["frame_num"])
    df["x_num"] = to_num(df["x_num"])
    df["y_num"] = to_num(df["y_num"])
    df["score_num"] = to_num(df["score_num"]).fillna(0.0)
    df["tracklet_id_005A"] = to_num(df["tracklet_id_005A"]).fillna(0).astype(int)

    if "display_source_005A2" not in df.columns:
        df["display_source_005A2"] = "orig"

    df = df.dropna(subset=["frame_num", "x_num", "y_num"]).copy()
    df["frame_num"] = df["frame_num"].astype(int)

    return df


def arbitrate_review(
    g: pd.DataFrame,
    max_link_gap_frames: int,
    max_step_px: float,
    very_high_score: float,
    continuity_weight: float,
    same_tracklet_bonus: float,
    orig_bonus: float,
    interp_penalty: float,
) -> pd.DataFrame:
    rows = []

    g = g.sort_values(["frame_num", "score_num"], ascending=[True, False]).copy()

    prev = None
    segment_id = 0

    for frame, cands in g.groupby("frame_num", dropna=False):
        frame = int(frame)
        cands = cands.copy()

        scored = []

        for _, r in cands.iterrows():
            x = float(r["x_num"])
            y = float(r["y_num"])
            score = float(r["score_num"])
            tid = int(r["tracklet_id_005A"])
            src = str(r.get("display_source_005A2", "orig"))

            rank = score
            dist = None
            dt = None
            linkable = False
            new_segment_reason = "start"

            if src == "orig":
                rank += orig_bonus
            else:
                rank -= interp_penalty

            if prev is not None:
                pf, px, py, ptid, pseg = prev
                dt = frame - pf
                dist = math.hypot(x - px, y - py)

                if dt <= 0:
                    linkable = False
                    new_segment_reason = "non_monotonic"
                elif dt > max_link_gap_frames:
                    linkable = False
                    new_segment_reason = f"gap>{max_link_gap_frames}"
                elif dist <= max_step_px:
                    linkable = True
                    new_segment_reason = "continue"
                    continuity = 1.0 - min(1.0, dist / max(1e-9, max_step_px))
                    rank += continuity_weight * continuity

                    if tid == ptid:
                        rank += same_tracklet_bonus
                elif score >= very_high_score and dt <= 2:
                    # Cas rare : point très confiant mais saut visuel. On accepte,
                    # mais on démarre un nouveau segment pour ne pas tracer une ligne absurde.
                    linkable = False
                    new_segment_reason = "high_score_jump_new_segment"
                    rank += 0.05
                else:
                    linkable = False
                    new_segment_reason = f"jump>{max_step_px}"
                    rank -= 0.65

            scored.append({
                "rank_005A3": rank,
                "dist_prev_005A3": dist,
                "dt_prev_005A3": dt,
                "linkable_005A3": int(linkable),
                "new_segment_reason_005A3": new_segment_reason,
                "row": r.to_dict(),
            })

        scored = sorted(scored, key=lambda x: x["rank_005A3"], reverse=True)
        best = scored[0]

        r = best["row"]
        tid = int(r["tracklet_id_005A"])

        if prev is None:
            segment_id += 1
        else:
            if not bool(best["linkable_005A3"]):
                segment_id += 1

        r["display_rank_005A3"] = round(float(best["rank_005A3"]), 6)
        r["dist_prev_005A3"] = "" if best["dist_prev_005A3"] is None else round(float(best["dist_prev_005A3"]), 3)
        r["dt_prev_005A3"] = "" if best["dt_prev_005A3"] is None else int(best["dt_prev_005A3"])
        r["linkable_005A3"] = int(best["linkable_005A3"])
        r["display_segment_id_005A3"] = int(segment_id)
        r["new_segment_reason_005A3"] = str(best["new_segment_reason_005A3"])
        r["candidates_same_frame_005A3"] = int(len(cands))

        rows.append(r)

        prev = (
            frame,
            float(r["x_num"]),
            float(r["y_num"]),
            tid,
            segment_id,
        )

    return pd.DataFrame(rows)


def arbitrate_all(
    display: pd.DataFrame,
    max_link_gap_frames: int,
    max_step_px: float,
    very_high_score: float,
    continuity_weight: float,
    same_tracklet_bonus: float,
    orig_bonus: float,
    interp_penalty: float,
) -> pd.DataFrame:
    parts = []

    for review_id, g in display.groupby("review_id", dropna=False):
        out = arbitrate_review(
            g,
            max_link_gap_frames=max_link_gap_frames,
            max_step_px=max_step_px,
            very_high_score=very_high_score,
            continuity_weight=continuity_weight,
            same_tracklet_bonus=same_tracklet_bonus,
            orig_bonus=orig_bonus,
            interp_penalty=interp_penalty,
        )
        parts.append(out)

    return pd.concat(parts, ignore_index=True) if parts else display.iloc[0:0].copy()


def draw_point(frame, x, y, score, src, draw_labels: bool):
    x = int(round(float(x)))
    y = int(round(float(y)))

    if src == "interp":
        color = (70, 180, 70)
        radius = 4
        label = f"FILL {score:.2f}"
    else:
        color = (0, 255, 0)
        radius = 6
        label = f"BALL {score:.2f}"

    cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)

    if src != "interp":
        cv2.line(frame, (x - 9, y), (x + 9, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - 9), (x, y + 9), color, 2, cv2.LINE_AA)

    if draw_labels:
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
    points: pd.DataFrame,
    out_path: Path,
    max_video_frames: int,
    trail_frames: int,
    no_ball_grace_frames: int,
    draw_labels: bool,
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

    by_frame = {
        int(r["frame_num"]): r
        for _, r in points.iterrows()
    }

    trail = []
    current_segment = None
    last_ball_frame = -999999

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        r = by_frame.get(fidx)

        if r is not None:
            seg = int(r["display_segment_id_005A3"])

            # Reset trail si nouveau segment. C'est le point clé.
            if current_segment is None or seg != current_segment:
                trail = []
                current_segment = seg

            x = float(r["x_num"])
            y = float(r["y_num"])
            score = float(r["score_num"])
            src = str(r.get("display_source_005A2", "orig"))

            trail.append((fidx, x, y, score, src))
            trail = trail[-trail_frames:]
            last_ball_frame = fidx

        # Trace uniquement le segment actif, jamais tous les tracklets en parallèle.
        if len(trail) >= 2:
            for a, b in zip(trail[:-1], trail[1:]):
                fa, ax, ay, _, _ = a
                fb, bx, by, _, _ = b

                if fb - fa > trail_frames:
                    continue

                cv2.line(
                    frame,
                    (int(round(ax)), int(round(ay))),
                    (int(round(bx)), int(round(by))),
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

        if trail:
            _, x, y, score, src = trail[-1]
            draw_point(frame, x, y, score, src, draw_labels=draw_labels)

        if r is not None:
            state = "BALL_OK"
            color = (0, 255, 0)
        elif fidx - last_ball_frame <= no_ball_grace_frames:
            state = "BALL_HOLD"
            color = (90, 180, 90)
        else:
            state = "NO_BALL"
            color = (95, 95, 180)

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
            "005A3 single-ball display: one point/frame, active segment only",
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
    ap.add_argument("--display-points", required=True)
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--max-link-gap-frames", type=int, default=8)
    ap.add_argument("--max-step-px", type=float, default=95.0)
    ap.add_argument("--very-high-score", type=float, default=0.94)
    ap.add_argument("--continuity-weight", type=float, default=1.20)
    ap.add_argument("--same-tracklet-bonus", type=float, default=0.25)
    ap.add_argument("--orig-bonus", type=float, default=0.18)
    ap.add_argument("--interp-penalty", type=float, default=0.08)

    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=20)
    ap.add_argument("--max-video-frames", type=int, default=3000)
    ap.add_argument("--trail-frames", type=int, default=12)
    ap.add_argument("--no-ball-grace-frames", type=int, default=5)
    ap.add_argument("--draw-labels", action="store_true")

    args = ap.parse_args()

    root = Path.cwd()

    display_path = Path(args.display_points)
    review_path = Path(args.review_summary)
    out_dir = Path(args.out_dir)

    if not display_path.is_absolute():
        display_path = root / display_path
    if not review_path.is_absolute():
        review_path = root / review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    display = pd.read_csv(display_path).fillna("")
    reviews = pd.read_csv(review_path).fillna("")

    display = prep(display)

    selected = arbitrate_all(
        display,
        max_link_gap_frames=args.max_link_gap_frames,
        max_step_px=args.max_step_px,
        very_high_score=args.very_high_score,
        continuity_weight=args.continuity_weight,
        same_tracklet_bonus=args.same_tracklet_bonus,
        orig_bonus=args.orig_bonus,
        interp_penalty=args.interp_penalty,
    )

    review_rows = []

    for review_id, g in selected.groupby("review_id", dropna=False):
        review_id = str(review_id)

        raw = reviews[reviews["review_id"].astype(str).eq(review_id)]
        base = raw.iloc[0].to_dict() if len(raw) else {}

        cand_counts = to_num(g["candidates_same_frame_005A3"]).dropna()

        review_rows.append({
            "review_id": review_id,
            "video_id": str(base.get("video_id", "")),
            "rally_id": str(base.get("rally_id", "")),
            "selected_points_005A3": int(len(g)),
            "orig_points_005A3": int((g["display_source_005A2"].astype(str).eq("orig")).sum()),
            "interp_points_005A3": int((g["display_source_005A2"].astype(str).eq("interp")).sum()),
            "display_segments_005A3": int(g["display_segment_id_005A3"].nunique()),
            "multi_candidate_frames_005A3": int((to_num(g["candidates_same_frame_005A3"]).fillna(1) > 1).sum()),
            "candidates_same_frame_med_005A3": round(float(cand_counts.median()), 3) if len(cand_counts) else "",
            "clip_path": str(base.get("clip_path", "")),
            "overlay_005A3": "",
        })

    review_out = pd.DataFrame(review_rows)

    overlays = []

    if args.make_video:
        viz = review_out.sort_values(
            ["selected_points_005A3", "multi_candidate_frames_005A3"],
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

            pts = selected[selected["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{i:03d}_{review_id}_005A3_single_ball_overlay.mp4"

            ok = make_overlay(
                review_id=review_id,
                clip_path=clip_path,
                points=pts,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
                trail_frames=args.trail_frames,
                no_ball_grace_frames=args.no_ball_grace_frames,
                draw_labels=args.draw_labels,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})
                review_out.loc[
                    review_out["review_id"].astype(str).eq(review_id),
                    "overlay_005A3",
                ] = str(out_video)
                print(f"overlay {i}/{len(viz)} {review_id} -> {out_video}")

    out_selected = out_dir / "single_ball_display_points_005A3.csv"
    out_reviews = out_dir / "single_ball_display_review_summary_005A3.csv"
    out_json = out_dir / "single_ball_display_summary_005A3.json"

    selected.to_csv(out_selected, index=False, encoding="utf-8")
    review_out.to_csv(out_reviews, index=False, encoding="utf-8")

    multi_frames = int((to_num(selected["candidates_same_frame_005A3"]).fillna(1) > 1).sum())

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "select_one_display_ball_per_frame_after_005A2_visual_interpolation",
        "display_points": str(display_path),
        "review_summary": str(review_path),
        "params": {
            "max_link_gap_frames": args.max_link_gap_frames,
            "max_step_px": args.max_step_px,
            "very_high_score": args.very_high_score,
            "continuity_weight": args.continuity_weight,
            "same_tracklet_bonus": args.same_tracklet_bonus,
            "orig_bonus": args.orig_bonus,
            "interp_penalty": args.interp_penalty,
            "trail_frames": args.trail_frames,
            "no_ball_grace_frames": args.no_ball_grace_frames,
            "draw_labels": args.draw_labels,
        },
        "reviews": int(review_out["review_id"].nunique()) if len(review_out) else 0,
        "input_display_points_005A2": int(len(display)),
        "selected_points_005A3": int(len(selected)),
        "reduced_points_005A3": int(len(display) - len(selected)),
        "multi_candidate_selected_frames_005A3": multi_frames,
        "overlays": overlays,
        "outputs": {
            "selected_points": str(out_selected),
            "review_summary": str(out_reviews),
        },
        "next": "If visual overlay is readable, freeze 005A/005A2/005A3 stack and move to 005B camera cut detection."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005A3 status=OK")
    print("reviews=", summary["reviews"])
    print("input_display_points_005A2=", summary["input_display_points_005A2"])
    print("selected_points_005A3=", summary["selected_points_005A3"])
    print("reduced_points_005A3=", summary["reduced_points_005A3"])
    print("multi_candidate_selected_frames_005A3=", summary["multi_candidate_selected_frames_005A3"])
    print("overlays=", len(overlays))
    print("wrote", out_selected)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
