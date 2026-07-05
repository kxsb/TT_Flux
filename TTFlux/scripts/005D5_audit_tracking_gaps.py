from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd


VERSION = "005D5"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def find_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"missing column among {names}")


def build_clip_map(scene: pd.DataFrame) -> dict[str, str]:
    out = {}

    if "clip_path" not in scene.columns:
        return out

    for review_id, g in scene.groupby("review_id", dropna=False):
        vals = [str(x) for x in g["clip_path"].tolist() if str(x).strip()]
        if vals:
            out[str(review_id)] = vals[0]

    return out


def classify_gap(orig_g: pd.DataFrame, frame_col: str, gap_start: int, gap_end: int) -> dict:
    inside = orig_g[
        to_num(orig_g[frame_col]).fillna(-1).astype(int).between(gap_start, gap_end)
    ].copy()

    if inside.empty:
        return {
            "gap_kind": "NO_SOURCE_POINT",
            "source_points_in_gap": 0,
            "low_score_points": 0,
            "safe_points": 0,
            "metric_points": 0,
            "reason": "detector_or_upstream_tracker_did_not_emit_points",
        }

    low_score = 0
    safe = 0
    metric = 0

    if "ball_table_score_005D2" in inside.columns:
        low_score = int(to_num(inside["ball_table_score_005D2"]).fillna(0).lt(0.55).sum())

    if "ball_table_safe_005D2" in inside.columns:
        safe = int(to_num(inside["ball_table_safe_005D2"]).fillna(0).astype(int).eq(1).sum())

    if "table_project_ok_005C9B" in inside.columns:
        metric = int(to_num(inside["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1).sum())

    if safe == 0 and low_score > 0:
        return {
            "gap_kind": "FILTERED_LOW_SCORE",
            "source_points_in_gap": int(len(inside)),
            "low_score_points": low_score,
            "safe_points": safe,
            "metric_points": metric,
            "reason": "source_points_exist_but_table_score_rejected_them",
        }

    if safe > 0:
        return {
            "gap_kind": "BRIDGE_TOO_STRICT_OR_SEGMENT_SPLIT",
            "source_points_in_gap": int(len(inside)),
            "low_score_points": low_score,
            "safe_points": safe,
            "metric_points": metric,
            "reason": "safe_points_exist_inside_gap_but_relink_did_not_connect",
        }

    return {
        "gap_kind": "SOURCE_POINTS_UNUSABLE",
        "source_points_in_gap": int(len(inside)),
        "low_score_points": low_score,
        "safe_points": safe,
        "metric_points": metric,
        "reason": "source_points_exist_but_not_usable_by_current_policy",
    }


def audit_review(orig_g: pd.DataFrame, clean_g: pd.DataFrame, frame_col: str, review_id: str, seg: int):
    clean_g = clean_g.copy()
    orig_g = orig_g.copy()

    clean_g[frame_col] = to_num(clean_g[frame_col]).fillna(-1).astype(int)
    orig_g[frame_col] = to_num(orig_g[frame_col]).fillna(-1).astype(int)

    clean_g = clean_g.sort_values(frame_col).copy()

    rows = []
    frames = clean_g[frame_col].tolist()

    for a, b in zip(frames[:-1], frames[1:]):
        gap_len = int(b - a - 1)

        if gap_len <= 0:
            continue

        gap_start = int(a + 1)
        gap_end = int(b - 1)

        c = classify_gap(orig_g, frame_col, gap_start, gap_end)

        rows.append({
            "review_id": str(review_id),
            "camera_segment_id_005B2": int(seg),
            "prev_frame": int(a),
            "next_frame": int(b),
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_len": gap_len,
            **c,
        })

    return rows


def open_writer(path: Path, fps: float, w: int, h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"cannot open writer: {path}")
    return wr


def draw_label(img, text, y):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def draw_point(img, x, y, color, radius=6):
    x = int(round(float(x)))
    y = int(round(float(y)))
    cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)


def render_gap_focus(
    review_id: str,
    orig_g: pd.DataFrame,
    clean_g: pd.DataFrame,
    gaps: list[dict],
    clip_path: str,
    out_dir: Path,
    frame_col: str,
    x_col: str,
    y_col: str,
    max_frames: int,
):
    if not gaps:
        return None

    clip_path = Path(str(clip_path))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("skip cannot_open_video", review_id, clip_path)
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        cap.release()
        return None

    gaps = sorted(gaps, key=lambda r: r["gap_len"], reverse=True)

    first = max(0, int(min(g["gap_start"] for g in gaps[:8])) - 35)
    last = min(n_frames - 1, int(max(g["gap_end"] for g in gaps[:8])) + 35)

    if max_frames > 0:
        last = min(last, first + max_frames - 1)

    orig_g = orig_g.copy()
    clean_g = clean_g.copy()

    orig_g[frame_col] = to_num(orig_g[frame_col]).fillna(-1).astype(int)
    clean_g[frame_col] = to_num(clean_g[frame_col]).fillna(-1).astype(int)

    orig_by_frame = {int(k): v.copy() for k, v in orig_g.groupby(frame_col)}
    clean_by_frame = {int(k): v.copy() for k, v in clean_g.groupby(frame_col)}

    out_path = out_dir / "gap_focus" / f"{review_id}_005D5_gap_focus.mp4"
    wr = open_writer(out_path, fps=fps, w=w, h=h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    gap_by_frame = {}

    for g in gaps:
        for f in range(int(g["gap_start"]), int(g["gap_end"]) + 1):
            gap_by_frame.setdefault(f, []).append(g)

    frame_idx = first
    written = 0

    while frame_idx <= last:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        active_gaps = gap_by_frame.get(frame_idx, [])

        rows_o = orig_by_frame.get(frame_idx)

        if rows_o is not None and len(rows_o):
            for _, r in rows_o.iterrows():
                try:
                    x = float(r[x_col])
                    y = float(r[y_col])
                except Exception:
                    continue

                score = float(to_num(pd.Series([r.get("ball_table_score_005D2", 0.42)])).fillna(0.42).iloc[0])
                safe = int(to_num(pd.Series([r.get("ball_table_safe_005D2", 0)])).fillna(0).iloc[0]) == 1
                metric = int(to_num(pd.Series([r.get("table_project_ok_005C9B", 0)])).fillna(0).iloc[0]) == 1

                if safe:
                    color = (0, 255, 0)       # source safe
                elif metric:
                    color = (0, 0, 255)       # source rejetée par score table
                else:
                    color = (255, 220, 40)    # source non métrique

                draw_point(frame, x, y, color, radius=5)

        rows_c = clean_by_frame.get(frame_idx)

        if rows_c is not None and len(rows_c):
            for _, r in rows_c.iterrows():
                try:
                    x = float(r[x_col])
                    y = float(r[y_col])
                except Exception:
                    continue

                interp = int(to_num(pd.Series([r.get("is_interpolated_005D4", 0)])).fillna(0).iloc[0]) == 1
                color = (255, 0, 255) if interp else (0, 255, 0)
                draw_point(frame, x, y, color, radius=7 if interp else 6)

        if active_gaps:
            g0 = active_gaps[0]
            draw_label(frame, f"{review_id} GAP {g0['gap_kind']} len={g0['gap_len']} src={g0['source_points_in_gap']}", 30)
            draw_label(frame, f"{g0['reason']}", 58)
        else:
            draw_label(frame, f"{review_id} 005D5 gap audit", 30)
            draw_label(frame, "green=safe/relinked magenta=interp red=source low-score cyan=no-metric", 58)

        wr.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    wr.release()

    return {
        "review_id": str(review_id),
        "overlay": str(out_path),
        "first_frame": first,
        "last_frame": frame_idx - 1,
        "written_frames": written,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-points", default="runs/rally_ball_table_score_005D2/ball_points_table_scored_005D2.csv")
    ap.add_argument("--relinked-points", default="runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_gap_audit_005D5")
    ap.add_argument("--min-gap", type=int, default=8)
    ap.add_argument("--make-videos", action="store_true")
    ap.add_argument("--max-videos", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=1200)
    args = ap.parse_args()

    root = Path.cwd()

    scored_path = Path(args.scored_points)
    relinked_path = Path(args.relinked_points)
    scene_path = Path(args.scene_tables)
    out_dir = Path(args.out_dir)

    if not scored_path.is_absolute():
        scored_path = root / scored_path
    if not relinked_path.is_absolute():
        relinked_path = root / relinked_path
    if not scene_path.is_absolute():
        scene_path = root / scene_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(scored_path).fillna("")
    relinked = pd.read_csv(relinked_path).fillna("")
    scene = pd.read_csv(scene_path).fillna("") if scene_path.is_file() else pd.DataFrame()

    clip_map = build_clip_map(scene)

    frame_col = find_col(scored, ["frame_num", "frame", "frame_idx"])
    seg_col = find_col(scored, ["camera_segment_id_005B2", "camera_segment_id"])
    x_col = find_col(scored, ["x_num", "x", "cx"])
    y_col = find_col(scored, ["y_num", "y", "cy"])

    rel_seg_col = find_col(relinked, ["camera_segment_id_005B2", "camera_segment_id"])

    scored[seg_col] = to_num(scored[seg_col]).fillna(1).astype(int)
    relinked[rel_seg_col] = to_num(relinked[rel_seg_col]).fillna(1).astype(int)

    all_gap_rows = []

    for (review_id, seg), clean_g in relinked.groupby(["review_id", rel_seg_col], dropna=False):
        orig_g = scored[
            scored["review_id"].astype(str).eq(str(review_id))
            & scored[seg_col].astype(int).eq(int(seg))
        ].copy()

        if orig_g.empty:
            continue

        rows = audit_review(
            orig_g=orig_g,
            clean_g=clean_g,
            frame_col=frame_col,
            review_id=review_id,
            seg=seg,
        )

        all_gap_rows.extend(rows)

    gaps = pd.DataFrame(all_gap_rows)

    if not gaps.empty:
        gaps = gaps[gaps["gap_len"].astype(int).ge(args.min_gap)].copy()
        gaps = gaps.sort_values(["gap_len", "source_points_in_gap"], ascending=[False, False]).copy()

    out_gaps = out_dir / "ball_tracking_gap_audit_005D5.csv"
    out_json = out_dir / "ball_tracking_gap_audit_summary_005D5.json"

    gaps.to_csv(out_gaps, index=False, encoding="utf-8")

    overlays = []

    if args.make_videos and not gaps.empty:
        chosen = []

        for r in gaps["review_id"].astype(str).tolist():
            if r not in chosen:
                chosen.append(r)
            if len(chosen) >= args.max_videos:
                break

        for review_id in chosen:
            orig_g = scored[scored["review_id"].astype(str).eq(str(review_id))].copy()
            clean_g = relinked[relinked["review_id"].astype(str).eq(str(review_id))].copy()
            review_gaps = gaps[gaps["review_id"].astype(str).eq(str(review_id))].to_dict("records")

            clip_path = clip_map.get(str(review_id))

            if not clip_path:
                print("skip no_clip_path", review_id)
                continue

            info = render_gap_focus(
                review_id=review_id,
                orig_g=orig_g,
                clean_g=clean_g,
                gaps=review_gaps,
                clip_path=clip_path,
                out_dir=out_dir,
                frame_col=frame_col,
                x_col=x_col,
                y_col=y_col,
                max_frames=args.max_frames,
            )

            if info:
                overlays.append(info)
                print("overlay", info["overlay"])

    kind_counts = gaps["gap_kind"].astype(str).value_counts().to_dict() if not gaps.empty else {}

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "audit_tracking_losses_after_table_aware_relink_to_separate_missing_source_points_from_over_filtering",
        "scored_points": str(scored_path),
        "relinked_points": str(relinked_path),
        "scene_tables": str(scene_path),
        "min_gap": args.min_gap,
        "gaps_total": int(len(gaps)),
        "gap_kind_counts": {str(k): int(v) for k, v in kind_counts.items()},
        "gap_len_median": round(float(gaps["gap_len"].median()), 4) if not gaps.empty else 0,
        "gap_len_max": int(gaps["gap_len"].max()) if not gaps.empty else 0,
        "source_points_in_gap_total": int(gaps["source_points_in_gap"].sum()) if not gaps.empty else 0,
        "low_score_points_in_gaps_total": int(gaps["low_score_points"].sum()) if not gaps.empty else 0,
        "safe_points_in_gaps_total": int(gaps["safe_points"].sum()) if not gaps.empty else 0,
        "outputs": {
            "gaps": str(out_gaps),
            "overlays": overlays,
        },
        "next": "If most gaps are FILTERED_LOW_SCORE, tune score/gate. If most are NO_SOURCE_POINT, improve candidate detector/ranker before relink."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D5 status=OK")
    print("gaps_total=", summary["gaps_total"])
    print("gap_kind_counts=", json.dumps(summary["gap_kind_counts"], ensure_ascii=False))
    print("gap_len_median=", summary["gap_len_median"])
    print("gap_len_max=", summary["gap_len_max"])
    print("source_points_in_gap_total=", summary["source_points_in_gap_total"])
    print("low_score_points_in_gaps_total=", summary["low_score_points_in_gaps_total"])
    print("safe_points_in_gaps_total=", summary["safe_points_in_gaps_total"])
    print("overlays=", len(overlays))
    print("wrote", out_gaps)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
