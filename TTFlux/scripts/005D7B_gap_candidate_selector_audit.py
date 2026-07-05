from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7B_gap_candidate_selector_audit"


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except Exception:
        return default


def find_col(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def source_bonus(src: str) -> float:
    s = (src or "").lower()
    bonus = 0.0
    if "motion_bright" in s:
        bonus += 0.22
    if "bright" in s:
        bonus += 0.08
    if "motion" in s:
        bonus += 0.04
    return min(0.30, bonus)


def selector_score(c: dict) -> float:
    raw_score = safe_float(c.get("score"), 0.0)
    dist = safe_float(c.get("dist_pred"), np.nan)
    area = safe_float(c.get("area"), 0.0)
    aspect = safe_float(c.get("aspect"), 9.0)

    if math.isfinite(dist):
        pred_prox = max(0.0, 1.0 - min(dist, 260.0) / 260.0)
    else:
        pred_prox = 0.25

    # balle : petit blob compact. Assez permissif, car flou possible.
    area_bonus = max(0.0, 1.0 - abs(area - 18.0) / 120.0)
    compact_bonus = max(0.0, 1.0 - min(aspect, 5.0) / 5.0)

    return (
        0.50 * raw_score
        + 0.30 * pred_prox
        + 0.12 * source_bonus(str(c.get("source", "")))
        + 0.05 * area_bonus
        + 0.03 * compact_bonus
    )


def read_track_points(track_csv: Path | None, sequence_key: str | None) -> dict[int, tuple[float, float, float]]:
    if track_csv is None or not track_csv.exists():
        return {}

    rows = read_csv_rows(track_csv)
    if not rows:
        return {}

    cols = list(rows[0].keys())
    frame_col = find_col(cols, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(cols, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(cols, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(cols, ["score", "conf", "confidence", "prob"])
    seq_col = find_col(cols, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])

    if frame_col is None or x_col is None or y_col is None:
        return {}

    if seq_col and sequence_key:
        filtered = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]
        if filtered:
            rows = filtered

    out = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            out[fr] = (x, y, sc)

    return out


def group_candidates(rows: list[dict]) -> dict[int, list[dict]]:
    by_frame = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue
        r["_selector_score"] = selector_score(r)
        by_frame.setdefault(fr, []).append(r)

    for fr in by_frame:
        by_frame[fr].sort(key=lambda x: safe_float(x.get("_selector_score"), 0.0), reverse=True)

    return by_frame


def make_selected_rows(by_frame: dict[int, list[dict]]) -> list[dict]:
    selected = []
    for fr in sorted(by_frame):
        cands = by_frame[fr]
        if not cands:
            continue

        best = cands[0].copy()
        best["selector_score"] = f'{safe_float(best.get("_selector_score"), 0.0):.6f}'
        best["rank"] = 1
        best["candidate_count_this_frame"] = len(cands)

        dist = safe_float(best.get("dist_pred"), np.nan)
        sc = safe_float(best.get("score"), 0.0)
        src = str(best.get("source", ""))

        flags = []
        if math.isfinite(dist) and dist > 120:
            flags.append("far_from_prediction")
        if sc < 0.42:
            flags.append("low_raw_score")
        if "motion_bright" not in src.lower():
            flags.append("not_motion_bright")
        if len(cands) >= 10:
            flags.append("crowded_frame")

        best["audit_flags"] = "|".join(flags)
        selected.append(best)

    return selected


def infer_paths(candidate_rows: list[dict], args):
    video = Path(args.video) if args.video else None
    track = Path(args.track_csv) if args.track_csv else None
    sequence = args.sequence_key or ""

    if candidate_rows:
        r0 = candidate_rows[0]
        if video is None and r0.get("video"):
            video = Path(str(r0["video"]))
        if track is None and r0.get("track_csv"):
            track = Path(str(r0["track_csv"]))
        if not sequence and r0.get("chosen_sequence"):
            sequence = str(r0["chosen_sequence"])

    return video, track, sequence


def draw_overlay_frame(
    frame,
    frame_idx: int,
    cands: list[dict],
    selected: dict | None,
    base_xy: tuple[float, float, float] | None,
):
    out = frame.copy()

    if base_xy is not None:
        x, y, _ = base_xy
        cv2.circle(out, (int(round(x)), int(round(y))), 5, (180, 180, 180), 1)
        cv2.putText(out, "base", (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    if cands:
        pred_x = safe_float(cands[0].get("pred_x"), np.nan)
        pred_y = safe_float(cands[0].get("pred_y"), np.nan)
        if math.isfinite(pred_x) and math.isfinite(pred_y):
            cv2.drawMarker(
                out,
                (int(round(pred_x)), int(round(pred_y))),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )

    for idx, c in enumerate(cands[:12]):
        x = safe_float(c.get("x"))
        y = safe_float(c.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        src = str(c.get("source", "")).lower()
        if "motion_bright" in src:
            color = (0, 255, 255)
        elif "bright" in src:
            color = (0, 180, 255)
        else:
            color = (0, 220, 0)

        radius = 4 if idx > 2 else 6
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, 1)
        cv2.putText(
            out,
            f"{idx+1}",
            (int(round(x)) + 5, int(round(y)) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    if selected is not None:
        sx = safe_float(selected.get("x"))
        sy = safe_float(selected.get("y"))
        if math.isfinite(sx) and math.isfinite(sy):
            cv2.circle(out, (int(round(sx)), int(round(sy))), 12, (255, 0, 255), 2)
            cv2.putText(
                out,
                "SELECTED",
                (int(round(sx)) + 12, int(round(sy)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

    label = f"{PATCH_ID} | frame={frame_idx} | candidates={len(cands)}"
    if selected is not None:
        label += f" | sel_score={selected.get('selector_score', '')} | flags={selected.get('audit_flags', '')}"

    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 1180), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def write_overlay(video_path: Path, out_path: Path, by_frame, selected_by_frame, base_track, full_video: bool):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir la vidéo pour overlay: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"WARN: impossible d'ouvrir VideoWriter: {out_path}")
        cap.release()
        return False

    target_frames = set(by_frame.keys())
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        if full_video or frame_idx in target_frames:
            out = draw_overlay_frame(
                frame=frame,
                frame_idx=frame_idx,
                cands=by_frame.get(frame_idx, []),
                selected=selected_by_frame.get(frame_idx),
                base_xy=base_track.get(frame_idx),
            )
            writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", default="runs/005D7A_gap_candidates/005D7A_gap_candidates.csv")
    ap.add_argument("--video", default="")
    ap.add_argument("--track-csv", default="")
    ap.add_argument("--sequence-key", default="")
    ap.add_argument("--out-dir", default="runs/005D7B_gap_selected")
    ap.add_argument("--full-video-overlay", action="store_true")
    args = ap.parse_args()

    candidates_csv = Path(args.candidates_csv)
    if not candidates_csv.exists():
        raise FileNotFoundError(f"CSV candidats introuvable: {candidates_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(candidates_csv)
    video_path, track_csv, sequence = infer_paths(rows, args)

    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

    by_frame = group_candidates(rows)
    selected = make_selected_rows(by_frame)
    selected_by_frame = {safe_int(r.get("frame")): r for r in selected}

    base_track = read_track_points(track_csv, sequence)

    selected_fields = [
        "patch",
        "video",
        "track_csv",
        "chosen_sequence",
        "gap_id",
        "gap_start",
        "gap_end",
        "frame",
        "rank",
        "candidate_count_this_frame",
        "x",
        "y",
        "r",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "aspect",
        "source",
        "score",
        "selector_score",
        "mean_gray",
        "max_gray",
        "mean_motion",
        "pred_x",
        "pred_y",
        "pred_mode",
        "dist_pred",
        "audit_flags",
        "roi_mode",
    ]

    for r in selected:
        r["patch"] = PATCH_ID

    selected_csv = out_dir / "005D7B_gap_selected_top1.csv"
    write_csv(selected_csv, selected, selected_fields)

    # CSV preview track : base + points choisis dans les gaps.
    merged_rows = []
    for fr in sorted(base_track):
        x, y, sc = base_track[fr]
        merged_rows.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "score": f"{sc:.6f}",
            "point_source": "base_005D4",
            "gap_id": "",
            "audit_flags": "",
        })

    for r in selected:
        fr = safe_int(r.get("frame"))
        if fr in base_track:
            continue
        merged_rows.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{safe_float(r.get('x')):.3f}",
            "y": f"{safe_float(r.get('y')):.3f}",
            "score": f"{safe_float(r.get('selector_score'), 0.0):.6f}",
            "point_source": "gapfill_005D7B_PREVIEW",
            "gap_id": r.get("gap_id", ""),
            "audit_flags": r.get("audit_flags", ""),
        })

    merged_rows.sort(key=lambda r: safe_int(r.get("frame")))
    merged_csv = out_dir / "005D7B_gapfilled_preview_points.csv"
    write_csv(
        merged_csv,
        merged_rows,
        ["sequence_key", "frame", "x", "y", "score", "point_source", "gap_id", "audit_flags"],
    )

    overlay_path = out_dir / "005D7B_gap_selected_overlay.mp4"
    overlay_ok = write_overlay(
        video_path=video_path,
        out_path=overlay_path,
        by_frame=by_frame,
        selected_by_frame=selected_by_frame,
        base_track=base_track,
        full_video=args.full_video_overlay,
    )

    dists = [safe_float(r.get("dist_pred")) for r in selected if math.isfinite(safe_float(r.get("dist_pred")))]
    raw_scores = [safe_float(r.get("score"), 0.0) for r in selected]
    selector_scores = [safe_float(r.get("selector_score"), 0.0) for r in selected]
    flagged = [r for r in selected if str(r.get("audit_flags", "")).strip()]

    source_counts = {}
    for r in selected:
        src = str(r.get("source", ""))
        source_counts[src] = source_counts.get(src, 0) + 1

    summary = {
        "patch": PATCH_ID,
        "input_candidates_csv": str(candidates_csv),
        "video": str(video_path),
        "track_csv": str(track_csv) if track_csv else None,
        "sequence_key": sequence,
        "gap_frames": len(by_frame),
        "selected_count": len(selected),
        "candidate_total": len(rows),
        "candidate_mean_per_gap_frame": len(rows) / max(1, len(by_frame)),
        "selected_dist_pred_mean": float(np.mean(dists)) if dists else None,
        "selected_dist_pred_median": float(np.median(dists)) if dists else None,
        "selected_dist_pred_p90": float(np.percentile(dists, 90)) if dists else None,
        "selected_raw_score_mean": float(np.mean(raw_scores)) if raw_scores else None,
        "selected_selector_score_mean": float(np.mean(selector_scores)) if selector_scores else None,
        "flagged_selected_count": len(flagged),
        "flagged_selected_ratio": len(flagged) / max(1, len(selected)),
        "source_counts": source_counts,
        "selected_csv": str(selected_csv),
        "merged_preview_csv": str(merged_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "note": "Preview seulement. Pas un tracking final, pas Viterbi.",
    }

    summary_path = out_dir / "005D7B_gap_selected_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"candidates_csv = {candidates_csv}")
    print(f"video          = {video_path}")
    print(f"track_csv      = {track_csv}")
    print(f"sequence       = {sequence}")
    print("=" * 72)
    print("")
    print("OK 005D7B")
    print(f"selected_csv = {selected_csv}")
    print(f"preview_csv  = {merged_csv}")
    print(f"summary      = {summary_path}")
    print(f"overlay      = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"gap_frames={len(by_frame)} selected={len(selected)} candidates_total={len(rows)}")
    print(f"mean_candidates_per_frame={summary['candidate_mean_per_gap_frame']:.2f}")
    print(f"flagged_selected_count={len(flagged)} ratio={summary['flagged_selected_ratio']:.3f}")
    if dists:
        print(f"selected_dist_pred_median={summary['selected_dist_pred_median']:.2f} p90={summary['selected_dist_pred_p90']:.2f}")


if __name__ == "__main__":
    main()
