from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7E_export_gapfilled_preview_track"


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


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def infer_paths(safe_rows: list[dict], args):
    track_csv = Path(args.track_csv) if args.track_csv else None
    video = Path(args.video) if args.video else None
    sequence = args.sequence_key or ""

    if safe_rows:
        r0 = safe_rows[0]
        if track_csv is None and r0.get("track_csv"):
            track_csv = Path(str(r0["track_csv"]))
        if video is None and r0.get("video"):
            video = Path(str(r0["video"]))
        if not sequence and r0.get("chosen_sequence"):
            sequence = str(r0["chosen_sequence"])

    return track_csv, video, sequence


def get_video_info(video_path: Path | None):
    if video_path is None or not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def rows_for_sequence(rows, fields, sequence_key):
    seq_col = find_col(fields, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])
    if seq_col and sequence_key:
        return [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key], seq_col
    return rows, seq_col


def frame_xy_map(rows, fields):
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables dans le track source.")

    out = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            out[fr] = (x, y, r)

    return out, frame_col, x_col, y_col


def compute_missing_runs(present_frames: set[int], total_frames: int, min_gap_len: int = 1):
    runs = []
    start = None

    for fr in range(total_frames):
        missing = fr not in present_frames
        if missing and start is None:
            start = fr
        elif not missing and start is not None:
            end = fr - 1
            if end - start + 1 >= min_gap_len:
                runs.append((start, end, end - start + 1))
            start = None

    if start is not None:
        end = total_frames - 1
        if end - start + 1 >= min_gap_len:
            runs.append((start, end, end - start + 1))

    return runs


def continuity_stats(points: dict[int, tuple[float, float, dict]]):
    frames = sorted(points)
    steps = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue
        ax, ay, _ = points[a]
        bx, by, _ = points[b]
        dist = math.hypot(bx - ax, by - ay)
        steps.append({
            "from": a,
            "to": b,
            "dt": dt,
            "dist": dist,
            "speed": dist / dt,
        })

    if not steps:
        return {
            "step_count": 0,
            "speed_median": None,
            "speed_p90": None,
            "speed_p95": None,
            "speed_max": None,
            "large_step_count": 0,
            "largest_steps": [],
        }

    speeds = [s["speed"] for s in steps]
    large = [s for s in steps if s["speed"] > 180 or s["dist"] > 260]
    largest = sorted(steps, key=lambda s: s["speed"], reverse=True)[:12]

    return {
        "step_count": len(steps),
        "speed_median": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "large_step_count": len(large),
        "largest_steps": largest,
    }


def build_injected_row(template_row, fields, frame_col, x_col, y_col, score_col, seq_col, sequence_key, safe_row):
    out = {k: template_row.get(k, "") for k in fields}

    fr = safe_int(safe_row.get("frame"))
    x = safe_float(safe_row.get("x"))
    y = safe_float(safe_row.get("y"))
    sc = safe_float(safe_row.get("strict_score"), safe_float(safe_row.get("score"), 0.0))

    out[frame_col] = str(fr)
    out[x_col] = f"{x:.6f}"
    out[y_col] = f"{y:.6f}"

    if score_col:
        out[score_col] = f"{sc:.6f}"

    if seq_col and sequence_key:
        out[seq_col] = sequence_key

    out["point_source"] = "gapfill_005D7E_SAFE_PREVIEW"
    out["fill_patch"] = PATCH_ID
    out["fill_from_patch"] = "005D7D"
    out["fill_safe_status"] = safe_row.get("safe_status", "")
    out["fill_safe_reason"] = safe_row.get("safe_reason", "")
    out["fill_gap_id"] = safe_row.get("gap_id", "")
    out["fill_bridge_error_px"] = safe_row.get("bridge_error_px", "")
    out["fill_strict_score"] = safe_row.get("strict_score", "")
    out["fill_source"] = safe_row.get("source", "")
    out["fill_dist_pred"] = safe_row.get("dist_pred", "")

    return out


def draw_overlay(video_path, overlay_path, before_points, after_points, injected_frames, total_frames, full_video=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir vidéo overlay: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        print(f"WARN: impossible d'écrire overlay: {overlay_path}")
        return False

    target = set()
    for fr in injected_frames:
        for k in range(fr - 18, fr + 19):
            if 0 <= k < total_frames:
                target.add(k)

    frame_idx = -1
    trail = 18

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if not full_video and frame_idx not in target:
            continue

        out = frame.copy()

        # Before trail = gris
        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            if fr in before_points:
                x, y, _ = before_points[fr]
                cv2.circle(out, (int(round(x)), int(round(y))), 3, (165, 165, 165), -1)

        # After trail = cyan/rose
        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            if fr in after_points:
                x, y, row = after_points[fr]
                src = row.get("point_source", "")
                if src.startswith("gapfill"):
                    cv2.circle(out, (int(round(x)), int(round(y))), 7, (255, 0, 255), 2)
                    cv2.putText(
                        out,
                        "FILL",
                        (int(round(x)) + 8, int(round(y)) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 220, 255), -1)

        label = f"{PATCH_ID} | frame={frame_idx} | injected_points={len(injected_frames)} | gris=before cyan/rose=after"
        cv2.rectangle(out, (8, 8), (min(w - 8, 1180), 42), (0, 0, 0), -1)
        cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe-csv", default="runs/005D7D_safe_gapfill/005D7D_safe_gapfill_accepted.csv")
    ap.add_argument("--track-csv", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--sequence-key", default="")
    ap.add_argument("--out-dir", default="runs/005D7E_gapfilled_preview")
    ap.add_argument("--full-video-overlay", action="store_true")
    args = ap.parse_args()

    safe_csv = Path(args.safe_csv)
    if not safe_csv.exists():
        raise FileNotFoundError(f"safe_csv introuvable: {safe_csv}")

    safe_rows, safe_fields = read_csv_rows(safe_csv)
    safe_rows = [r for r in safe_rows if r.get("safe_status") == "accepted_safe"]

    track_csv, video_path, sequence_key = infer_paths(safe_rows, args)

    if track_csv is None or not track_csv.exists():
        raise FileNotFoundError(f"track_csv introuvable: {track_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, fields = read_csv_rows(track_csv)
    seq_rows, seq_col = rows_for_sequence(all_rows, fields, sequence_key)

    before_points, frame_col, x_col, y_col = frame_xy_map(seq_rows, fields)
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if not seq_rows:
        raise RuntimeError("Aucune ligne source trouvée pour la séquence.")

    template_row = seq_rows[0]

    extra_fields = [
        "point_source",
        "fill_patch",
        "fill_from_patch",
        "fill_safe_status",
        "fill_safe_reason",
        "fill_gap_id",
        "fill_bridge_error_px",
        "fill_strict_score",
        "fill_source",
        "fill_dist_pred",
    ]

    out_fields = list(fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    safe_by_frame = {}
    skipped_existing = []
    for r in safe_rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue
        if fr in before_points:
            skipped_existing.append(fr)
            continue
        safe_by_frame[fr] = r

    injected_rows = []
    for fr in sorted(safe_by_frame):
        injected_rows.append(build_injected_row(
            template_row=template_row,
            fields=out_fields,
            frame_col=frame_col,
            x_col=x_col,
            y_col=y_col,
            score_col=score_col,
            seq_col=seq_col,
            sequence_key=sequence_key,
            safe_row=safe_by_frame[fr],
        ))

    # Préserver tous les autres clips/séquences + injecter seulement la séquence cible.
    merged_rows = []
    for r in all_rows:
        rr = dict(r)
        for f in extra_fields:
            rr.setdefault(f, "")
        rr.setdefault("point_source", "base_005D4")
        merged_rows.append(rr)

    merged_rows.extend(injected_rows)

    def sort_key(r):
        seq = str(r.get(seq_col, "")) if seq_col else ""
        return (seq, safe_int(r.get(frame_col)), str(r.get("point_source", "")))

    merged_rows.sort(key=sort_key)

    # Points après injection pour la séquence cible uniquement.
    after_seq_rows = []
    if seq_col and sequence_key:
        after_seq_rows = [r for r in merged_rows if str(r.get(seq_col, "")).strip() == sequence_key]
    else:
        after_seq_rows = merged_rows

    after_points, _, _, _ = frame_xy_map(after_seq_rows, out_fields)

    video_info = get_video_info(video_path)
    if video_info and video_info["frames"] > 0:
        total_frames = video_info["frames"]
    else:
        total_frames = max(after_points.keys()) + 1 if after_points else 0

    missing_before = compute_missing_runs(set(before_points.keys()), total_frames)
    missing_after = compute_missing_runs(set(after_points.keys()), total_frames)

    cont_before = continuity_stats(before_points)
    cont_after = continuity_stats(after_points)

    output_csv = out_dir / "005D7E_ball_points_gapfilled_preview.csv"
    injected_csv = out_dir / "005D7E_injected_points_only.csv"
    summary_path = out_dir / "005D7E_gapfilled_preview_summary.json"
    overlay_path = out_dir / "005D7E_gapfilled_preview_overlay.mp4"

    write_csv(output_csv, merged_rows, out_fields)
    write_csv(injected_csv, injected_rows, out_fields)

    overlay_ok = False
    if video_path and video_path.exists():
        overlay_ok = draw_overlay(
            video_path=video_path,
            overlay_path=overlay_path,
            before_points=before_points,
            after_points=after_points,
            injected_frames=sorted(safe_by_frame),
            total_frames=total_frames,
            full_video=args.full_video_overlay,
        )

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "safe_csv": str(safe_csv),
        "video": str(video_path) if video_path else None,
        "sequence_key": sequence_key,
        "total_video_frames": total_frames,
        "source_rows_total": len(all_rows),
        "source_sequence_points": len(before_points),
        "safe_rows_input": len(safe_rows),
        "injected_count": len(injected_rows),
        "skipped_existing_frames": skipped_existing,
        "preview_rows_total": len(merged_rows),
        "preview_sequence_points": len(after_points),
        "missing_before_count": sum(x[2] for x in missing_before),
        "missing_after_count": sum(x[2] for x in missing_after),
        "missing_before_runs": [{"start": a, "end": b, "len": n} for a, b, n in missing_before],
        "missing_after_runs": [{"start": a, "end": b, "len": n} for a, b, n in missing_after],
        "continuity_before": cont_before,
        "continuity_after": cont_after,
        "output_csv": str(output_csv),
        "injected_csv": str(injected_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "note": "Preview propre. Ne remplace pas le track source 005D4.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"safe_csv  = {safe_csv}")
    print(f"video     = {video_path}")
    print(f"sequence  = {sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005D7E")
    print(f"output_csv   = {output_csv}")
    print(f"injected_csv = {injected_csv}")
    print(f"summary      = {summary_path}")
    print(f"overlay      = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"source_sequence_points={len(before_points)}")
    print(f"injected_count={len(injected_rows)}")
    print(f"preview_sequence_points={len(after_points)}")
    print(f"missing_before_count={summary['missing_before_count']}")
    print(f"missing_after_count={summary['missing_after_count']}")
    print(f"missing_before_runs={summary['missing_before_runs']}")
    print(f"missing_after_runs={summary['missing_after_runs']}")
    print(f"continuity_after_speed_p95={cont_after.get('speed_p95')}")
    print(f"continuity_after_large_step_count={cont_after.get('large_step_count')}")


if __name__ == "__main__":
    main()
