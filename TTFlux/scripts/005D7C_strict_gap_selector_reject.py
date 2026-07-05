from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7C_strict_gap_selector_reject"


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


def source_class(src: str) -> str:
    s = (src or "").lower()
    if "motion_bright" in s:
        return "motion_bright"
    if "bright" in s and "motion" in s:
        return "bright_motion"
    if "bright" in s:
        return "bright"
    if "motion" in s:
        return "motion"
    return "other"


def source_bonus(src: str) -> float:
    cls = source_class(src)
    if cls == "motion_bright":
        return 0.32
    if cls == "bright_motion":
        return 0.24
    if cls == "bright":
        return 0.12
    if cls == "motion":
        return -0.18
    return -0.10


def candidate_score(c: dict, max_dist: float) -> float:
    raw = safe_float(c.get("score"), 0.0)
    dist = safe_float(c.get("dist_pred"), 9999.0)
    area = safe_float(c.get("area"), 0.0)
    aspect = safe_float(c.get("aspect"), 9.0)

    prox = max(0.0, 1.0 - min(dist, max_dist) / max_dist)
    area_bonus = max(0.0, 1.0 - abs(area - 18.0) / 120.0)
    compact = max(0.0, 1.0 - min(aspect, 5.0) / 5.0)

    return (
        0.44 * prox
        + 0.28 * raw
        + 0.18 * source_bonus(str(c.get("source", "")))
        + 0.06 * area_bonus
        + 0.04 * compact
    )


def gate_reason(c: dict, args) -> tuple[bool, str, str]:
    dist = safe_float(c.get("dist_pred"), 9999.0)
    raw = safe_float(c.get("score"), 0.0)
    area = safe_float(c.get("area"), 0.0)
    aspect = safe_float(c.get("aspect"), 9.0)
    src = str(c.get("source", ""))
    cls = source_class(src)

    if not math.isfinite(dist):
        return False, "reject_no_dist_pred", cls

    if area < args.min_area or area > args.max_area:
        return False, "reject_area", cls

    if aspect > args.max_aspect:
        return False, "reject_aspect", cls

    # Tier A : très proche de la prédiction, source visuelle acceptable.
    if dist <= args.strict_px and cls in {"motion_bright", "bright_motion", "bright"}:
        return True, "accept_strict_near_pred", cls

    # Tier B : proche et très bon signal motion+bright.
    if dist <= args.relaxed_px and cls in {"motion_bright", "bright_motion"} and raw >= args.min_raw_relaxed:
        return True, "accept_relaxed_motion_bright", cls

    # Tier C : fallback bright, mais seulement assez proche.
    if dist <= args.bright_px and cls == "bright" and raw >= args.min_raw_bright:
        return True, "accept_bright_near_pred", cls

    # On évite volontairement les purs motion : pieds, raquette, jambes, reflets.
    if cls == "motion":
        return False, "reject_pure_motion", cls

    if dist > args.relaxed_px:
        return False, "reject_far_from_pred", cls

    return False, "reject_low_confidence", cls


def group_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    out = {}
    for r in rows:
        fr = safe_int(r.get("frame"))
        if fr < 0:
            continue
        out.setdefault(fr, []).append(r)
    return out


def select_frame(fr: int, cands: list[dict], args) -> dict:
    accepted = []
    rejected = []

    for c in cands:
        ok, reason, cls = gate_reason(c, args)
        cc = dict(c)
        cc["_gate_reason"] = reason
        cc["_source_class"] = cls
        cc["_strict_score"] = candidate_score(cc, args.relaxed_px)
        if ok:
            accepted.append(cc)
        else:
            rejected.append(cc)

    accepted.sort(key=lambda x: safe_float(x.get("_strict_score"), -999), reverse=True)
    rejected.sort(key=lambda x: safe_float(x.get("_strict_score"), -999), reverse=True)

    if accepted:
        best = accepted[0]
        status = "accepted"
    else:
        best = rejected[0] if rejected else {}
        status = "rejected"

    out = dict(best)
    out["frame"] = fr
    out["strict_status"] = status
    out["strict_score"] = f"{safe_float(best.get('_strict_score'), 0.0):.6f}" if best else ""
    out["strict_gate_reason"] = best.get("_gate_reason", "no_candidate") if best else "no_candidate"
    out["source_class"] = best.get("_source_class", "") if best else ""
    out["accepted_count_this_frame"] = len(accepted)
    out["rejected_count_this_frame"] = len(rejected)
    out["candidate_count_this_frame"] = len(cands)

    return out


def infer_paths(rows: list[dict], args):
    video = Path(args.video) if args.video else None
    track = Path(args.track_csv) if args.track_csv else None
    sequence = args.sequence_key or ""

    if rows:
        r0 = rows[0]
        if video is None and r0.get("video"):
            video = Path(str(r0["video"]))
        if track is None and r0.get("track_csv"):
            track = Path(str(r0["track_csv"]))
        if not sequence and r0.get("chosen_sequence"):
            sequence = str(r0["chosen_sequence"])

    return video, track, sequence


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


def draw_frame(frame, frame_idx, cands, selected, base_xy):
    out = frame.copy()

    # Prédiction interpolée.
    pred_x = pred_y = np.nan
    if cands:
        pred_x = safe_float(cands[0].get("pred_x"), np.nan)
        pred_y = safe_float(cands[0].get("pred_y"), np.nan)

    if math.isfinite(pred_x) and math.isfinite(pred_y):
        cv2.drawMarker(
            out,
            (int(round(pred_x)), int(round(pred_y))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )
        cv2.circle(out, (int(round(pred_x)), int(round(pred_y))), 55, (0, 0, 180), 1)
        cv2.circle(out, (int(round(pred_x)), int(round(pred_y))), 95, (0, 0, 120), 1)

    if base_xy is not None:
        bx, by, _ = base_xy
        cv2.circle(out, (int(round(bx)), int(round(by))), 5, (180, 180, 180), 1)
        cv2.putText(out, "base", (int(bx) + 7, int(by) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Tous les candidats, petits marqueurs.
    sorted_cands = sorted(cands, key=lambda c: safe_float(c.get("dist_pred"), 9999.0))
    for i, c in enumerate(sorted_cands[:16]):
        x = safe_float(c.get("x"))
        y = safe_float(c.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        cls = source_class(str(c.get("source", "")))
        if cls == "motion_bright":
            color = (0, 255, 255)
        elif cls == "bright_motion":
            color = (0, 210, 255)
        elif cls == "bright":
            color = (0, 150, 255)
        elif cls == "motion":
            color = (0, 180, 0)
        else:
            color = (160, 160, 160)

        cv2.circle(out, (int(round(x)), int(round(y))), 4, color, 1)
        cv2.putText(
            out,
            f"d{int(round(safe_float(c.get('dist_pred'), 0)))}",
            (int(round(x)) + 5, int(round(y)) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    # Sélection stricte : magenta si accepté, gris si rejeté.
    if selected:
        sx = safe_float(selected.get("x"))
        sy = safe_float(selected.get("y"))
        status = selected.get("strict_status", "rejected")
        if math.isfinite(sx) and math.isfinite(sy):
            color = (255, 0, 255) if status == "accepted" else (120, 120, 120)
            cv2.circle(out, (int(round(sx)), int(round(sy))), 13, color, 2)
            cv2.putText(
                out,
                "ACCEPT" if status == "accepted" else "REJECT",
                (int(round(sx)) + 14, int(round(sy)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    label = (
        f"{PATCH_ID} | frame={frame_idx} | "
        f"status={selected.get('strict_status', '') if selected else 'none'} | "
        f"reason={selected.get('strict_gate_reason', '') if selected else ''} | "
        f"cands={len(cands)}"
    )
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 1220), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def write_overlay(video_path: Path, out_path: Path, by_frame, selected_by_frame, base_track, full_video: bool):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir la vidéo: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"WARN: impossible d'écrire l'overlay: {out_path}")
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
            out = draw_frame(
                frame=frame,
                frame_idx=frame_idx,
                cands=by_frame.get(frame_idx, []),
                selected=selected_by_frame.get(frame_idx, {}),
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
    ap.add_argument("--out-dir", default="runs/005D7C_strict_gap_selected")
    ap.add_argument("--full-video-overlay", action="store_true")

    # Seuils volontairement stricts : on préfère laisser un trou plutôt qu'injecter un faux point.
    ap.add_argument("--strict-px", type=float, default=58.0)
    ap.add_argument("--relaxed-px", type=float, default=92.0)
    ap.add_argument("--bright-px", type=float, default=72.0)
    ap.add_argument("--min-raw-relaxed", type=float, default=0.56)
    ap.add_argument("--min-raw-bright", type=float, default=0.62)
    ap.add_argument("--min-area", type=float, default=2.0)
    ap.add_argument("--max-area", type=float, default=120.0)
    ap.add_argument("--max-aspect", type=float, default=3.4)

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

    by_frame = group_by_frame(rows)

    selected_rows = []
    for fr in sorted(by_frame):
        selected_rows.append(select_frame(fr, by_frame[fr], args))

    selected_by_frame = {safe_int(r.get("frame")): r for r in selected_rows}
    accepted = [r for r in selected_rows if r.get("strict_status") == "accepted"]
    rejected = [r for r in selected_rows if r.get("strict_status") != "accepted"]

    fields = [
        "patch",
        "strict_status",
        "strict_gate_reason",
        "source_class",
        "strict_score",
        "video",
        "track_csv",
        "chosen_sequence",
        "gap_id",
        "gap_start",
        "gap_end",
        "frame",
        "candidate_count_this_frame",
        "accepted_count_this_frame",
        "rejected_count_this_frame",
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
        "mean_gray",
        "max_gray",
        "mean_motion",
        "pred_x",
        "pred_y",
        "pred_mode",
        "dist_pred",
        "roi_mode",
    ]

    for r in selected_rows:
        r["patch"] = PATCH_ID

    selected_csv = out_dir / "005D7C_strict_selected_all_frames.csv"
    accepted_csv = out_dir / "005D7C_strict_accepted_only.csv"
    write_csv(selected_csv, selected_rows, fields)
    write_csv(accepted_csv, accepted, fields)

    base_track = read_track_points(track_csv, sequence)

    preview_rows = []
    for fr in sorted(base_track):
        x, y, sc = base_track[fr]
        preview_rows.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "score": f"{sc:.6f}",
            "point_source": "base_005D4",
            "gap_id": "",
            "strict_status": "",
            "strict_gate_reason": "",
        })

    for r in accepted:
        fr = safe_int(r.get("frame"))
        if fr in base_track:
            continue
        preview_rows.append({
            "sequence_key": sequence,
            "frame": fr,
            "x": f"{safe_float(r.get('x')):.3f}",
            "y": f"{safe_float(r.get('y')):.3f}",
            "score": f"{safe_float(r.get('strict_score'), 0.0):.6f}",
            "point_source": "gapfill_005D7C_STRICT_PREVIEW",
            "gap_id": r.get("gap_id", ""),
            "strict_status": r.get("strict_status", ""),
            "strict_gate_reason": r.get("strict_gate_reason", ""),
        })

    preview_rows.sort(key=lambda r: safe_int(r.get("frame")))
    preview_csv = out_dir / "005D7C_strict_gapfilled_preview_points.csv"
    write_csv(
        preview_csv,
        preview_rows,
        ["sequence_key", "frame", "x", "y", "score", "point_source", "gap_id", "strict_status", "strict_gate_reason"],
    )

    overlay_path = out_dir / "005D7C_strict_selector_overlay.mp4"
    overlay_ok = write_overlay(
        video_path=video_path,
        out_path=overlay_path,
        by_frame=by_frame,
        selected_by_frame=selected_by_frame,
        base_track=base_track,
        full_video=args.full_video_overlay,
    )

    accepted_dists = [
        safe_float(r.get("dist_pred"))
        for r in accepted
        if math.isfinite(safe_float(r.get("dist_pred")))
    ]

    reason_counts = {}
    source_counts = {}
    for r in selected_rows:
        reason_counts[r.get("strict_gate_reason", "")] = reason_counts.get(r.get("strict_gate_reason", ""), 0) + 1
        source_counts[r.get("source_class", "")] = source_counts.get(r.get("source_class", ""), 0) + 1

    summary = {
        "patch": PATCH_ID,
        "input_candidates_csv": str(candidates_csv),
        "video": str(video_path),
        "track_csv": str(track_csv) if track_csv else None,
        "sequence_key": sequence,
        "gap_frames": len(by_frame),
        "candidate_total": len(rows),
        "candidate_mean_per_gap_frame": len(rows) / max(1, len(by_frame)),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_ratio": len(accepted) / max(1, len(by_frame)),
        "accepted_dist_pred_mean": float(np.mean(accepted_dists)) if accepted_dists else None,
        "accepted_dist_pred_median": float(np.median(accepted_dists)) if accepted_dists else None,
        "accepted_dist_pred_p90": float(np.percentile(accepted_dists, 90)) if accepted_dists else None,
        "reason_counts": reason_counts,
        "source_counts": source_counts,
        "selected_csv": str(selected_csv),
        "accepted_csv": str(accepted_csv),
        "preview_csv": str(preview_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "params": vars(args),
        "note": "Strict preview seulement. Aucun merge final. Les frames rejetées restent des gaps.",
    }

    summary_path = out_dir / "005D7C_strict_selector_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"candidates_csv = {candidates_csv}")
    print(f"video          = {video_path}")
    print(f"track_csv      = {track_csv}")
    print(f"sequence       = {sequence}")
    print("=" * 72)
    print("")
    print("OK 005D7C")
    print(f"selected_csv = {selected_csv}")
    print(f"accepted_csv = {accepted_csv}")
    print(f"preview_csv  = {preview_csv}")
    print(f"summary      = {summary_path}")
    print(f"overlay      = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"gap_frames={len(by_frame)} candidates_total={len(rows)}")
    print(f"accepted_count={len(accepted)} rejected_count={len(rejected)} accepted_ratio={summary['accepted_ratio']:.3f}")
    if accepted_dists:
        print(f"accepted_dist_pred_median={summary['accepted_dist_pred_median']:.2f} p90={summary['accepted_dist_pred_p90']:.2f}")
    print(f"reason_counts={reason_counts}")
    print(f"source_counts={source_counts}")


if __name__ == "__main__":
    main()
