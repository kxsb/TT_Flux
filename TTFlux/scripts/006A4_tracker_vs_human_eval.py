from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006A4_tracker_vs_human_eval"

ROOT = Path.cwd()
LABEL_DIR = ROOT / "runs" / "006A_human_frame_labels"
OUT_DIR = LABEL_DIR / "006A4_tracker_vs_human_eval"

POSITION_CSV = LABEL_DIR / "006A2_position_training_rows.csv"
VISIBILITY_CSV = LABEL_DIR / "006A2_visibility_training_rows.csv"

EXCLUDE_PATH_TOKENS = [
    "006A_human_frame_labels",
    "external_data",
    "ttnet_image_extract",
    "integrity",
    "manifest",
    "summary",
    "dataset",
    "registry",
]

FRAME_COLS = ["video_frame", "frame", "frame_id", "frame_idx", "img_frame_id", "fid"]
X_COLS = ["x", "ball_x", "pred_x", "cx", "center_x", "x_px"]
Y_COLS = ["y", "ball_y", "pred_y", "cy", "center_y", "y_px"]
SCORE_COLS = ["score", "confidence", "conf", "prob", "probability", "p", "likelihood", "selected_score"]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def as_int(v):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def as_float(v):
    try:
        return float(str(v).strip().replace(",", "."))
    except Exception:
        return None


def norm_bool(v):
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "visible"}:
        return "1"
    if s in {"0", "false", "no", "n", "invisible"}:
        return "0"
    return ""


def find_col(fieldnames, candidates):
    lower = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def csv_header(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def looks_like_prediction_csv(path: Path):
    s = rel(path).lower()
    if any(tok in s for tok in EXCLUDE_PATH_TOKENS):
        return False

    header = csv_header(path)
    if not header:
        return False

    fcol = find_col(header, FRAME_COLS)
    xcol = find_col(header, X_COLS)
    ycol = find_col(header, Y_COLS)

    return bool(fcol and xcol and ycol)


def extract_prediction_rows(path: Path, max_rows=None):
    rows = read_csv(path)
    if max_rows:
        rows = rows[:max_rows]

    if not rows:
        return [], {}

    fieldnames = list(rows[0].keys())

    fcol = find_col(fieldnames, FRAME_COLS)
    xcol = find_col(fieldnames, X_COLS)
    ycol = find_col(fieldnames, Y_COLS)
    scol = find_col(fieldnames, SCORE_COLS)

    if not (fcol and xcol and ycol):
        return [], {}

    out = []
    for i, r in enumerate(rows):
        frame = as_int(r.get(fcol))
        x = as_float(r.get(xcol))
        y = as_float(r.get(ycol))
        score = as_float(r.get(scol)) if scol else None

        if frame is None or x is None or y is None:
            continue

        out.append({
            "source_csv": rel(path),
            "source_row": i + 1,
            "frame": frame,
            "x": x,
            "y": y,
            "score": score if score is not None else "",
            "raw_score_col": scol or "",
        })

    meta = {
        "frame_col": fcol,
        "x_col": xcol,
        "y_col": ycol,
        "score_col": scol or "",
        "raw_rows": len(rows),
        "valid_rows": len(out),
    }

    return out, meta


def best_prediction_by_frame(pred_rows: list[dict]):
    grouped = {}

    for r in pred_rows:
        frame = r["frame"]
        if frame not in grouped:
            grouped[frame] = r
            continue

        old = grouped[frame]

        old_score = as_float(old.get("score"))
        new_score = as_float(r.get("score"))

        if old_score is None and new_score is not None:
            grouped[frame] = r
        elif old_score is not None and new_score is not None and new_score > old_score:
            grouped[frame] = r

    return grouped


def find_pred_for_frame(pred_by_frame, frame: int, tolerance: int):
    if frame in pred_by_frame:
        return pred_by_frame[frame], 0

    if tolerance <= 0:
        return None, None

    best = None
    best_dt = None

    for dt in range(1, tolerance + 1):
        for ff in [frame - dt, frame + dt]:
            if ff in pred_by_frame:
                cand = pred_by_frame[ff]
                if best is None:
                    best = cand
                    best_dt = abs(ff - frame)

    return best, best_dt


def scan_prediction_sources(position_frames, visibility_frames):
    candidates = []

    for path in (ROOT / "runs").rglob("*.csv"):
        if not looks_like_prediction_csv(path):
            continue

        try:
            size_mb = path.stat().st_size / 1024 / 1024
        except Exception:
            size_mb = 0

        # Evite de charger des monstres absurdes ici.
        if size_mb > 250:
            continue

        pred_rows, meta = extract_prediction_rows(path)
        if not pred_rows:
            continue

        frames = {r["frame"] for r in pred_rows}

        overlap_position = len(set(position_frames) & frames)
        overlap_visibility = len(set(visibility_frames) & frames)

        candidates.append({
            "path": rel(path),
            "size_mb": round(size_mb, 3),
            "valid_rows": len(pred_rows),
            "unique_frames": len(frames),
            "overlap_position_frames": overlap_position,
            "overlap_visibility_frames": overlap_visibility,
            "frame_col": meta.get("frame_col", ""),
            "x_col": meta.get("x_col", ""),
            "y_col": meta.get("y_col", ""),
            "score_col": meta.get("score_col", ""),
        })

    candidates.sort(
        key=lambda r: (
            r["overlap_position_frames"],
            r["overlap_visibility_frames"],
            r["unique_frames"],
            r["valid_rows"],
        ),
        reverse=True,
    )

    return candidates


def imwrite_unicode(path: Path, img, jpg_quality=92):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def make_sheet(images, cell_w=320, cell_h=180, cols=4):
    if not images:
        return None

    rows = math.ceil(len(images) / cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        thumb = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = thumb

    return sheet


def get_video_path(rows):
    paths = sorted({r.get("video_path", "").strip() for r in rows if r.get("video_path", "").strip()})
    if not paths:
        return None
    return ROOT / paths[0]


def draw_eval_sheets(position_rows, eval_rows, video_path: Path):
    if not video_path or not video_path.exists():
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    by_uid = {r["label_uid"]: r for r in eval_rows}
    thumbs = []
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for idx, hr in enumerate(position_rows, 1):
        uid = hr["label_uid"]
        er = by_uid.get(uid)
        frame = as_int(hr.get("video_frame"))
        hx = as_float(hr.get("x"))
        hy = as_float(hr.get("y"))

        if frame is None or hx is None or hy is None:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok or img is None:
            continue

        overlay = img.copy()
        hx_i = int(round(hx))
        hy_i = int(round(hy))

        # Humain = rouge
        cv2.circle(overlay, (hx_i, hy_i), 14, (0, 0, 255), 2)
        cv2.drawMarker(overlay, (hx_i, hy_i), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)

        label = f"{idx:02d} f={frame} human=red"

        if er and er.get("matched") == "1":
            px_i = int(round(float(er["pred_x"])))
            py_i = int(round(float(er["pred_y"])))

            # Prediction = vert
            cv2.circle(overlay, (px_i, py_i), 14, (0, 255, 0), 2)
            cv2.drawMarker(overlay, (px_i, py_i), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 28, 2)
            cv2.line(overlay, (hx_i, hy_i), (px_i, py_i), (0, 255, 255), 2)

            label += f" pred=green err={float(er['dist_px']):.1f}px"
        else:
            label += " pred=missing"

        cv2.putText(
            overlay,
            label,
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out_img = frames_dir / f"{idx:02d}_frame_{frame:06d}_eval.jpg"
        imwrite_unicode(out_img, overlay)

        thumbs.append(overlay)

    cap.release()

    sheet = make_sheet(thumbs, cell_w=320, cell_h=180, cols=4)
    sheet_path = OUT_DIR / "006A4_tracker_vs_human_contact_sheet.jpg"

    if sheet is not None:
        imwrite_unicode(sheet_path, sheet)

    return {
        "contact_sheet": str(sheet_path),
        "frames_dir": str(frames_dir),
        "sheet_count": len(thumbs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", default="", help="Prediction/tracking CSV. If omitted, auto-scan runs/*.csv")
    parser.add_argument("--frame-tolerance", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    position_rows = read_csv(POSITION_CSV)
    visibility_rows = read_csv(VISIBILITY_CSV)

    position_rows = [r for r in position_rows if r.get("usable_for_position") == "1"]
    visibility_rows = [r for r in visibility_rows if r.get("usable_for_visibility") == "1"]

    position_frames = [as_int(r.get("video_frame")) for r in position_rows]
    position_frames = [f for f in position_frames if f is not None]

    visibility_frames = [as_int(r.get("video_frame")) for r in visibility_rows]
    visibility_frames = [f for f in visibility_frames if f is not None]

    source_audit = scan_prediction_sources(position_frames, visibility_frames)
    write_csv(OUT_DIR / "006A4_prediction_source_audit.csv", source_audit)

    if args.pred_csv:
        pred_csv = ROOT / args.pred_csv
    else:
        if not source_audit:
            raise SystemExit("No prediction CSV found. Check runs/ or pass --pred-csv manually.")
        pred_csv = ROOT / source_audit[0]["path"]

    pred_rows, pred_meta = extract_prediction_rows(pred_csv)
    pred_by_frame = best_prediction_by_frame(pred_rows)

    eval_position = []

    for hr in position_rows:
        frame = as_int(hr.get("video_frame"))
        hx = as_float(hr.get("x"))
        hy = as_float(hr.get("y"))

        pred, dt = find_pred_for_frame(pred_by_frame, frame, args.frame_tolerance)

        row = {
            "label_uid": hr.get("label_uid", ""),
            "frame": frame,
            "human_x": hx,
            "human_y": hy,
            "matched": "0",
            "pred_frame": "",
            "pred_x": "",
            "pred_y": "",
            "pred_score": "",
            "frame_dt": "",
            "dx": "",
            "dy": "",
            "dist_px": "",
            "hit_10": "0",
            "hit_25": "0",
            "hit_50": "0",
            "hit_100": "0",
        }

        if pred is not None:
            px = float(pred["x"])
            py = float(pred["y"])
            dx = px - hx
            dy = py - hy
            dist = math.sqrt(dx * dx + dy * dy)

            row.update({
                "matched": "1",
                "pred_frame": pred["frame"],
                "pred_x": round(px, 3),
                "pred_y": round(py, 3),
                "pred_score": pred.get("score", ""),
                "frame_dt": dt,
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "dist_px": round(dist, 3),
                "hit_10": "1" if dist <= 10 else "0",
                "hit_25": "1" if dist <= 25 else "0",
                "hit_50": "1" if dist <= 50 else "0",
                "hit_100": "1" if dist <= 100 else "0",
            })

        eval_position.append(row)

    eval_visibility = []

    for hr in visibility_rows:
        frame = as_int(hr.get("video_frame"))
        visible = norm_bool(hr.get("visible"))

        pred, dt = find_pred_for_frame(pred_by_frame, frame, args.frame_tolerance)
        has_pred = pred is not None

        status = ""
        if visible == "1" and has_pred:
            status = "tp_visible_predicted"
        elif visible == "1" and not has_pred:
            status = "fn_visible_missing"
        elif visible == "0" and has_pred:
            status = "fp_invisible_predicted"
        elif visible == "0" and not has_pred:
            status = "tn_invisible_empty"

        eval_visibility.append({
            "label_uid": hr.get("label_uid", ""),
            "frame": frame,
            "visible": visible,
            "has_prediction": "1" if has_pred else "0",
            "pred_frame": pred["frame"] if pred else "",
            "pred_x": round(float(pred["x"]), 3) if pred else "",
            "pred_y": round(float(pred["y"]), 3) if pred else "",
            "pred_score": pred.get("score", "") if pred else "",
            "frame_dt": dt if pred else "",
            "status": status,
        })

    write_csv(OUT_DIR / "006A4_position_eval.csv", eval_position)
    write_csv(OUT_DIR / "006A4_visibility_eval.csv", eval_visibility)

    dists = [float(r["dist_px"]) for r in eval_position if r.get("matched") == "1" and r.get("dist_px") != ""]
    matched = len(dists)
    total_pos = len(eval_position)

    def avg(vals):
        return sum(vals) / len(vals) if vals else None

    def median(vals):
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        if n % 2:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])

    vis_counts = {}
    for r in eval_visibility:
        vis_counts[r["status"]] = vis_counts.get(r["status"], 0) + 1

    video_path = get_video_path(position_rows)
    sheet_info = draw_eval_sheets(position_rows, eval_position, video_path)

    summary = {
        "patch": PATCH_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_csv": rel(pred_csv),
        "prediction_meta": pred_meta,
        "prediction_rows": len(pred_rows),
        "prediction_unique_frames": len(pred_by_frame),
        "frame_tolerance": args.frame_tolerance,
        "position_total": total_pos,
        "position_matched": matched,
        "position_missing": total_pos - matched,
        "mean_px": round(avg(dists), 3) if dists else None,
        "median_px": round(median(dists), 3) if dists else None,
        "max_px": round(max(dists), 3) if dists else None,
        "hit_10": sum(1 for r in eval_position if r.get("hit_10") == "1"),
        "hit_25": sum(1 for r in eval_position if r.get("hit_25") == "1"),
        "hit_50": sum(1 for r in eval_position if r.get("hit_50") == "1"),
        "hit_100": sum(1 for r in eval_position if r.get("hit_100") == "1"),
        "visibility_counts": vis_counts,
        "source_audit_csv": str(OUT_DIR / "006A4_prediction_source_audit.csv"),
        "position_eval_csv": str(OUT_DIR / "006A4_position_eval.csv"),
        "visibility_eval_csv": str(OUT_DIR / "006A4_visibility_eval.csv"),
        "contact_sheet": sheet_info.get("contact_sheet", ""),
        "ok_for_next": matched >= 5,
    }

    summary_path = OUT_DIR / "006A4_tracker_vs_human_summary.json"
    md_path = OUT_DIR / "006A4_tracker_vs_human_summary.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# 006A4 tracker vs human eval")
    md.append("")
    md.append(f"- Prediction CSV: `{summary['prediction_csv']}`")
    md.append(f"- Prediction rows: **{summary['prediction_rows']}**")
    md.append(f"- Prediction unique frames: **{summary['prediction_unique_frames']}**")
    md.append(f"- Frame tolerance: **{args.frame_tolerance}**")
    md.append("")
    md.append("## Position")
    md.append("")
    md.append(f"- Total: **{total_pos}**")
    md.append(f"- Matched: **{matched}**")
    md.append(f"- Missing: **{total_pos - matched}**")
    md.append(f"- Mean px: **{summary['mean_px']}**")
    md.append(f"- Median px: **{summary['median_px']}**")
    md.append(f"- Max px: **{summary['max_px']}**")
    md.append(f"- Hit <=10px: **{summary['hit_10']} / {total_pos}**")
    md.append(f"- Hit <=25px: **{summary['hit_25']} / {total_pos}**")
    md.append(f"- Hit <=50px: **{summary['hit_50']} / {total_pos}**")
    md.append(f"- Hit <=100px: **{summary['hit_100']} / {total_pos}**")
    md.append("")
    md.append("## Visibility")
    md.append("")
    for k, v in sorted(vis_counts.items()):
        md.append(f"- `{k}`: **{v}**")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Source audit: `{summary['source_audit_csv']}`")
    md.append(f"- Position eval: `{summary['position_eval_csv']}`")
    md.append(f"- Visibility eval: `{summary['visibility_eval_csv']}`")
    md.append(f"- Contact sheet: `{summary['contact_sheet']}`")
    md.append("")
    md.append(f"- ok_for_next: **{summary['ok_for_next']}**")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006A4")
    print(f"summary = {summary_path}")
    print(f"source_audit = {OUT_DIR / '006A4_prediction_source_audit.csv'}")
    print(f"position_eval = {OUT_DIR / '006A4_position_eval.csv'}")
    print(f"visibility_eval = {OUT_DIR / '006A4_visibility_eval.csv'}")
    print(f"contact_sheet = {summary['contact_sheet']}")
    print("")
    print(f"prediction_csv={summary['prediction_csv']}")
    print(f"position matched={matched}/{total_pos}")
    print(f"median_px={summary['median_px']} mean_px={summary['mean_px']} max_px={summary['max_px']}")
    print(f"hit25={summary['hit_25']}/{total_pos} hit50={summary['hit_50']}/{total_pos} hit100={summary['hit_100']}/{total_pos}")
    print(f"visibility={vis_counts}")
    print(f"ok_for_next={summary['ok_for_next']}")


if __name__ == "__main__":
    main()
