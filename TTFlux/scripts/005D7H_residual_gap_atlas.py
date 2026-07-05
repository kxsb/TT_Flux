from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7H_residual_gap_atlas"


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


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def imwrite_unicode(path: Path, img) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_track_map(path: Path, sequence_key: str):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    seq_col = find_col(fields, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Colonnes frame/x/y introuvables: {path}")

    if seq_col and sequence_key:
        rows = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]

    out = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            out[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "score": sc,
                "point_source": str(r.get("point_source", "")),
            }

    return out


def read_candidates(path: Path):
    rows, _ = read_csv(path)
    by_frame = {}

    for r in rows:
        fr = safe_int(r.get("frame"))
        x = safe_float(r.get("x"))
        y = safe_float(r.get("y"))
        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        by_frame.setdefault(fr, []).append(r)

    for fr in by_frame:
        by_frame[fr].sort(key=lambda r: safe_float(r.get("score"), 0.0), reverse=True)

    return by_frame


def nearest_neighbors(track, fr):
    frames = sorted(track)
    left = None
    right = None

    for f in frames:
        if f < fr:
            left = f
        elif f > fr:
            right = f
            break

    return left, right


def interp(track, lf, rf, fr):
    if lf is None or rf is None or lf == rf:
        return None
    a = track[lf]
    b = track[rf]
    t = (fr - lf) / float(rf - lf)
    x = a["x"] * (1 - t) + b["x"] * t
    y = a["y"] * (1 - t) + b["y"] * t
    return x, y, t


def crop_around(img, cx, cy, size=220):
    h, w = img.shape[:2]
    half = size // 2
    x1 = max(0, int(round(cx)) - half)
    y1 = max(0, int(round(cy)) - half)
    x2 = min(w, int(round(cx)) + half)
    y2 = min(h, int(round(cy)) + half)

    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return img.copy(), 0, 0

    return crop, x1, y1


def draw_frame(frame, fr, pred, cands, left_fr, right_fr):
    out = frame.copy()

    if pred:
        px, py, _ = pred
        cv2.drawMarker(out, (int(round(px)), int(round(py))), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(out, (int(round(px)), int(round(py))), 55, (0, 0, 180), 1)
        cv2.circle(out, (int(round(px)), int(round(py))), 95, (0, 0, 120), 1)

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
        elif "motion" in src:
            color = (0, 220, 0)
        else:
            color = (150, 150, 150)

        cv2.circle(out, (int(round(x)), int(round(y))), 6, color, 1)
        cv2.putText(out, f"{idx+1}", (int(round(x)) + 7, int(round(y)) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    label = f"{PATCH_ID} | frame={fr} | left={left_fr} right={right_fr} | cands={len(cands)}"
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 1100), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_crop(crop, offset_x, offset_y, pred, cands):
    out = crop.copy()

    if pred:
        px, py, _ = pred
        lx = int(round(px)) - offset_x
        ly = int(round(py)) - offset_y
        cv2.drawMarker(out, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(out, (lx, ly), 55, (0, 0, 180), 1)

    for idx, c in enumerate(cands[:8]):
        x = safe_float(c.get("x"))
        y = safe_float(c.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        lx = int(round(x)) - offset_x
        ly = int(round(y)) - offset_y
        if 0 <= lx < out.shape[1] and 0 <= ly < out.shape[0]:
            cv2.circle(out, (lx, ly), 6, (255, 0, 255), 1)
            cv2.putText(out, f"{idx+1}", (lx + 7, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)

    return out


def source_class(src):
    s = str(src).lower()
    if "motion_bright" in s:
        return "motion_bright"
    if "bright" in s:
        return "bright"
    if "motion" in s:
        return "motion"
    return "other"


def candidate_diagnostics(cands, pred):
    out = []
    for rank, c in enumerate(cands, 1):
        x = safe_float(c.get("x"))
        y = safe_float(c.get("y"))
        if pred and math.isfinite(x) and math.isfinite(y):
            px, py, _ = pred
            dist = math.hypot(x - px, y - py)
        else:
            dist = np.nan

        cc = dict(c)
        cc["rank"] = rank
        cc["diag_source_class"] = source_class(c.get("source", ""))
        cc["diag_dist_interp"] = "" if not math.isfinite(dist) else f"{dist:.3f}"
        out.append(cc)

    return out


def make_contact_sheet(items, out_path):
    thumbs = []
    labels = []

    for it in items:
        img = imread_unicode(Path(it["crop_img"]))
        if img is None:
            continue
        img = cv2.resize(img, (220, 220))
        thumbs.append(img)
        labels.append(f"f{it['frame']} {it['status']}")

    if not thumbs:
        return False

    cols = 5
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 250, cols * 220, 3), np.uint8)

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y = r * 250
        x = c * 220
        sheet[y:y+220, x:x+220] = img
        cv2.putText(sheet, labels[i][:28], (x + 5, y + 242), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    return imwrite_unicode(out_path, sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-summary", default="runs/005D7F_target_gap_validation/005D7F_target_gap_validation_summary.json")
    ap.add_argument("--track-csv", default="runs/005D7G_gapfilled_canonical/005D7G_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--candidates-csv", default="runs/005D7A_gap_candidates/005D7A_gap_candidates.csv")
    ap.add_argument("--out-dir", default="runs/005D7H_residual_gap_atlas")
    args = ap.parse_args()

    validation_path = Path(args.validation_summary)
    track_csv = Path(args.track_csv)
    candidates_csv = Path(args.candidates_csv)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)
    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not candidates_csv.exists():
        raise FileNotFoundError(candidates_csv)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    video_path = Path(validation["video"])
    sequence_key = validation["sequence_key"]
    residual_frames = [int(x) for x in validation["still_missing_frames"]]
    filled_frames = [int(x) for x in validation["filled_frames"]]

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    track = read_track_map(track_csv, sequence_key)
    cands_by_frame = read_candidates(candidates_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    atlas_rows = []
    candidate_rows = []

    for fr in residual_frames + filled_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        lf, rf = nearest_neighbors(track, fr)
        pred = interp(track, lf, rf, fr) if lf is not None and rf is not None else None
        cands = cands_by_frame.get(fr, [])

        # Tri diagnostic par distance à l'interpolation si possible.
        if pred:
            px, py, _ = pred
            cands = sorted(cands, key=lambda c: math.hypot(safe_float(c.get("x"), 9999) - px, safe_float(c.get("y"), 9999) - py))
        else:
            cands = sorted(cands, key=lambda c: safe_float(c.get("score"), 0.0), reverse=True)

        status = "still_missing" if fr in residual_frames else "filled_005D7G"

        full = draw_frame(frame, fr, pred, cands, lf, rf)
        frame_path = frames_dir / f"frame_{fr:04d}_{status}.jpg"
        imwrite_unicode(frame_path, full)

        if pred:
            px, py, _ = pred
        elif cands:
            px, py = safe_float(cands[0].get("x"), frame.shape[1] / 2), safe_float(cands[0].get("y"), frame.shape[0] / 2)
        else:
            px, py = frame.shape[1] / 2, frame.shape[0] / 2

        crop, ox, oy = crop_around(frame, px, py, size=240)
        crop_drawn = draw_crop(crop, ox, oy, pred, cands)
        crop_path = crops_dir / f"crop_{fr:04d}_{status}.jpg"
        imwrite_unicode(crop_path, crop_drawn)

        best = cands[0] if cands else {}
        best_dist = ""
        if pred and best:
            best_dist = math.hypot(safe_float(best.get("x")) - pred[0], safe_float(best.get("y")) - pred[1])

        atlas_rows.append({
            "frame": fr,
            "status": status,
            "left_frame": "" if lf is None else lf,
            "right_frame": "" if rf is None else rf,
            "pred_x": "" if not pred else f"{pred[0]:.3f}",
            "pred_y": "" if not pred else f"{pred[1]:.3f}",
            "candidate_count": len(cands),
            "best_source": best.get("source", ""),
            "best_score": best.get("score", ""),
            "best_x": best.get("x", ""),
            "best_y": best.get("y", ""),
            "best_dist_interp": "" if best_dist == "" else f"{best_dist:.3f}",
            "frame_img": str(frame_path),
            "crop_img": str(crop_path),
        })

        for cc in candidate_diagnostics(cands[:12], pred):
            row = {
                "frame": fr,
                "status": status,
                "left_frame": "" if lf is None else lf,
                "right_frame": "" if rf is None else rf,
                "pred_x": "" if not pred else f"{pred[0]:.3f}",
                "pred_y": "" if not pred else f"{pred[1]:.3f}",
            }
            for k in ["rank", "x", "y", "r", "area", "source", "score", "dist_pred", "diag_dist_interp", "diag_source_class", "mean_gray", "max_gray", "mean_motion"]:
                row[k] = cc.get(k, "")
            candidate_rows.append(row)

    cap.release()

    atlas_csv = out_dir / "005D7H_residual_gap_atlas.csv"
    cand_csv = out_dir / "005D7H_residual_gap_candidates_ranked.csv"
    sheet_path = out_dir / "005D7H_residual_gap_contact_sheet.jpg"

    write_csv(atlas_csv, atlas_rows, [
        "frame", "status", "left_frame", "right_frame",
        "pred_x", "pred_y", "candidate_count",
        "best_source", "best_score", "best_x", "best_y", "best_dist_interp",
        "frame_img", "crop_img",
    ])

    write_csv(cand_csv, candidate_rows, [
        "frame", "status", "left_frame", "right_frame",
        "pred_x", "pred_y",
        "rank", "x", "y", "r", "area", "source", "score",
        "dist_pred", "diag_dist_interp", "diag_source_class",
        "mean_gray", "max_gray", "mean_motion",
    ])

    make_contact_sheet(atlas_rows, sheet_path)

    # HTML léger.
    html_path = out_dir / "005D7H_residual_gap_atlas.html"

    cards = []
    for r in atlas_rows:
        crop_rel = Path(r["crop_img"]).relative_to(out_dir).as_posix()
        frame_rel = Path(r["frame_img"]).relative_to(out_dir).as_posix()
        cards.append(f"""
        <div class="card {html.escape(r['status'])}">
          <h3>Frame {r['frame']} — {html.escape(r['status'])}</h3>
          <p>
            left={r['left_frame']} right={r['right_frame']} |
            pred=({r['pred_x']}, {r['pred_y']}) |
            cands={r['candidate_count']} |
            best={html.escape(str(r['best_source']))} score={html.escape(str(r['best_score']))} dist={r['best_dist_interp']}
          </p>
          <a href="{frame_rel}"><img src="{crop_rel}"></a>
        </div>
        """)

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{PATCH_ID}</title>
<style>
body {{
  font-family: Arial, sans-serif;
  background:#111;
  color:#eee;
  margin:24px;
}}
h1 {{ margin-bottom: 4px; }}
.meta {{ color:#bbb; margin-bottom: 20px; }}
.grid {{
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
  gap:16px;
}}
.card {{
  border:1px solid #333;
  background:#1b1b1b;
  padding:12px;
  border-radius:10px;
}}
.card h3 {{ margin:0 0 8px 0; font-size:16px; }}
.card p {{ color:#bbb; font-size:12px; line-height:1.35; }}
.card img {{
  width:100%;
  border-radius:8px;
  border:1px solid #444;
}}
.filled_005D7G {{ border-color:#aa44ff; }}
.still_missing {{ border-color:#884444; }}
a {{ color:#9cf; }}
</style>
</head>
<body>
<h1>{PATCH_ID}</h1>
<div class="meta">
sequence={html.escape(sequence_key)} |
residual={len(residual_frames)} |
filled_reference={len(filled_frames)} |
track={html.escape(str(track_csv))}
<br>
<a href="{sheet_path.name}">contact sheet</a> |
<a href="{atlas_csv.name}">atlas csv</a> |
<a href="{cand_csv.name}">candidates csv</a>
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    summary = {
        "patch": PATCH_ID,
        "validation_summary": str(validation_path),
        "track_csv": str(track_csv),
        "candidates_csv": str(candidates_csv),
        "video": str(video_path),
        "sequence_key": sequence_key,
        "residual_frame_count": len(residual_frames),
        "filled_reference_count": len(filled_frames),
        "atlas_csv": str(atlas_csv),
        "candidates_ranked_csv": str(cand_csv),
        "contact_sheet": str(sheet_path),
        "html": str(html_path),
        "residual_frames": residual_frames,
        "filled_reference_frames": filled_frames,
    }

    summary_path = out_dir / "005D7H_residual_gap_atlas_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"validation = {validation_path}")
    print(f"track_csv  = {track_csv}")
    print(f"candidates = {candidates_csv}")
    print(f"video      = {video_path}")
    print(f"sequence   = {sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005D7H")
    print(f"summary       = {summary_path}")
    print(f"html          = {html_path}")
    print(f"contact_sheet = {sheet_path}")
    print(f"atlas_csv     = {atlas_csv}")
    print(f"candidates    = {cand_csv}")
    print("")
    print(f"residual_frame_count={len(residual_frames)}")
    print(f"filled_reference_count={len(filled_frames)}")


if __name__ == "__main__":
    main()
