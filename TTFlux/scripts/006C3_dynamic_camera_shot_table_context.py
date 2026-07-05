from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006C3_dynamic_camera_shot_table_context"


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 50.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def frame_signature(img):
    h, w = img.shape[:2]
    roi = img[int(h * 0.05):int(h * 0.95), int(w * 0.03):int(w * 0.97)]
    small = cv2.resize(roi, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist /= hist.sum() + 1e-6

    return gray, hist


def sig_diff(a, b):
    ga, ha = a
    gb, hb = b
    d_gray = float(np.mean(np.abs(ga - gb)))
    d_hist = float(cv2.compareHist(ha.astype(np.float32), hb.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
    return 0.65 * d_gray + 0.35 * d_hist


def detect_camera_cuts(video_path: Path, step=3, min_shot_len=35):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sigs = []

    for fr in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok or img is None:
            continue
        sigs.append((fr, frame_signature(img)))

    cap.release()

    diffs = []
    for (f0, s0), (f1, s1) in zip(sigs[:-1], sigs[1:]):
        diffs.append((f1, sig_diff(s0, s1)))

    if not diffs:
        return [0], [], 0.0

    vals = np.array([d for _, d in diffs], dtype=np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    threshold = max(0.32, med + 7.0 * mad)

    raw_cuts = [fr for fr, d in diffs if d >= threshold]

    cuts = [0]
    for c in raw_cuts:
        if c - cuts[-1] >= min_shot_len:
            cuts.append(c)

    # Supprimer les shots trop courts en fusionnant.
    merged = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] >= min_shot_len:
            merged.append(c)

    return merged, diffs, threshold


def quad_from_cols(row, prefix, suffix):
    cols = [
        f"{prefix}_tl_x_{suffix}", f"{prefix}_tl_y_{suffix}",
        f"{prefix}_tr_x_{suffix}", f"{prefix}_tr_y_{suffix}",
        f"{prefix}_br_x_{suffix}", f"{prefix}_br_y_{suffix}",
        f"{prefix}_bl_x_{suffix}", f"{prefix}_bl_y_{suffix}",
    ]
    vals = [safe_float(row.get(c), np.nan) for c in cols]
    if all(math.isfinite(v) for v in vals):
        q = np.array([
            [vals[0], vals[1]],
            [vals[2], vals[3]],
            [vals[4], vals[5]],
            [vals[6], vals[7]],
        ], dtype=np.float32)
        if cv2.contourArea(q.reshape(-1, 1, 2)) > 500:
            return q, f"{prefix}_{suffix}"
    return None, ""


def extract_quad(row):
    ordered = [
        ("opt", "005C9A"),
        ("snap", "005C7A"),
        ("quad", "005C5C2"),
        ("quad", "005C3"),
        ("quad", "005C1"),
    ]
    for prefix, suffix in ordered:
        q, src = quad_from_cols(row, prefix, suffix)
        if q is not None:
            return q, src
    return None, ""


def get_review_id(row):
    for k in ["review_id", "rally_id", "rally_id_vote", "sequence_key", "clip_id"]:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def load_table_templates(fit_csv: Path):
    rows, _ = read_csv(fit_csv)
    templates = []

    for i, row in enumerate(rows):
        quad, qsrc = extract_quad(row)
        if quad is None:
            continue

        templates.append({
            "template_id": i,
            "review_id": get_review_id(row),
            "quad_source": qsrc,
            "quad": quad,
            "quality": safe_float(row.get("opt_score_005C9A"), 0.0),
            "video_id": str(row.get("video_id", "")),
            "clip_path": str(row.get("clip_path", "")),
        })

    return templates


def blue_table_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 30, 35], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def score_quad_on_frame(img, quad):
    h, w = img.shape[:2]
    q = quad.astype(np.int32)

    poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly, [q.reshape(-1, 1, 2)], 255)

    area = int(np.count_nonzero(poly))
    if area < 800:
        return 0.0, {}

    blue = blue_table_mask(img)

    inside_blue = int(np.count_nonzero(cv2.bitwise_and(blue, blue, mask=poly)))
    inside_ratio = inside_blue / max(1, area)

    # Support bord : les bords de table ont souvent de l'arête.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_mask, [q.reshape(-1, 1, 2)], True, 255, 9)
    edge_support = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=edge_mask)))
    edge_ratio = edge_support / max(1, int(np.count_nonzero(edge_mask)))

    # Pénaliser les quads trop hors image.
    outside = np.sum((quad[:, 0] < -20) | (quad[:, 0] > w + 20) | (quad[:, 1] < -20) | (quad[:, 1] > h + 20))
    outside_penalty = 0.10 * float(outside)

    score = 0.72 * inside_ratio + 0.28 * min(1.0, edge_ratio * 6.0) - outside_penalty

    return float(score), {
        "inside_blue_ratio": float(inside_ratio),
        "edge_ratio": float(edge_ratio),
        "outside_points": int(outside),
        "poly_area": int(area),
    }


def read_frame(cap, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
    ok, img = cap.read()
    if not ok or img is None:
        return None
    return img


def score_template_on_shot(video_path, start, end, template):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    span = max(1, end - start + 1)
    sample_frames = sorted(set([
        start,
        start + int(span * 0.33),
        start + int(span * 0.66),
        end,
    ]))

    vals = []
    details = []

    for fr in sample_frames:
        img = read_frame(cap, fr)
        if img is None:
            continue
        s, d = score_quad_on_frame(img, template["quad"])
        vals.append(s)
        d["frame"] = fr
        details.append(d)

    cap.release()

    if not vals:
        return 0.0, details

    return float(np.median(vals)), details


def select_table_for_shot(video_path, start, end, templates, min_score=0.18):
    scored = []

    for t in templates:
        score, details = score_template_on_shot(video_path, start, end, t)
        scored.append({
            "template": t,
            "score": score,
            "details": details,
        })

    scored.sort(key=lambda x: (x["score"], x["template"]["quality"]), reverse=True)

    best = scored[0] if scored else None
    if best is None or best["score"] < min_score:
        return None, scored[:5]

    return best, scored[:5]


def compute_lines(quad):
    tl, tr, br, bl = quad
    # Lignes indicatives image : centre longueur + centre largeur.
    net_line = np.array([(tl + bl) * 0.5, (tr + br) * 0.5], dtype=np.float32)
    center_line = np.array([(tl + tr) * 0.5, (bl + br) * 0.5], dtype=np.float32)
    return net_line, center_line


def draw_text_bg(img, text, org, scale=0.48, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_context(img, shot_row):
    out = img.copy()

    if shot_row["table_context_valid"] == "1":
        quad = np.array(json.loads(shot_row["table_quad"]), dtype=np.float32)
        q = quad.astype(np.int32)
        net, center = compute_lines(quad)

        cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        cv2.line(out, tuple(net[0].astype(int)), tuple(net[1].astype(int)), (255, 255, 0), 2)
        cv2.line(out, tuple(center[0].astype(int)), tuple(center[1].astype(int)), (0, 180, 255), 2)

        label = (
            f"shot#{shot_row['shot_id']} TABLE "
            f"tpl={shot_row['template_id']} {shot_row['template_review_id']} "
            f"score={float(shot_row['table_score']):.3f}"
        )
        color = (255, 255, 255)
    else:
        label = f"shot#{shot_row['shot_id']} NO_TABLE_CONTEXT"
        color = (80, 80, 255)

    draw_text_bg(out, label, (12, 28), color=color)
    return out


def make_shot_contact(video_path, shot_rows, out_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    rows_img = []

    for sr in shot_rows:
        start = int(sr["start_frame"])
        end = int(sr["end_frame"])
        span = max(1, end - start + 1)
        samples = [start, start + span // 2, end]

        panels = []
        for fr in samples:
            img = read_frame(cap, fr)
            if img is None:
                continue
            out = draw_context(img, sr)
            tw = 360
            scale = tw / out.shape[1]
            thumb = cv2.resize(out, (tw, int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            panels.append(thumb)

        if not panels:
            continue

        h = max(p.shape[0] for p in panels)
        w = sum(p.shape[1] for p in panels)
        row_img = np.zeros((h + 30, w, 3), dtype=np.uint8)

        x = 0
        for p in panels:
            row_img[:p.shape[0], x:x+p.shape[1]] = p
            x += p.shape[1]

        txt = f"shot#{sr['shot_id']} f={start}->{end} valid={sr['table_context_valid']} score={sr['table_score']} tpl={sr['template_review_id']}"
        cv2.putText(row_img, txt, (6, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)
        rows_img.append(row_img)

    cap.release()

    if not rows_img:
        return

    width = max(r.shape[1] for r in rows_img)
    height = sum(r.shape[0] + 8 for r in rows_img)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)

    y = 0
    for r in rows_img:
        sheet[y:y+r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 8

    imwrite_unicode(out_path, sheet)


def make_review_video(video_path, shot_rows, out_video):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire {out_video}")

    shot_by_frame = {}
    for sr in shot_rows:
        for fr in range(int(sr["start_frame"]), int(sr["end_frame"]) + 1):
            shot_by_frame[fr] = sr

    written = 0
    fr = 0
    while fr < total:
        ok, img = cap.read()
        if not ok or img is None:
            break

        sr = shot_by_frame.get(fr, shot_rows[-1])
        out = draw_context(img, sr)
        writer.write(out)

        written += 1
        fr += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": total,
        "written_frames": written,
        "duration_sec": written / fps if fps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--fit-csv", default="runs/rally_table_reference_fit_005C9A/table_reference_fit_005C9A.csv")
    ap.add_argument("--out-dir", default="runs/006C3_dynamic_camera_table_context")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--min-shot-len", type=int, default=35)
    ap.add_argument("--min-table-score", type=float, default=0.18)
    args = ap.parse_args()

    video_path = Path(args.video)
    fit_csv = Path(args.fit_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not fit_csv.exists():
        raise FileNotFoundError(fit_csv)

    info = video_info(video_path)
    templates = load_table_templates(fit_csv)

    if not templates:
        raise RuntimeError("Aucun template table 005C9A exploitable.")

    cuts, diffs, threshold = detect_camera_cuts(
        video_path,
        step=args.step,
        min_shot_len=args.min_shot_len,
    )

    # Shots.
    starts = cuts
    ends = []
    for i, s in enumerate(starts):
        if i + 1 < len(starts):
            ends.append(starts[i + 1] - 1)
        else:
            ends.append(info["frames"] - 1)

    shot_rows = []
    top_debug = []

    for shot_id, (start, end) in enumerate(zip(starts, ends)):
        best, top = select_table_for_shot(
            video_path,
            start,
            end,
            templates,
            min_score=args.min_table_score,
        )

        if best is None:
            row = {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "duration_frames": end - start + 1,
                "table_context_valid": "0",
                "table_score": "0.000000",
                "template_id": "",
                "template_review_id": "",
                "quad_source": "",
                "table_quad": "",
                "reason": "NO_VALID_TABLE_TEMPLATE_FOR_THIS_CAMERA_SHOT",
            }
        else:
            t = best["template"]
            row = {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "duration_frames": end - start + 1,
                "table_context_valid": "1",
                "table_score": f"{best['score']:.6f}",
                "template_id": t["template_id"],
                "template_review_id": t["review_id"],
                "quad_source": t["quad_source"],
                "table_quad": json.dumps([[float(x), float(y)] for x, y in t["quad"]], ensure_ascii=False),
                "reason": "MATCHED_EXISTING_TABLE_TEMPLATE_BY_VISUAL_SUPPORT",
            }

        shot_rows.append(row)

        top_debug.append({
            "shot_id": shot_id,
            "start": start,
            "end": end,
            "top": [
                {
                    "score": float(x["score"]),
                    "template_id": int(x["template"]["template_id"]),
                    "review_id": x["template"]["review_id"],
                    "quad_source": x["template"]["quad_source"],
                    "quality": float(x["template"]["quality"]),
                    "details": x["details"],
                }
                for x in top
            ],
        })

    timeline_csv = out_dir / "006C3_camera_table_context_timeline.csv"
    fields = [
        "shot_id",
        "start_frame",
        "end_frame",
        "duration_frames",
        "table_context_valid",
        "table_score",
        "template_id",
        "template_review_id",
        "quad_source",
        "table_quad",
        "reason",
    ]
    write_csv(timeline_csv, shot_rows, fields)

    contact = out_dir / "006C3_camera_table_context_contact.jpg"
    review_video = out_dir / "006C3_camera_table_context_review.mp4"
    debug_json = out_dir / "006C3_camera_table_context_debug.json"

    make_shot_contact(video_path, shot_rows, contact)
    video_meta = make_review_video(video_path, shot_rows, review_video)

    summary = {
        "patch": PATCH_ID,
        "video": str(video_path),
        "fit_csv": str(fit_csv),
        "video_info": info,
        "template_count": len(templates),
        "cut_threshold": threshold,
        "cuts": cuts,
        "shot_count": len(shot_rows),
        "valid_table_shot_count": sum(1 for r in shot_rows if r["table_context_valid"] == "1"),
        "timeline_csv": str(timeline_csv),
        "contact": str(contact),
        "review_video": str(review_video),
        "video_meta": video_meta,
        "top_debug": top_debug,
        "note": "Table context dynamique par shot caméra. Les close-ups peuvent être NO_TABLE_CONTEXT.",
    }

    debug_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    next_txt = out_dir / "NEXT_DYNAMIC_TABLE_CONTEXT_CSV.txt"
    next_txt.write_text(str(timeline_csv), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006C3")
    print(f"timeline_csv = {timeline_csv}")
    print(f"contact      = {contact}")
    print(f"review_video = {review_video}")
    print(f"debug_json   = {debug_json}")
    print("")
    print(f"frames={info['frames']} fps={info['fps']}")
    print(f"template_count={len(templates)}")
    print(f"cuts={cuts}")
    print(f"shot_count={len(shot_rows)}")
    print(f"valid_table_shot_count={summary['valid_table_shot_count']}")
    print("")
    for r in shot_rows:
        print(
            f"shot#{r['shot_id']} f={r['start_frame']}->{r['end_frame']} "
            f"valid={r['table_context_valid']} score={r['table_score']} "
            f"tpl={r['template_review_id']} reason={r['reason']}"
        )


if __name__ == "__main__":
    main()
