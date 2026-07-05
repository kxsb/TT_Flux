from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "006C_FIX_restore_existing_table_context_005C9B"


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


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


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except Exception:
        return default


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 50.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def load_align_summary(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_clip_range(row: dict):
    txt = " ".join(str(v) for v in row.values())
    m = re.search(r"_f(\d+)_to_f(\d+)", txt)
    if not m:
        m = re.search(r"f(\d+)_to_f(\d+)", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def row_has_video_id(row: dict, video_id: str):
    if not video_id:
        return False
    txt = " ".join(str(v) for v in row.values())
    return video_id in txt


def row_review_id(row: dict):
    for k in ["review_id", "rally_id", "rally_id_vote", "sequence_key", "clip_id"]:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def row_quality(row: dict):
    keys = [
        "opt_score_005C9A",
        "metric_quality_005C7B",
        "table_object_confidence_005C6B",
        "snap_confidence_005C7A",
        "table_confidence_005C1",
        "rating_1_10",
    ]
    val = 0.0
    for k in keys:
        x = safe_float(row.get(k), np.nan)
        if math.isfinite(x):
            if k == "rating_1_10":
                val += x / 10.0
            else:
                val += x
    return val


def rank_row(row: dict, video_id: str, target_review_id: str, target_offset: int, target_end: int):
    score = 0.0
    reasons = []

    rid = row_review_id(row)
    if target_review_id and target_review_id in rid:
        score += 10000
        reasons.append("exact_review_id")

    if row_has_video_id(row, video_id):
        score += 1000
        reasons.append("same_video_id")

    cr = extract_clip_range(row)
    if cr:
        a, b = cr
        overlap = max(0, min(b, target_end) - max(a, target_offset))
        contains_start = a <= target_offset <= b
        contains_mid = a <= ((target_offset + target_end) // 2) <= b

        if contains_start:
            score += 5000
            reasons.append("contains_start_offset")

        if contains_mid:
            score += 3000
            reasons.append("contains_mid_offset")

        if overlap > 0:
            score += 2000 + overlap / max(1, target_end - target_offset)
            reasons.append(f"overlap={overlap}")

        mid = (a + b) / 2.0
        target_mid = (target_offset + target_end) / 2.0
        dist = abs(mid - target_mid)
        score -= dist * 0.01
        reasons.append(f"range={a}-{b},dist={dist:.1f}")

    q = row_quality(row)
    score += q * 100.0
    reasons.append(f"quality={q:.3f}")

    status_txt = " ".join(str(row.get(k, "")) for k in row.keys() if "status" in k.lower() or "reason" in k.lower())
    if "OK" in status_txt or "ok" in status_txt:
        score += 50
        reasons.append("status_ok")

    return score, reasons


def parse_homography(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, list) and len(arr) == 9:
            H = np.array(arr, dtype=np.float32).reshape(3, 3)
            if np.all(np.isfinite(H)):
                return H
    except Exception:
        return None
    return None


def extract_homographies(row: dict):
    suffixes = ["005C9A", "005C7A", "005C5C2", "005C3", "005C1"]

    for suf in suffixes:
        img_key = f"H_img_to_table_{suf}"
        table_key = f"H_table_to_img_{suf}"

        H_img = parse_homography(row.get(img_key))
        H_tab = parse_homography(row.get(table_key))

        if H_img is not None or H_tab is not None:
            return {
                "source_suffix": suf,
                "H_img_to_table": H_img.tolist() if H_img is not None else None,
                "H_table_to_img": H_tab.tolist() if H_tab is not None else None,
            }

    return {
        "source_suffix": "",
        "H_img_to_table": None,
        "H_table_to_img": None,
    }


def extract_quad(row: dict):
    candidates = [
        ("005C9A_opt", "opt_tl_x_005C9A", "opt_tl_y_005C9A", "opt_tr_x_005C9A", "opt_tr_y_005C9A", "opt_br_x_005C9A", "opt_br_y_005C9A", "opt_bl_x_005C9A", "opt_bl_y_005C9A"),
        ("005C7A_snap", "snap_tl_x_005C7A", "snap_tl_y_005C7A", "snap_tr_x_005C7A", "snap_tr_y_005C7A", "snap_br_x_005C7A", "snap_br_y_005C7A", "snap_bl_x_005C7A", "snap_bl_y_005C7A"),
        ("005C5C2_quad", "quad_tl_x_005C5C2", "quad_tl_y_005C5C2", "quad_tr_x_005C5C2", "quad_tr_y_005C5C2", "quad_br_x_005C5C2", "quad_br_y_005C5C2", "quad_bl_x_005C5C2", "quad_bl_y_005C5C2"),
        ("005C3_quad", "quad_tl_x_005C3", "quad_tl_y_005C3", "quad_tr_x_005C3", "quad_tr_y_005C3", "quad_br_x_005C3", "quad_br_y_005C3", "quad_bl_x_005C3", "quad_bl_y_005C3"),
        ("005C1_quad", "quad_tl_x_005C1", "quad_tl_y_005C1", "quad_tr_x_005C1", "quad_tr_y_005C1", "quad_br_x_005C1", "quad_br_y_005C1", "quad_bl_x_005C1", "quad_bl_y_005C1"),
    ]

    for source, *cols in candidates:
        vals = [safe_float(row.get(c), np.nan) for c in cols]
        if all(math.isfinite(v) for v in vals):
            quad = np.array([
                [vals[0], vals[1]],
                [vals[2], vals[3]],
                [vals[4], vals[5]],
                [vals[6], vals[7]],
            ], dtype=np.float32)

            area = cv2.contourArea(quad.reshape(-1, 1, 2))
            if area > 500:
                return source, quad

    return "", None


def perspective_points(H, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(1, -1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def compute_reference_lines(row: dict, quad: np.ndarray, homos: dict):
    tl, tr, br, bl = quad

    # Fallback image : net = moitié longueur table, centerline = moitié largeur table.
    net_line = np.array([(tl + tr) * 0.5, (bl + br) * 0.5], dtype=np.float32)
    center_line = np.array([(tl + bl) * 0.5, (tr + br) * 0.5], dtype=np.float32)

    H_tab = None
    if homos.get("H_table_to_img") is not None:
        H_tab = np.array(homos["H_table_to_img"], dtype=np.float32)

    L = safe_float(row.get("table_length_m_005C3"), np.nan)
    if not math.isfinite(L):
        L = safe_float(row.get("table_length_m_005C1"), 2.74)

    W = safe_float(row.get("table_width_m_005C3"), np.nan)
    if not math.isfinite(W):
        W = safe_float(row.get("table_width_m_005C1"), 1.525)

    if H_tab is not None and math.isfinite(L) and math.isfinite(W):
        try:
            net_line = perspective_points(H_tab, [[L / 2.0, 0.0], [L / 2.0, W]])
            center_line = perspective_points(H_tab, [[0.0, W / 2.0], [L, W / 2.0]])
        except Exception:
            pass

    return net_line.astype(np.float32), center_line.astype(np.float32)


def draw_text_bg(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_table_context(img, quad, net_line, center_line, label):
    out = img.copy()

    q = quad.astype(np.int32)
    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 3)

    nl = net_line.astype(np.int32)
    cl = center_line.astype(np.int32)

    cv2.line(out, tuple(nl[0]), tuple(nl[1]), (255, 255, 0), 2)
    cv2.line(out, tuple(cl[0]), tuple(cl[1]), (0, 180, 255), 2)

    names = ["TL", "TR", "BR", "BL"]
    for name, p in zip(names, q):
        cv2.circle(out, tuple(p), 7, (0, 255, 0), -1)
        cv2.putText(out, name, tuple(p + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    draw_text_bg(out, label, (12, 28), scale=0.55)

    return out


def make_review_assets(video_path: Path, out_dir: Path, quad, net_line, center_line, label, max_frames=500):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    max_frames = min(max_frames, total)

    review_video = out_dir / "006C_FIX_restored_table_context_review.mp4"
    writer = cv2.VideoWriter(str(review_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire {review_video}")

    sample_frames = sorted(set([
        0,
        max(0, int(total * 0.20)),
        max(0, int(total * 0.40)),
        max(0, int(total * 0.60)),
        max(0, int(total * 0.80)),
        max(0, total - 1),
    ]))

    thumbs = []
    written = 0
    idx = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out = draw_table_context(frame, quad, net_line, center_line, f"{label} | f={idx}")
        writer.write(out)

        if idx in sample_frames:
            th_w = 420
            scale = th_w / w
            thumb = cv2.resize(out, (th_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)

        written += 1
        idx += 1

    writer.release()
    cap.release()

    contact = out_dir / "006C_FIX_restored_table_context_contact.jpg"
    if thumbs:
        cols = 2
        rows = int(math.ceil(len(thumbs) / cols))
        tw = max(t.shape[1] for t in thumbs)
        th = max(t.shape[0] for t in thumbs)
        sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for i, t in enumerate(thumbs):
            r = i // cols
            c = i % cols
            y = r * th
            x = c * tw
            sheet[y:y+t.shape[0], x:x+t.shape[1]] = t
        imwrite_unicode(contact, sheet)

    return {
        "review_video": str(review_video),
        "contact": str(contact),
        "fps": fps,
        "width": w,
        "height": h,
        "total_frames": total,
        "written_frames": written,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--align-summary", default="runs/005F3_align_clean_video/005F3_align_clean_video_summary.json")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--fit-csv", default="runs/rally_table_reference_fit_005C9A/table_reference_fit_005C9A.csv")
    ap.add_argument("--video-id", default="-0bM0t0qS8Q")
    ap.add_argument("--review-id", default="RLY0074")
    ap.add_argument("--out-dir", default="runs/006C_FIX_restore_existing_table_context")
    args = ap.parse_args()

    clean_video = Path(args.clean_video)
    align_summary_path = Path(args.align_summary)
    scene_tables = Path(args.scene_tables)
    fit_csv = Path(args.fit_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clean_video.exists():
        raise FileNotFoundError(clean_video)

    align = load_align_summary(align_summary_path)
    info = video_info(clean_video)

    best_offset = safe_int(align.get("best_offset"), 0)
    target_start = best_offset
    target_end = best_offset + info["frames"] - 1

    source_path = None
    rows = []
    fields = []

    if scene_tables.exists():
        rows, fields = read_csv(scene_tables)
        source_path = scene_tables

    if not rows and fit_csv.exists():
        rows, fields = read_csv(fit_csv)
        source_path = fit_csv

    if not rows:
        raise RuntimeError(
            "Aucune source table trouvée. Vérifie 005C9B/005C9A : "
            f"{scene_tables} ou {fit_csv}"
        )

    ranked = []
    for i, row in enumerate(rows):
        score, reasons = rank_row(
            row,
            video_id=args.video_id,
            target_review_id=args.review_id,
            target_offset=target_start,
            target_end=target_end,
        )
        src, quad = extract_quad(row)
        if quad is None:
            continue

        ranked.append({
            "idx": i,
            "rank_score": score,
            "rank_reasons": "|".join(reasons),
            "quad_source": src,
            "review_id": row_review_id(row),
            "clip_range": extract_clip_range(row),
            "row": row,
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)

    if not ranked:
        raise RuntimeError("Aucune ligne table exploitable avec quad trouvée.")

    selected = ranked[0]
    row = selected["row"]
    quad_source, quad = extract_quad(row)
    homos = extract_homographies(row)
    net_line, center_line = compute_reference_lines(row, quad, homos)

    candidate_rows = []
    for r in ranked[:30]:
        rr = {
            "rank_score": r["rank_score"],
            "rank_reasons": r["rank_reasons"],
            "quad_source": r["quad_source"],
            "review_id": r["review_id"],
            "clip_range": str(r["clip_range"]),
        }
        candidate_rows.append(rr)

    candidates_csv = out_dir / "006C_FIX_table_context_candidates.csv"
    write_csv(candidates_csv, candidate_rows, ["rank_score", "rank_reasons", "quad_source", "review_id", "clip_range"])

    selected_row_csv = out_dir / "006C_FIX_selected_table_row.csv"
    write_csv(selected_row_csv, [row], fields)

    context_json = out_dir / "006C_FIX_existing_table_context.json"

    label = f"{PATCH_ID} | source={source_path.name} | quad={quad_source}"
    review_meta = make_review_assets(clean_video, out_dir, quad, net_line, center_line, label, max_frames=500)

    context = {
        "patch": PATCH_ID,
        "status": "RESTORED_EXISTING_TABLE_CONTEXT_NOT_REDETECTED",
        "clean_video": str(clean_video),
        "video_info": info,
        "align_summary": str(align_summary_path),
        "best_offset_in_full_video": best_offset,
        "target_full_frame_start": target_start,
        "target_full_frame_end": target_end,
        "table_source_csv": str(source_path),
        "selected_rank_score": selected["rank_score"],
        "selected_rank_reasons": selected["rank_reasons"],
        "selected_review_id": selected["review_id"],
        "selected_clip_range": selected["clip_range"],
        "quad_source": quad_source,
        "table_quad_order": "top_left, top_right, bottom_right, bottom_left",
        "table_quad": [[float(x), float(y)] for x, y in quad],
        "net_line": [[float(x), float(y)] for x, y in net_line],
        "center_line": [[float(x), float(y)] for x, y in center_line],
        "homography_source_suffix": homos.get("source_suffix"),
        "H_img_to_table": homos.get("H_img_to_table"),
        "H_table_to_img": homos.get("H_table_to_img"),
        "candidates_csv": str(candidates_csv),
        "selected_row_csv": str(selected_row_csv),
        "review_video": review_meta["review_video"],
        "contact": review_meta["contact"],
        "review_meta": review_meta,
        "warning": "Si selected_clip_range ne recouvre pas target_full_frame_start/end, vérifier visuellement. Mais on réutilise bien l'ancien contexte table, pas une nouvelle heuristique HSV.",
    }

    context_json.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    restore_txt = out_dir / "NEXT_TABLE_CONTEXT_JSON.txt"
    restore_txt.write_text(str(context_json), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006C_FIX")
    print(f"context_json = {context_json}")
    print(f"contact      = {review_meta['contact']}")
    print(f"review_video = {review_meta['review_video']}")
    print(f"candidates   = {candidates_csv}")
    print(f"selected_row = {selected_row_csv}")
    print("")
    print(f"table_source_csv={source_path}")
    print(f"target_full_frame_start={target_start}")
    print(f"target_full_frame_end={target_end}")
    print(f"selected_review_id={selected['review_id']}")
    print(f"selected_clip_range={selected['clip_range']}")
    print(f"selected_rank_score={selected['rank_score']:.3f}")
    print(f"selected_rank_reasons={selected['rank_reasons']}")
    print(f"quad_source={quad_source}")
    print(f"quad={context['table_quad']}")
    print("")
    print("À ouvrir :")
    print(f"  {review_meta['contact']}")
    print(f"  {review_meta['review_video']}")


if __name__ == "__main__":
    main()
