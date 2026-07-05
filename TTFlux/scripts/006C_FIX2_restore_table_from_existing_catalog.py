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


PATCH_ID = "006C_FIX2_restore_table_from_existing_catalog"


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


def extract_clip_range(row: dict):
    txt = " ".join(str(v) for v in row.values())
    m = re.search(r"_f(\d+)_to_f(\d+)", txt)
    if not m:
        m = re.search(r"f(\d+)_to_f(\d+)", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def row_text(row: dict):
    return " ".join(str(v) for v in row.values())


def get_review_id(row: dict):
    for k in ["review_id", "rally_id", "rally_id_vote", "sequence_key", "clip_id"]:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def row_quality(row: dict):
    keys = [
        "opt_score_005C9A",
        "snap_confidence_005C7A",
        "snap_confidence_005C7A_vote",
        "table_object_confidence_005C6B",
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


def quad_from_cols(row: dict, prefix: str, suffix: str):
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


def quad_from_exact_cols(row: dict):
    candidates = [
        ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"),
        ("table_tl_x", "table_tl_y", "table_tr_x", "table_tr_y", "table_br_x", "table_br_y", "table_bl_x", "table_bl_y"),
        ("quad_tl_x", "quad_tl_y", "quad_tr_x", "quad_tr_y", "quad_br_x", "quad_br_y", "quad_bl_x", "quad_bl_y"),
    ]
    for cols in candidates:
        vals = [safe_float(row.get(c), np.nan) for c in cols]
        if all(math.isfinite(v) for v in vals):
            q = np.array([
                [vals[0], vals[1]],
                [vals[2], vals[3]],
                [vals[4], vals[5]],
                [vals[6], vals[7]],
            ], dtype=np.float32)
            if cv2.contourArea(q.reshape(-1, 1, 2)) > 500:
                return q, "exact_cols"
    return None, ""


def normalize_quad_array(obj):
    """
    Accepte :
    - [[x,y],[x,y],[x,y],[x,y]]
    - [x,y,x,y,x,y,x,y]
    - [{"x":..,"y":..}, ...]
    """
    if isinstance(obj, dict):
        for key in ["quad", "polygon", "corners", "points", "table_quad"]:
            if key in obj:
                return normalize_quad_array(obj[key])
        return None

    if isinstance(obj, list):
        if len(obj) == 4 and all(isinstance(p, (list, tuple, dict)) for p in obj):
            pts = []
            for p in obj:
                if isinstance(p, dict):
                    x = safe_float(p.get("x"), np.nan)
                    y = safe_float(p.get("y"), np.nan)
                else:
                    if len(p) < 2:
                        return None
                    x = safe_float(p[0], np.nan)
                    y = safe_float(p[1], np.nan)
                if not (math.isfinite(x) and math.isfinite(y)):
                    return None
                pts.append([x, y])
            q = np.array(pts, dtype=np.float32)
            if cv2.contourArea(q.reshape(-1, 1, 2)) > 500:
                return q

        if len(obj) == 8:
            vals = [safe_float(x, np.nan) for x in obj]
            if all(math.isfinite(v) for v in vals):
                q = np.array([
                    [vals[0], vals[1]],
                    [vals[2], vals[3]],
                    [vals[4], vals[5]],
                    [vals[6], vals[7]],
                ], dtype=np.float32)
                if cv2.contourArea(q.reshape(-1, 1, 2)) > 500:
                    return q

    return None


def quad_from_jsonish(row: dict):
    for k, v in row.items():
        kl = k.lower()
        if not any(tok in kl for tok in ["quad", "polygon", "corner", "table_points", "table_quad"]):
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        if "[" not in s and "{" not in s:
            continue
        try:
            obj = ast.literal_eval(s)
        except Exception:
            continue
        q = normalize_quad_array(obj)
        if q is not None:
            return q, f"jsonish:{k}"
    return None, ""


def extract_quad(row: dict):
    # Priorité : projection optimisée validée, puis snap, puis anciens quads.
    ordered = [
        ("opt", "005C9B"),
        ("opt", "005C9A"),
        ("table", "005C9B"),
        ("scene_table", "005C9B"),
        ("final", "005C9B"),
        ("snap", "005C7A"),
        ("quad", "005C5C2"),
        ("quad", "005C3"),
        ("quad", "005C1"),
    ]

    for prefix, suffix in ordered:
        q, src = quad_from_cols(row, prefix, suffix)
        if q is not None:
            return q, src

    q, src = quad_from_exact_cols(row)
    if q is not None:
        return q, src

    q, src = quad_from_jsonish(row)
    if q is not None:
        return q, src

    return None, ""


def rank_candidate(row: dict, source_name: str, video_id: str, review_id: str, target_start: int, target_end: int):
    score = 0.0
    reasons = []

    txt = row_text(row)
    rid = get_review_id(row)

    if review_id and review_id in rid:
        score += 10000
        reasons.append("exact_review_id")

    if video_id and video_id in txt:
        score += 1200
        reasons.append("same_video_id")

    cr = extract_clip_range(row)
    if cr:
        a, b = cr
        overlap = max(0, min(b, target_end) - max(a, target_start))
        if overlap > 0:
            score += 5000 + overlap
            reasons.append(f"clip_overlap={overlap}")

        mid = (a + b) / 2.0
        target_mid = (target_start + target_end) / 2.0
        dist = abs(mid - target_mid)
        score -= dist * 0.03
        reasons.append(f"clip_range={a}-{b},dist={dist:.1f}")

    q = row_quality(row)
    score += q * 100.0
    reasons.append(f"quality={q:.3f}")

    if source_name.endswith("005C9B"):
        score += 100
        reasons.append("source_005C9B")
    if source_name.endswith("005C9A"):
        score += 80
        reasons.append("source_005C9A")

    status = txt.lower()
    if "ok" in status or "strict" in status:
        score += 30
        reasons.append("status_okish")

    return score, "|".join(reasons)


def draw_text_bg(img, text, org, scale=0.52, color=(255, 255, 255), thickness=1):
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 5, y - th - 8), (x + tw + 5, y + base + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def compute_lines(quad):
    tl, tr, br, bl = quad
    # Pour l’instant : lignes image simples. Les features métriques viendront après si H disponible.
    net_line = np.array([(tl + bl) * 0.5, (tr + br) * 0.5], dtype=np.float32)
    center_line = np.array([(tl + tr) * 0.5, (bl + br) * 0.5], dtype=np.float32)
    return net_line, center_line


def draw_table(img, quad, label):
    out = img.copy()
    q = quad.astype(np.int32)
    net, center = compute_lines(quad)

    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    cv2.line(out, tuple(net[0].astype(int)), tuple(net[1].astype(int)), (255, 255, 0), 2)
    cv2.line(out, tuple(center[0].astype(int)), tuple(center[1].astype(int)), (0, 180, 255), 2)

    for name, p in zip(["TL", "TR", "BR", "BL"], q):
        cv2.circle(out, tuple(p), 7, (0, 255, 0), -1)
        cv2.putText(out, name, tuple(p + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    draw_text_bg(out, label, (12, 28))
    return out


def read_frame(cap, fr):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
    ok, img = cap.read()
    if not ok or img is None:
        return None
    return img


def make_candidate_contact(video_path: Path, candidates: list[dict], out_path: Path, max_candidates=12):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_frames = [0, int(total * 0.33), int(total * 0.66)]

    rows = []

    for rank, c in enumerate(candidates[:max_candidates]):
        panels = []
        quad = np.array(c["quad"], dtype=np.float32)

        for fr in sample_frames:
            img = read_frame(cap, fr)
            if img is None:
                continue
            label = f"rank#{rank} {c['source_name']} {c['quad_source']} {c['review_id']} score={c['rank_score']:.1f}"
            out = draw_table(img, quad, label)
            th_w = 320
            scale = th_w / out.shape[1]
            thumb = cv2.resize(out, (th_w, int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            panels.append(thumb)

        if not panels:
            continue

        h = max(p.shape[0] for p in panels)
        w = sum(p.shape[1] for p in panels)
        row_img = np.zeros((h + 26, w, 3), dtype=np.uint8)

        x = 0
        for p in panels:
            row_img[0:p.shape[0], x:x+p.shape[1]] = p
            x += p.shape[1]

        txt = f"#{rank} rank={c['rank_score']:.1f} reasons={c['rank_reasons'][:120]}"
        cv2.putText(row_img, txt, (6, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        rows.append(row_img)

    cap.release()

    if not rows:
        raise RuntimeError("Aucune image candidate produite.")

    width = max(r.shape[1] for r in rows)
    height = sum(r.shape[0] + 8 for r in rows)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)

    y = 0
    for r in rows:
        sheet[y:y+r.shape[0], 0:r.shape[1]] = r
        y += r.shape[0] + 8

    imwrite_unicode(out_path, sheet)


def make_review_video(video_path: Path, quad, out_video: Path, label: str, max_frames=500):
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

    written = 0
    max_frames = min(total, max_frames)

    while written < max_frames:
        ok, img = cap.read()
        if not ok or img is None:
            break

        out = draw_table(img, quad, f"{label} | f={written}")
        writer.write(out)
        written += 1

    cap.release()
    writer.release()

    return {"written_frames": written, "fps": fps, "width": w, "height": h, "duration_sec": written / fps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-video", default="runs/005F3_align_clean_video/005F3_clean_aligned_segment.mp4")
    ap.add_argument("--align-summary", default="runs/005F3_align_clean_video/005F3_align_clean_video_summary.json")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--fit-csv", default="runs/rally_table_reference_fit_005C9A/table_reference_fit_005C9A.csv")
    ap.add_argument("--video-id", default="-0bM0t0qS8Q")
    ap.add_argument("--review-id", default="RLY0074")
    ap.add_argument("--pick-rank", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/006C_FIX2_restore_table_catalog")
    args = ap.parse_args()

    clean_video = Path(args.clean_video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clean_video.exists():
        raise FileNotFoundError(clean_video)

    info = video_info(clean_video)

    align = {}
    align_path = Path(args.align_summary)
    if align_path.exists():
        align = json.loads(align_path.read_text(encoding="utf-8"))

    best_offset = safe_int(align.get("best_offset"), 0)
    target_start = best_offset
    target_end = best_offset + info["frames"] - 1

    sources = [
        ("005C9B_scene_objects", Path(args.scene_tables)),
        ("005C9A_reference_fit", Path(args.fit_csv)),
    ]

    candidates = []
    source_debug = []

    for source_name, path in sources:
        rows, fields = read_csv(path)
        source_debug.append({
            "source_name": source_name,
            "path": str(path),
            "exists": path.exists(),
            "rows": len(rows),
            "columns": fields,
        })

        for idx, row in enumerate(rows):
            quad, quad_source = extract_quad(row)
            if quad is None:
                continue

            rank_score, rank_reasons = rank_candidate(
                row=row,
                source_name=source_name,
                video_id=args.video_id,
                review_id=args.review_id,
                target_start=target_start,
                target_end=target_end,
            )

            candidates.append({
                "source_name": source_name,
                "source_path": str(path),
                "row_idx": idx,
                "review_id": get_review_id(row),
                "clip_range": extract_clip_range(row),
                "quad_source": quad_source,
                "rank_score": float(rank_score),
                "rank_reasons": rank_reasons,
                "quality": row_quality(row),
                "quad": [[float(x), float(y)] for x, y in quad],
            })

    candidates.sort(key=lambda c: c["rank_score"], reverse=True)

    debug_json = out_dir / "006C_FIX2_source_debug.json"
    debug_json.write_text(json.dumps(source_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    if not candidates:
        raise RuntimeError(
            "Aucun quad trouvé dans 005C9B/005C9A. "
            f"Debug colonnes écrit ici : {debug_json}"
        )

    pick_rank = max(0, min(args.pick_rank, len(candidates) - 1))
    selected = candidates[pick_rank]
    selected_quad = np.array(selected["quad"], dtype=np.float32)
    net, center = compute_lines(selected_quad)

    candidates_csv = out_dir / "006C_FIX2_table_candidates.csv"
    write_csv(
        candidates_csv,
        [
            {
                "rank": i,
                "rank_score": c["rank_score"],
                "source_name": c["source_name"],
                "row_idx": c["row_idx"],
                "review_id": c["review_id"],
                "clip_range": str(c["clip_range"]),
                "quad_source": c["quad_source"],
                "rank_reasons": c["rank_reasons"],
            }
            for i, c in enumerate(candidates)
        ],
        ["rank", "rank_score", "source_name", "row_idx", "review_id", "clip_range", "quad_source", "rank_reasons"],
    )

    contact = out_dir / "006C_FIX2_table_candidates_contact.jpg"
    make_candidate_contact(clean_video, candidates, contact, max_candidates=12)

    review_video = out_dir / "006C_FIX2_selected_table_review.mp4"
    label = f"{PATCH_ID} rank#{pick_rank} {selected['source_name']} {selected['quad_source']} {selected['review_id']}"
    review_meta = make_review_video(clean_video, selected_quad, review_video, label, max_frames=500)

    context = {
        "patch": PATCH_ID,
        "status": "RESTORED_FROM_EXISTING_TABLE_CATALOG_NO_NEW_DETECTION",
        "clean_video": str(clean_video),
        "video_info": info,
        "align_best_offset": best_offset,
        "target_full_frame_start": target_start,
        "target_full_frame_end": target_end,
        "selected_rank": pick_rank,
        "selected": selected,
        "table_quad_order": "top_left, top_right, bottom_right, bottom_left",
        "table_quad": selected["quad"],
        "net_line": [[float(x), float(y)] for x, y in net],
        "center_line": [[float(x), float(y)] for x, y in center],
        "candidate_count": len(candidates),
        "candidates_csv": str(candidates_csv),
        "contact": str(contact),
        "review_video": str(review_video),
        "review_meta": review_meta,
        "source_debug": str(debug_json),
        "warning": "Si le rank#0 ne colle pas visuellement, ouvrir le contact sheet puis relancer avec --pick-rank N.",
    }

    context_json = out_dir / "006C_FIX2_table_context.json"
    context_json.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    next_txt = out_dir / "NEXT_TABLE_CONTEXT_JSON.txt"
    next_txt.write_text(str(context_json), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006C_FIX2")
    print(f"context_json = {context_json}")
    print(f"contact      = {contact}")
    print(f"review_video = {review_video}")
    print(f"candidates   = {candidates_csv}")
    print(f"source_debug = {debug_json}")
    print("")
    print(f"candidate_count={len(candidates)}")
    print(f"selected_rank={pick_rank}")
    print(f"selected_source={selected['source_name']}")
    print(f"selected_review_id={selected['review_id']}")
    print(f"selected_clip_range={selected['clip_range']}")
    print(f"selected_quad_source={selected['quad_source']}")
    print(f"selected_rank_score={selected['rank_score']:.3f}")
    print(f"selected_rank_reasons={selected['rank_reasons']}")
    print("")
    print("Si la table sélectionnée est mauvaise :")
    print("  1) ouvre le contact sheet")
    print("  2) repère le bon rank")
    print("  3) relance avec --pick-rank N")


if __name__ == "__main__":
    main()
