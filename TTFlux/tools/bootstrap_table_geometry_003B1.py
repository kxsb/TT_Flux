from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        frame_idx = max(0, min(int(frame_idx), frame_count - 1))
    else:
        frame_idx = max(0, int(frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()

    if not ok or img is None:
        return None

    return img


def fnum(v: Any, default: float | None = None) -> float | None:
    try:
        s = str(v or "").strip().replace(",", ".")
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def valid_point(p: Any) -> bool:
    return isinstance(p, dict) and p.get("x") is not None and p.get("y") is not None


def normalize_quad(q: Any) -> list[Any]:
    labels = ["front_left", "front_right", "back_right", "back_left"]
    if not isinstance(q, list):
        q = []

    out = []
    for i in range(4):
        if i < len(q) and valid_point(q[i]):
            p = q[i]
            out.append({
                "label": p.get("label", labels[i]),
                "x": float(p["x"]),
                "y": float(p["y"]),
            })
        else:
            out.append(None)
    return out


def quad_complete(q: Any) -> bool:
    q = normalize_quad(q)
    return sum(1 for p in q if valid_point(p)) == 4


def quad_to_np(q: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[float(p["x"]), float(p["y"])] for p in q], dtype=np.float32)


def quad_area(q: list[dict[str, Any]]) -> float:
    if not quad_complete(q):
        return 0.0
    pts = quad_to_np(q)
    return float(abs(cv2.contourArea(pts)))


def order_quad_from_points(points: np.ndarray) -> list[dict[str, float]]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) != 4:
        raise ValueError("need 4 points")

    # Convention TTFlux :
    # front = bord bas dans l'image, back = bord haut.
    by_y = sorted(pts.tolist(), key=lambda p: p[1])
    top = sorted(by_y[:2], key=lambda p: p[0])
    bottom = sorted(by_y[2:], key=lambda p: p[0])

    back_left = top[0]
    back_right = top[1]
    front_left = bottom[0]
    front_right = bottom[1]

    return [
        {"label": "front_left", "x": round(float(front_left[0]), 2), "y": round(float(front_left[1]), 2)},
        {"label": "front_right", "x": round(float(front_right[0]), 2), "y": round(float(front_right[1]), 2)},
        {"label": "back_right", "x": round(float(back_right[0]), 2), "y": round(float(back_right[1]), 2)},
        {"label": "back_left", "x": round(float(back_left[0]), 2), "y": round(float(back_left[1]), 2)},
    ]


def polygon_quality(q: list[dict[str, Any]], w: int, h: int) -> dict[str, float]:
    if not quad_complete(q):
        return {
            "area_norm": 0.0,
            "convex": 0.0,
            "center_score": 0.0,
            "aspect_box": 0.0,
            "aspect_score": 0.0,
            "quality": 0.0,
        }

    pts = quad_to_np(q)
    area = abs(cv2.contourArea(pts))
    area_norm = area / max(1.0, float(w * h))
    convex = 1.0 if cv2.isContourConvex(pts.astype(np.int32)) else 0.0

    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # La table est généralement dans le tiers central / bas, mais on reste large.
    cx_score = max(0.0, 1.0 - abs((cx / w) - 0.5) / 0.55)
    cy_score = max(0.0, 1.0 - abs((cy / h) - 0.55) / 0.55)
    center_score = 0.5 * cx_score + 0.5 * cy_score

    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]
    if rw <= 1 or rh <= 1:
        aspect_box = 0.0
    else:
        aspect_box = max(rw, rh) / max(1.0, min(rw, rh))

    # En perspective, le ratio apparent peut être loin du ratio réel.
    # On pénalise seulement les cas extrêmes.
    if 1.05 <= aspect_box <= 5.5:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - min(abs(aspect_box - 1.8), abs(aspect_box - 4.0)) / 4.0)

    area_score = 0.0
    if 0.015 <= area_norm <= 0.45:
        area_score = 1.0
    elif area_norm < 0.015:
        area_score = area_norm / 0.015
    else:
        area_score = max(0.0, 1.0 - (area_norm - 0.45) / 0.40)

    quality = (
        0.30 * area_score
        + 0.25 * convex
        + 0.25 * center_score
        + 0.20 * aspect_score
    )

    return {
        "area_norm": round(area_norm, 6),
        "convex": convex,
        "center_score": round(center_score, 6),
        "aspect_box": round(aspect_box, 6),
        "aspect_score": round(aspect_score, 6),
        "quality": round(float(quality), 6),
    }


def infer_frame_range(row: dict[str, str]) -> tuple[int | None, int | None]:
    for a, b in [
        ("first_frame", "last_frame"),
        ("frame_first", "frame_last"),
        ("start_frame", "end_frame"),
    ]:
        if row.get(a) and row.get(b):
            try:
                return int(float(row[a])), int(float(row[b]))
            except Exception:
                pass

    blob = " ".join(str(v) for v in row.values())
    m = re.search(r"f(\d+)_to_f(\d+)", blob)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def collect_probe_frames(final_csv: Path, clips: list[dict[str, Any]]) -> dict[str, list[int]]:
    rows = read_csv(final_csv)
    out: dict[str, set[int]] = {}

    for c in clips:
        frame_count = int(c.get("frame_count") or 0)
        base = {0}
        if frame_count > 0:
            base.update([frame_count // 4, frame_count // 2, int(frame_count * 0.75)])
        out[c["clip_id"]] = base

    for row in rows:
        blob = " ".join(str(v) for v in row.values())

        for c in clips:
            stem = Path(c["filename"]).stem
            if stem not in blob:
                continue

            f1, f2 = infer_frame_range(row)
            if f1 is None or f2 is None:
                continue

            mid = int(round((f1 + f2) / 2))
            out[c["clip_id"]].update([f1, mid, f2])

    return {cid: sorted(v) for cid, v in out.items()}


def load_manual_annotations(path: Path) -> dict[str, Any]:
    data = read_json(path) or {}
    clips = data.get("clips")
    if not isinstance(clips, dict):
        return {}
    return clips


def get_manual_quad(manual_by_clip: dict[str, Any], clip_id: str) -> tuple[list[Any] | None, int]:
    c = manual_by_clip.get(clip_id)
    if not isinstance(c, dict):
        return None, 0

    if isinstance(c.get("table_keyframes"), list) and c["table_keyframes"]:
        # Garde la keyframe la plus complète.
        best = None
        best_count = -1
        for kf in c["table_keyframes"]:
            q = normalize_quad(kf.get("table_quad"))
            count = sum(1 for p in q if valid_point(p))
            if count > best_count:
                best = kf
                best_count = count
        if best:
            return normalize_quad(best.get("table_quad")), int(float(best.get("frame") or 0))

    q = normalize_quad(c.get("table_quad"))
    return q, int(float(c.get("frame_ref") or 0))


def detect_color_candidates(img: np.ndarray, frame: int, clip_id: str, max_candidates: int = 8) -> list[dict[str, Any]]:
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masque large : table bleue/verte/saturée, avec exclusion du très sombre/noir pur et du très blanc.
    # L'objectif est de proposer des hypothèses, pas de valider seul.
    lower1 = np.array([35, 35, 25], dtype=np.uint8)
    upper1 = np.array([145, 255, 235], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)

    # Limite souple : on évite le haut complet de l'image, souvent public/fond.
    roi = np.zeros_like(mask)
    roi[int(h * 0.18):int(h * 0.95), :] = 255
    mask = cv2.bitwise_and(mask, roi)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_norm = area / max(1.0, float(w * h))

        if area_norm < 0.008 or area_norm > 0.55:
            continue

        hull = cv2.convexHull(contour)
        peri = float(cv2.arcLength(hull, True))
        approx = cv2.approxPolyDP(hull, 0.025 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(hull)
            pts = cv2.boxPoints(rect).astype(np.float32)

        try:
            quad = order_quad_from_points(pts)
        except Exception:
            continue

        q = polygon_quality(quad, w, h)

        if q["quality"] <= 0.25:
            continue

        # Bonus si la zone contient des pixels table-like.
        x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        if bw <= 0 or bh <= 0:
            continue

        local_mask = mask[y:y + bh, x:x + bw]
        color_fill = float((local_mask > 0).mean())

        score = q["quality"] + 0.20 * min(1.0, color_fill * 2.0)

        candidates.append({
            "clip_id": clip_id,
            "frame": frame,
            "source": "auto_color_contour",
            "table_quad": quad,
            "score": round(float(score), 6),
            "quality": q,
            "color_fill": round(color_fill, 6),
        })

    candidates = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)
    return candidates[:max_candidates]


def draw_quad(img: np.ndarray, quad: list[Any], title: str, score: float | None = None) -> np.ndarray:
    out = img.copy()
    q = normalize_quad(quad)

    if quad_complete(q):
        pts = quad_to_np(q).astype(np.int32)
        fill = out.copy()
        cv2.fillPoly(fill, [pts], (80, 230, 120))
        out = cv2.addWeighted(fill, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, (80, 230, 120), 3, cv2.LINE_AA)

        labels = ["1 FL", "2 FR", "3 BR", "4 BL"]
        for i, p in enumerate(q):
            x = int(round(float(p["x"])))
            y = int(round(float(p["y"])))
            cv2.circle(out, (x, y), 8, (80, 230, 120), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, labels[i], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, labels[i], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    text = title if score is None else f"{title} score={score:.3f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text[:110], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build_bootstrap(
    payload: dict[str, Any],
    priors: dict[str, Any],
    manual_by_clip: dict[str, Any],
    final_csv: Path,
    out_dir: Path,
    ignore_manual: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clips = payload.get("clips", [])
    probe_frames_by_clip = collect_probe_frames(final_csv, clips)
    overlays_dir = out_dir / "overlays"
    candidates_dir = out_dir / "candidate_overlays"

    models: dict[str, Any] = {}
    summary_rows = []

    for clip in clips:
        clip_id = clip["clip_id"]
        video_path = Path(str(clip["video_path"]))
        frame_count = int(clip.get("frame_count") or 0)
        w = int(clip.get("width") or 0)
        h = int(clip.get("height") or 0)

        candidates: list[dict[str, Any]] = []

        manual_quad, manual_frame = get_manual_quad(manual_by_clip, clip_id)
        manual_complete = manual_quad is not None and quad_complete(manual_quad)

        if manual_complete and clip_id not in ignore_manual:
            q = polygon_quality(manual_quad, w, h)
            candidates.append({
                "clip_id": clip_id,
                "frame": manual_frame,
                "source": "manual_seed_003A",
                "table_quad": manual_quad,
                "score": round(0.80 + 0.20 * q["quality"], 6),
                "quality": q,
                "color_fill": "",
            })

        frames = set(probe_frames_by_clip.get(clip_id, []))
        if manual_frame is not None:
            frames.add(int(manual_frame))

        # Limite : quelques frames de test par clip.
        frame_list = sorted(f for f in frames if f >= 0)
        if frame_count > 0:
            frame_list = [min(f, frame_count - 1) for f in frame_list]
        frame_list = sorted(set(frame_list))[:12]

        for frame in frame_list:
            img = read_video_frame(video_path, frame)
            if img is None:
                continue

            auto_candidates = detect_color_candidates(img, frame=frame, clip_id=clip_id, max_candidates=5)
            candidates.extend(auto_candidates)

            for idx, cand in enumerate(auto_candidates[:3], start=1):
                overlay = draw_quad(img, cand["table_quad"], f"{clip_id} f{frame} {cand['source']} #{idx}", float(cand["score"]))
                overlay_path = candidates_dir / f"{clip_id}_f{frame}_cand{idx}.jpg"
                imwrite_unicode(overlay_path, overlay, quality=90)
                cand["candidate_overlay"] = str(overlay_path.relative_to(out_dir))

        candidates = sorted(candidates, key=lambda c: float(c["score"]), reverse=True)

        chosen = candidates[0] if candidates else None

        if chosen:
            frame = int(chosen.get("frame") or 0)
            img = read_video_frame(video_path, frame)
            overlay_rel = ""

            if img is not None:
                overlay = draw_quad(img, chosen["table_quad"], f"CHOSEN {clip_id} {chosen['source']} f{frame}", float(chosen["score"]))
                overlay_path = overlays_dir / f"table_bootstrap_{clip_id}.jpg"
                imwrite_unicode(overlay_path, overlay, quality=92)
                overlay_rel = str(overlay_path.relative_to(out_dir))

            model = {
                "clip_id": clip_id,
                "filename": clip.get("filename", ""),
                "video_path": clip.get("video_path", ""),
                "width": w,
                "height": h,
                "frame_ref": frame,
                "table_quad": chosen["table_quad"],
                "source": chosen["source"],
                "confidence": round(float(chosen["score"]), 6),
                "overlay": overlay_rel,
                "quality": chosen.get("quality", {}),
                "candidates_count": len(candidates),
                "candidates": candidates[:12],
                "ignored_manual": clip_id in ignore_manual,
                "manual_complete_available": bool(manual_complete),
            }
        else:
            model = {
                "clip_id": clip_id,
                "filename": clip.get("filename", ""),
                "video_path": clip.get("video_path", ""),
                "width": w,
                "height": h,
                "frame_ref": 0,
                "table_quad": [None, None, None, None],
                "source": "missing",
                "confidence": 0.0,
                "overlay": "",
                "quality": {},
                "candidates_count": 0,
                "candidates": [],
                "ignored_manual": clip_id in ignore_manual,
                "manual_complete_available": bool(manual_complete),
            }

        models[clip_id] = model
        summary_rows.append({
            "clip_id": clip_id,
            "filename": clip.get("filename", ""),
            "source": model["source"],
            "confidence": model["confidence"],
            "candidates": model["candidates_count"],
            "manual_complete_available": model["manual_complete_available"],
            "ignored_manual": model["ignored_manual"],
        })

    complete = [cid for cid, m in models.items() if quad_complete(m.get("table_quad"))]
    missing = [cid for cid, m in models.items() if not quad_complete(m.get("table_quad"))]

    result = {
        "version": "003B1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Semi-automatic table geometry bootstrap using manual seeds, regulatory priors, and broad color/contour hypotheses.",
        "priors_version": priors.get("version", ""),
        "table_ratio_length_over_width": priors.get("regulatory_objects", {}).get("table", {}).get("aspect_ratio_length_over_width", ""),
        "ignored_manual_clip_ids": sorted(ignore_manual),
        "table_models": models,
    }

    summary = {
        "clips_total": len(models),
        "clips_with_table_model": len(complete),
        "clips_missing_table_model": len(missing),
        "complete_clip_ids": complete,
        "missing_clip_ids": missing,
        "models": summary_rows,
        "warning": "003B1 is a bootstrap/audit layer, not a final detector. Validate overlays before using table_context features.",
    }

    return result, summary


def write_html(path: Path, result: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = []

    for clip_id, model in result["table_models"].items():
        status = "ok" if quad_complete(model.get("table_quad")) else "bad"
        overlay_html = ""
        if model.get("overlay"):
            overlay_html = f'<img src="{html.escape(model["overlay"])}" loading="lazy">'

        cand_rows = []
        for c in model.get("candidates", [])[:8]:
            cand_rows.append(f"""
<tr>
<td>{html.escape(str(c.get("source", "")))}</td>
<td>{html.escape(str(c.get("frame", "")))}</td>
<td>{html.escape(str(c.get("score", "")))}</td>
<td>{html.escape(str(c.get("quality", {}).get("area_norm", "")))}</td>
<td>{html.escape(str(c.get("quality", {}).get("aspect_box", "")))}</td>
<td>{html.escape(str(c.get("candidate_overlay", "")))}</td>
</tr>
""")

        rows.append(f"""
<section>
<h2>{html.escape(clip_id)} · <span class="{status}">{html.escape(str(model.get("source", "")))}</span></h2>
<p><b>Fichier :</b> {html.escape(str(model.get("filename", "")))}</p>
<p><b>Frame :</b> {html.escape(str(model.get("frame_ref", "")))} · <b>Confiance :</b> {html.escape(str(model.get("confidence", "")))}</p>
<p><b>Manual disponible :</b> {html.escape(str(model.get("manual_complete_available", "")))} · <b>Manual ignoré :</b> {html.escape(str(model.get("ignored_manual", "")))}</p>
<pre>{html.escape(json.dumps(model.get("table_quad", []), ensure_ascii=False))}</pre>
{overlay_html}
<h3>Candidats top</h3>
<table>
<thead><tr><th>source</th><th>frame</th><th>score</th><th>area</th><th>aspect</th><th>overlay</th></tr></thead>
<tbody>{''.join(cand_rows)}</tbody>
</table>
</section>
""")

    model_rows = "".join(
        f"<tr><td>{html.escape(str(m['clip_id']))}</td><td>{html.escape(str(m['source']))}</td><td>{html.escape(str(m['confidence']))}</td><td>{html.escape(str(m['candidates']))}</td><td>{html.escape(str(m['ignored_manual']))}</td></tr>"
        for m in summary["models"]
    )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux table bootstrap 003B1</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:13px}}
code,pre{{color:#dce4ff}}
pre{{white-space:pre-wrap;word-break:break-word;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
.ok{{color:#74d99f}}
.bad{{color:#ff7a7a}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux · table bootstrap 003B1</h1>

<section>
<h2>Résumé</h2>
<table>
<tr><th>clips_total</th><td>{summary["clips_total"]}</td></tr>
<tr><th>clips_with_table_model</th><td>{summary["clips_with_table_model"]}</td></tr>
<tr><th>clips_missing_table_model</th><td>{summary["clips_missing_table_model"]}</td></tr>
<tr><th>warning</th><td class="warn">{html.escape(summary["warning"])}</td></tr>
</table>

<table>
<thead><tr><th>clip</th><th>source</th><th>confidence</th><th>candidates</th><th>manual ignoré</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>
</section>

{''.join(rows)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="runs/batch_001E/table_scene_003A/table_annotation_payload_003A.json")
    parser.add_argument("--priors", default="runs/batch_001E/scene_priors_003B0/scene_priors_003B0.json")
    parser.add_argument("--manual-json", default="runs/batch_001E/table_scene_003A/table_annotations_003A_merged.json")
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/table_bootstrap_003B1")
    parser.add_argument("--ignore-manual", default="")
    args = parser.parse_args()

    payload = read_json(Path(args.payload))
    if not payload or not isinstance(payload.get("clips"), list):
        raise SystemExit("[003B1] Payload invalide ou introuvable.")

    priors = read_json(Path(args.priors)) or {}
    manual_by_clip = load_manual_annotations(Path(args.manual_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore_manual = {
        x.strip()
        for x in str(args.ignore_manual or "").split(",")
        if x.strip()
    }

    result, summary = build_bootstrap(
        payload=payload,
        priors=priors,
        manual_by_clip=manual_by_clip,
        final_csv=Path(args.final_csv),
        out_dir=out_dir,
        ignore_manual=ignore_manual,
    )

    result_path = out_dir / "table_bootstrap_models_003B1.json"
    summary_path = out_dir / "table_bootstrap_summary_003B1.json"
    html_path = out_dir / "table_bootstrap_003B1.html"

    write_json(result_path, result)
    write_json(summary_path, summary)
    write_html(html_path, result, summary)

    print(f"[003B1] clips total       : {summary['clips_total']}")
    print(f"[003B1] with table model  : {summary['clips_with_table_model']}")
    print(f"[003B1] missing           : {summary['clips_missing_table_model']}")
    print(f"[003B1] ignored manual    : {sorted(ignore_manual)}")
    print(f"[003B1] out dir           : {out_dir}")
    print(f"[003B1] html              : {html_path}")


if __name__ == "__main__":
    main()
