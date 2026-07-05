from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "003J0"

VIDEO_COL_PRIORITY = [
    "mp4",
    "table_mp4",
    "video",
    "overlay_mp4",
    "segment_mp4",
]

TABLE_BAD_STATUSES = {"BAD_OR_UNRELIABLE", "WEAK", "MISSING_MODEL"}

HSV_PROFILES = [
    # table bleue / cyan / verte sombre, volontairement large.
    ("blue_cyan", (80, 35, 35), (135, 255, 255)),
    ("green_cyan", (45, 30, 30), (105, 255, 255)),
    ("dark_blue", (90, 25, 20), (135, 255, 180)),
    ("wide_cool", (40, 25, 25), (140, 255, 230)),
]

MIN_AREA_RATIO = 0.015
MAX_AREA_RATIO = 0.55


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def resolve_path(value, project_root: Path, run_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw:
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)
        candidates.append(run_dir / p)

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def choose_video(row: pd.Series, project_root: Path, run_dir: Path) -> tuple[Path | None, str]:
    for col in VIDEO_COL_PRIORITY:
        if col not in row.index:
            continue
        p = resolve_path(row.get(col), project_root, run_dir)
        if p and p.suffix.lower() in {".mp4", ".webm", ".mov", ".avi"}:
            return p, col
    return None, ""


def sample_frames(video_path: Path, max_frames: int = 13) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"error": "VideoCapture failed"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total <= 0:
        indices = list(range(max_frames))
    else:
        indices = np.linspace(0, max(0, total - 1), num=min(max_frames, total), dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    return frames, {
        "frame_count": total,
        "fps": fps,
        "width": width,
        "height": height,
        "sampled": len(frames),
    }


def median_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    stack = np.stack(resized, axis=0).astype(np.float32)
    med = np.median(stack, axis=0).astype(np.uint8)
    return med


def build_mask(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = []
    profiles = []

    for name, low, high in HSV_PROFILES:
        lo = np.array(low, dtype=np.uint8)
        hi = np.array(high, dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        masks.append(m)
        profiles.append({"name": name, "pixels": int(np.count_nonzero(m))})

    mask = np.zeros_like(masks[0])
    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    # Nettoyage : on ferme les trous, puis on retire le bruit.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)

    return mask, profiles


def contour_score(cnt: np.ndarray, frame_w: int, frame_h: int) -> dict:
    area = float(cv2.contourArea(cnt))
    frame_area = float(frame_w * frame_h)
    if frame_area <= 0:
        return {"valid": False, "reason": "empty_frame"}

    area_ratio = area / frame_area
    if area_ratio < MIN_AREA_RATIO:
        return {"valid": False, "reason": "too_small", "area_ratio": area_ratio}
    if area_ratio > MAX_AREA_RATIO:
        return {"valid": False, "reason": "too_large", "area_ratio": area_ratio}

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return {"valid": False, "reason": "bad_bbox", "area_ratio": area_ratio}

    aspect = w / max(1.0, h)

    # Table en perspective : souvent large, mais on garde large pour ne pas rater.
    if aspect < 0.8 or aspect > 7.0:
        return {
            "valid": False,
            "reason": "bad_aspect",
            "area_ratio": area_ratio,
            "aspect": aspect,
        }

    rect_area = float(w * h)
    extent = area / max(1.0, rect_area)

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(1.0, hull_area)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

    # Bonus si forme polygonale simple.
    vertex_count = int(len(approx))
    vertex_bonus = 1.0
    if 4 <= vertex_count <= 8:
        vertex_bonus = 1.25
    elif vertex_count > 14:
        vertex_bonus = 0.75

    # Pénalité si trop haut/bas, mais pas éliminatoire.
    center_y = y + h / 2
    y_norm = center_y / max(1.0, frame_h)
    y_bonus = 1.0
    if 0.18 <= y_norm <= 0.82:
        y_bonus = 1.15
    else:
        y_bonus = 0.75

    score = (
        area_ratio * 100.0
        + extent * 15.0
        + solidity * 12.0
        + min(aspect, 4.0) * 2.0
    ) * vertex_bonus * y_bonus

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)

    return {
        "valid": True,
        "score": float(score),
        "area": area,
        "area_ratio": float(area_ratio),
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "aspect": float(aspect),
        "extent": float(extent),
        "solidity": float(solidity),
        "vertex_count": vertex_count,
        "center_y_norm": float(y_norm),
        "box": box.tolist(),
    }


def detect_table(frame: np.ndarray) -> tuple[dict, np.ndarray]:
    h, w = frame.shape[:2]
    mask, profiles = build_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        s = contour_score(cnt, w, h)
        if s.get("valid"):
            s["contour"] = cnt
            candidates.append(s)

    candidates.sort(key=lambda d: d["score"], reverse=True)

    if not candidates:
        return {
            "v2_status": "NO_CANDIDATE",
            "v2_profile_pixels": profiles,
            "v2_candidate_count": 0,
        }, mask

    best = candidates[0]
    cnt = best.pop("contour")

    return {
        "v2_status": "CANDIDATE",
        "v2_profile_pixels": profiles,
        "v2_candidate_count": len(candidates),
        **best,
    }, mask


def draw_diagnostic(frame: np.ndarray, mask: np.ndarray, result: dict, row: pd.Series) -> np.ndarray:
    vis = frame.copy()

    # Mask en overlay rouge léger.
    color_mask = np.zeros_like(vis)
    color_mask[:, :, 2] = mask
    vis = cv2.addWeighted(vis, 0.78, color_mask, 0.22, 0)

    if result.get("v2_status") == "CANDIDATE":
        box = np.array(result.get("box", []), dtype=np.int32)
        if box.shape == (4, 2):
            cv2.polylines(vis, [box], isClosed=True, color=(0, 255, 255), thickness=3)

        x = int(result.get("bbox_x", 0))
        y = int(result.get("bbox_y", 0))
        bw = int(result.get("bbox_w", 0))
        bh = int(result.get("bbox_h", 0))
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    rid = str(row.get("review_id_003G", "?"))
    old_status = str(row.get("table_reality_status_003I", "?"))
    old_score = row.get("table_context_score_003B2", "")
    old_inside = row.get("point_inside_table_quad_ratio", "")
    old_dist = row.get("point_distance_to_table_px_median", "")

    lines = [
        f"{rid} | old={old_status} | v2={result.get('v2_status')}",
        f"old_score={old_score} inside={old_inside} dist={old_dist}",
    ]

    if result.get("v2_status") == "CANDIDATE":
        lines.append(
            f"v2_score={result.get('score'):.2f} area={result.get('area_ratio'):.3f} "
            f"aspect={result.get('aspect'):.2f} vertices={result.get('vertex_count')}"
        )

    y0 = 28
    for i, text in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(vis, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def html_escape(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html(path: Path, rows: pd.DataFrame, summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.candidate{border-color:rgba(116,217,159,.7)}
.card.no{border-color:rgba(255,80,80,.7);box-shadow:inset 4px 0 0 rgba(255,80,80,.8)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:270px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.candidate{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.no{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.badge.old{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
img{display:block;width:900px;max-width:100%;border:1px solid #2b303b;border-radius:10px;background:#05060a}
.muted{color:#aab2c5}
"""

    cards = []
    for _, r in rows.iterrows():
        rid = str(r["review_id_003J0"])
        v2 = str(r["v2_status"])
        old = str(r.get("table_reality_status_003I", ""))

        cls = "card candidate" if v2 == "CANDIDATE" else "card no"
        badge = "badge candidate" if v2 == "CANDIDATE" else "badge no"

        img_rel = r.get("diagnostic_png_rel", "")

        cols = [
            "target_class_003G",
            "would_reject_shadow_003G",
            "table_reality_status_003I",
            "table_context_score_003B2",
            "point_inside_table_quad_ratio",
            "point_distance_to_table_px_median",
            "v2_status",
            "score",
            "area_ratio",
            "aspect",
            "extent",
            "solidity",
            "vertex_count",
            "center_y_norm",
            "v2_candidate_count",
            "video_source_col",
            "video_path",
        ]

        trs = []
        for c in cols:
            if c in r.index and not pd.isna(r[c]):
                trs.append(f"<tr><th>{html_escape(c)}</th><td>{html_escape(r[c])}</td></tr>")

        img_html = ""
        if isinstance(img_rel, str) and img_rel:
            img_html = f"<a href='{html_escape(img_rel)}'><img src='{html_escape(img_rel)}'></a>"
        else:
            img_html = "<p class='muted'>Aucun diagnostic PNG.</p>"

        cards.append(f"""
<div class="{cls}">
<h2>{html_escape(rid)}</h2>
<div>
  <span class="{badge}">{html_escape(v2)}</span>
  <span class="badge old">old={html_escape(old)}</span>
</div>
<div class="grid">
  <div>
    <h3>Mesures</h3>
    <table><tbody>{''.join(trs)}</tbody></table>
  </div>
  <div>
    <h3>Diagnostic image</h3>
    {img_html}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003J0 table detector repair probe</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003J0 table detector repair probe</h1>

<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>

{''.join(cards)}

</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/batch_001E")
    ap.add_argument("--only-bad", action="store_true", default=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    input_csv = run_dir / "table_reality_audit_003I.csv"
    input_json = run_dir / "table_reality_audit_summary_003I.json"

    if not input_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {input_csv}")
    if not input_json.is_file():
        raise SystemExit(f"Fichier introuvable: {input_json}")

    i_summary = json.loads(input_json.read_text(encoding="utf-8"))
    df = read_csv(input_csv)

    if "review_id_003G" not in df.columns:
        raise SystemExit("Colonne review_id_003G manquante.")
    if "table_reality_status_003I" not in df.columns:
        raise SystemExit("Colonne table_reality_status_003I manquante.")

    df["review_id_003G"] = df["review_id_003G"].astype(str)

    if args.all:
        work = df.copy()
    else:
        work = df[df["table_reality_status_003I"].isin(TABLE_BAD_STATUSES)].copy()

    asset_dir = run_dir / "table_detector_repair_probe_003J0_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in work.iterrows():
        rid = str(row["review_id_003G"])
        video_path, video_col = choose_video(row, project_root, run_dir)

        base = {
            "review_id_003J0": rid,
            "video_source_col": video_col,
            "video_path": str(video_path) if video_path else "",
        }

        for c in row.index:
            if c not in base:
                base[c] = row[c]

        if not video_path:
            base.update({
                "v2_status": "NO_VIDEO",
                "diagnostic_png": "",
                "diagnostic_png_rel": "",
            })
            records.append(base)
            continue

        frames, meta = sample_frames(video_path)
        for k, v in meta.items():
            base[f"video_{k}"] = v

        med = median_frame(frames)
        if med is None:
            base.update({
                "v2_status": "NO_FRAME",
                "diagnostic_png": "",
                "diagnostic_png_rel": "",
            })
            records.append(base)
            continue

        result, mask = detect_table(med)
        diag = draw_diagnostic(med, mask, result, row)

        diag_path = asset_dir / f"{rid}_table_v2_probe.png"
        imwrite_unicode(diag_path, diag)

        base.update(result)
        base["diagnostic_png"] = str(diag_path)
        base["diagnostic_png_rel"] = os.path.relpath(diag_path, run_dir).replace("\\", "/")

        # Sérialisation propre des listes.
        if "box" in base:
            base["box"] = json.dumps(base["box"], ensure_ascii=False)
        if "v2_profile_pixels" in base:
            base["v2_profile_pixels"] = json.dumps(base["v2_profile_pixels"], ensure_ascii=False)

        records.append(base)

    out = pd.DataFrame(records)

    candidate_ids = out[out["v2_status"].eq("CANDIDATE")]["review_id_003J0"].astype(str).tolist()
    no_candidate_ids = out[~out["v2_status"].eq("CANDIDATE")]["review_id_003J0"].astype(str).tolist()

    old_bad_count = int(len(work))
    candidate_count = int(len(candidate_ids))

    # Ce n'est pas un vrai repair encore : on mesure seulement s'il existe une piste.
    if candidate_count == 0:
        status = "WARN"
    else:
        status = "OK"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "source_csv": str(input_csv),
        "source_json": str(input_json),
        "input_003I_status": i_summary.get("status"),
        "policy": "probe_only_no_table_model_write_no_arbiter_change",
        "rows_tested": old_bad_count,
        "v2_candidate_count": candidate_count,
        "v2_candidate_ids": candidate_ids,
        "v2_no_candidate_ids": no_candidate_ids,
        "asset_dir": str(asset_dir),
        "interpretation": (
            "Si les PNG montrent que le contour jaune/vert colle mieux à la vraie table, "
            "on pourra transformer ce probe en table detector v2. "
            "Sinon il faut passer à une annotation manuelle de coins table pour quelques clips."
        ),
    }

    out_csv = run_dir / "table_detector_repair_probe_003J0.csv"
    out_json = run_dir / "table_detector_repair_probe_summary_003J0.json"
    out_html = run_dir / "table_detector_repair_probe_003J0.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print(f"003J0 status={status}")
    print(f"rows_tested={old_bad_count}")
    print(f"v2_candidate_count={candidate_count}")
    print("v2_candidate_ids=" + (",".join(candidate_ids) or "-"))
    print("v2_no_candidate_ids=" + (",".join(no_candidate_ids) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)
    print("assets", asset_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
