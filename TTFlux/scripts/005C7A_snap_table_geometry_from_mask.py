from __future__ import annotations

import argparse
import html
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C7A"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def read_frame(clip_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def resize_max_w(img, max_w: int):
    h, w = img.shape[:2]
    if max_w <= 0 or w <= max_w:
        return img.copy(), 1.0

    scale = max_w / max(1, w)
    out = cv2.resize(img, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def feature_image(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    bgr = img_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    lab /= 255.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(1, w - 1)
    yy /= max(1, h - 1)

    feats = np.dstack([
        bgr[:, :, 0],
        bgr[:, :, 1],
        bgr[:, :, 2],
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2],
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        xx,
        yy,
    ])

    return feats.reshape(-1, feats.shape[-1])


def predict_table_mask(model, img_bgr, threshold: float, chunk_size: int):
    feats = feature_image(img_bgr)
    n = len(feats)

    probs = np.zeros((n,), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        pp = model.predict_proba(feats[start:end])
        probs[start:end] = pp[:, 1].astype(np.float32)

    h, w = img_bgr.shape[:2]
    prob_img = probs.reshape(h, w)

    mask = (prob_img >= threshold).astype(np.uint8) * 255

    k1 = np.ones((3, 3), np.uint8)
    k2 = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    return prob_img, mask


def old_quad_proc(row: dict, scale: float):
    try:
        q = np.asarray([
            [float(row["quad_tl_x_005C3"]), float(row["quad_tl_y_005C3"])],
            [float(row["quad_tr_x_005C3"]), float(row["quad_tr_y_005C3"])],
            [float(row["quad_br_x_005C3"]), float(row["quad_br_y_005C3"])],
            [float(row["quad_bl_x_005C3"]), float(row["quad_bl_y_005C3"])],
        ], dtype=np.float32)
        return q * float(scale)
    except Exception:
        return np.zeros((4, 2), dtype=np.float32)


def quad_area(q):
    q = np.asarray(q, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(q)))


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def polygon_iou_mask(poly: np.ndarray, mask: np.ndarray):
    h, w = mask.shape[:2]
    pm = np.zeros((h, w), dtype=np.uint8)

    try:
        cv2.fillConvexPoly(pm, np.round(poly).astype(np.int32), 255)
    except Exception:
        return 0.0, 0.0, 0.0, pm

    m = mask > 0
    p = pm > 0

    inter = float(np.logical_and(m, p).sum())
    union = float(np.logical_or(m, p).sum())
    poly_area = float(p.sum())
    mask_area = float(m.sum())

    iou = inter / max(1.0, union)
    mask_in_poly = inter / max(1.0, mask_area)
    poly_support = inter / max(1.0, poly_area)

    return iou, mask_in_poly, poly_support, pm


def fit_quad_from_contour(cnt, mask):
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))

    if hull_area <= 1:
        return None, "bad_hull"

    candidates = []

    # 1) approxPolyDP : cherche un vrai quad de contour.
    for eps_ratio in [0.012, 0.016, 0.022, 0.030, 0.040, 0.055, 0.075, 0.10]:
        eps = eps_ratio * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True)

        if len(approx) == 4:
            q = order_quad(approx.reshape(4, 2).astype(np.float32))
            area = quad_area(q)
            iou, mask_in_poly, poly_support, _ = polygon_iou_mask(q, mask)

            candidates.append({
                "quad": q,
                "method": f"approx4_eps{eps_ratio}",
                "area": area,
                "iou": iou,
                "mask_in_poly": mask_in_poly,
                "poly_support": poly_support,
            })

    # 2) minAreaRect fallback : utile quand le masque est presque rectangulaire mais pas assez polygonal.
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    q = order_quad(box)
    area = quad_area(q)
    iou, mask_in_poly, poly_support, _ = polygon_iou_mask(q, mask)

    candidates.append({
        "quad": q,
        "method": "minAreaRect_fallback",
        "area": area,
        "iou": iou,
        "mask_in_poly": mask_in_poly,
        "poly_support": poly_support,
    })

    if not candidates:
        return None, "no_candidate"

    # On favorise un polygone qui couvre le masque mais qui n'englobe pas trop de non-table.
    candidates = sorted(
        candidates,
        key=lambda c: (
            0.46 * c["poly_support"]
            + 0.34 * c["mask_in_poly"]
            + 0.20 * c["iou"]
            - 0.06 * (1.0 if c["method"].startswith("minAreaRect") else 0.0)
        ),
        reverse=True,
    )

    return candidates[0], "ok"


def connected_components_candidates(mask: np.ndarray, old_quad_mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    total = max(1, h * w)

    out = []

    old = old_quad_mask > 0

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < total * 0.004:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < w * 0.06 or bh < h * 0.025:
            continue

        comp_mask = np.zeros_like(mask)
        cv2.drawContours(comp_mask, [cnt], -1, 255, -1)

        comp = comp_mask > 0
        old_overlap = float(np.logical_and(comp, old).sum()) / max(1.0, float(comp.sum()))

        aspect = bw / max(1, bh)
        fill = area / max(1.0, bw * bh)

        # Table broadcast : souvent plus large que haute, mais plans serrés peuvent être atypiques.
        aspect_score = 1.0
        if aspect < 0.65:
            aspect_score = max(0.0, aspect / 0.65)
        elif aspect > 8.0:
            aspect_score = max(0.0, 8.0 / aspect)

        score = (
            0.38 * min(1.0, area / (total * 0.09))
            + 0.26 * old_overlap
            + 0.18 * min(1.0, fill / 0.75)
            + 0.18 * aspect_score
        )

        out.append({
            "cnt": cnt,
            "area": area,
            "area_ratio": area / total,
            "bbox": (x, y, bw, bh),
            "aspect": aspect,
            "fill": fill,
            "old_overlap": old_overlap,
            "score": score,
        })

    out = sorted(out, key=lambda r: r["score"], reverse=True)
    return out


def table_pts_m():
    return np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)


def compute_homographies_from_src_quad(q_src):
    q_src = np.asarray(q_src, dtype=np.float32).reshape(4, 2)
    t = table_pts_m()
    H_img_to_table = cv2.getPerspectiveTransform(q_src, t)
    H_table_to_img = cv2.getPerspectiveTransform(t, q_src)
    return H_img_to_table, H_table_to_img


def classify_snap(cand, fit, inside_ratio: float, points_projected: int, min_metric_conf: float):
    if cand is None or fit is None:
        return "BAD_TABLE_MASK", "no_component_or_quad", 0.0

    area_ratio = float(cand["area_ratio"])
    old_overlap = float(cand["old_overlap"])
    fill = float(cand["fill"])

    poly_support = float(fit.get("poly_support", 0.0))
    mask_in_poly = float(fit.get("mask_in_poly", 0.0))
    iou = float(fit.get("iou", 0.0))
    method = str(fit.get("method", ""))

    fallback_penalty = 0.12 if method.startswith("minAreaRect") else 0.0

    conf = (
        0.24 * min(1.0, area_ratio / 0.085)
        + 0.24 * poly_support
        + 0.20 * mask_in_poly
        + 0.14 * iou
        + 0.10 * min(1.0, fill / 0.75)
        + 0.08 * min(1.0, max(0.0, inside_ratio))
        - fallback_penalty
    )

    if conf >= min_metric_conf and poly_support >= 0.42 and mask_in_poly >= 0.45:
        if points_projected >= 20 and inside_ratio >= 0.72:
            return "SNAP_METRIC_TABLE_OK", "mask_quad_and_projection_agree", conf
        return "SNAP_METRIC_CANDIDATE", "mask_quad_ok_projection_uncertain", conf

    if area_ratio >= 0.012 and mask_in_poly >= 0.20:
        return "SNAP_PARTIAL_TABLE", "mask_table_visible_but_quad_uncertain", conf

    return "BAD_TABLE_MASK", "weak_mask_geometry", conf


def draw_poly(img, q, color, label):
    if q is None:
        return

    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    qi = np.round(q).astype(int)

    for a, b in zip(qi, np.vstack([qi[1:], qi[:1]])):
        cv2.line(img, tuple(a), tuple(b), color, 3, cv2.LINE_AA)

    for i, p in enumerate(qi):
        cv2.circle(img, tuple(p), 6, color, -1, cv2.LINE_AA)

    cv2.putText(
        img,
        label,
        (int(qi[0, 0]) + 8, int(qi[0, 1]) + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def make_overlay(img, prob_img, mask, old_quad, snap_quad, status_text):
    heat = np.clip(prob_img * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    ov = cv2.addWeighted(img, 0.62, heat_color, 0.38, 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)

    draw_poly(ov, old_quad, (0, 255, 255), "old")
    draw_poly(ov, snap_quad, (255, 0, 255), "snap")

    cv2.putText(
        ov,
        status_text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return ov


def canonical_warp(img, q_proc, out_w=520, out_h=936):
    if q_proc is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    dst = np.asarray([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(np.asarray(q_proc, dtype=np.float32), dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h))

    cv2.rectangle(warped, (0, 0), (out_w - 1, out_h - 1), (0, 255, 255), 3)
    cv2.line(warped, (0, out_h // 2), (out_w - 1, out_h // 2), (0, 180, 255), 2)
    cv2.line(warped, (out_w // 2, 0), (out_w // 2, out_h - 1), (255, 255, 0), 2)

    return warped


def hconcat_resized(left, right, right_w=360):
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]

    scale = right_w / max(1, w2)
    r = cv2.resize(right, (right_w, int(round(h2 * scale))), interpolation=cv2.INTER_AREA)

    h = max(h1, r.shape[0])

    def pad(img):
        dh = h - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    return cv2.hconcat([pad(left), pad(r)])


def build_html(cards):
    parts = []

    for c in cards:
        parts.append(f"""
<section class="card {html.escape(str(c["status"]))}">
  <h2>{html.escape(str(c["review_id"]))} · seg {c["seg"]} · {html.escape(str(c["status"]))}</h2>
  <p>
    reason={html.escape(str(c["reason"]))}
    · conf={c["conf"]}
    · area={c["area_ratio"]}
    · poly_support={c["poly_support"]}
    · mask_in_poly={c["mask_in_poly"]}
    · old_status={html.escape(str(c["old_status"]))}
  </p>
  <img src="images/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C7A snapped table geometry</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
.card.SNAP_METRIC_TABLE_OK{{border-color:#2d8a4d}}
.card.SNAP_METRIC_CANDIDATE{{border-color:#4a8a8a}}
.card.SNAP_PARTIAL_TABLE{{border-color:#9b842e}}
.card.BAD_TABLE_MASK{{border-color:#9b3939}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005C7A · snap géométrique table depuis masque appris</h1>
<p>Vert = masque table appris · jaune = ancien quad · magenta = quad snapped · droite = warp canonique du snap.</p>
</header>
{''.join(parts)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", default="runs/rally_table_mask_classifier_005C6B/table_object_classification_005C6B.csv")
    ap.add_argument("--model", default="runs/rally_table_mask_classifier_005C6B/table_mask_rf_model_005C6B.pkl")
    ap.add_argument("--out-dir", default="runs/rally_table_snap_geometry_005C7A")
    ap.add_argument("--mask-max-w", type=int, default=640)
    ap.add_argument("--prob-threshold", type=float, default=0.55)
    ap.add_argument("--chunk-size", type=int, default=250000)
    ap.add_argument("--min-metric-conf", type=float, default=0.54)
    ap.add_argument("--make-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=90)
    args = ap.parse_args()

    root = Path.cwd()

    classification_path = Path(args.classification)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    if not classification_path.is_absolute():
        classification_path = root / classification_path
    if not model_path.is_absolute():
        model_path = root / model_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(classification_path).fillna("")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    rows = []
    cards = []

    for i, (_, r) in enumerate(table.iterrows(), start=1):
        obj = r.to_dict()
        review_id = str(obj["review_id"])
        seg = int(to_num(pd.Series([obj["camera_segment_id_005B2"]])).fillna(1).iloc[0])

        clip_path = Path(str(obj["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([obj.get("mask_frame_005C6B", obj.get("best_frame_005C1", obj.get("first_frame", 0)))])).fillna(0).iloc[0])

        frame = read_frame(clip_path, frame_idx)

        if frame is None:
            out = dict(obj)
            out.update({
                "snap_status_005C7A": "BAD_TABLE_MASK",
                "snap_reason_005C7A": "missing_frame",
            })
            rows.append(out)
            continue

        proc, scale = resize_max_w(frame, args.mask_max_w)
        prob_img, mask = predict_table_mask(model, proc, args.prob_threshold, args.chunk_size)

        old_q = old_quad_proc(obj, scale=scale)

        old_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        try:
            cv2.fillConvexPoly(old_mask, np.round(old_q).astype(np.int32), 255)
        except Exception:
            pass

        comps = connected_components_candidates(mask, old_mask)

        cand = comps[0] if comps else None
        fit = None
        fit_reason = "no_component"

        if cand is not None:
            fit, fit_reason = fit_quad_from_contour(cand["cnt"], mask)

        inside_ratio = float(to_num(pd.Series([obj.get("audit_inside_ratio", obj.get("inside_ratio", 0))])).fillna(0).iloc[0])
        points_projected = int(to_num(pd.Series([obj.get("audit_points_projected", obj.get("points_projected", 0))])).fillna(0).iloc[0])

        status, reason, conf = classify_snap(
            cand,
            fit,
            inside_ratio=inside_ratio,
            points_projected=points_projected,
            min_metric_conf=args.min_metric_conf,
        )

        snap_q_proc = None
        snap_q_src = np.zeros((4, 2), dtype=np.float32)

        if fit is not None:
            snap_q_proc = fit["quad"]
            snap_q_src = snap_q_proc / max(1e-9, scale)

        try:
            H_img_to_table, H_table_to_img = compute_homographies_from_src_quad(snap_q_src)
            h_ok = int(fit is not None and status in {"SNAP_METRIC_TABLE_OK", "SNAP_METRIC_CANDIDATE"})
        except Exception:
            H_img_to_table = np.eye(3, dtype=np.float64)
            H_table_to_img = np.eye(3, dtype=np.float64)
            h_ok = 0

        out = dict(obj)
        out.update({
            "snap_status_005C7A": status,
            "snap_reason_005C7A": reason,
            "snap_confidence_005C7A": round(float(conf), 6),
            "snap_fit_reason_005C7A": fit_reason,
            "snap_method_005C7A": "" if fit is None else str(fit["method"]),
            "snap_homography_ok_005C7A": h_ok,
            "snap_mask_component_area_ratio_005C7A": "" if cand is None else round(float(cand["area_ratio"]), 6),
            "snap_mask_component_aspect_005C7A": "" if cand is None else round(float(cand["aspect"]), 6),
            "snap_mask_component_fill_005C7A": "" if cand is None else round(float(cand["fill"]), 6),
            "snap_mask_old_overlap_005C7A": "" if cand is None else round(float(cand["old_overlap"]), 6),
            "snap_quad_iou_005C7A": "" if fit is None else round(float(fit["iou"]), 6),
            "snap_mask_in_poly_005C7A": "" if fit is None else round(float(fit["mask_in_poly"]), 6),
            "snap_poly_support_005C7A": "" if fit is None else round(float(fit["poly_support"]), 6),

            "snap_tl_x_005C7A": round(float(snap_q_src[0, 0]), 3),
            "snap_tl_y_005C7A": round(float(snap_q_src[0, 1]), 3),
            "snap_tr_x_005C7A": round(float(snap_q_src[1, 0]), 3),
            "snap_tr_y_005C7A": round(float(snap_q_src[1, 1]), 3),
            "snap_br_x_005C7A": round(float(snap_q_src[2, 0]), 3),
            "snap_br_y_005C7A": round(float(snap_q_src[2, 1]), 3),
            "snap_bl_x_005C7A": round(float(snap_q_src[3, 0]), 3),
            "snap_bl_y_005C7A": round(float(snap_q_src[3, 1]), 3),

            "H_img_to_table_005C7A": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C7A": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),
        })

        rows.append(out)

        if args.make_images:
            title = f"{review_id} seg={seg} {status} conf={conf:.2f}"
            ov = make_overlay(proc, prob_img, mask, old_q, snap_q_proc, title)
            warp = canonical_warp(proc, snap_q_proc)
            combo = hconcat_resized(ov, warp, right_w=360)

            fname = f"{i:03d}_{review_id}_seg{seg}_005C7A_snap.jpg"
            out_img = out_dir / "images" / fname
            imwrite_unicode(out_img, combo)

            cards.append({
                "review_id": review_id,
                "seg": seg,
                "status": status,
                "reason": reason,
                "conf": round(float(conf), 4),
                "area_ratio": "" if cand is None else round(float(cand["area_ratio"]), 4),
                "poly_support": "" if fit is None else round(float(fit["poly_support"]), 4),
                "mask_in_poly": "" if fit is None else round(float(fit["mask_in_poly"]), 4),
                "old_status": str(obj.get("table_object_status_005C6B", "")),
                "image": str(out_img),
            })

    out_df = pd.DataFrame(rows)

    status_order = {
        "BAD_TABLE_MASK": 0,
        "SNAP_PARTIAL_TABLE": 1,
        "SNAP_METRIC_CANDIDATE": 2,
        "SNAP_METRIC_TABLE_OK": 3,
    }

    out_df["_status_order"] = out_df["snap_status_005C7A"].astype(str).map(status_order).fillna(9)
    out_df = out_df.sort_values(
        ["_status_order", "snap_confidence_005C7A", "review_id", "camera_segment_id_005B2"],
        ascending=[True, True, True, True],
    ).drop(columns=["_status_order"]).copy()

    out_csv = out_dir / "table_object_snap_geometry_005C7A.csv"
    out_json = out_dir / "table_snap_geometry_summary_005C7A.json"
    out_html = out_dir / "table_snap_geometry_report_005C7A.html"

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    if args.make_images:
        cards = sorted(
            cards,
            key=lambda c: (
                status_order.get(str(c["status"]), 9),
                float(c["conf"]),
                str(c["review_id"]),
                int(c["seg"]),
            ),
        )[:args.max_images]

        out_html.write_text(build_html(cards), encoding="utf-8")

    counts = out_df["snap_status_005C7A"].astype(str).value_counts().to_dict()
    old_counts = out_df["table_object_status_005C6B"].astype(str).value_counts().to_dict() if "table_object_status_005C6B" in out_df.columns else {}

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "snap_regular_table_quadrilateral_from_learned_table_mask_component",
        "classification": str(classification_path),
        "model": str(model_path),
        "params": {
            "mask_max_w": args.mask_max_w,
            "prob_threshold": args.prob_threshold,
            "min_metric_conf": args.min_metric_conf,
        },
        "objects_total": int(len(out_df)),
        "old_status_counts_005C6B": old_counts,
        "snap_status_counts_005C7A": counts,
        "outputs": {
            "csv": str(out_csv),
            "html": str(out_html) if args.make_images else "",
        },
        "next": "Review snapped quads. Promote SNAP_METRIC_TABLE_OK to metric table objects; keep SNAP_PARTIAL_TABLE as local table masks only."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C7A status=OK")
    print("objects_total=", summary["objects_total"])
    print("old_status_counts_005C6B=", json.dumps(old_counts, ensure_ascii=False))
    print("snap_status_counts_005C7A=", json.dumps(counts, ensure_ascii=False))
    print("wrote", out_csv)
    if args.make_images:
        print("wrote", out_html)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
