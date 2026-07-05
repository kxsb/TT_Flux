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


VERSION = "005C9A"

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


def table_corners_m():
    return np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)


def table_reference_lines_m():
    lines = []

    # Contour officiel.
    corners = [
        (-TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W,  TABLE_HALF_L),
    ]

    for a, b in zip(corners, corners[1:] + corners[:1]):
        lines.append(("outer", [a, b]))

    # Filet.
    lines.append(("net", [(-TABLE_HALF_W, 0.0), (TABLE_HALF_W, 0.0)]))

    # Ligne centrale doubles.
    lines.append(("center", [(0.0, -TABLE_HALF_L), (0.0, TABLE_HALF_L)]))

    # Grille légère interne, moins pondérée.
    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        lines.append(("grid", [(float(x), -TABLE_HALF_L), (float(x), TABLE_HALF_L)]))

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        lines.append(("grid", [(-TABLE_HALF_W, float(y)), (TABLE_HALF_W, float(y))]))

    return lines


def snap_quad_src(row) -> np.ndarray:
    return np.asarray([
        [float(row["snap_tl_x_005C7A"]), float(row["snap_tl_y_005C7A"])],
        [float(row["snap_tr_x_005C7A"]), float(row["snap_tr_y_005C7A"])],
        [float(row["snap_br_x_005C7A"]), float(row["snap_br_y_005C7A"])],
        [float(row["snap_bl_x_005C7A"]), float(row["snap_bl_y_005C7A"])],
    ], dtype=np.float32)


def old_quad_src(row) -> np.ndarray:
    return np.asarray([
        [float(row["quad_tl_x_005C3"]), float(row["quad_tl_y_005C3"])],
        [float(row["quad_tr_x_005C3"]), float(row["quad_tr_y_005C3"])],
        [float(row["quad_br_x_005C3"]), float(row["quad_br_y_005C3"])],
        [float(row["quad_bl_x_005C3"]), float(row["quad_bl_y_005C3"])],
    ], dtype=np.float32)


def quad_area(q: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2))))


def is_convex_quad(q: np.ndarray) -> bool:
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    return bool(cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)))


def homography_table_to_img(q_img: np.ndarray):
    return cv2.getPerspectiveTransform(table_corners_m(), np.asarray(q_img, dtype=np.float32))


def homography_img_to_table(q_img: np.ndarray):
    return cv2.getPerspectiveTransform(np.asarray(q_img, dtype=np.float32), table_corners_m())


def project_points(H, pts_m):
    pts = np.asarray(pts_m, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def polygon_mask(q: np.ndarray, shape_hw):
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 255)
    except Exception:
        pass
    return m


def line_support_maps(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    white = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 135)).astype(np.uint8)
    edge = (cv2.Canny(gray, 55, 145) > 0).astype(np.uint8)

    # léger épaississement pour tolérer les décalages.
    k = np.ones((3, 3), np.uint8)
    white = cv2.dilate(white, k, iterations=1)
    edge = cv2.dilate(edge, k, iterations=1)

    return white, edge


def draw_reference_lines_mask(q: np.ndarray, shape_hw, thickness_outer=5, thickness_inner=4):
    h, w = shape_hw
    line_mask = np.zeros((h, w), dtype=np.uint8)
    weighted_mask = np.zeros((h, w), dtype=np.float32)

    try:
        H = homography_table_to_img(q)
    except Exception:
        return line_mask, weighted_mask

    for name, line in table_reference_lines_m():
        pp = project_points(H, line)
        p0 = (int(round(pp[0, 0])), int(round(pp[0, 1])))
        p1 = (int(round(pp[1, 0])), int(round(pp[1, 1])))

        if name == "outer":
            weight = 1.00
            thick = thickness_outer
        elif name == "net":
            weight = 0.85
            thick = thickness_inner
        elif name == "center":
            weight = 0.55
            thick = max(2, thickness_inner - 1)
        else:
            weight = 0.25
            thick = 2

        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.line(tmp, p0, p1, 255, thick, cv2.LINE_AA)

        line_mask[tmp > 0] = 255
        weighted_mask[tmp > 0] = np.maximum(weighted_mask[tmp > 0], weight)

    return line_mask, weighted_mask


def score_quad(frame_bgr, prob_img, table_mask, white_map, edge_map, q, initial_area):
    h, w = table_mask.shape[:2]
    total = max(1, h * w)

    q = np.asarray(q, dtype=np.float32).reshape(4, 2)

    if not is_convex_quad(q):
        return -999.0, {}

    area = quad_area(q)
    if area <= total * 0.003:
        return -999.0, {}

    if area >= total * 0.78:
        return -999.0, {}

    # coins trop loin hors image => rejet doux.
    margin = max(w, h) * 0.22
    if (
        (q[:, 0] < -margin).any()
        or (q[:, 0] > w + margin).any()
        or (q[:, 1] < -margin).any()
        or (q[:, 1] > h + margin).any()
    ):
        return -999.0, {}

    pm = polygon_mask(q, table_mask.shape[:2])
    p = pm > 0
    m = table_mask > 0

    if p.sum() <= 10:
        return -999.0, {}

    inter = float(np.logical_and(p, m).sum())
    poly_area = float(p.sum())
    mask_area = float(max(1, m.sum()))

    inside_prob = float(prob_img[p].mean()) if p.any() else 0.0
    poly_support = inter / max(1.0, poly_area)
    mask_coverage = inter / max(1.0, mask_area)

    outside_mask = float(np.logical_and(~p, m).sum()) / max(1.0, mask_area)

    line_mask, weighted_line_mask = draw_reference_lines_mask(q, table_mask.shape[:2])
    l = line_mask > 0

    if l.any():
        white_support = float((white_map[l] * weighted_line_mask[l]).sum() / max(1e-6, weighted_line_mask[l].sum()))
        edge_support = float((edge_map[l] * weighted_line_mask[l]).sum() / max(1e-6, weighted_line_mask[l].sum()))
        line_score = 0.58 * white_support + 0.42 * edge_support
    else:
        white_support = 0.0
        edge_support = 0.0
        line_score = 0.0

    # éviter que l’optimiseur fasse grossir absurdement le quad.
    area_ratio_to_init = area / max(1.0, initial_area)
    area_penalty = abs(math.log(max(1e-6, area_ratio_to_init))) * 0.055

    # ratio géométrique image très extrême = suspect, mais pas bloquant.
    x, y, bw, bh = cv2.boundingRect(q.reshape(-1, 1, 2).astype(np.float32))
    bbox_aspect = bw / max(1, bh)
    aspect_penalty = 0.0
    if bbox_aspect < 0.55:
        aspect_penalty = (0.55 - bbox_aspect) * 0.10
    elif bbox_aspect > 7.5:
        aspect_penalty = (bbox_aspect - 7.5) * 0.025

    score = (
        0.34 * inside_prob
        + 0.23 * poly_support
        + 0.17 * mask_coverage
        + 0.22 * line_score
        - 0.10 * outside_mask
        - area_penalty
        - aspect_penalty
    )

    parts = {
        "score": round(float(score), 6),
        "inside_prob": round(float(inside_prob), 6),
        "poly_support": round(float(poly_support), 6),
        "mask_coverage": round(float(mask_coverage), 6),
        "outside_mask": round(float(outside_mask), 6),
        "white_support": round(float(white_support), 6),
        "edge_support": round(float(edge_support), 6),
        "line_score": round(float(line_score), 6),
        "area_ratio_to_init": round(float(area_ratio_to_init), 6),
        "bbox_aspect": round(float(bbox_aspect), 6),
    }

    return float(score), parts


def perturb_candidates(q0, rng, n_random=450):
    q0 = np.asarray(q0, dtype=np.float32).reshape(4, 2)
    c = q0.mean(axis=0)

    candidates = []

    def add(q):
        candidates.append(np.asarray(q, dtype=np.float32).reshape(4, 2))

    add(q0)

    # Scale autour du centre.
    for sx in [0.92, 0.96, 1.00, 1.04, 1.08, 1.12]:
        for sy in [0.92, 0.96, 1.00, 1.04, 1.08]:
            add(c + (q0 - c) * np.asarray([sx, sy], dtype=np.float32))

    # Petits shifts globaux.
    for dx in [-18, -9, 0, 9, 18]:
        for dy in [-18, -9, 0, 9, 18]:
            add(q0 + np.asarray([dx, dy], dtype=np.float32))

    # Edge/corner jitter progressif.
    for sigma in [4, 8, 14, 22, 32]:
        for _ in range(max(15, n_random // 5)):
            noise = rng.normal(0, sigma, size=(4, 2)).astype(np.float32)

            # 40% du temps, bruit plus cohérent par bord.
            if rng.random() < 0.40:
                edge_noise = rng.normal(0, sigma, size=(4, 2)).astype(np.float32)
                noise[0] = 0.5 * (edge_noise[0] + edge_noise[3])
                noise[1] = 0.5 * (edge_noise[1] + edge_noise[2])
                noise[2] = 0.5 * (edge_noise[1] + edge_noise[2])
                noise[3] = 0.5 * (edge_noise[0] + edge_noise[3])

            add(q0 + noise)

    # Expansion/contraction différenciée vers chaque coin.
    for fac in [0.88, 0.94, 1.00, 1.06, 1.14, 1.22]:
        add(c + (q0 - c) * fac)

    # Dédoublonnage grossier.
    uniq = []
    seen = set()

    for q in candidates:
        key = tuple(np.round(q.reshape(-1) / 2.0).astype(int).tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)

    return uniq


def optimize_quad(frame_bgr, prob_img, mask, q0, n_random, seed=42):
    white_map, edge_map = line_support_maps(frame_bgr)

    rng = np.random.default_rng(seed)
    initial_area = max(1.0, quad_area(q0))

    best_q = q0.copy()
    best_score, best_parts = score_quad(frame_bgr, prob_img, mask, white_map, edge_map, best_q, initial_area)

    candidates = perturb_candidates(q0, rng, n_random=n_random)

    for q in candidates:
        score, parts = score_quad(frame_bgr, prob_img, mask, white_map, edge_map, q, initial_area)
        if score > best_score:
            best_score = score
            best_q = q.copy()
            best_parts = parts

    # deuxième passe locale autour du best.
    candidates2 = perturb_candidates(best_q, rng, n_random=max(120, n_random // 2))

    for q in candidates2:
        score, parts = score_quad(frame_bgr, prob_img, mask, white_map, edge_map, q, initial_area)
        if score > best_score:
            best_score = score
            best_q = q.copy()
            best_parts = parts

    return best_q, best_score, best_parts


def draw_reference(frame, q, color, label):
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)

    try:
        H = homography_table_to_img(q)
    except Exception:
        return

    def pt(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    # Grille officielle.
    for name, line in table_reference_lines_m():
        pp = project_points(H, line)

        if name == "outer":
            col = color
            thick = 3
        elif name == "net":
            col = (0, 180, 255)
            thick = 3
        elif name == "center":
            col = (255, 255, 0)
            thick = 2
        else:
            col = (80, 180, 180)
            thick = 1

        cv2.line(frame, pt(pp[0]), pt(pp[1]), col, thick, cv2.LINE_AA)

    for p in q:
        cv2.circle(frame, pt(p), 6, color, -1, cv2.LINE_AA)

    cv2.putText(frame, label, pt(q[0] + np.asarray([8, 22])), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def canonical_warp(frame, q, out_w=520, out_h=936):
    dst = np.asarray([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    try:
        H = cv2.getPerspectiveTransform(np.asarray(q, dtype=np.float32), dst)
        warped = cv2.warpPerspective(frame, H, (out_w, out_h))
    except Exception:
        warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cv2.rectangle(warped, (0, 0), (out_w - 1, out_h - 1), (0, 255, 255), 3)
    cv2.line(warped, (0, out_h // 2), (out_w - 1, out_h // 2), (0, 180, 255), 2)
    cv2.line(warped, (out_w // 2, 0), (out_w // 2, out_h - 1), (255, 255, 0), 2)

    return warped


def hconcat_report(left, right, right_w=380):
    h2, w2 = right.shape[:2]
    scale = right_w / max(1, w2)
    right = cv2.resize(right, (right_w, int(round(h2 * scale))), interpolation=cv2.INTER_AREA)

    h = max(left.shape[0], right.shape[0])

    def pad(img):
        dh = h - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    return cv2.hconcat([pad(left), pad(right)])


def build_html(cards):
    parts = []

    for c in cards:
        parts.append(f"""
<section class="card">
  <h2>{html.escape(c["review_id"])} · seg {c["seg"]} · rating {c["rating"]}/10</h2>
  <p>
    init={c["init_score"]} · opt={c["opt_score"]} · gain={c["gain"]}
    · line={c["line_score"]} · poly={c["poly_support"]} · maskcov={c["mask_coverage"]}
  </p>
  <img src="images/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C9A reference table optimizer</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005C9A · optimisation table de référence projetée</h1>
<p>Jaune = ancien quad · magenta = snap initial · vert = projection optimisée du modèle-table officiel.</p>
</header>
{''.join(parts)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="runs/rally_table_projection_vote_summary_005C8B/table_projection_votes_accepted_005C8B.csv")
    ap.add_argument("--model", default="runs/rally_table_mask_classifier_005C6B/table_mask_rf_model_005C6B.pkl")
    ap.add_argument("--out-dir", default="runs/rally_table_reference_fit_005C9A")
    ap.add_argument("--mask-max-w", type=int, default=640)
    ap.add_argument("--prob-threshold", type=float, default=0.55)
    ap.add_argument("--chunk-size", type=int, default=250000)
    ap.add_argument("--n-random", type=int, default=420)
    ap.add_argument("--make-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=80)
    args = ap.parse_args()

    root = Path.cwd()

    accepted_path = Path(args.accepted)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    if not accepted_path.is_absolute():
        accepted_path = root / accepted_path
    if not model_path.is_absolute():
        model_path = root / model_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    accepted = pd.read_csv(accepted_path).fillna("")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    rows = []
    cards = []

    for i, (_, r) in enumerate(accepted.iterrows(), start=1):
        row = r.to_dict()

        review_id = str(row["review_id"])
        seg = int(to_num(pd.Series([row["camera_segment_id"]])).fillna(1).iloc[0])
        rating = int(to_num(pd.Series([row.get("rating_1_10", 0)])).fillna(0).iloc[0])

        clip_path = Path(str(row["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([
            row.get("metric_frame_005C7B", row.get("mask_frame_005C6B", row.get("frame", 0)))
        ])).fillna(0).iloc[0])

        frame = read_frame(clip_path, frame_idx)

        if frame is None:
            out = dict(row)
            out["fit_status_005C9A"] = "missing_frame"
            rows.append(out)
            continue

        proc, scale = resize_max_w(frame, args.mask_max_w)

        prob_img, table_mask = predict_table_mask(
            model,
            proc,
            threshold=args.prob_threshold,
            chunk_size=args.chunk_size,
        )

        q_init_src = snap_quad_src(row)
        q_init = q_init_src * scale

        white_map, edge_map = line_support_maps(proc)
        init_score, init_parts = score_quad(
            proc,
            prob_img,
            table_mask,
            white_map,
            edge_map,
            q_init,
            initial_area=max(1.0, quad_area(q_init)),
        )

        q_opt, opt_score, opt_parts = optimize_quad(
            frame_bgr=proc,
            prob_img=prob_img,
            mask=table_mask,
            q0=q_init,
            n_random=args.n_random,
            seed=42 + i,
        )

        q_opt_src = q_opt / max(1e-9, scale)

        try:
            H_img_to_table = homography_img_to_table(q_opt_src)
            H_table_to_img = homography_table_to_img(q_opt_src)
            h_ok = 1
        except Exception:
            H_img_to_table = np.eye(3, dtype=np.float64)
            H_table_to_img = np.eye(3, dtype=np.float64)
            h_ok = 0

        gain = float(opt_score - init_score)

        out = dict(row)
        out.update({
            "fit_status_005C9A": "OK" if h_ok else "BAD_H",
            "fit_frame_005C9A": frame_idx,
            "fit_scale_005C9A": round(float(scale), 8),
            "init_score_005C9A": round(float(init_score), 6),
            "opt_score_005C9A": round(float(opt_score), 6),
            "opt_gain_005C9A": round(float(gain), 6),

            "opt_inside_prob_005C9A": opt_parts.get("inside_prob", ""),
            "opt_poly_support_005C9A": opt_parts.get("poly_support", ""),
            "opt_mask_coverage_005C9A": opt_parts.get("mask_coverage", ""),
            "opt_outside_mask_005C9A": opt_parts.get("outside_mask", ""),
            "opt_white_support_005C9A": opt_parts.get("white_support", ""),
            "opt_edge_support_005C9A": opt_parts.get("edge_support", ""),
            "opt_line_score_005C9A": opt_parts.get("line_score", ""),
            "opt_area_ratio_to_init_005C9A": opt_parts.get("area_ratio_to_init", ""),
            "opt_bbox_aspect_005C9A": opt_parts.get("bbox_aspect", ""),

            "opt_tl_x_005C9A": round(float(q_opt_src[0, 0]), 3),
            "opt_tl_y_005C9A": round(float(q_opt_src[0, 1]), 3),
            "opt_tr_x_005C9A": round(float(q_opt_src[1, 0]), 3),
            "opt_tr_y_005C9A": round(float(q_opt_src[1, 1]), 3),
            "opt_br_x_005C9A": round(float(q_opt_src[2, 0]), 3),
            "opt_br_y_005C9A": round(float(q_opt_src[2, 1]), 3),
            "opt_bl_x_005C9A": round(float(q_opt_src[3, 0]), 3),
            "opt_bl_y_005C9A": round(float(q_opt_src[3, 1]), 3),

            "H_img_to_table_005C9A": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C9A": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),
        })

        rows.append(out)

        if args.make_images:
            old_q = old_quad_src(row) * scale

            left = proc.copy()
            draw_reference(left, old_q, (0, 255, 255), "old")
            draw_reference(left, q_init, (255, 0, 255), "snap")
            draw_reference(left, q_opt, (0, 255, 0), "opt_ref")

            cv2.putText(
                left,
                f"{review_id} seg={seg} rating={rating}/10 init={init_score:.3f} opt={opt_score:.3f} gain={gain:.3f}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            warp = canonical_warp(proc, q_opt)
            report = hconcat_report(left, warp)

            fname = f"{i:03d}_{review_id}_seg{seg}_005C9A_ref_fit.jpg"
            out_img = out_dir / "images" / fname
            imwrite_unicode(out_img, report)

            cards.append({
                "review_id": review_id,
                "seg": seg,
                "rating": rating,
                "init_score": round(float(init_score), 4),
                "opt_score": round(float(opt_score), 4),
                "gain": round(float(gain), 4),
                "line_score": opt_parts.get("line_score", ""),
                "poly_support": opt_parts.get("poly_support", ""),
                "mask_coverage": opt_parts.get("mask_coverage", ""),
                "image": str(out_img),
            })

    out_df = pd.DataFrame(rows)

    out_csv = out_dir / "table_reference_fit_005C9A.csv"
    out_json = out_dir / "table_reference_fit_summary_005C9A.json"
    out_html = out_dir / "table_reference_fit_report_005C9A.html"

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    if args.make_images:
        cards = sorted(cards, key=lambda c: (c["rating"], c["gain"]), reverse=True)[:args.max_images]
        out_html.write_text(build_html(cards), encoding="utf-8")

    ok_count = int((out_df["fit_status_005C9A"].astype(str).eq("OK")).sum()) if len(out_df) else 0
    gain_med = float(to_num(out_df["opt_gain_005C9A"]).median()) if len(out_df) else 0.0
    gain_pos = int((to_num(out_df["opt_gain_005C9A"]).fillna(0) > 0).sum()) if len(out_df) else 0

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "optimize_official_reference_table_projection_against_learned_table_mask_and_line_support",
        "accepted": str(accepted_path),
        "model": str(model_path),
        "params": {
            "mask_max_w": args.mask_max_w,
            "prob_threshold": args.prob_threshold,
            "n_random": args.n_random,
        },
        "items": int(len(out_df)),
        "fit_ok": ok_count,
        "positive_gain": gain_pos,
        "gain_median": round(float(gain_med), 6),
        "outputs": {
            "csv": str(out_csv),
            "html": str(out_html) if args.make_images else "",
        },
        "next": "Review 005C9A report. If optimized green reference is consistently better than magenta snap, promote 005C9A as table object source."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C9A status=OK")
    print("items=", summary["items"])
    print("fit_ok=", summary["fit_ok"])
    print("positive_gain=", summary["positive_gain"])
    print("gain_median=", summary["gain_median"])
    print("wrote", out_csv)
    if args.make_images:
        print("wrote", out_html)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
