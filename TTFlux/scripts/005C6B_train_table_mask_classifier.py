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


VERSION = "005C6B"


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


def feature_points(img_bgr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    xs = np.clip(xs.astype(int), 0, w - 1)
    ys = np.clip(ys.astype(int), 0, h - 1)

    bgr = img_bgr[ys, xs].astype(np.float32) / 255.0
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    hsv = hsv_img[ys, xs].astype(np.float32)
    lab = lab_img[ys, xs].astype(np.float32)

    hsv[:, 0] /= 180.0
    hsv[:, 1] /= 255.0
    hsv[:, 2] /= 255.0
    lab /= 255.0

    xn = xs.astype(np.float32) / max(1, w - 1)
    yn = ys.astype(np.float32) / max(1, h - 1)

    return np.column_stack([
        bgr[:, 0],
        bgr[:, 1],
        bgr[:, 2],
        hsv[:, 0],
        hsv[:, 1],
        hsv[:, 2],
        lab[:, 0],
        lab[:, 1],
        lab[:, 2],
        xn,
        yn,
    ])


def sample_circle_pixels(x: float, y: float, r: float, w: int, h: int, max_samples: int):
    x0 = max(0, int(math.floor(x - r)))
    x1 = min(w - 1, int(math.ceil(x + r)))
    y0 = max(0, int(math.floor(y - r)))
    y1 = min(h - 1, int(math.ceil(y + r)))

    xs = []
    ys = []
    rr = max(1.0, float(r))

    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            if (xx - x) ** 2 + (yy - y) ** 2 <= rr ** 2:
                xs.append(xx)
                ys.append(yy)

    if not xs:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)

    if len(xs) > max_samples:
        idx = np.linspace(0, len(xs) - 1, max_samples).astype(int)
        xs = xs[idx]
        ys = ys[idx]

    return xs, ys


def collect_training_samples(seeds: pd.DataFrame, max_samples_per_seed: int):
    Xs = []
    ys = []
    rows = []

    root = Path.cwd()
    cache = {}

    seeds = seeds.copy()
    seeds = seeds[seeds["label"].astype(str).isin(["table", "non_table"])].copy()

    for i, r in seeds.iterrows():
        label = str(r["label"])
        y_val = 1 if label == "table" else 0

        clip_path = Path(str(r["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([r["frame"]])).fillna(0).iloc[0])
        key = (str(clip_path), frame_idx)

        if key not in cache:
            cache[key] = read_frame(clip_path, frame_idx)

        frame = cache[key]
        if frame is None:
            continue

        h, w = frame.shape[:2]

        x = float(r["x_src"])
        y = float(r["y_src"])
        rad = float(r.get("radius_src", 9))

        xs, yy = sample_circle_pixels(
            x=x,
            y=y,
            r=rad,
            w=w,
            h=h,
            max_samples=max_samples_per_seed,
        )

        if len(xs) == 0:
            continue

        feats = feature_points(frame, xs, yy)

        Xs.append(feats)
        ys.append(np.full((len(feats),), y_val, dtype=np.int32))

        rows.append({
            "review_id": str(r["review_id"]),
            "camera_segment_id": int(to_num(pd.Series([r["camera_segment_id"]])).fillna(1).iloc[0]),
            "frame": frame_idx,
            "label": label,
            "pixels_sampled": int(len(feats)),
        })

    if not Xs:
        raise SystemExit("Aucun sample table/non_table trouvé. Vérifie table_mask_seed_points_005C6A.csv")

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int32)

    return X, y, pd.DataFrame(rows)


def balance_samples(X, y, max_per_class: int):
    rng = np.random.default_rng(42)
    keep = []

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(idx.tolist())

    keep = np.asarray(sorted(keep), dtype=np.int64)
    return X[keep], y[keep]


def train_classifier(X, y):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise SystemExit(f"scikit-learn indisponible: {e}")

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )

    metrics = {}

    if len(np.unique(y)) < 2:
        raise SystemExit("Il faut au moins des points table ET non_table.")

    if len(y) >= 2000:
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.22,
            random_state=42,
            stratify=y,
        )

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        metrics = {
            "train_samples": int(len(Xtr)),
            "test_samples": int(len(Xte)),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        }

        # Refit sur tout.
        model.fit(X, y)
    else:
        model.fit(X, y)
        pred = model.predict(X)
        metrics = {
            "train_samples": int(len(X)),
            "test_samples": 0,
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        }

    return model, metrics


def quad_mask_from_obj(obj: dict, scale: float, shape_hw):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        quad = np.asarray([
            [float(obj["quad_tl_x_005C3"]), float(obj["quad_tl_y_005C3"])],
            [float(obj["quad_tr_x_005C3"]), float(obj["quad_tr_y_005C3"])],
            [float(obj["quad_br_x_005C3"]), float(obj["quad_br_y_005C3"])],
            [float(obj["quad_bl_x_005C3"]), float(obj["quad_bl_y_005C3"])],
        ], dtype=np.float32)

        quad *= float(scale)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    except Exception:
        pass

    return mask


def largest_component(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, None

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)

    return cnt, area, (x, y, w, h)


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


def classify_object(
    obj: dict,
    prob_img,
    mask,
    quad_mask,
    audit_row: dict | None,
    min_mask_area_ratio: float,
    metric_inside_min: float,
    metric_points_min: int,
):
    h, w = mask.shape[:2]
    total = max(1, h * w)

    table_pixels = int((mask > 0).sum())
    mask_area_ratio = table_pixels / total

    cnt, comp_area, bbox = largest_component(mask)
    comp_area_ratio = comp_area / total

    quad_pixels = int((quad_mask > 0).sum())
    mask_in_quad = int(((mask > 0) & (quad_mask > 0)).sum())

    quad_table_ratio = mask_in_quad / max(1, quad_pixels)
    mask_in_quad_ratio = mask_in_quad / max(1, table_pixels)

    if quad_pixels > 0:
        quad_prob_mean = float(prob_img[quad_mask > 0].mean())
    else:
        quad_prob_mean = 0.0

    inside_ratio = -1.0
    points_projected = 0

    if audit_row is not None:
        inside_ratio = float(audit_row.get("inside_ratio", -1.0))
        points_projected = int(float(audit_row.get("points_projected", 0)))

    # Heuristique volontairement prudente.
    if mask_area_ratio < min_mask_area_ratio or comp_area_ratio < min_mask_area_ratio * 0.60:
        status = "BAD_TABLE_OBJECT"
        reason = "table_mask_too_small"
    elif (
        points_projected >= metric_points_min
        and inside_ratio >= metric_inside_min
        and quad_table_ratio >= 0.32
        and quad_prob_mean >= 0.50
    ):
        status = "METRIC_TABLE_OK"
        reason = "inside_ratio_and_mask_support_quad"
    elif (
        quad_table_ratio >= 0.18
        or mask_in_quad_ratio >= 0.20
        or comp_area_ratio >= min_mask_area_ratio * 1.8
    ):
        status = "PARTIAL_TABLE"
        reason = "table_visible_but_metric_uncertain"
    else:
        status = "BAD_TABLE_OBJECT"
        reason = "weak_quad_mask_support"

    confidence = (
        0.30 * min(1.0, mask_area_ratio / 0.12)
        + 0.25 * min(1.0, comp_area_ratio / 0.08)
        + 0.25 * min(1.0, quad_table_ratio / 0.55)
        + 0.20 * max(0.0, min(1.0, inside_ratio if inside_ratio >= 0 else 0.0))
    )

    return {
        "table_mask_area_ratio_005C6B": round(float(mask_area_ratio), 6),
        "table_component_area_ratio_005C6B": round(float(comp_area_ratio), 6),
        "quad_table_ratio_005C6B": round(float(quad_table_ratio), 6),
        "mask_in_quad_ratio_005C6B": round(float(mask_in_quad_ratio), 6),
        "quad_prob_mean_005C6B": round(float(quad_prob_mean), 6),
        "audit_inside_ratio": round(float(inside_ratio), 6),
        "audit_points_projected": int(points_projected),
        "table_object_status_005C6B": status,
        "table_object_reason_005C6B": reason,
        "table_object_confidence_005C6B": round(float(confidence), 6),
        "bbox_x_005C6B": "" if bbox is None else int(bbox[0]),
        "bbox_y_005C6B": "" if bbox is None else int(bbox[1]),
        "bbox_w_005C6B": "" if bbox is None else int(bbox[2]),
        "bbox_h_005C6B": "" if bbox is None else int(bbox[3]),
    }


def overlay_mask(img, prob_img, mask, quad_mask, status_text):
    base = img.copy()

    heat = np.clip(prob_img * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)

    overlay = cv2.addWeighted(base, 0.62, heat_color, 0.38, 0)

    # Contour masque table.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # Quad actuel.
    cnts_q, _ = cv2.findContours(quad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts_q, -1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        overlay,
        status_text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def build_html(items):
    cards = []

    for r in items:
        cards.append(f"""
<section class="card {html.escape(str(r["status"]))}">
  <h2>{html.escape(str(r["review_id"]))} · seg {r["camera_segment_id"]} · {html.escape(str(r["status"]))}</h2>
  <p>
    reason={html.escape(str(r["reason"]))}
    · conf={r["confidence"]}
    · mask_area={r["mask_area"]}
    · quad_ratio={r["quad_ratio"]}
    · inside={r["inside_ratio"]}
  </p>
  <img src="images/{html.escape(Path(r["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C6B table mask classifier</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
.card.METRIC_TABLE_OK{{border-color:#2d8a4d}}
.card.PARTIAL_TABLE{{border-color:#9b842e}}
.card.BAD_TABLE_OBJECT{{border-color:#9b3939}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005C6B · table/non-table mask classifier</h1>
<p>Vert = masque table appris. Jaune = quad/homographie actuelle. Heatmap = probabilité table.</p>
</header>
{''.join(cards)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="runs/rally_table_mask_seed_005C6A/table_mask_seed_points_005C6A.csv")
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C5C2_merged/table_objects_merged_005C5C2.csv")
    ap.add_argument("--audit-csv", default="runs/rally_table_object_005C5C2_merged_audit/table_object_canonical_audit_005C4.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_mask_classifier_005C6B")

    ap.add_argument("--max-samples-per-seed", type=int, default=420)
    ap.add_argument("--max-train-per-class", type=int, default=70000)
    ap.add_argument("--mask-max-w", type=int, default=640)
    ap.add_argument("--prob-threshold", type=float, default=0.55)
    ap.add_argument("--chunk-size", type=int, default=250000)

    ap.add_argument("--min-mask-area-ratio", type=float, default=0.012)
    ap.add_argument("--metric-inside-min", type=float, default=0.75)
    ap.add_argument("--metric-points-min", type=int, default=20)

    ap.add_argument("--make-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=80)

    args = ap.parse_args()

    root = Path.cwd()

    seeds_path = Path(args.seeds)
    table_path = Path(args.table_objects)
    audit_path = Path(args.audit_csv)
    out_dir = Path(args.out_dir)

    if not seeds_path.is_absolute():
        seeds_path = root / seeds_path
    if not table_path.is_absolute():
        table_path = root / table_path
    if not audit_path.is_absolute():
        audit_path = root / audit_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = pd.read_csv(seeds_path).fillna("")
    table = pd.read_csv(table_path).fillna("")
    audit = pd.read_csv(audit_path).fillna("") if audit_path.is_file() else pd.DataFrame()

    labels = seeds["label"].astype(str).value_counts().to_dict()

    usable = seeds[seeds["label"].astype(str).isin(["table", "non_table"])].copy()

    print("005C6B collect training samples...")
    print("seed labels=", json.dumps(labels, ensure_ascii=False))

    X, y, sample_summary = collect_training_samples(
        usable,
        max_samples_per_seed=args.max_samples_per_seed,
    )

    Xb, yb = balance_samples(X, y, max_per_class=args.max_train_per_class)

    print("samples_raw=", len(y))
    print("samples_balanced=", len(yb))
    print("class_counts=", json.dumps({str(k): int(v) for k, v in pd.Series(yb).value_counts().to_dict().items()}))

    model, model_metrics = train_classifier(Xb, yb)

    model_path = out_dir / "table_mask_rf_model_005C6B.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    sample_summary_path = out_dir / "table_mask_training_samples_005C6B.csv"
    sample_summary.to_csv(sample_summary_path, index=False, encoding="utf-8")

    audit_map = {}
    if not audit.empty:
        audit = audit.copy()
        audit["camera_segment_id_005B2"] = to_num(audit["camera_segment_id"]).fillna(1).astype(int)

        for _, r in audit.iterrows():
            key = (str(r["review_id"]), int(r["camera_segment_id_005B2"]))
            audit_map[key] = r.to_dict()

    rows = []
    html_items = []

    for i, (_, r) in enumerate(table.iterrows(), start=1):
        obj = r.to_dict()
        review_id = str(obj["review_id"])
        seg = int(to_num(pd.Series([obj["camera_segment_id_005B2"]])).fillna(1).iloc[0])

        clip_path = Path(str(obj["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([obj.get("best_frame_005C1", obj.get("first_frame", 0))])).fillna(0).iloc[0])

        frame = read_frame(clip_path, frame_idx)
        if frame is None:
            out = dict(obj)
            out.update({
                "table_object_status_005C6B": "BAD_TABLE_OBJECT",
                "table_object_reason_005C6B": "missing_frame",
            })
            rows.append(out)
            continue

        proc, scale = resize_max_w(frame, args.mask_max_w)

        prob_img, mask = predict_table_mask(
            model,
            proc,
            threshold=args.prob_threshold,
            chunk_size=args.chunk_size,
        )

        quad_mask = quad_mask_from_obj(obj, scale=scale, shape_hw=mask.shape[:2])
        audit_row = audit_map.get((review_id, seg))

        cls = classify_object(
            obj=obj,
            prob_img=prob_img,
            mask=mask,
            quad_mask=quad_mask,
            audit_row=audit_row,
            min_mask_area_ratio=args.min_mask_area_ratio,
            metric_inside_min=args.metric_inside_min,
            metric_points_min=args.metric_points_min,
        )

        out = dict(obj)
        out.update(cls)
        out["mask_frame_005C6B"] = frame_idx
        out["mask_scale_005C6B"] = round(float(scale), 6)
        rows.append(out)

        if args.make_images:
            status = cls["table_object_status_005C6B"]
            status_text = f"{review_id} seg={seg} {status} conf={cls['table_object_confidence_005C6B']:.2f}"

            ov = overlay_mask(proc, prob_img, mask, quad_mask, status_text=status_text)

            fname = f"{i:03d}_{review_id}_seg{seg}_005C6B_mask.jpg"
            out_img = out_dir / "images" / fname
            imwrite_unicode(out_img, ov)

            html_items.append({
                "review_id": review_id,
                "camera_segment_id": seg,
                "status": status,
                "reason": cls["table_object_reason_005C6B"],
                "confidence": cls["table_object_confidence_005C6B"],
                "mask_area": cls["table_mask_area_ratio_005C6B"],
                "quad_ratio": cls["quad_table_ratio_005C6B"],
                "inside_ratio": cls["audit_inside_ratio"],
                "image": str(out_img),
            })

    classified = pd.DataFrame(rows)

    # Priorité affichage : BAD/PARTIAL d’abord, puis METRIC.
    status_order = {"BAD_TABLE_OBJECT": 0, "PARTIAL_TABLE": 1, "METRIC_TABLE_OK": 2}
    classified["_status_order"] = classified["table_object_status_005C6B"].astype(str).map(status_order).fillna(9)
    classified = classified.sort_values(
        ["_status_order", "table_object_confidence_005C6B", "review_id", "camera_segment_id_005B2"],
        ascending=[True, True, True, True],
    ).drop(columns=["_status_order"]).copy()

    # Re-écrire les images dans l’ordre prioritaire si demandé.
    if args.make_images:
        html_items = sorted(
            html_items,
            key=lambda x: (
                status_order.get(str(x["status"]), 9),
                float(x["confidence"]),
                str(x["review_id"]),
                int(x["camera_segment_id"]),
            ),
        )[:args.max_images]

    out_csv = out_dir / "table_object_classification_005C6B.csv"
    out_summary = out_dir / "table_mask_classifier_summary_005C6B.json"
    out_html = out_dir / "table_mask_classifier_report_005C6B.html"

    classified.to_csv(out_csv, index=False, encoding="utf-8")

    if args.make_images:
        out_html.write_text(build_html(html_items), encoding="utf-8")

    counts = classified["table_object_status_005C6B"].astype(str).value_counts().to_dict()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "train_table_vs_non_table_pixel_classifier_from_human_seed_points_ignore_edge_label",
        "seeds": str(seeds_path),
        "table_objects": str(table_path),
        "audit_csv": str(audit_path),
        "labels_raw": labels,
        "labels_used": usable["label"].astype(str).value_counts().to_dict(),
        "samples_raw": int(len(y)),
        "samples_balanced": int(len(yb)),
        "class_counts_balanced": {str(k): int(v) for k, v in pd.Series(yb).value_counts().to_dict().items()},
        "model_metrics": model_metrics,
        "params": {
            "max_samples_per_seed": args.max_samples_per_seed,
            "max_train_per_class": args.max_train_per_class,
            "mask_max_w": args.mask_max_w,
            "prob_threshold": args.prob_threshold,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "metric_inside_min": args.metric_inside_min,
            "metric_points_min": args.metric_points_min,
        },
        "objects_total": int(len(classified)),
        "status_counts": counts,
        "outputs": {
            "model": str(model_path),
            "training_samples": str(sample_summary_path),
            "classification": str(out_csv),
            "html": str(out_html) if args.make_images else "",
        },
        "next": "Review BAD/PARTIAL/METRIC report. Then use METRIC_TABLE_OK only for metric projection; PARTIAL_TABLE only for local table mask."
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C6B status=OK")
    print("labels_raw=", json.dumps(labels, ensure_ascii=False))
    print("samples_raw=", summary["samples_raw"])
    print("samples_balanced=", summary["samples_balanced"])
    print("model_metrics=", json.dumps(model_metrics, ensure_ascii=False))
    print("objects_total=", summary["objects_total"])
    print("status_counts=", json.dumps(counts, ensure_ascii=False))
    print("wrote", model_path)
    print("wrote", sample_summary_path)
    print("wrote", out_csv)
    if args.make_images:
        print("wrote", out_html)
    print("wrote", out_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
