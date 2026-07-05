from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


POS_LABEL = "human_visible_positive"
NEG_LABEL = "auto_far_negative"
PARTIAL_LABEL = "auto_near_partial"


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols: list[str] = []
    seen = set()

    preferred = [
        "sample_id",
        "label",
        "review_id",
        "clip_id",
        "segment_name",
        "frame",
        "tubelet_path",
        "source_video",
        "source_center",
        "missing_interp_count",
        "score_002F",
        "pred_002F",
        "correct_002F",
        "y_true",
        "crop_path",
        "dist_offset0",
        "dist_best",
        "human_decision_002A",
        "classification_001Z",
    ]

    for c in preferred:
        cols.append(c)
        seen.add(c)

    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


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


def resolve_path(value: str, root: Path) -> Path | None:
    value = str(value or "").strip()
    if not value:
        return None

    p = Path(value)

    if p.is_absolute() and p.exists():
        return p

    p1 = root / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


class VideoCache:
    def __init__(self) -> None:
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.counts: dict[str, int] = {}

    def read(self, video_path: Path, frame: int) -> np.ndarray | None:
        key = str(video_path)

        cap = self.caps.get(key)
        if cap is None:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                return None
            self.caps[key] = cap
            self.counts[key] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        count = self.counts.get(key, 0)
        if count > 0:
            frame = max(0, min(int(frame), count - 1))
        else:
            frame = max(0, int(frame))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()

        if not ok or img is None:
            return None

        return img

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()


def crop_image(img: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2

    cx = int(round(x))
    cy = int(round(y))

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    if crop.shape[0] != size or crop.shape[1] != size:
        out = np.zeros((size, size, 3), dtype=np.uint8)
        ox = (size - crop.shape[1]) // 2
        oy = (size - crop.shape[0]) // 2
        out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        crop = out

    return crop


def load_human_points(merged_csv: Path) -> dict[str, list[dict[str, float]]]:
    rows = read_csv(merged_csv)
    out: dict[str, list[dict[str, float]]] = {}

    for r in rows:
        if r.get("quality", "") != "visible":
            continue

        if str(r.get("visible", "")).strip() not in ("1", "1.0", "true", "True"):
            continue

        rid = r.get("review_id", "")
        frame = fnum(r.get("frame"), None)
        x = fnum(r.get("x"), None)
        y = fnum(r.get("y"), None)

        if not rid or frame is None or x is None or y is None:
            continue

        out.setdefault(rid, []).append({"frame": frame, "x": x, "y": y})

    for rid in out:
        out[rid] = sorted(out[rid], key=lambda p: p["frame"])

    return out


def load_track_points(track_csv: Path) -> list[dict[str, float]]:
    rows = read_csv(track_csv)
    pts = []

    for r in rows:
        frame = fnum(r.get("frame") or r.get("frame_idx") or r.get("f"), None)
        x = fnum(r.get("x") or r.get("cx") or r.get("ball_x"), None)
        y = fnum(r.get("y") or r.get("cy") or r.get("ball_y"), None)

        if frame is None or x is None or y is None:
            continue

        pts.append({"frame": frame, "x": x, "y": y})

    return sorted(pts, key=lambda p: p["frame"])


def load_auto_points(final_csv: Path, root: Path) -> dict[str, list[dict[str, float]]]:
    rows = read_csv(final_csv)
    out: dict[str, list[dict[str, float]]] = {}

    for r in rows:
        rid = r.get("review_id", "")
        csv_path = r.get("csv") or r.get("csv_path") or ""

        if not rid or not csv_path:
            continue

        p = resolve_path(csv_path, root)
        if p is None:
            continue

        out[rid] = load_track_points(p)

    return out


def interp_point(points: list[dict[str, float]], frame: float) -> tuple[float, float] | None:
    if not points:
        return None

    if frame < points[0]["frame"] or frame > points[-1]["frame"]:
        return None

    for p in points:
        if abs(p["frame"] - frame) < 1e-6:
            return p["x"], p["y"]

    lo = None
    hi = None

    for p in points:
        if p["frame"] <= frame:
            lo = p
        if p["frame"] >= frame:
            hi = p
            break

    if lo is None or hi is None:
        return None

    if abs(hi["frame"] - lo["frame"]) < 1e-6:
        return lo["x"], lo["y"]

    t = (frame - lo["frame"]) / (hi["frame"] - lo["frame"])
    x = lo["x"] + t * (hi["x"] - lo["x"])
    y = lo["y"] + t * (hi["y"] - lo["y"])

    return x, y


def build_tubelet_image(
    video_cache: VideoCache,
    video_path: Path,
    points: list[dict[str, float]],
    base_frame: int,
    fallback_x: float,
    fallback_y: float,
    offsets: list[int],
    crop_size: int,
) -> tuple[np.ndarray | None, int]:
    tiles = []
    missing = 0

    for off in offsets:
        f = base_frame + off
        p = interp_point(points, f)

        if p is None:
            x, y = fallback_x, fallback_y
            missing += 1
        else:
            x, y = p

        img = video_cache.read(video_path, f)
        if img is None:
            return None, missing

        tiles.append(crop_image(img, x, y, crop_size))

    return np.concatenate(tiles, axis=1), missing


def build_tubelets(
    manifest_002D: Path,
    merged_csv: Path,
    final_csv: Path,
    out_dir: Path,
    crop_size: int,
    offsets: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path.cwd()
    rows = read_csv(manifest_002D)

    human_points = load_human_points(merged_csv)
    auto_points = load_auto_points(final_csv, root)

    tubelet_dir = out_dir / "tubelets"
    video_cache = VideoCache()

    out_rows: list[dict[str, Any]] = []
    skipped = 0

    try:
        for i, row in enumerate(rows):
            label = row.get("label", "")
            rid = row.get("review_id", "")

            if label not in (POS_LABEL, NEG_LABEL, PARTIAL_LABEL):
                continue

            frame_f = fnum(row.get("frame"), None)
            x = fnum(row.get("x"), None)
            y = fnum(row.get("y"), None)

            if not rid or frame_f is None or x is None or y is None:
                skipped += 1
                continue

            video_path = resolve_path(row.get("source_video", ""), root)
            if video_path is None:
                skipped += 1
                continue

            if label == POS_LABEL:
                pts = human_points.get(rid, [])
                source_center = "human_interpolated_trace"
            else:
                pts = auto_points.get(rid, [])
                source_center = "auto_interpolated_tracker"

            frame = int(round(frame_f))

            tubelet, missing = build_tubelet_image(
                video_cache=video_cache,
                video_path=video_path,
                points=pts,
                base_frame=frame,
                fallback_x=x,
                fallback_y=y,
                offsets=offsets,
                crop_size=crop_size,
            )

            if tubelet is None:
                skipped += 1
                continue

            sample_id = row.get("sample_id", f"{label}_{rid}_f{frame}_{i:06d}")
            filename = safe_name(sample_id) + ".jpg"
            path = tubelet_dir / label / filename

            ok = imwrite_unicode(path, tubelet, quality=94)
            if not ok:
                skipped += 1
                continue

            item = dict(row)
            item["tubelet_path"] = str(path)
            item["source_center"] = source_center
            item["missing_interp_count"] = missing
            item["tubelet_offsets"] = ",".join(str(x) for x in offsets)
            item["tubelet_crop_size"] = crop_size

            out_rows.append(item)
    finally:
        video_cache.close()

    by_label: dict[str, int] = {}

    for r in out_rows:
        label = r.get("label", "")
        by_label[label] = by_label.get(label, 0) + 1

    summary = {
        "tubelets": len(out_rows),
        "skipped": skipped,
        "crop_size": crop_size,
        "offsets": offsets,
        "by_label": dict(sorted(by_label.items())),
        "inputs": {
            "manifest_002D": str(manifest_002D),
            "merged_csv": str(merged_csv),
            "final_csv": str(final_csv),
        },
    }

    return out_rows, summary


def extract_tubelet_features(img: np.ndarray) -> np.ndarray:
    # Tubelet = 5 crops horizontaux. On transforme en 5 tuiles 32x32.
    img = cv2.resize(img, (160, 32), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    raw = gray.flatten()

    tiles_g = []
    stats = []

    for i in range(5):
        x1 = i * 32
        x2 = x1 + 32

        g = gray[:, x1:x2]
        htile = hsv[:, x1:x2]

        tiles_g.append(g)

        _, s, v = cv2.split(htile)

        center_g = g[10:22, 10:22]
        center_s = s[10:22, 10:22].astype(np.float32) / 255.0
        center_v = v[10:22, 10:22].astype(np.float32) / 255.0

        stats.extend([
            float(center_g.mean()),
            float(center_g.std()),
            float(center_g.max()),
            float(center_s.mean()),
            float(center_v.mean()),
            float(center_v.max()),
            float(((center_v > 0.60) & (center_s < 0.45)).mean()),
            float(((center_v > 0.45) & (center_s > 0.25)).mean()),
        ])

    # Descripteurs temporels simples : différence visuelle entre crops successifs.
    for a, b in zip(tiles_g[:-1], tiles_g[1:]):
        d = np.abs(a - b)
        stats.extend([
            float(d.mean()),
            float(d.std()),
            float(d.max()),
        ])

    # Histogramme global HSV.
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))

    return np.concatenate([raw, np.array(stats, dtype=np.float32), hist]).astype(np.float32)


def load_tubelet_samples(manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    usable = []
    feats = []
    labels = []

    for row in manifest_rows:
        label = row.get("label", "")

        if label not in (POS_LABEL, NEG_LABEL, PARTIAL_LABEL):
            continue

        p = Path(str(row.get("tubelet_path", "")))
        img = imread_unicode(p)

        if img is None:
            continue

        if label == POS_LABEL:
            y = 1
        elif label == NEG_LABEL:
            y = 0
        else:
            y = -1

        usable.append(dict(row))
        feats.append(extract_tubelet_features(img))
        labels.append(y)

    if not feats:
        raise SystemExit("[002F] Aucun tubelet lisible.")

    return usable, np.vstack(feats), np.array(labels, dtype=np.int32)


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X - mu) / sd, mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def centroid_score(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    pos = X_train[y_train == 1]
    neg = X_train[y_train == 0]

    pos_c = pos.mean(axis=0)
    neg_c = neg.mean(axis=0)

    d_pos = np.sqrt(((X_eval - pos_c) ** 2).sum(axis=1))
    d_neg = np.sqrt(((X_eval - neg_c) ** 2).sum(axis=1))

    return d_neg - d_pos


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    xs = sorted(set(float(x) for x in scores))

    if not xs:
        return 0.0

    mids = [xs[0] - 1e-6]
    mids.extend((xs[i] + xs[i + 1]) / 2.0 for i in range(len(xs) - 1))
    mids.append(xs[-1] + 1e-6)

    best_t = mids[0]
    best_bal = -1.0

    for t in mids:
        pred = (scores >= t).astype(np.int32)

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)

        if bal > best_bal:
            best_bal = bal
            best_t = t

    return float(best_t)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(1, len(y))
    recall_pos = tp / max(1, tp + fn)
    recall_neg = tn / max(1, tn + fp)
    precision_pos = tp / max(1, tp + fp)
    bal_acc = 0.5 * (recall_pos + recall_neg)

    return {
        "n": int(len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "positive_recall": round(recall_pos, 4),
        "negative_recall": round(recall_neg, 4),
        "positive_precision": round(precision_pos, 4),
    }


def leave_one_review_cv(rows: list[dict[str, Any]], X: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trainable = y >= 0
    review_ids = sorted(set(r.get("review_id", "") for r, yy in zip(rows, y) if yy >= 0))

    pred_rows = []
    folds = []

    for rid in review_ids:
        test_mask = np.array([(r.get("review_id", "") == rid and yy >= 0) for r, yy in zip(rows, y)], dtype=bool)
        train_mask = trainable & ~test_mask

        if test_mask.sum() == 0:
            continue

        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(set(y_train.tolist())) < 2:
            continue

        X_train_std, mu, sd = standardize_train(X[train_mask])
        X_test_std = apply_standardize(X[test_mask], mu, sd)

        train_scores = centroid_score(X_train_std, y_train, X_train_std)
        threshold = best_threshold(train_scores, y_train)

        scores = centroid_score(X_train_std, y_train, X_test_std)
        pred = (scores >= threshold).astype(np.int32)

        m = metrics(y_test, pred)
        m["review_id"] = rid
        m["threshold"] = round(float(threshold), 6)
        folds.append(m)

        idxs = np.where(test_mask)[0]

        for idx, score, pp in zip(idxs, scores, pred):
            item = dict(rows[int(idx)])
            item["y_true"] = int(y[int(idx)])
            item["score_002F"] = round(float(score), 6)
            item["pred_002F"] = int(pp)
            item["correct_002F"] = int(pp == y[int(idx)])
            pred_rows.append(item)

    y_all = np.array([int(r["y_true"]) for r in pred_rows], dtype=np.int32)
    p_all = np.array([int(r["pred_002F"]) for r in pred_rows], dtype=np.int32)

    return pred_rows, {
        "cv_mode": "leave_one_review_id_out",
        "folds": folds,
        "overall": metrics(y_all, p_all) if len(y_all) else {},
    }


def global_score(rows: list[dict[str, Any]], X: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_mask = y >= 0
    y_train = y[train_mask]

    X_train_std, mu, sd = standardize_train(X[train_mask])
    X_all_std = apply_standardize(X, mu, sd)

    train_scores = centroid_score(X_train_std, y_train, X_train_std)
    threshold = best_threshold(train_scores, y_train)

    scores = centroid_score(X_train_std, y_train, X_all_std)

    out = []

    for row, yy, score in zip(rows, y, scores):
        item = dict(row)
        item["y_true"] = "" if int(yy) < 0 else int(yy)
        item["score_002F"] = round(float(score), 6)

        if int(yy) >= 0:
            pred = int(score >= threshold)
            item["pred_002F"] = pred
            item["correct_002F"] = int(pred == int(yy))
        else:
            item["pred_002F"] = "score_only_partial"
            item["correct_002F"] = ""

        out.append(item)

    train_pred = (train_scores >= threshold).astype(np.int32)

    return out, {
        "threshold": round(float(threshold), 6),
        "train_on_all_metrics": metrics(y_train, train_pred),
    }


def make_tile(path: Path, text: str, correct: bool | None = None) -> np.ndarray:
    img = imread_unicode(path)

    if img is None:
        img = np.zeros((96, 96 * 5, 3), dtype=np.uint8)

    img = cv2.resize(img, (96 * 5, 96), interpolation=cv2.INTER_AREA)

    pad = 30
    out = np.zeros((96 + pad, 96 * 5, 3), dtype=np.uint8)
    out[:96] = img

    if correct is None:
        color = (230, 230, 230)
    elif correct:
        color = (80, 220, 80)
    else:
        color = (80, 80, 255)

    cv2.rectangle(out, (1, 1), (out.shape[1] - 2, 94), color, 2)

    cv2.putText(
        out,
        text[:54],
        (6, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    return out


def write_sheet(path: Path, rows: list[dict[str, Any]], max_items: int = 30) -> None:
    rows = rows[:max_items]

    if not rows:
        return

    tiles = []

    for r in rows:
        p = Path(str(r.get("tubelet_path", "")))
        text = f'{r.get("review_id","")} {r.get("score_002F","")}'
        c = r.get("correct_002F", "")
        correct = None if c == "" else bool(int(c))
        tiles.append(make_tile(p, text, correct))

    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.zeros((len(tiles) * tile_h, tile_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        y = i * tile_h
        sheet[y:y + tile_h, :] = tile

    imwrite_unicode(path, sheet, quality=92)


def write_label_sheet(path: Path, rows: list[dict[str, Any]], label: str, max_items: int = 30) -> None:
    selected = [r for r in rows if r.get("label") == label][:max_items]
    write_sheet(path, selected, max_items=max_items)


def write_html(path: Path, summary: dict[str, Any], sheet_paths: dict[str, Path]) -> None:
    overall = summary["cv"].get("overall", {})
    global_m = summary["global"].get("train_on_all_metrics", {})

    fold_rows = []
    for f in summary["cv"].get("folds", []):
        fold_rows.append(f"""
<tr>
<td>{html.escape(str(f.get("review_id", "")))}</td>
<td>{html.escape(str(f.get("n", "")))}</td>
<td>{html.escape(str(f.get("balanced_accuracy", "")))}</td>
<td>{html.escape(str(f.get("accuracy", "")))}</td>
<td>{html.escape(str(f.get("positive_recall", "")))}</td>
<td>{html.escape(str(f.get("negative_recall", "")))}</td>
<td>{html.escape(str(f.get("tp", "")))}</td>
<td>{html.escape(str(f.get("tn", "")))}</td>
<td>{html.escape(str(f.get("fp", "")))}</td>
<td>{html.escape(str(f.get("fn", "")))}</td>
</tr>
""")

    label_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["build"]["by_label"].items()
    )

    sheet_sections = []
    for name, p in sheet_paths.items():
        sheet_sections.append(f"""
<section>
<h2>{html.escape(name)}</h2>
<img src="{html.escape(p.name)}" loading="lazy">
</section>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux temporal tubelets 002F</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
code{{color:#dce4ff}}
.warn{{color:#ffd37a}}
.muted{{color:#9ea7b8}}
</style>
</head>
<body>
<h1>TTFlux temporal tubelets 002F</h1>

<section>
<h2>Résumé</h2>
<p>Tubelets : <b>{summary["build"]["tubelets"]}</b> · skipped : <b>{summary["build"]["skipped"]}</b></p>
<p>Offsets : <code>{html.escape(str(summary["build"]["offsets"]))}</code></p>
<table>
<thead><tr><th>label</th><th>count</th></tr></thead>
<tbody>{label_rows}</tbody>
</table>
</section>

<section>
<h2>Separabilité temporelle</h2>
<p>CV balanced accuracy : <b>{html.escape(str(overall.get("balanced_accuracy", "")))}</b></p>
<p>CV accuracy : <b>{html.escape(str(overall.get("accuracy", "")))}</b></p>
<p>CV false positives : <b>{html.escape(str(summary["cv_error_counts"]["false_positive"]))}</b></p>
<p>CV false negatives : <b>{html.escape(str(summary["cv_error_counts"]["false_negative"]))}</b></p>
<p>Train-on-all balanced accuracy : <b>{html.escape(str(global_m.get("balanced_accuracy", "")))}</b></p>
<p class="warn">
À comparer à 002E : si la balanced accuracy augmente nettement, le signal temporel est utile.
Si ça reste faible, il faut plus de segments humains ou une représentation plus structurée qu'un simple classifieur de contact-sheet.
</p>
</section>

<section>
<h2>Résultats par segment tenu à l'écart</h2>
<table>
<thead>
<tr>
<th>review_id</th><th>n</th><th>balanced acc</th><th>accuracy</th><th>recall positif</th><th>recall négatif</th>
<th>tp</th><th>tn</th><th>fp</th><th>fn</th>
</tr>
</thead>
<tbody>{''.join(fold_rows)}</tbody>
</table>
</section>

{''.join(sheet_sections)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-002D", default="runs/batch_001E/supervised_crops_002D/supervised_crops_manifest_002D.csv")
    parser.add_argument("--analysis-dir", default="runs/batch_001E/human_analysis_001Z")
    parser.add_argument("--final-csv", default="runs/batch_001E/final_decisions_002B/final_decisions_002B_all.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/temporal_tubelets_002F")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--offsets", default="-4,-2,0,2,4")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]

    manifest_002D = Path(args.manifest_002D)
    analysis_dir = Path(args.analysis_dir)
    merged_csv = analysis_dir / "human_annotations_001Z_merged.csv"
    final_csv = Path(args.final_csv)

    tubelet_rows, build_summary = build_tubelets(
        manifest_002D=manifest_002D,
        merged_csv=merged_csv,
        final_csv=final_csv,
        out_dir=out_dir,
        crop_size=args.crop_size,
        offsets=offsets,
    )

    tubelet_manifest = out_dir / "tubelets_manifest_002F.csv"
    write_csv(tubelet_manifest, tubelet_rows)

    rows, X, y = load_tubelet_samples(tubelet_rows)

    cv_rows, cv_summary = leave_one_review_cv(rows, X, y)
    global_rows, global_summary = global_score(rows, X, y)

    write_csv(out_dir / "tubelet_predictions_cv_002F.csv", cv_rows)
    write_csv(out_dir / "tubelet_predictions_global_002F.csv", global_rows)

    false_pos = [r for r in cv_rows if r.get("y_true") == 0 and r.get("pred_002F") == 1]
    false_neg = [r for r in cv_rows if r.get("y_true") == 1 and r.get("pred_002F") == 0]
    correct_pos = [r for r in cv_rows if r.get("y_true") == 1 and r.get("pred_002F") == 1]
    correct_neg = [r for r in cv_rows if r.get("y_true") == 0 and r.get("pred_002F") == 0]

    partial_rows = [r for r in global_rows if r.get("label") == PARTIAL_LABEL]
    partial_rows = sorted(partial_rows, key=lambda r: float(r.get("score_002F", 0)), reverse=True)

    sheets: dict[str, Path] = {}

    sheet_specs = {
        "label_positive_tubelets": [r for r in tubelet_rows if r.get("label") == POS_LABEL],
        "label_negative_tubelets": [r for r in tubelet_rows if r.get("label") == NEG_LABEL],
        "label_partial_tubelets": [r for r in tubelet_rows if r.get("label") == PARTIAL_LABEL],
        "false_positive_negative_pred_ball": false_pos,
        "false_negative_positive_pred_negative": false_neg,
        "correct_positive_examples": correct_pos,
        "correct_negative_examples": correct_neg,
        "partial_high_score_examples": partial_rows[:30],
        "partial_low_score_examples": list(reversed(partial_rows))[:30],
    }

    for name, sheet_rows in sheet_specs.items():
        p = out_dir / f"{name}_002F.jpg"
        write_sheet(p, sheet_rows, max_items=30)
        if p.exists():
            sheets[name] = p

    summary = {
        "build": build_summary,
        "rows_loaded": len(rows),
        "used_for_cv": int((y >= 0).sum()),
        "partial_score_only": int((y < 0).sum()),
        "cv": cv_summary,
        "global": global_summary,
        "cv_error_counts": {
            "false_positive": len(false_pos),
            "false_negative": len(false_neg),
            "correct_positive": len(correct_pos),
            "correct_negative": len(correct_neg),
        },
        "comparison_hint": {
            "previous_002E_balanced_accuracy": 0.6061,
            "previous_002E_accuracy": 0.5964,
            "previous_002E_false_positive": 38,
            "previous_002E_false_negative": 73,
        },
    }

    (out_dir / "temporal_tubelets_summary_002F.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "temporal_tubelets_002F.html", summary, sheets)

    print(f"[002F] tubelets          : {build_summary['tubelets']}")
    print(f"[002F] skipped           : {build_summary['skipped']}")
    print(f"[002F] by_label          : {build_summary['by_label']}")
    print(f"[002F] rows loaded       : {summary['rows_loaded']}")
    print(f"[002F] used for cv       : {summary['used_for_cv']}")
    print(f"[002F] partial score only: {summary['partial_score_only']}")
    print(f"[002F] cv overall        : {summary['cv'].get('overall', {})}")
    print(f"[002F] errors            : {summary['cv_error_counts']}")
    print(f"[002F] out dir           : {out_dir}")
    print(f"[002F] html              : {out_dir / 'temporal_tubelets_002F.html'}")


if __name__ == "__main__":
    main()
