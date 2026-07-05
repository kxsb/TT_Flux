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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(f, dialect=dialect)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = []
    seen = set()

    preferred = [
        "sample_id", "review_id", "label", "y_true", "score_002E", "pred_002E",
        "correct_002E", "split_role_002E", "crop_path", "frame", "x", "y",
        "human_decision_002A", "classification_001Z", "dist_offset0", "dist_best",
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


def resolve_path(value: str, root: Path) -> Path | None:
    p = Path(str(value).strip())

    if p.is_absolute() and p.exists():
        return p

    p1 = root / p
    if p1.exists():
        return p1

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None



def imread_unicode(path: Path) -> np.ndarray | None:
    """Lecture image compatible chemins Windows avec accents."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 92) -> bool:
    """Écriture image compatible chemins Windows avec accents."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def extract_features(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32)
    s = s.astype(np.float32)
    v = v.astype(np.float32)

    # Raw shape faible résolution
    gray_flat = gray.flatten()

    # Centre du crop : la balle est censée être au centre
    c = img[10:22, 10:22]
    chsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    ch, cs, cv = cv2.split(chsv)

    center_stats = np.array([
        gray[10:22, 10:22].mean(),
        gray[10:22, 10:22].std(),
        gray[10:22, 10:22].max(),
        float(cs.mean()) / 255.0,
        float(cv.mean()) / 255.0,
        float(cv.max()) / 255.0,
        float(((cv > 155) & (cs < 110)).mean()),  # blanc / clair peu saturé
        float(((cv > 120) & (cs > 70)).mean()),   # objet clair saturé
    ], dtype=np.float32)

    # Histogrammes globaux simples
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist = hist / max(1.0, float(hist.sum()))

    return np.concatenate([gray_flat, center_stats, hist]).astype(np.float32)


def load_samples(manifest: Path, root: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = read_csv(manifest)

    usable = []
    feats = []
    labels = []

    for row in rows:
        label = row.get("label", "")

        if label not in (POS_LABEL, NEG_LABEL, PARTIAL_LABEL):
            continue

        crop_path = resolve_path(row.get("crop_path", ""), root)
        if crop_path is None:
            continue

        img = imread_unicode(crop_path)
        if img is None:
            continue

        y = 1 if label == POS_LABEL else 0 if label == NEG_LABEL else -1

        item = dict(row)
        item["_crop_path_resolved"] = str(crop_path)
        item["_y"] = y

        usable.append(item)
        feats.append(extract_features(img))
        labels.append(y)

    if not feats:
        raise SystemExit(
            "[002E] Aucun crop lisible. Vérifie le manifest 002D et les chemins. "
            "Si le chemin contient un accent, ce patch 002E2 doit être appliqué."
        )

    return usable, np.vstack(feats), np.array(labels, dtype=np.int32)


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X - mu) / sd, mu, sd


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def centroid_score(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    pos = X[y == 1]
    neg = X[y == 0]

    pos_c = pos.mean(axis=0)
    neg_c = neg.mean(axis=0)

    d_pos = np.sqrt(((X_eval - pos_c) ** 2).sum(axis=1))
    d_neg = np.sqrt(((X_eval - neg_c) ** 2).sum(axis=1))

    # plus haut = plus proche des positifs que des négatifs
    return d_neg - d_pos


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = sorted(set(float(x) for x in scores))

    if not candidates:
        return 0.0

    mids = []
    for i in range(len(candidates) - 1):
        mids.append((candidates[i] + candidates[i + 1]) / 2.0)

    mids = [candidates[0] - 1e-6] + mids + [candidates[-1] + 1e-6]

    best_t = mids[0]
    best_score = -1.0

    for t in mids:
        pred = (scores >= t).astype(np.int32)

        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)

        if bal > best_score:
            best_score = bal
            best_t = t

    return float(best_t)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
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
    train_mask_all = y >= 0
    review_ids = sorted(set(r.get("review_id", "") for r, yy in zip(rows, y) if yy >= 0))

    pred_rows = []
    fold_summaries = []

    for rid in review_ids:
        test_mask = np.array([(r.get("review_id", "") == rid and yy >= 0) for r, yy in zip(rows, y)], dtype=bool)
        train_mask = train_mask_all & ~test_mask

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

        test_scores = centroid_score(X_train_std, y_train, X_test_std)
        test_pred = (test_scores >= threshold).astype(np.int32)

        m = metrics(y_test, test_pred)
        m["review_id"] = rid
        m["threshold"] = round(float(threshold), 6)
        fold_summaries.append(m)

        idxs = np.where(test_mask)[0]

        for idx, score, pred in zip(idxs, test_scores, test_pred):
            item = dict(rows[int(idx)])
            item["y_true"] = int(y[int(idx)])
            item["score_002E"] = round(float(score), 6)
            item["pred_002E"] = int(pred)
            item["correct_002E"] = int(pred == y[int(idx)])
            item["split_role_002E"] = f"leave_one_review_test:{rid}"
            pred_rows.append(item)

    y_all = np.array([int(r["y_true"]) for r in pred_rows], dtype=np.int32)
    p_all = np.array([int(r["pred_002E"]) for r in pred_rows], dtype=np.int32)

    summary = {
        "cv_mode": "leave_one_review_id_out",
        "folds": fold_summaries,
        "overall": metrics(y_all, p_all) if len(y_all) else {},
    }

    return pred_rows, summary


def score_all_with_global_model(rows: list[dict[str, Any]], X: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_mask = y >= 0
    y_train = y[train_mask]

    X_train_std, mu, sd = standardize_train(X[train_mask])
    X_all_std = apply_standardize(X, mu, sd)

    train_scores = centroid_score(X_train_std, y_train, X_train_std)
    threshold = best_threshold(train_scores, y_train)

    all_scores = centroid_score(X_train_std, y_train, X_all_std)

    out = []

    for row, yy, score in zip(rows, y, all_scores):
        item = dict(row)
        item["y_true"] = "" if int(yy) < 0 else int(yy)
        item["score_002E"] = round(float(score), 6)

        if int(yy) >= 0:
            pred = int(score >= threshold)
            item["pred_002E"] = pred
            item["correct_002E"] = int(pred == int(yy))
        else:
            item["pred_002E"] = "score_only_partial"
            item["correct_002E"] = ""

        item["split_role_002E"] = "global_score_all"
        out.append(item)

    global_pred = (train_scores >= threshold).astype(np.int32)

    summary = {
        "threshold": round(float(threshold), 6),
        "train_on_all_metrics": metrics(y_train, global_pred),
    }

    return out, summary


def make_tile(crop_path: Path, text: str, good: bool, size: int = 96) -> np.ndarray:
    img = imread_unicode(crop_path)
    if img is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    color = (80, 220, 80) if good else (80, 80, 255)
    cv2.rectangle(img, (1, 1), (size - 2, size - 2), color, 2)
    cv2.drawMarker(img, (size // 2, size // 2), color, cv2.MARKER_CROSS, 15, 1, cv2.LINE_AA)

    pad = 28
    out = np.zeros((size + pad, size, 3), dtype=np.uint8)
    out[:size] = img

    cv2.putText(out, text[:20], (3, size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)

    return out


def write_sheet(path: Path, rows: list[dict[str, Any]], title_field: str, max_items: int = 80, cols: int = 8) -> None:
    rows = rows[:max_items]
    if not rows:
        return

    tiles = []

    for r in rows:
        crop_path = Path(str(r.get("_crop_path_resolved", r.get("crop_path", ""))))
        text = f'{r.get("review_id","")} {r.get(title_field,"")}'
        good = bool(int(r.get("correct_002E", 0))) if str(r.get("correct_002E", "")).isdigit() else True
        tiles.append(make_tile(crop_path, text, good))

    tile_h, tile_w = tiles[0].shape[:2]
    n_cols = min(cols, len(tiles))
    n_rows = int(math.ceil(len(tiles) / n_cols))

    sheet = np.zeros((n_rows * tile_h, n_cols * tile_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        y = (i // n_cols) * tile_h
        x = (i % n_cols) * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile

    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(path, sheet, quality=92)


def write_html(path: Path, summary: dict[str, Any], sheet_paths: dict[str, Path]) -> None:
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

    sheets = []

    for name, p in sheet_paths.items():
        sheets.append(f"""
<section>
<h2>{html.escape(name)}</h2>
<img src="{html.escape(p.name)}" loading="lazy">
</section>
""")

    overall = summary["cv"].get("overall", {})
    global_m = summary["global"].get("train_on_all_metrics", {})

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux crop separability 002E</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{border-bottom:1px solid #2b303b;padding:7px 8px;vertical-align:top;font-size:13px}}
th{{background:#20242e;text-align:left}}
img{{max-width:100%;border-radius:12px;border:1px solid #2b303b;background:#000}}
code{{color:#dce4ff}}
.muted{{color:#9ea7b8}}
.warn{{color:#ffd37a}}
</style>
</head>
<body>
<h1>TTFlux crop separability 002E</h1>

<section>
<h2>Résumé</h2>
<p>Validation : <b>leave-one-review-id-out</b></p>
<p>Overall balanced accuracy : <b>{html.escape(str(overall.get("balanced_accuracy", "")))}</b></p>
<p>Overall accuracy : <b>{html.escape(str(overall.get("accuracy", "")))}</b></p>
<p>Train-on-all balanced accuracy : <b>{html.escape(str(global_m.get("balanced_accuracy", "")))}</b></p>
<p class="warn">
Si le score leave-one-review est faible, les crops actuels ne suffisent pas encore à apprendre une apparence stable de balle.
Si le train-on-all est bon mais le leave-one-review faible, il y a sur-apprentissage par segment.
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

{''.join(sheets)}
</body>
</html>
"""

    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="runs/batch_001E/supervised_crops_002D/supervised_crops_manifest_002D.csv")
    parser.add_argument("--out-dir", default="runs/batch_001E/crop_separability_002E")
    args = parser.parse_args()

    root = Path.cwd()
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, X, y = load_samples(manifest, root)

    cv_rows, cv_summary = leave_one_review_cv(rows, X, y)
    global_rows, global_summary = score_all_with_global_model(rows, X, y)

    write_csv(out_dir / "crop_predictions_cv_002E.csv", cv_rows)
    write_csv(out_dir / "crop_predictions_global_002E.csv", global_rows)

    false_pos = [r for r in cv_rows if r.get("y_true") == 0 and r.get("pred_002E") == 1]
    false_neg = [r for r in cv_rows if r.get("y_true") == 1 and r.get("pred_002E") == 0]
    correct_pos = [r for r in cv_rows if r.get("y_true") == 1 and r.get("pred_002E") == 1]
    correct_neg = [r for r in cv_rows if r.get("y_true") == 0 and r.get("pred_002E") == 0]

    partial_scored = [r for r in global_rows if r.get("label") == PARTIAL_LABEL]
    partial_scored = sorted(partial_scored, key=lambda r: float(r.get("score_002E", 0)), reverse=True)

    sheets: dict[str, Path] = {}

    sheet_specs = {
        "false_positive_auto_far_pred_ball": false_pos,
        "false_negative_human_ball_pred_negative": false_neg,
        "correct_positive_examples": correct_pos,
        "correct_negative_examples": correct_neg,
        "partial_high_score_examples": partial_scored[:80],
        "partial_low_score_examples": list(reversed(partial_scored))[:80],
    }

    for name, sheet_rows in sheet_specs.items():
        p = out_dir / f"{name}_002E.jpg"
        write_sheet(p, sheet_rows, "score_002E")
        if p.exists():
            sheets[name] = p

    summary = {
        "input_manifest": str(manifest),
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
    }

    (out_dir / "crop_separability_summary_002E.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_html(out_dir / "crop_separability_002E.html", summary, sheets)

    print(f"[002E] rows loaded        : {summary['rows_loaded']}")
    print(f"[002E] used for cv        : {summary['used_for_cv']}")
    print(f"[002E] partial score only : {summary['partial_score_only']}")
    print(f"[002E] cv overall         : {summary['cv'].get('overall', {})}")
    print(f"[002E] errors             : {summary['cv_error_counts']}")
    print(f"[002E] out dir            : {out_dir}")
    print(f"[002E] html               : {out_dir / 'crop_separability_002E.html'}")


if __name__ == "__main__":
    main()
