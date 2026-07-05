from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
except Exception as exc:
    raise SystemExit(
        "scikit-learn/joblib absent. Lance : python -m pip install scikit-learn joblib\n"
        f"Erreur: {repr(exc)}"
    )


VERSION = "004N"


NUMERIC_FEATURES = [
    "cand_x",
    "cand_y",
    "area",
    "bbox_w",
    "bbox_h",
    "aspect",
    "fill",
    "compact",
    "candidate_score_004M",
    "patch_gray_mean",
    "patch_gray_std",
    "patch_s_mean",
    "patch_v_mean",
    "patch_motion_mean",
    "patch_motion_max",
]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def make_click_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["review_id"].astype(str)
        + "|"
        + df["local_frame"].astype(str)
        + "|"
        + pd.to_numeric(df["click_x"], errors="coerce").round(2).astype(str)
        + "|"
        + pd.to_numeric(df["click_y"], errors="coerce").round(2).astype(str)
    )


def build_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    x = pd.DataFrame(index=df.index)

    for c in NUMERIC_FEATURES:
        if c in df.columns:
            x[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            x[c] = 0.0

    if "pass_name" in df.columns:
        dummies = pd.get_dummies(df["pass_name"].fillna("").astype(str), prefix="pass")
        x = pd.concat([x, dummies], axis=1)

    if feature_cols is None:
        feature_cols = list(x.columns)
    else:
        for c in feature_cols:
            if c not in x.columns:
                x[c] = 0.0
        x = x[feature_cols]

    return x, feature_cols


def sample_train_rows(train_df: pd.DataFrame, neg_ratio: int, seed: int) -> pd.DataFrame:
    # Positif strict = candidat <=20 px du clic.
    # Négatif sûr = candidat >50 px du clic.
    # Zone 20-50 px ignorée à l'entraînement pour ne pas brouiller.
    pos = train_df[train_df["dist_to_click_num"] <= 20].copy()
    neg = train_df[train_df["dist_to_click_num"] > 50].copy()

    if pos.empty:
        raise RuntimeError("Aucun positif <=20 px dans train_df")

    n_neg = min(len(neg), len(pos) * neg_ratio)
    neg_s = neg.sample(n=n_neg, random_state=seed) if len(neg) > n_neg else neg

    out = pd.concat([pos, neg_s], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["y"] = (out["dist_to_click_num"] <= 20).astype(int)
    return out


def fit_model(train_df: pd.DataFrame, feature_cols: list[str], seed: int):
    sampled = sample_train_rows(train_df, neg_ratio=8, seed=seed)
    x, _ = build_features(sampled, feature_cols)

    y = sampled["y"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=280,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(x, y)

    return clf, sampled


def evaluate(df: pd.DataFrame, score_col: str, prefix: str) -> dict:
    if df.empty:
        return {}

    rows = []

    for key, g in df.groupby("click_key", dropna=False):
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        dists = pd.to_numeric(g["dist_to_click"], errors="coerce").fillna(1e9).to_numpy()

        row = {
            "click_key": key,
            "review_id": str(g.iloc[0].get("review_id", "")),
            "n_candidates": int(len(g)),
        }

        for k in [1, 3, 5, 10, 20, 50]:
            kk = min(k, len(g))
            top = dists[:kk]
            row[f"{prefix}_hit10_top{k}"] = int(np.any(top <= 10))
            row[f"{prefix}_hit20_top{k}"] = int(np.any(top <= 20))
            row[f"{prefix}_hit50_top{k}"] = int(np.any(top <= 50))

        idx20 = np.where(dists <= 20)[0]
        idx50 = np.where(dists <= 50)[0]

        row[f"{prefix}_best_rank20"] = int(idx20[0] + 1) if len(idx20) else ""
        row[f"{prefix}_best_rank50"] = int(idx50[0] + 1) if len(idx50) else ""
        row[f"{prefix}_best_dist"] = round(float(np.min(dists)), 3) if len(dists) else ""

        rows.append(row)

    ev = pd.DataFrame(rows)

    def rate(col):
        return round(float(pd.to_numeric(ev[col], errors="coerce").fillna(0).mean()), 4) if col in ev.columns else 0.0

    def med(col):
        vals = pd.to_numeric(ev[col], errors="coerce").dropna()
        return round(float(vals.median()), 3) if len(vals) else None

    return {
        "clicks": int(len(ev)),
        "hit10_top1": rate(f"{prefix}_hit10_top1"),
        "hit20_top1": rate(f"{prefix}_hit20_top1"),
        "hit50_top1": rate(f"{prefix}_hit50_top1"),
        "hit10_top5": rate(f"{prefix}_hit10_top5"),
        "hit20_top5": rate(f"{prefix}_hit20_top5"),
        "hit50_top5": rate(f"{prefix}_hit50_top5"),
        "hit10_top20": rate(f"{prefix}_hit10_top20"),
        "hit20_top20": rate(f"{prefix}_hit20_top20"),
        "hit50_top20": rate(f"{prefix}_hit50_top20"),
        "hit10_top50": rate(f"{prefix}_hit10_top50"),
        "hit20_top50": rate(f"{prefix}_hit20_top50"),
        "hit50_top50": rate(f"{prefix}_hit50_top50"),
        "best_rank20_med": med(f"{prefix}_best_rank20"),
        "best_rank50_med": med(f"{prefix}_best_rank50"),
        "best_dist_med": med(f"{prefix}_best_dist"),
    }


def make_folds(reviews: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    arr = np.array(reviews, dtype=object)
    rng.shuffle(arr)

    folds = []
    for part in np.array_split(arr, n_folds):
        folds.append([str(x) for x in part.tolist()])

    return folds


def write_html(path: Path, summary: dict):
    fold_rows = []

    for f in summary.get("cv_folds", []):
        fold_rows.append(
            "<tr>"
            f"<td>{esc(f.get('fold'))}</td>"
            f"<td>{esc(', '.join(f.get('test_reviews', [])))}</td>"
            f"<td>{esc(f.get('baseline', {}).get('hit50_top5'))}</td>"
            f"<td>{esc(f.get('model', {}).get('hit50_top5'))}</td>"
            f"<td>{esc(f.get('baseline', {}).get('hit50_top20'))}</td>"
            f"<td>{esc(f.get('model', {}).get('hit50_top20'))}</td>"
            f"<td>{esc(f.get('baseline', {}).get('best_rank50_med'))}</td>"
            f"<td>{esc(f.get('model', {}).get('best_rank50_med'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004N ball ranker</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e}}
</style>
</head>
<body>
<h1>TTFlux · 004N ball ranker</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Cross-validation par review_id</h2>
<table>
<thead>
<tr>
<th>fold</th><th>test reviews</th>
<th>baseline ≤50 top5</th><th>model ≤50 top5</th>
<th>baseline ≤50 top20</th><th>model ≤50 top20</th>
<th>baseline rank50 med</th><th>model rank50 med</th>
</tr>
</thead>
<tbody>{''.join(fold_rows)}</tbody>
</table>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="runs/multicandidate_audit_004M/multicandidates_004M.csv")
    ap.add_argument("--out-dir", default="runs/ball_ranker_004N")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = Path.cwd()

    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = root / candidates_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(candidates_path).fillna("")

    required = {"review_id", "local_frame", "click_x", "click_y", "dist_to_click"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Colonnes manquantes: {sorted(missing)}")

    df["dist_to_click_num"] = pd.to_numeric(df["dist_to_click"], errors="coerce")
    df = df.dropna(subset=["dist_to_click_num"]).copy()

    df["click_key"] = make_click_key(df)
    df["baseline_score_004N"] = pd.to_numeric(df["candidate_score_004M"], errors="coerce").fillna(0.0)

    _, feature_cols = build_features(df, None)

    reviews = sorted(df["review_id"].astype(str).unique().tolist())
    n_folds = max(2, min(args.folds, len(reviews)))

    folds = make_folds(reviews, n_folds=n_folds, seed=args.seed)

    cv_summaries = []
    scored_parts = []

    print("004N candidates=", len(df))
    print("004N reviews=", len(reviews))
    print("004N features=", len(feature_cols))

    for fold_i, test_reviews in enumerate(folds, start=1):
        test_mask = df["review_id"].astype(str).isin(test_reviews)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()

        clf, sampled = fit_model(train_df, feature_cols, seed=args.seed + fold_i)

        x_test, _ = build_features(test_df, feature_cols)
        test_df["model_score_004N"] = clf.predict_proba(x_test)[:, 1]

        baseline = evaluate(test_df, "baseline_score_004N", "baseline")
        model = evaluate(test_df, "model_score_004N", "model")

        cv_summaries.append({
            "fold": fold_i,
            "test_reviews": test_reviews,
            "train_rows_sampled": int(len(sampled)),
            "train_pos": int(sampled["y"].sum()),
            "train_neg": int((sampled["y"] == 0).sum()),
            "baseline": baseline,
            "model": model,
        })

        scored_parts.append(test_df)

        print(
            f"  fold {fold_i}/{n_folds} "
            f"baseline50top5={baseline.get('hit50_top5')} "
            f"model50top5={model.get('hit50_top5')} "
            f"baseline50top20={baseline.get('hit50_top20')} "
            f"model50top20={model.get('hit50_top20')}"
        )

    cv_scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()

    # Entraînement final sur tout le goldset.
    final_clf, final_sampled = fit_model(df, feature_cols, seed=args.seed + 99)
    x_all, _ = build_features(df, feature_cols)
    df["model_score_004N"] = final_clf.predict_proba(x_all)[:, 1]

    baseline_all = evaluate(df, "baseline_score_004N", "baseline")
    model_all = evaluate(df, "model_score_004N", "model")
    cv_model = evaluate(cv_scored, "model_score_004N", "cv_model") if not cv_scored.empty else {}
    cv_baseline = evaluate(cv_scored, "baseline_score_004N", "cv_baseline") if not cv_scored.empty else {}

    model_path = out_dir / "ball_ranker_004N.joblib"
    joblib.dump(
        {
            "version": VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": final_clf,
            "feature_cols": feature_cols,
            "numeric_features": NUMERIC_FEATURES,
        },
        model_path,
    )

    out_scores = out_dir / "candidate_ranker_scores_004N.csv"
    out_cv = out_dir / "candidate_ranker_cv_scores_004N.csv"
    out_json = out_dir / "ball_ranker_summary_004N.json"
    out_html = out_dir / "ball_ranker_004N.html"

    df.to_csv(out_scores, index=False, encoding="utf-8")
    cv_scored.to_csv(out_cv, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "manual_goldset_ranker_experiment_not_live",
        "candidates_csv": str(candidates_path),
        "candidate_rows": int(len(df)),
        "reviews": int(len(reviews)),
        "clicks": int(df["click_key"].nunique()),
        "features": feature_cols,
        "final_train_sampled_rows": int(len(final_sampled)),
        "final_train_pos": int(final_sampled["y"].sum()),
        "final_train_neg": int((final_sampled["y"] == 0).sum()),
        "baseline_all": baseline_all,
        "model_all_in_sample": model_all,
        "cv_baseline": cv_baseline,
        "cv_model": cv_model,
        "cv_folds": cv_summaries,
        "model_path": str(model_path),
        "next": "If CV model improves top5/top20 ranking, apply 004N ranker to new candidates and rebuild trajectories.",
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary)

    print("004N status=OK")
    print("candidate_rows=", len(df))
    print("reviews=", len(reviews))
    print("clicks=", df["click_key"].nunique())
    print("baseline_all=", json.dumps(baseline_all, ensure_ascii=False))
    print("model_all_in_sample=", json.dumps(model_all, ensure_ascii=False))
    print("cv_baseline=", json.dumps(cv_baseline, ensure_ascii=False))
    print("cv_model=", json.dumps(cv_model, ensure_ascii=False))
    print("model_path=", model_path)
    print("wrote", out_scores)
    print("wrote", out_cv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
