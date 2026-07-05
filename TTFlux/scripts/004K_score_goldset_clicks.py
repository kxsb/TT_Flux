from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "004K"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def dist(x1, y1, x2, y2) -> float:
    try:
        return float(math.hypot(float(x1) - float(x2), float(y1) - float(y2)))
    except Exception:
        return float("nan")


def nearest_point(df: pd.DataFrame, frame: int, x: float, y: float, frame_window: int = 2) -> dict:
    if df is None or df.empty:
        return {
            "found": False,
            "dist": np.nan,
            "frame": "",
            "x": "",
            "y": "",
            "frame_delta": "",
            "score": "",
        }

    sub = df[
        (df["frame_num"] >= frame - frame_window) &
        (df["frame_num"] <= frame + frame_window)
    ].copy()

    if sub.empty:
        return {
            "found": False,
            "dist": np.nan,
            "frame": "",
            "x": "",
            "y": "",
            "frame_delta": "",
            "score": "",
        }

    sub["dist_to_click"] = np.sqrt((sub["x_num"] - x) ** 2 + (sub["y_num"] - y) ** 2)
    sub["frame_delta_abs"] = (sub["frame_num"] - frame).abs()
    sub = sub.sort_values(["dist_to_click", "frame_delta_abs"])

    r = sub.iloc[0]

    score = ""
    for col in ["final_score_004E", "score"]:
        if col in sub.columns:
            score = r.get(col, "")
            break

    return {
        "found": True,
        "dist": round(float(r["dist_to_click"]), 3),
        "frame": int(r["frame_num"]),
        "x": round(float(r["x_num"]), 3),
        "y": round(float(r["y_num"]), 3),
        "frame_delta": int(r["frame_num"] - frame),
        "score": score,
    }


def load_selected_segment_points(run_dir: Path, manifest_row: pd.Series) -> pd.DataFrame:
    rel_csv = str(manifest_row.get("csv", "")).strip()
    if not rel_csv:
        return pd.DataFrame()

    p = run_dir / rel_csv
    if not p.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    cols = {c.lower(): c for c in df.columns}
    if not {"frame", "x", "y"}.issubset(cols.keys()):
        return pd.DataFrame()

    out = df.copy()
    out["frame_num"] = to_num(out[cols["frame"]])
    out["x_num"] = to_num(out[cols["x"]])
    out["y_num"] = to_num(out[cols["y"]])
    out = out.dropna(subset=["frame_num", "x_num", "y_num"])
    out["frame_num"] = out["frame_num"].astype(int)

    return out


def summarize_group(g: pd.DataFrame) -> dict:
    def med(col):
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        return round(float(vals.median()), 3) if len(vals) else None

    def rate(col, threshold):
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        if not len(vals):
            return None
        return round(float((vals <= threshold).mean()), 4)

    return {
        "clicks": int(len(g)),
        "reservoir_dist_med": med("reservoir_nearest_dist"),
        "reservoir_hit_10": rate("reservoir_nearest_dist", 10),
        "reservoir_hit_20": rate("reservoir_nearest_dist", 20),
        "reservoir_hit_50": rate("reservoir_nearest_dist", 50),
        "selected_dist_med": med("selected_nearest_dist"),
        "selected_hit_10": rate("selected_nearest_dist", 10),
        "selected_hit_20": rate("selected_nearest_dist", 20),
        "selected_hit_50": rate("selected_nearest_dist", 50),
    }


def write_html(path: Path, summary: dict, per_review: pd.DataFrame, sample_clicks: pd.DataFrame):
    review_rows = []
    for _, r in per_review.iterrows():
        cls = "good" if (r.get("selected_hit_20") or 0) >= 0.6 else "bad" if (r.get("selected_hit_50") or 0) < 0.3 else "mid"
        review_rows.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('video_id'))}</td>"
            f"<td>{esc(r.get('clip_id'))}</td>"
            f"<td>{esc(r.get('clicks'))}</td>"
            f"<td>{esc(r.get('reservoir_dist_med'))}</td>"
            f"<td>{esc(r.get('reservoir_hit_20'))}</td>"
            f"<td>{esc(r.get('reservoir_hit_50'))}</td>"
            f"<td>{esc(r.get('selected_dist_med'))}</td>"
            f"<td>{esc(r.get('selected_hit_20'))}</td>"
            f"<td>{esc(r.get('selected_hit_50'))}</td>"
            "</tr>"
        )

    click_rows = []
    for _, r in sample_clicks.head(300).iterrows():
        click_rows.append(
            "<tr>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('source_frame'))}</td>"
            f"<td>{esc(r.get('x'))}</td>"
            f"<td>{esc(r.get('y'))}</td>"
            f"<td>{esc(r.get('reservoir_nearest_dist'))}</td>"
            f"<td>{esc(r.get('selected_nearest_dist'))}</td>"
            f"<td>{esc(r.get('reservoir_nearest_x'))},{esc(r.get('reservoir_nearest_y'))}</td>"
            f"<td>{esc(r.get('selected_nearest_x'))},{esc(r.get('selected_nearest_y'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 004K goldset scorer</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
.wrap{{overflow:auto;max-height:75vh;border:1px solid #2b303b;border-radius:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e;position:sticky;top:0}}
tr.good td{{background:rgba(116,217,159,.08)}}
tr.mid td{{background:rgba(255,200,80,.08)}}
tr.bad td{{background:rgba(255,80,80,.10)}}
</style>
</head>
<body>
<h1>TTFlux · 004K goldset scorer</h1>
<section><h2>Résumé</h2><pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre></section>
<section>
<h2>Score par segment annoté</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>review</th><th>video</th><th>clip</th><th>clicks</th>
<th>reservoir med</th><th>reservoir ≤20</th><th>reservoir ≤50</th>
<th>selected med</th><th>selected ≤20</th><th>selected ≤50</th>
</tr>
</thead>
<tbody>{''.join(review_rows)}</tbody>
</table>
</div>
</section>
<section>
<h2>Échantillon clics</h2>
<div class="wrap">
<table>
<thead>
<tr>
<th>review</th><th>frame</th><th>click x</th><th>click y</th>
<th>reservoir dist</th><th>selected dist</th><th>reservoir xy</th><th>selected xy</th>
</tr>
</thead>
<tbody>{''.join(click_rows)}</tbody>
</table>
</div>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", default="runs/ball_goldset_004J/ball_clicks_004J.csv")
    ap.add_argument("--raw-points", default="runs/dataset_points_004E_full240/raw_tracking_points_004E.csv")
    ap.add_argument("--manifest", default="runs/batch_004F_full240/operational_manifest_001T2.csv")
    ap.add_argument("--run-dir", default="runs/batch_004F_full240")
    ap.add_argument("--out-dir", default="runs/goldset_score_004K")
    ap.add_argument("--frame-window", type=int, default=2)
    args = ap.parse_args()

    root = Path.cwd()

    clicks_path = Path(args.clicks)
    raw_path = Path(args.raw_points)
    manifest_path = Path(args.manifest)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    if not clicks_path.is_absolute():
        clicks_path = root / clicks_path
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    clicks = pd.read_csv(clicks_path).fillna("")
    raw = pd.read_csv(raw_path)
    manifest = pd.read_csv(manifest_path).fillna("")

    clicks = clicks[clicks["visibility"].astype(str).eq("ball")].copy()

    for c in ["x", "y", "source_frame"]:
        clicks[c] = to_num(clicks[c])
    clicks = clicks.dropna(subset=["x", "y", "source_frame"])
    clicks["source_frame"] = clicks["source_frame"].astype(int)

    raw["frame_num"] = to_num(raw["frame"])
    raw["x_num"] = to_num(raw["x"])
    raw["y_num"] = to_num(raw["y"])
    raw = raw.dropna(subset=["clip_id", "frame_num", "x_num", "y_num"])
    raw["clip_id"] = raw["clip_id"].astype(str)
    raw["frame_num"] = raw["frame_num"].astype(int)

    manifest_by_review = {
        str(r["review_id"]): r for _, r in manifest.iterrows()
    }

    raw_by_clip = {
        str(k): g.copy() for k, g in raw.groupby("clip_id", dropna=False)
    }

    selected_by_review = {}
    for review_id, row in manifest_by_review.items():
        selected_by_review[review_id] = load_selected_segment_points(run_dir, row)

    scored = []

    print("004K clicks_ball=", len(clicks))

    for i, (_, c) in enumerate(clicks.iterrows(), start=1):
        review_id = str(c["review_id"])
        clip_id = str(c["clip_id"])
        frame = int(c["source_frame"])
        x = float(c["x"])
        y = float(c["y"])

        reservoir = nearest_point(
            raw_by_clip.get(clip_id, pd.DataFrame()),
            frame,
            x,
            y,
            frame_window=args.frame_window,
        )

        selected = nearest_point(
            selected_by_review.get(review_id, pd.DataFrame()),
            frame,
            x,
            y,
            frame_window=args.frame_window,
        )

        row = c.to_dict()

        row.update({
            "reservoir_found": reservoir["found"],
            "reservoir_nearest_dist": reservoir["dist"],
            "reservoir_nearest_frame": reservoir["frame"],
            "reservoir_nearest_x": reservoir["x"],
            "reservoir_nearest_y": reservoir["y"],
            "reservoir_frame_delta": reservoir["frame_delta"],
            "reservoir_score": reservoir["score"],
            "selected_found": selected["found"],
            "selected_nearest_dist": selected["dist"],
            "selected_nearest_frame": selected["frame"],
            "selected_nearest_x": selected["x"],
            "selected_nearest_y": selected["y"],
            "selected_frame_delta": selected["frame_delta"],
            "selected_score": selected["score"],
        })

        scored.append(row)

        if i % 100 == 0 or i == len(clicks):
            print(f"  scored {i}/{len(clicks)}")

    scored_df = pd.DataFrame(scored)

    per_review_rows = []
    for review_id, g in scored_df.groupby("review_id", dropna=False):
        s = summarize_group(g)
        first = g.iloc[0]
        s.update({
            "review_id": review_id,
            "video_id": first.get("video_id", ""),
            "clip_id": first.get("clip_id", ""),
            "segment_idx": first.get("segment_idx", ""),
        })
        per_review_rows.append(s)

    per_review = pd.DataFrame(per_review_rows)

    # tri : d’abord les meilleurs selected_hit_20, puis ceux où reservoir existe mais selected échoue.
    if not per_review.empty:
        per_review["selected_hit_20_sort"] = pd.to_numeric(per_review["selected_hit_20"], errors="coerce").fillna(-1)
        per_review["reservoir_hit_20_sort"] = pd.to_numeric(per_review["reservoir_hit_20"], errors="coerce").fillna(-1)
        per_review = per_review.sort_values(
            ["selected_hit_20_sort", "reservoir_hit_20_sort", "clicks"],
            ascending=[False, False, False],
        ).drop(columns=["selected_hit_20_sort", "reservoir_hit_20_sort"])

    global_summary = summarize_group(scored_df)

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "score_manual_ball_clicks_against_raw_reservoir_and_selected_segments",
        "clicks_csv": str(clicks_path),
        "raw_points_csv": str(raw_path),
        "manifest_csv": str(manifest_path),
        "ball_clicks_scored": int(len(scored_df)),
        "reviews_scored": int(scored_df["review_id"].nunique() if len(scored_df) else 0),
        "frame_window": args.frame_window,
        "global": global_summary,
        "interpretation": {
            "reservoir": "If reservoir_hit_20/50 is low, raw 004E rarely contains the real ball.",
            "selected": "If selected_hit_20/50 is low but reservoir is higher, the segment selector/refiner is wrong.",
            "next": "Use this to train or tune a refiner that prefers manual ball positions."
        }
    }

    out_clicks = out_dir / "goldset_click_scores_004K.csv"
    out_reviews = out_dir / "goldset_review_scores_004K.csv"
    out_json = out_dir / "goldset_score_summary_004K.json"
    out_html = out_dir / "goldset_score_004K.html"

    scored_df.to_csv(out_clicks, index=False, encoding="utf-8")
    per_review.to_csv(out_reviews, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, summary, per_review, scored_df)

    print("004K status=OK")
    print("ball_clicks_scored=", len(scored_df))
    print("reviews_scored=", summary["reviews_scored"])
    print("global=", json.dumps(global_summary, ensure_ascii=False))
    print("wrote", out_clicks)
    print("wrote", out_reviews)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
