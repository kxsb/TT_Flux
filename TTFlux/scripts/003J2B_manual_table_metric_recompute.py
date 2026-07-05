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


VERSION = "003J2B"

TRACK_CSV_COL_PRIORITY = [
    "csv",
    "track_csv_resolved_003C",
    "track_csv_resolved_003B2",
    "track_csv_resolved_002G2",
    "table_csv",
]

X_COL_CANDIDATES = ["x", "cx", "ball_x", "center_x", "point_x", "px"]
Y_COL_CANDIDATES = ["y", "cy", "ball_y", "center_y", "point_y", "py"]


def resolve_path(value, project_root: Path, run_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None

    raw = value.strip().strip('"').strip("'")
    if not raw:
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = [p] if p.is_absolute() else [project_root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def boolish(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "hit", "reject"}


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def get_manual_polygon(row: pd.Series) -> np.ndarray | None:
    pts = []

    for i in range(1, 5):
        x = num(row.get(f"ordered_q{i}_x"))
        y = num(row.get(f"ordered_q{i}_y"))

        if not np.isfinite(x) or not np.isfinite(y):
            return None

        pts.append([x, y])

    arr = np.array(pts, dtype=np.float32)
    if arr.shape != (4, 2):
        return None

    area = abs(cv2.contourArea(arr.reshape(-1, 1, 2)))
    if area <= 5:
        return None

    return arr


def choose_track_csv(row: pd.Series, project_root: Path, run_dir: Path) -> tuple[Path | None, str]:
    for col in TRACK_CSV_COL_PRIORITY:
        if col not in row.index:
            continue

        p = resolve_path(row.get(col), project_root, run_dir)
        if p and p.suffix.lower() == ".csv":
            return p, col

    return None, ""


def find_xy_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lower = {str(c).lower(): c for c in df.columns}

    x_col = None
    y_col = None

    for c in X_COL_CANDIDATES:
        if c in lower:
            x_col = lower[c]
            break

    for c in Y_COL_CANDIDATES:
        if c in lower:
            y_col = lower[c]
            break

    if x_col and y_col:
        return x_col, y_col

    # Fallback : colonnes numériques avec noms courts contenant x/y.
    numeric_cols = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="raise")
            numeric_cols.append(c)
        except Exception:
            pass

    for c in numeric_cols:
        low = str(c).lower()
        if x_col is None and ("x" == low or low.endswith("_x") or "x_" in low):
            x_col = c
        if y_col is None and ("y" == low or low.endswith("_y") or "y_" in low):
            y_col = c

    return x_col, y_col


def point_metrics(points: np.ndarray, polygon: np.ndarray) -> dict:
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)

    signed_distances = []
    unsigned_distances = []
    inside_flags = []

    for x, y in points:
        d = float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
        signed_distances.append(d)
        unsigned_distances.append(abs(d))
        inside_flags.append(d >= 0)

    if not signed_distances:
        return {
            "manual_track_point_count_003J2B": 0,
            "manual_inside_ratio_003J2B": np.nan,
            "manual_distance_px_median_003J2B": np.nan,
            "manual_distance_px_p90_003J2B": np.nan,
            "manual_signed_distance_px_median_003J2B": np.nan,
        }

    sd = np.array(signed_distances, dtype=np.float32)
    ud = np.array(unsigned_distances, dtype=np.float32)
    inside = np.array(inside_flags, dtype=bool)

    return {
        "manual_track_point_count_003J2B": int(len(points)),
        "manual_inside_ratio_003J2B": round(float(np.mean(inside)), 6),
        "manual_distance_px_median_003J2B": round(float(np.median(ud)), 3),
        "manual_distance_px_p90_003J2B": round(float(np.percentile(ud, 90)), 3),
        "manual_signed_distance_px_median_003J2B": round(float(np.median(sd)), 3),
        "manual_inside_count_003J2B": int(np.sum(inside)),
        "manual_outside_count_003J2B": int(len(points) - np.sum(inside)),
    }


def classify_manual_table(row: dict) -> tuple[str, list[str]]:
    reasons = []

    valid = boolish(row.get("manual_table_valid_003J1B"))
    if not valid:
        return "NO_MANUAL_TABLE", ["manual_table_invalid"]

    n = int(row.get("manual_track_point_count_003J2B") or 0)
    inside = num(row.get("manual_inside_ratio_003J2B"))
    dist = num(row.get("manual_distance_px_median_003J2B"))
    signed = num(row.get("manual_signed_distance_px_median_003J2B"))

    if n <= 0:
        return "NO_TRACK_POINTS", ["no_track_points"]

    if inside >= 0.50:
        reasons.append("inside_ratio_good")
    elif inside >= 0.20:
        reasons.append("inside_ratio_partial")
    else:
        reasons.append("inside_ratio_low")

    if dist <= 20:
        reasons.append("distance_good")
    elif dist <= 80:
        reasons.append("distance_mid")
    else:
        reasons.append("distance_high")

    if signed >= 0:
        reasons.append("median_signed_inside")
    else:
        reasons.append("median_signed_outside")

    if inside >= 0.50 and dist <= 80:
        return "MANUAL_TABLE_ALIGNED", reasons

    if inside >= 0.20 and dist <= 120:
        return "MANUAL_TABLE_PARTIAL", reasons

    return "MANUAL_TABLE_NOT_ALIGNED", reasons


def build_comparison(row: dict) -> dict:
    old_inside = num(row.get("point_inside_table_quad_ratio"))
    old_dist = num(row.get("point_distance_to_table_px_median"))
    manual_inside = num(row.get("manual_inside_ratio_003J2B"))
    manual_dist = num(row.get("manual_distance_px_median_003J2B"))

    return {
        "delta_inside_manual_minus_old_003J2B": (
            round(manual_inside - old_inside, 6)
            if np.isfinite(old_inside) and np.isfinite(manual_inside)
            else np.nan
        ),
        "delta_distance_old_minus_manual_003J2B": (
            round(old_dist - manual_dist, 3)
            if np.isfinite(old_dist) and np.isfinite(manual_dist)
            else np.nan
        ),
    }


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.aligned{border-color:rgba(116,217,159,.75);box-shadow:inset 4px 0 0 rgba(116,217,159,.85)}
.card.partial{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
.card.bad{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:280px;background:#20242e}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.aligned{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.partial{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
.badge.bad{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
"""

    cards = []

    cols_left = [
        "review_id",
        "target_class",
        "would_reject_shadow_003G",
        "table_reality_status_003I",
        "manual_table_status_003J2B",
        "manual_table_reasons_003J2B",
        "track_csv_source_col_003J2B",
        "track_csv_path_003J2B",
    ]

    cols_right = [
        "point_inside_table_quad_ratio",
        "point_distance_to_table_px_median",
        "manual_inside_ratio_003J2B",
        "manual_distance_px_median_003J2B",
        "manual_distance_px_p90_003J2B",
        "manual_signed_distance_px_median_003J2B",
        "delta_inside_manual_minus_old_003J2B",
        "delta_distance_old_minus_manual_003J2B",
        "manual_track_point_count_003J2B",
        "manual_table_area_ratio_ordered_003J1B",
    ]

    for _, r in df.iterrows():
        status = str(r.get("manual_table_status_003J2B", ""))
        if status == "MANUAL_TABLE_ALIGNED":
            cls = "card aligned"
            badge = "badge aligned"
        elif status == "MANUAL_TABLE_PARTIAL":
            cls = "card partial"
            badge = "badge partial"
        else:
            cls = "card bad"
            badge = "badge bad"

        def table(cols):
            trs = []
            for c in cols:
                if c in r.index:
                    trs.append(f"<tr><th>{c}</th><td>{r.get(c, '')}</td></tr>")
            return "<table><tbody>" + "".join(trs) + "</tbody></table>"

        cards.append(f"""
<div class="{cls}">
<h2>{r.get('review_id', '')}</h2>
<div><span class="{badge}">{status}</span></div>
<div class="grid">
  <div>
    <h3>Contexte</h3>
    {table(cols_left)}
  </div>
  <div>
    <h3>Comparaison ancienne table / table manuelle</h3>
    {table(cols_right)}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003J2B manual table metric recompute</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003J2B manual table metric recompute</h1>

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
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    project_root = Path.cwd()
    run_dir = Path(args.run)

    manual_csv = run_dir / "manual_table_points_003J1B.csv"
    sim_csv = run_dir / "shadow_filter_simulation_003G.csv"

    if not manual_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {manual_csv}")
    if not sim_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {sim_csv}")

    manual_df = pd.read_csv(manual_csv)
    sim_df = pd.read_csv(sim_csv)

    if "review_id" not in manual_df.columns:
        raise SystemExit("Colonne review_id manquante dans manual_table_points_003J1B.csv")
    if "review_id_003G" not in sim_df.columns:
        raise SystemExit("Colonne review_id_003G manquante dans shadow_filter_simulation_003G.csv")

    manual_df["review_id"] = manual_df["review_id"].astype(str)
    sim_df["review_id_003G"] = sim_df["review_id_003G"].astype(str)

    rows = []

    sim_by_id = {str(r["review_id_003G"]): r for _, r in sim_df.iterrows()}

    for _, m in manual_df.iterrows():
        rid = str(m["review_id"])

        if str(m.get("annotation_action", "")).lower() != "saved":
            continue

        s = sim_by_id.get(rid)
        if s is None:
            continue

        row = {}
        for c in s.index:
            row[c] = s[c]
        for c in m.index:
            row[c] = m[c]

        poly = get_manual_polygon(m)

        if poly is None:
            row["manual_table_status_003J2B"] = "NO_MANUAL_TABLE"
            row["manual_table_reasons_003J2B"] = "manual_polygon_invalid"
            rows.append(row)
            continue

        track_path, track_col = choose_track_csv(s, project_root, run_dir)
        row["track_csv_source_col_003J2B"] = track_col
        row["track_csv_path_003J2B"] = str(track_path) if track_path else ""

        if not track_path:
            row["manual_table_status_003J2B"] = "NO_TRACK_CSV"
            row["manual_table_reasons_003J2B"] = "track_csv_not_found"
            rows.append(row)
            continue

        try:
            tdf = pd.read_csv(track_path)
        except Exception as e:
            row["manual_table_status_003J2B"] = "TRACK_CSV_READ_ERROR"
            row["manual_table_reasons_003J2B"] = str(e)
            rows.append(row)
            continue

        x_col, y_col = find_xy_columns(tdf)

        row["track_x_col_003J2B"] = x_col or ""
        row["track_y_col_003J2B"] = y_col or ""

        if not x_col or not y_col:
            row["manual_table_status_003J2B"] = "NO_XY_COLUMNS"
            row["manual_table_reasons_003J2B"] = "x_y_columns_not_found"
            rows.append(row)
            continue

        x = pd.to_numeric(tdf[x_col], errors="coerce").to_numpy(dtype=np.float32)
        y = pd.to_numeric(tdf[y_col], errors="coerce").to_numpy(dtype=np.float32)
        ok = np.isfinite(x) & np.isfinite(y)
        points = np.stack([x[ok], y[ok]], axis=1)

        metrics = point_metrics(points, poly)
        row.update(metrics)

        comparison = build_comparison(row)
        row.update(comparison)

        status, reasons = classify_manual_table(row)
        row["manual_table_status_003J2B"] = status
        row["manual_table_reasons_003J2B"] = ",".join(reasons)

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise SystemExit("Aucune ligne recalculée.")

    by_status = (
        out.groupby("manual_table_status_003J2B")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
    )

    old_by_status = (
        out.groupby("table_reality_status_003I")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
        if "table_reality_status_003I" in out.columns else {}
    )

    aligned_ids = out[out["manual_table_status_003J2B"].eq("MANUAL_TABLE_ALIGNED")]["review_id"].astype(str).tolist()
    partial_ids = out[out["manual_table_status_003J2B"].eq("MANUAL_TABLE_PARTIAL")]["review_id"].astype(str).tolist()
    bad_ids = out[
        ~out["manual_table_status_003J2B"].isin(["MANUAL_TABLE_ALIGNED", "MANUAL_TABLE_PARTIAL"])
    ]["review_id"].astype(str).tolist()

    status = "OK" if aligned_ids or partial_ids else "WARN"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "manual_table_metric_recompute_only_no_arbiter_change",
        "source_manual_csv": str(manual_csv),
        "source_simulation_csv": str(sim_csv),
        "rows_recomputed": int(len(out)),
        "old_table_reality_status_by_id": old_by_status,
        "manual_table_status_by_id": by_status,
        "aligned_ids": aligned_ids,
        "partial_ids": partial_ids,
        "bad_or_error_ids": bad_ids,
        "interpretation": (
            "Si les segments passent de BAD/WEAK à ALIGNED/PARTIAL avec les points manuels, "
            "le problème est bien le modèle table automatique, pas la logique player/kinematic. "
            "On peut ensuite générer un jeu de calibration pour table_detector_v3."
        ),
    }

    out_csv = run_dir / "manual_table_metric_recompute_003J2B.csv"
    out_json = run_dir / "manual_table_metric_recompute_summary_003J2B.json"
    out_html = run_dir / "manual_table_metric_recompute_003J2B.html"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out, summary)

    print(f"003J2B status={status}")
    print("manual_table_status_by_id=" + json.dumps(by_status, ensure_ascii=False))
    print("aligned_ids=" + (",".join(aligned_ids) or "-"))
    print("partial_ids=" + (",".join(partial_ids) or "-"))
    print("bad_or_error_ids=" + (",".join(bad_ids) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and status == "WARN":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
