from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "003J3"

X_COL_CANDIDATES = ["x", "cx", "ball_x", "center_x", "point_x", "px"]
Y_COL_CANDIDATES = ["y", "cy", "ball_y", "center_y", "point_y", "py"]


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))


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


def get_polygon(row: pd.Series) -> np.ndarray | None:
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

    if abs(cv2.contourArea(arr.reshape(-1, 1, 2))) <= 5:
        return None

    return arr


def sample_frames(video_path: Path, max_frames: int = 15) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"error": "VideoCapture failed"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

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
        "video_w": w,
        "video_h": h,
        "sampled": len(frames),
    }


def median_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)

    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    stack = np.stack(resized, axis=0).astype(np.float32)

    return np.median(stack, axis=0).astype(np.uint8)


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

    numeric_cols = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_cols.append(c)

    for c in numeric_cols:
        low = str(c).lower()
        if x_col is None and (low == "x" or low.endswith("_x") or "x_" in low):
            x_col = c
        if y_col is None and (low == "y" or low.endswith("_y") or "y_" in low):
            y_col = c

    return x_col, y_col


def point_polygon_metrics(points: np.ndarray, polygon: np.ndarray) -> dict:
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)

    signed = []
    inside = []

    for x, y in points:
        d = float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
        signed.append(d)
        inside.append(d >= 0)

    if len(points) == 0:
        return {
            "inside_ratio": np.nan,
            "distance_median": np.nan,
            "distance_p90": np.nan,
            "signed_median": np.nan,
        }

    signed = np.array(signed, dtype=np.float32)
    inside = np.array(inside, dtype=bool)
    dist = np.abs(signed)

    return {
        "inside_ratio": round(float(np.mean(inside)), 6),
        "distance_median": round(float(np.median(dist)), 3),
        "distance_p90": round(float(np.percentile(dist, 90)), 3),
        "signed_median": round(float(np.median(signed)), 3),
    }


def coord_sanity(points: np.ndarray, img_w: int, img_h: int) -> dict:
    if len(points) == 0:
        return {
            "coord_status_003J3": "NO_POINTS",
            "out_of_bounds_ratio_003J3": np.nan,
            "coord_x_min_003J3": np.nan,
            "coord_x_max_003J3": np.nan,
            "coord_y_min_003J3": np.nan,
            "coord_y_max_003J3": np.nan,
        }

    x = points[:, 0]
    y = points[:, 1]

    in_bounds = (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h)
    oob_ratio = float(1.0 - np.mean(in_bounds))

    if oob_ratio > 0.50:
        status = "COORD_MISMATCH_LIKELY"
    elif oob_ratio > 0.05:
        status = "COORD_MISMATCH_POSSIBLE"
    else:
        status = "COORD_OK"

    return {
        "coord_status_003J3": status,
        "out_of_bounds_ratio_003J3": round(oob_ratio, 6),
        "coord_x_min_003J3": round(float(np.min(x)), 3),
        "coord_x_max_003J3": round(float(np.max(x)), 3),
        "coord_y_min_003J3": round(float(np.min(y)), 3),
        "coord_y_max_003J3": round(float(np.max(y)), 3),
    }


def draw_overlay(
    frame: np.ndarray,
    polygon: np.ndarray,
    points: np.ndarray,
    rid: str,
    target: str,
    status: str,
    metrics: dict,
    coord_info: dict,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Table manuelle : remplissage léger + contour jaune.
    poly_i = polygon.astype(np.int32).reshape(-1, 1, 2)
    fill = vis.copy()
    cv2.fillPoly(fill, [poly_i], color=(0, 210, 210))
    vis = cv2.addWeighted(vis, 0.78, fill, 0.22, 0)
    cv2.polylines(vis, [poly_i], isClosed=True, color=(0, 255, 255), thickness=3)

    for i, (x, y) in enumerate(polygon):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), -1)
        cv2.circle(vis, (int(x), int(y)), 10, (0, 0, 0), 2)
        cv2.putText(
            vis,
            f"Q{i+1}",
            (int(x) + 10, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Q{i+1}",
            (int(x) + 10, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Track : rouge si hors table, vert si dedans.
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)

    for idx, (x, y) in enumerate(points):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        xi = int(round(float(x)))
        yi = int(round(float(y)))

        if xi < -1000 or yi < -1000 or xi > w + 1000 or yi > h + 1000:
            continue

        d = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        color = (0, 255, 0) if d >= 0 else (0, 0, 255)

        radius = 5
        if idx == 0 or idx == len(points) - 1:
            radius = 8

        cv2.circle(vis, (xi, yi), radius, color, -1)
        cv2.circle(vis, (xi, yi), radius + 1, (0, 0, 0), 1)

        if idx == 0:
            cv2.putText(vis, "START", (xi + 8, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        elif idx == len(points) - 1:
            cv2.putText(vis, "END", (xi + 8, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Trajectoire.
    if len(points) >= 2:
        pts = points[np.isfinite(points).all(axis=1)].astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts.reshape(-1, 1, 2)], isClosed=False, color=(255, 255, 255), thickness=1)

    lines = [
        f"{rid} | {target} | {status}",
        f"inside={metrics.get('inside_ratio')} dist_med={metrics.get('distance_median')} signed_med={metrics.get('signed_median')}",
        f"coord={coord_info.get('coord_status_003J3')} oob={coord_info.get('out_of_bounds_ratio_003J3')}",
        "yellow=manual table | green=track inside | red=track outside",
    ]

    y0 = 28
    for i, text in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(vis, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def write_html(path: Path, df: pd.DataFrame, summary: dict) -> None:
    css = """
body{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}
section,.card{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}
.card.ok{border-color:rgba(116,217,159,.65)}
.card.mismatch{border-color:rgba(255,80,80,.75);box-shadow:inset 4px 0 0 rgba(255,80,80,.85)}
.card.outside{border-color:rgba(255,200,80,.75);box-shadow:inset 4px 0 0 rgba(255,200,80,.85)}
h1,h2,h3{margin-top:0}
pre{white-space:pre-wrap;color:#dce4ff;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-bottom:1px solid #2b303b;padding:8px;text-align:left;vertical-align:top;font-size:12px}
th{width:270px;background:#20242e}
.grid{display:grid;grid-template-columns:430px 1fr;gap:16px}
@media(max-width:950px){.grid{grid-template-columns:1fr}}
.badge{display:inline-block;padding:3px 8px;border:1px solid #3a4050;border-radius:999px;margin-right:6px;font-size:12px;background:#20242e}
.badge.ok{border-color:rgba(116,217,159,.8);background:rgba(116,217,159,.08)}
.badge.mismatch{border-color:rgba(255,80,80,.8);background:rgba(255,80,80,.08)}
.badge.outside{border-color:rgba(255,200,80,.8);background:rgba(255,200,80,.08)}
img{display:block;width:920px;max-width:100%;border:1px solid #2b303b;border-radius:10px;background:#05060a}
"""

    cards = []

    cols = [
        "review_id",
        "target_class",
        "would_reject_shadow_003G",
        "manual_table_status_003J2B",
        "coord_status_003J3",
        "out_of_bounds_ratio_003J3",
        "inside_ratio_003J3",
        "distance_median_003J3",
        "distance_p90_003J3",
        "signed_median_003J3",
        "track_point_count_003J3",
        "track_csv_path_003J2B",
        "video_path",
    ]

    for _, r in df.iterrows():
        coord = str(r.get("coord_status_003J3", ""))
        inside = num(r.get("inside_ratio_003J3"))

        if coord.startswith("COORD_MISMATCH"):
            cls = "card mismatch"
            badge = "badge mismatch"
            badge_text = coord
        elif inside < 0.20:
            cls = "card outside"
            badge = "badge outside"
            badge_text = "TRACK_OUTSIDE_TABLE"
        else:
            cls = "card ok"
            badge = "badge ok"
            badge_text = "TRACK_HAS_TABLE_OVERLAP"

        trs = []
        for c in cols:
            if c in r.index:
                trs.append(f"<tr><th>{c}</th><td>{r.get(c, '')}</td></tr>")

        img = r.get("overlay_png_rel_003J3", "")
        if isinstance(img, str) and img:
            img_html = f"<a href='{img}'><img src='{img}'></a>"
        else:
            img_html = "<p>Pas d'image.</p>"

        cards.append(f"""
<div class="{cls}">
<h2>{r.get('review_id', '')}</h2>
<div><span class="{badge}">{badge_text}</span></div>
<div class="grid">
  <div>
    <h3>Diagnostic</h3>
    <table><tbody>{''.join(trs)}</tbody></table>
  </div>
  <div>
    <h3>Overlay table manuelle + track</h3>
    {img_html}
  </div>
</div>
</div>
""")

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003J3 manual table track visual sanity</title>
<style>{css}</style>
</head>
<body>
<h1>TTFlux · 003J3 manual table track visual sanity</h1>

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

    j2_csv = run_dir / "manual_table_metric_recompute_003J2B.csv"
    if not j2_csv.is_file():
        raise SystemExit(f"Fichier introuvable: {j2_csv}")

    df = pd.read_csv(j2_csv)

    required = [
        "review_id",
        "video_path",
        "track_csv_path_003J2B",
        "ordered_q1_x",
        "ordered_q1_y",
        "ordered_q2_x",
        "ordered_q2_y",
        "ordered_q3_x",
        "ordered_q3_y",
        "ordered_q4_x",
        "ordered_q4_y",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Colonnes manquantes dans 003J2B CSV: " + ", ".join(missing))

    asset_dir = run_dir / "manual_table_track_sanity_003J3_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in df.iterrows():
        rid = str(row["review_id"])
        out = row.to_dict()

        video_path = resolve_path(row.get("video_path"), project_root, run_dir)
        track_path = resolve_path(row.get("track_csv_path_003J2B"), project_root, run_dir)
        polygon = get_polygon(row)

        if video_path is None:
            out["coord_status_003J3"] = "NO_VIDEO"
            rows.append(out)
            continue

        if track_path is None:
            out["coord_status_003J3"] = "NO_TRACK_CSV"
            rows.append(out)
            continue

        if polygon is None:
            out["coord_status_003J3"] = "NO_POLYGON"
            rows.append(out)
            continue

        frames, meta = sample_frames(video_path)
        med = median_frame(frames)

        if med is None:
            out["coord_status_003J3"] = "NO_FRAME"
            rows.append(out)
            continue

        try:
            tdf = pd.read_csv(track_path)
        except Exception as e:
            out["coord_status_003J3"] = "TRACK_READ_ERROR"
            out["coord_error_003J3"] = str(e)
            rows.append(out)
            continue

        x_col, y_col = find_xy_columns(tdf)
        out["track_x_col_003J3"] = x_col or ""
        out["track_y_col_003J3"] = y_col or ""

        if not x_col or not y_col:
            out["coord_status_003J3"] = "NO_XY_COLUMNS"
            rows.append(out)
            continue

        x = pd.to_numeric(tdf[x_col], errors="coerce").to_numpy(dtype=np.float32)
        y = pd.to_numeric(tdf[y_col], errors="coerce").to_numpy(dtype=np.float32)
        ok = np.isfinite(x) & np.isfinite(y)

        points = np.stack([x[ok], y[ok]], axis=1)

        h, w = med.shape[:2]

        coord_info = coord_sanity(points, w, h)
        metrics = point_polygon_metrics(points, polygon)

        out.update(coord_info)
        out["track_point_count_003J3"] = int(len(points))
        out["inside_ratio_003J3"] = metrics["inside_ratio"]
        out["distance_median_003J3"] = metrics["distance_median"]
        out["distance_p90_003J3"] = metrics["distance_p90"]
        out["signed_median_003J3"] = metrics["signed_median"]

        overlay = draw_overlay(
            med,
            polygon,
            points,
            rid,
            str(row.get("target_class", "")),
            str(row.get("manual_table_status_003J2B", "")),
            metrics,
            coord_info,
        )

        png_path = asset_dir / f"{rid}_manual_table_track_overlay_003J3.png"
        imwrite_unicode(png_path, overlay)

        out["overlay_png_003J3"] = str(png_path)
        out["overlay_png_rel_003J3"] = os.path.relpath(png_path, run_dir).replace("\\", "/")

        rows.append(out)

    out_df = pd.DataFrame(rows)

    by_coord = (
        out_df.groupby("coord_status_003J3")["review_id"]
        .apply(lambda s: list(s.astype(str)))
        .to_dict()
        if "coord_status_003J3" in out_df.columns else {}
    )

    mismatch_ids = out_df[
        out_df["coord_status_003J3"].astype(str).str.startswith("COORD_MISMATCH")
    ]["review_id"].astype(str).tolist()

    coord_ok_df = out_df[out_df["coord_status_003J3"].eq("COORD_OK")].copy()
    outside_ids = coord_ok_df[
        pd.to_numeric(coord_ok_df["inside_ratio_003J3"], errors="coerce").fillna(0) < 0.20
    ]["review_id"].astype(str).tolist()

    overlap_ids = coord_ok_df[
        pd.to_numeric(coord_ok_df["inside_ratio_003J3"], errors="coerce").fillna(0) >= 0.20
    ]["review_id"].astype(str).tolist()

    if mismatch_ids:
        status = "WARN_COORD_MISMATCH"
    elif outside_ids:
        status = "OK_TRACK_OUTSIDE_TABLE_CONFIRMED"
    else:
        status = "OK_TABLE_OVERLAP_PRESENT"

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "policy": "visual_coordinate_sanity_only_no_arbiter_change",
        "source_csv": str(j2_csv),
        "rows": int(len(out_df)),
        "coord_status_by_id": by_coord,
        "coord_mismatch_ids": mismatch_ids,
        "coord_ok_track_outside_table_ids": outside_ids,
        "coord_ok_track_overlap_table_ids": overlap_ids,
        "asset_dir": str(asset_dir),
        "interpretation": (
            "Si coord_status=COORD_OK et les points rouges suivent bien la fausse trajectoire, "
            "les rejets 003E/003F sont confirmes par table manuelle. "
            "Si coord mismatch, il faut corriger le repere CSV/image avant toute conclusion."
        ),
    }

    out_csv = run_dir / "manual_table_track_sanity_003J3.csv"
    out_json = run_dir / "manual_table_track_sanity_summary_003J3.json"
    out_html = run_dir / "manual_table_track_sanity_003J3.html"

    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out_df, summary)

    print(f"003J3 status={status}")
    print("coord_status_by_id=" + json.dumps(by_coord, ensure_ascii=False))
    print("coord_mismatch_ids=" + (",".join(mismatch_ids) or "-"))
    print("coord_ok_track_outside_table_ids=" + (",".join(outside_ids) or "-"))
    print("coord_ok_track_overlap_table_ids=" + (",".join(overlap_ids) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    if args.strict and status == "WARN_COORD_MISMATCH":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
