from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "003R"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def resolve_path(value, root: Path, run_dir: Path) -> Path | None:
    if value is None:
        return None

    raw = str(value).strip().strip('"').strip("'")
    if not raw or raw.lower() == "nan":
        return None

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    candidates = [p] if p.is_absolute() else [root / p, run_dir / p]

    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except Exception:
            pass

    return None


def find_col(cols: list[str], names: list[str], contains: list[str] = []) -> str | None:
    low = {c.lower(): c for c in cols}

    for n in names:
        if n.lower() in low:
            return low[n.lower()]

    for c in cols:
        cl = c.lower()
        if any(s in cl for s in contains):
            return c

    return None


def load_manifest_rows(run_dir: Path, root: Path) -> list[dict]:
    candidates = [
        run_dir / "review_manifest_001T2_trajectory.csv",
        run_dir / "operational_manifest_001T2.csv",
        run_dir / "review_manifest_001T_raw.csv",
        run_dir / "review_manifest_001F.csv",
    ]

    manifest = None
    for p in candidates:
        if p.is_file():
            manifest = p
            break

    rows = []

    if manifest:
        df = pd.read_csv(manifest)
        cols = list(df.columns)

        rid_col = find_col(cols, ["review_id", "id"], ["review"])
        mp4_col = find_col(cols, ["mp4", "video", "video_path", "segment_mp4"], ["mp4"])
        csv_col = find_col(cols, ["csv", "track_csv", "segment_csv"], ["csv"])
        target_col = find_col(cols, ["target_class_003G", "target_class", "decision", "guess"], ["target", "decision", "guess"])

        for i, row in df.iterrows():
            rid = str(row.get(rid_col, f"R{i+1:04d}")).strip() if rid_col else f"R{i+1:04d}"

            csv_path = resolve_path(row.get(csv_col), root, run_dir) if csv_col else None
            mp4_path = resolve_path(row.get(mp4_col), root, run_dir) if mp4_col else None

            if csv_path is None and mp4_path is not None:
                guess_csv = mp4_path.with_suffix(".csv")
                if guess_csv.is_file():
                    csv_path = guess_csv.resolve()

            if mp4_path is None and csv_path is not None:
                guess_mp4 = csv_path.with_suffix(".mp4")
                if guess_mp4.is_file():
                    mp4_path = guess_mp4.resolve()

            if csv_path is None:
                continue

            rows.append({
                "review_id": rid,
                "target_class_003G": str(row.get(target_col, "")).strip() if target_col else "",
                "csv": str(csv_path),
                "mp4": str(mp4_path) if mp4_path else "",
                "manifest_source_003R": str(manifest),
            })

    if rows:
        return rows

    # Fallback brut : tous les validation_*.csv hors assets.
    for i, p in enumerate(sorted(run_dir.rglob("validation_*.csv")), start=1):
        if "assets" in {x.lower() for x in p.parts}:
            continue
        mp4 = p.with_suffix(".mp4")
        rows.append({
            "review_id": f"R{i:04d}",
            "target_class_003G": "",
            "csv": str(p.resolve()),
            "mp4": str(mp4.resolve()) if mp4.is_file() else "",
            "manifest_source_003R": "recursive_validation_csv_fallback",
        })

    return rows


def load_config_map(config_dir: Path) -> dict[str, dict]:
    out = {}

    if not config_dir.is_dir():
        return out

    for p in sorted(config_dir.glob("*.json")):
        if p.name.startswith("batch_"):
            continue

        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        out[p.stem] = cfg

        clip_id = str(cfg.get("clip_id", "")).strip()
        if clip_id:
            out[clip_id] = cfg

    return out


def config_for_row(row: dict, config_map: dict[str, dict]) -> dict | None:
    csv_path = Path(row["csv"])
    parent = csv_path.parent.name

    if parent in config_map:
        return config_map[parent]

    # parent = 01_v62_xxx ; clip_id = tout après le premier "_"
    parts = parent.split("_", 1)
    if len(parts) == 2 and parts[1] in config_map:
        return config_map[parts[1]]

    for key, cfg in config_map.items():
        clip_id = str(cfg.get("clip_id", ""))
        if clip_id and clip_id in parent:
            return cfg

    return None


def read_track_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)

    frame_col = find_col(cols, ["frame", "frame_idx", "f"], ["frame"])
    x_col = find_col(cols, ["x", "cx", "ball_x", "center_x", "point_x", "px"], ["x"])
    y_col = find_col(cols, ["y", "cy", "ball_y", "center_y", "point_y", "py"], ["y"])

    if not frame_col or not x_col or not y_col:
        raise RuntimeError(f"Colonnes frame/x/y introuvables dans {path}")

    out = pd.DataFrame({
        "frame": pd.to_numeric(df[frame_col], errors="coerce"),
        "x": pd.to_numeric(df[x_col], errors="coerce"),
        "y": pd.to_numeric(df[y_col], errors="coerce"),
    }).dropna()

    out = out.sort_values("frame").reset_index(drop=True)
    return out


def max_accel(track: pd.DataFrame) -> float:
    pts = track[["x", "y"]].to_numpy(dtype=np.float32)

    if len(pts) < 3:
        return np.nan

    acc = pts[2:] - 2 * pts[1:-1] + pts[:-2]
    vals = np.linalg.norm(acc, axis=1)

    if len(vals) == 0:
        return np.nan

    return float(np.nanmax(vals))


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    return cap


def get_frame(cap, frame_idx: int):
    if frame_idx < 0:
        frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()

    if not ok or frame is None:
        return None

    return frame


def detect_table_mask(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Table souvent bleue / cyan / verte. On reste large, puis on filtre par taille.
    lower1 = np.array([35, 35, 35], dtype=np.uint8)
    upper1 = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)

    # Évite le public/score en haut, et les bandes extrêmes.
    roi = np.zeros_like(mask)
    roi[int(h * 0.18): int(h * 0.95), int(w * 0.05): int(w * 0.95)] = 255
    mask = cv2.bitwise_and(mask, roi)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label = 0
    best_score = 0.0

    for lab in range(1, num_labels):
        x, y, ww, hh, area = stats[lab]

        if area < 2500:
            continue

        cx = x + ww / 2
        cy = y + hh / 2

        center_bonus = 1.0 - min(1.0, abs(cx - w / 2) / (w / 2))
        vertical_bonus = 1.0 - min(1.0, abs(cy - h * 0.58) / (h * 0.58))
        shape_bonus = min(2.0, ww / max(1, hh))

        score = float(area) * (0.5 + center_bonus) * (0.5 + vertical_bonus) * shape_bonus

        if score > best_score:
            best_score = score
            best_label = lab

    out = np.zeros_like(mask)

    if best_label:
        out[labels == best_label] = 255

        kernel2 = np.ones((13, 13), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel2)

    return out


def table_distance_median(video_path: Path, track: pd.DataFrame, use_absolute_frames: bool) -> tuple[float, int]:
    cap = open_video(video_path)
    if cap is None:
        return np.nan, 0

    try:
        if use_absolute_frames:
            med_frame = int(np.nanmedian(track["frame"].to_numpy()))
        else:
            med_frame = max(0, len(track) // 2)

        frame = get_frame(cap, med_frame)
        if frame is None:
            return np.nan, 0

        mask = detect_table_mask(frame)

        if int(mask.sum()) == 0:
            return np.nan, 0

        inside = mask > 0
        inv = (~inside).astype(np.uint8) * 255
        dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

        h, w = mask.shape[:2]
        distances = []

        for _, r in track.iterrows():
            x = int(round(num(r["x"])))
            y = int(round(num(r["y"])))

            if 0 <= x < w and 0 <= y < h:
                distances.append(float(dist_map[y, x]))

        if not distances:
            return np.nan, int(mask.sum() // 255)

        return float(np.nanmedian(distances)), int(mask.sum() // 255)

    finally:
        cap.release()


def point_in_big_motion_component(mask: np.ndarray, x: int, y: int, min_area: int = 650) -> bool:
    h, w = mask.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        return False

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    lab = labels[y, x]

    if lab > 0 and stats[lab, cv2.CC_STAT_AREA] >= min_area:
        return True

    # Cherche autour du point.
    r = 8
    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    labs = np.unique(labels[y0:y1, x0:x1])

    for lab in labs:
        if lab > 0 and stats[lab, cv2.CC_STAT_AREA] >= min_area:
            return True

    return False


def motion_ratio(video_path: Path, track: pd.DataFrame, use_absolute_frames: bool, max_points: int = 24) -> tuple[float, int]:
    cap = open_video(video_path)
    if cap is None:
        return np.nan, 0

    try:
        if len(track) == 0:
            return np.nan, 0

        sample = track.copy()
        if len(sample) > max_points:
            idxs = np.linspace(0, len(sample) - 1, max_points).round().astype(int)
            sample = sample.iloc[idxs].reset_index(drop=True)

        hits = 0
        total = 0

        kernel = np.ones((11, 11), np.uint8)

        for local_idx, r in sample.iterrows():
            if use_absolute_frames:
                f = int(round(num(r["frame"])))
            else:
                f = int(local_idx)

            f0 = max(0, f - 2)
            f1 = f + 2

            a = get_frame(cap, f0)
            b = get_frame(cap, f1)

            if a is None or b is None:
                continue

            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

            ga = cv2.GaussianBlur(ga, (7, 7), 0)
            gb = cv2.GaussianBlur(gb, (7, 7), 0)

            diff = cv2.absdiff(ga, gb)
            _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, kernel, iterations=2)

            x = int(round(num(r["x"])))
            y = int(round(num(r["y"])))

            if point_in_big_motion_component(mask, x, y):
                hits += 1

            total += 1

        if total == 0:
            return np.nan, 0

        return float(hits / total), total

    finally:
        cap.release()


def write_html(path: Path, rows: list[dict], summary: dict) -> None:
    trs = []

    for r in rows:
        hit = r.get("multi_object_shadow_hit_003E")
        cls = "hit" if hit else "miss"

        trs.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r.get('review_id'))}</td>"
            f"<td>{esc(r.get('target_class_003G'))}</td>"
            f"<td>{esc(r.get('multi_object_shadow_hit_003E'))}</td>"
            f"<td>{esc(r.get('inside_player_motion_mask_ratio_003C'))}</td>"
            f"<td>{esc(r.get('point_distance_to_table_px_median'))}</td>"
            f"<td>{esc(r.get('max_accel'))}</td>"
            f"<td>{esc(r.get('003R_status'))}</td>"
            f"<td>{esc(r.get('mp4'))}</td>"
            f"<td>{esc(r.get('csv'))}</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TTFlux 003R feature bridge</title>
<style>
body{{margin:0;padding:24px;background:#101218;color:#edf0f7;font-family:system-ui,Segoe UI,sans-serif}}
section{{background:#181b22;border:1px solid #2b303b;border-radius:16px;padding:16px;margin-bottom:16px}}
pre{{white-space:pre-wrap;background:#101218;border:1px solid #2b303b;border-radius:10px;padding:10px}}
table{{width:100%;border-collapse:collapse}}
td,th{{border-bottom:1px solid #2b303b;padding:8px;text-align:left;font-size:12px;vertical-align:top}}
th{{background:#20242e}}
tr.hit td{{background:rgba(255,200,80,.06)}}
a{{color:#b9cdfa}}
</style>
</head>
<body>
<h1>TTFlux · 003R feature bridge for 003N</h1>
<section>
<h2>Résumé</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</section>
<section>
<h2>Segments</h2>
<table>
<thead>
<tr>
<th>review_id</th>
<th>target</th>
<th>003R hit</th>
<th>player motion ratio</th>
<th>table dist</th>
<th>max accel</th>
<th>status</th>
<th>mp4</th>
<th>csv</th>
</tr>
</thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
</section>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--player-min", type=float, default=0.60)
    ap.add_argument("--table-distance-max", type=float, default=210.0)
    ap.add_argument("--accel-min", type=float, default=20.0)
    args = ap.parse_args()

    root = Path.cwd()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = root / config_dir

    rows_in = load_manifest_rows(run_dir, root)
    config_map = load_config_map(config_dir)

    out_rows = []

    for i, row in enumerate(rows_in, start=1):
        status = "OK"

        csv_path = Path(row["csv"])
        mp4_path = Path(row["mp4"]) if row.get("mp4") else None

        try:
            track = read_track_csv(csv_path)
        except Exception as exc:
            track = pd.DataFrame()
            status = f"TRACK_READ_ERROR:{repr(exc)}"

        cfg = config_for_row(row, config_map)
        source_video = None
        use_absolute_frames = False

        if cfg:
            vp = resolve_path(cfg.get("video_path"), root, run_dir)
            if vp:
                source_video = vp
                use_absolute_frames = True

        if source_video is None and mp4_path and mp4_path.is_file():
            source_video = mp4_path
            use_absolute_frames = False
            status = status if status != "OK" else "OK_USING_SEGMENT_MP4"

        ma = np.nan
        motion = np.nan
        motion_n = 0
        table_dist = np.nan
        table_mask_area = 0

        if len(track):
            ma = max_accel(track)

        if source_video is not None and len(track):
            try:
                motion, motion_n = motion_ratio(source_video, track, use_absolute_frames)
            except Exception as exc:
                status = f"{status};MOTION_ERROR:{repr(exc)}"

            try:
                table_dist, table_mask_area = table_distance_median(source_video, track, use_absolute_frames)
            except Exception as exc:
                status = f"{status};TABLE_ERROR:{repr(exc)}"

        else:
            status = f"{status};NO_VIDEO_OR_TRACK"

        player_pass = np.isfinite(motion) and motion >= args.player_min
        table_pass = np.isfinite(table_dist) and table_dist <= args.table_distance_max
        accel_pass = np.isfinite(ma) and ma >= args.accel_min

        rule_hit = bool(player_pass and table_pass and accel_pass)

        out = {
            **row,
            "review_id": row.get("review_id") or f"R{i:04d}",
            "003R_status": status,
            "003R_source_video": str(source_video) if source_video else "",
            "003R_use_absolute_frames": use_absolute_frames,
            "003R_track_point_count": int(len(track)),
            "inside_player_motion_mask_ratio_003C": round(float(motion), 6) if np.isfinite(motion) else np.nan,
            "point_distance_to_table_px_median": round(float(table_dist), 3) if np.isfinite(table_dist) else np.nan,
            "max_accel": round(float(ma), 6) if np.isfinite(ma) else np.nan,
            "003R_motion_points_used": motion_n,
            "003R_table_mask_area": int(table_mask_area),
            "003R_player_pass": bool(player_pass),
            "003R_table_pass": bool(table_pass),
            "003R_accel_pass": bool(accel_pass),
            "multi_object_shadow_hit_003E": bool(rule_hit),
            "would_reject_shadow_003G": bool(rule_hit),
            "003R_feature_note": "proxy bridge, not historical 003D",
        }

        out_rows.append(out)

        print(
            f"[003R] {i:02d}/{len(rows_in)} {out['review_id']} "
            f"hit={rule_hit} motion={out['inside_player_motion_mask_ratio_003C']} "
            f"table={out['point_distance_to_table_px_median']} accel={out['max_accel']} "
            f"status={status}"
        )

    out_csv = run_dir / "multi_object_arbiter_shadow_003E.csv"
    out_json = run_dir / "feature_bridge_summary_003R.json"
    out_html = run_dir / "feature_bridge_003R.html"

    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    hit_ids = df[df["multi_object_shadow_hit_003E"].astype(bool)]["review_id"].astype(str).tolist()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "OK",
        "policy": "proxy_feature_bridge_for_frozen_shadow_rule_no_live_filter_no_delete",
        "run_dir": str(run_dir),
        "config_dir": str(config_dir),
        "rows": int(len(df)),
        "hit_total": int(len(hit_ids)),
        "hit_ids": hit_ids,
        "rule_proxy": {
            "inside_player_motion_mask_ratio_003C_min": args.player_min,
            "point_distance_to_table_px_median_max": args.table_distance_max,
            "max_accel_min": args.accel_min,
        },
        "warning": "003R computes proxy features; not identical to original 003D/003C pipeline.",
        "out_csv_for_003N": str(out_csv),
        "out_html": str(out_html),
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_html, out_rows, summary)

    print("")
    print("003R status=OK")
    print("rows=", len(df))
    print("hit_total=", len(hit_ids))
    print("hit_ids=" + (",".join(hit_ids) or "-"))
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
