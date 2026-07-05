
from __future__ import annotations

import ast
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttflux.core.paths import project_root, legacy_root


DEFAULT_SCENE_TABLE_REL = Path("runs") / "rally_scene_table_objects_005C9B" / "scene_table_objects_005C9B.csv"
DEFAULT_TABLE_PROJECTED_BALL_REL = Path("runs") / "rally_scene_table_objects_005C9B" / "ball_points_scene_table_projected_005C9B.csv"
DEFAULT_REFERENCE_FIT_REL = Path("runs") / "rally_table_reference_fit_005C9A" / "table_reference_fit_005C9A.csv"
DEFAULT_PROJECTION_VOTES_REL = Path("runs") / "rally_table_projection_vote_005C8A" / "table_projection_votes_005C8A.csv"


def scene_table_csv_path() -> Path:
    return legacy_root() / DEFAULT_SCENE_TABLE_REL


def projected_ball_csv_path() -> Path:
    return legacy_root() / DEFAULT_TABLE_PROJECTED_BALL_REL


def reference_fit_csv_path() -> Path:
    return legacy_root() / DEFAULT_REFERENCE_FIT_REL


def projection_votes_csv_path() -> Path:
    return legacy_root() / DEFAULT_PROJECTION_VOTES_REL


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []

    encodings = ["utf-8-sig", "utf-8", "cp1252"]
    last_exc: Exception | None = None

    for enc in encodings:
        try:
            rows: list[dict[str, str]] = []
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
                    if limit is not None and len(rows) >= limit:
                        break
            return rows
        except Exception as exc:
            last_exc = exc

    if last_exc:
        raise last_exc

    return []


def _num(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def detect_frame_col(columns: list[str]) -> str | None:
    exact = [
        "frame",
        "video_frame",
        "frame_idx",
        "local_frame",
        "source_frame",
        "full_frame",
        "abs_frame",
        "image_frame",
    ]
    low = {c.lower(): c for c in columns}

    for name in exact:
        if name in low:
            return low[name]

    for c in columns:
        cl = c.lower()
        if cl.endswith("_frame") or cl.startswith("frame_") or cl == "f":
            return c

    for c in columns:
        if "frame" in c.lower():
            return c

    return None


def detect_interval_cols(columns: list[str]) -> list[tuple[str, str, str]]:
    low = {c.lower(): c for c in columns}
    pairs: list[tuple[str, str, str]] = []

    candidates = [
        ("local", "local_start", "local_end"),
        ("full", "full_start", "full_end"),
        ("frame", "first_frame", "last_frame"),
        ("frame", "start_frame", "end_frame"),
        ("range", "range_start", "range_end"),
        ("video", "video_start", "video_end"),
    ]

    for label, a, b in candidates:
        if a in low and b in low:
            pairs.append((label, low[a], low[b]))

    for c in columns:
        cl = c.lower()
        if cl.endswith("_start"):
            base = cl[:-6]
            end = base + "_end"
            if end in low:
                pair = (base, c, low[end])
                if pair not in pairs:
                    pairs.append(pair)

    return pairs


def detect_role_cols(columns: list[str]) -> list[str]:
    preferred = [
        "status",
        "quality",
        "quality_label",
        "object_status",
        "table_status",
        "object_type",
        "type",
        "kind",
        "label",
        "class",
        "scene_class",
        "trust",
        "review",
    ]
    low = {c.lower(): c for c in columns}
    out = []

    for p in preferred:
        if p in low:
            out.append(low[p])

    for c in columns:
        cl = c.lower()
        if c not in out and any(k in cl for k in ["status", "quality", "class", "label", "trust"]):
            out.append(c)

    return out[:8]



def detect_xy_pairs(columns: list[str]) -> list[tuple[str, str, str]]:
    cols = list(columns)
    low_map = {c.lower(): c for c in cols}
    pairs: list[tuple[str, str, str]] = []

    def add(label: str, xcol: str | None, ycol: str | None) -> None:
        if not xcol or not ycol:
            return

        if xcol not in cols:
            xcol = low_map.get(str(xcol).lower())
        if ycol not in cols:
            ycol = low_map.get(str(ycol).lower())

        if not xcol or not ycol:
            return

        key = (xcol, ycol)
        if any((a, b) == key for _, a, b in pairs):
            return

        bad_terms = ["range", "velocity", "speed", "score", "prob", "conf", "count", "width", "height"]
        if any(t in xcol.lower() for t in bad_terms) or any(t in ycol.lower() for t in bad_terms):
            return

        pairs.append((label, xcol, ycol))

    # Paires simples.
    add("xy", low_map.get("x"), low_map.get("y"))
    add("center", low_map.get("center_x"), low_map.get("center_y"))
    add("center", low_map.get("cx"), low_map.get("cy"))
    add("image", low_map.get("image_x"), low_map.get("image_y"))
    add("point", low_map.get("px"), low_map.get("py"))

    # Cas prioritaire TTFlux legacy :
    # quad_tl_x_005C9B / quad_tl_y_005C9B
    # quad_tr_x_005C1 / quad_tr_y_005C1
    corner_order = ["tl", "tr", "br", "bl", "top_left", "top_right", "bottom_right", "bottom_left"]
    suffixes = ["005C9B", "005C9A", "005C8B", "005C8A", "005C7B", "005C7A", "005C6B", "005C5B", "005C3", "005C1"]

    for suffix in suffixes:
        for corner in corner_order:
            add(
                f"quad_{corner}_{suffix}",
                low_map.get(f"quad_{corner}_x_{suffix}".lower()),
                low_map.get(f"quad_{corner}_y_{suffix}".lower()),
            )
            add(
                f"{corner}_{suffix}",
                low_map.get(f"{corner}_x_{suffix}".lower()),
                low_map.get(f"{corner}_y_{suffix}".lower()),
            )

    # Cas générique : prefix_x_suffix -> prefix_y_suffix.
    # Exemple : quad_tl_x_005C9B -> quad_tl_y_005C9B.
    for c in cols:
        cl = c.lower()

        m = re.match(r"^(?P<prefix>.+)_x(?P<suffix>_[a-z0-9]+)$", cl)
        if m:
            y_low = f"{m.group('prefix')}_y{m.group('suffix')}"
            ycol = low_map.get(y_low)
            if ycol:
                label = f"{m.group('prefix')}{m.group('suffix')}"
                add(label, c, ycol)

        m = re.match(r"^(?P<prefix>.+)x(?P<suffix>_[a-z0-9]+)$", cl)
        if m:
            y_low = f"{m.group('prefix')}y{m.group('suffix')}"
            ycol = low_map.get(y_low)
            if ycol:
                label = f"{m.group('prefix')}{m.group('suffix')}"
                add(label, c, ycol)

        # Cas non suffixé classique : something_x -> something_y.
        if cl.endswith("_x"):
            base = c[:-2]
            y = base + "_y"
            add(base, c, y if y in cols else low_map.get(y.lower()))

        # Cas x0/y0.
        m = re.match(r"^x([0-9]+)$", cl)
        if m:
            y = "y" + m.group(1)
            add("p" + m.group(1), c, low_map.get(y))

        # Cas p0_x/p0_y.
        m = re.match(r"^(.*?)([0-9]+)_x$", cl)
        if m:
            prefix = m.group(1)
            idx = m.group(2)
            y = f"{prefix}{idx}_y"
            add(f"{prefix}{idx}", c, low_map.get(y))

    # Réordonner si on a les 4 coins table.
    def rank(pair: tuple[str, str, str]) -> tuple[int, str]:
        label = pair[0].lower()
        xcol = pair[1].lower()
        key = label + " " + xcol

        if "quad_tl" in key or "_tl_" in key or "top_left" in key:
            return (0, label)
        if "quad_tr" in key or "_tr_" in key or "top_right" in key:
            return (1, label)
        if "quad_br" in key or "_br_" in key or "bottom_right" in key:
            return (2, label)
        if "quad_bl" in key or "_bl_" in key or "bottom_left" in key:
            return (3, label)

        return (50, label)

    pairs.sort(key=rank)
    return pairs[:64]


def _safe_parse_obj(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return None

    if not any(ch in s for ch in ["[", "{", "(", "]", "}"]):
        return None

    tries = [s]

    if "'" in s and '"' not in s:
        tries.append(s.replace("'", '"'))

    for candidate in tries:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _collect_points(obj: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def walk(x: Any) -> None:
        if x is None:
            return

        if isinstance(x, dict):
            keys = {str(k).lower(): k for k in x.keys()}

            x_key = None
            y_key = None
            for a in ["x", "u", "px", "image_x", "screen_x"]:
                if a in keys:
                    x_key = keys[a]
                    break
            for b in ["y", "v", "py", "image_y", "screen_y"]:
                if b in keys:
                    y_key = keys[b]
                    break

            if x_key is not None and y_key is not None:
                xv = _num(x.get(x_key))
                yv = _num(x.get(y_key))
                if xv is not None and yv is not None:
                    points.append((xv, yv))

            for v in x.values():
                walk(v)

        elif isinstance(x, (list, tuple)):
            if len(x) >= 2:
                xv = _num(x[0])
                yv = _num(x[1])
                if xv is not None and yv is not None:
                    points.append((xv, yv))
                    return
            for v in x:
                walk(v)

    walk(obj)

    clean: list[tuple[float, float]] = []
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if abs(x) > 10000 or abs(y) > 10000:
            continue
        key = (round(x, 3), round(y, 3))
        if key not in [(round(a, 3), round(b, 3)) for a, b in clean]:
            clean.append((x, y))

    return clean[:16]


def _geometry_from_explicit_pairs(row: dict[str, str], xy_pairs: list[tuple[str, str, str]]) -> list[tuple[float, float]]:
    pts = []
    for label, xcol, ycol in xy_pairs:
        x = _num(row.get(xcol))
        y = _num(row.get(ycol))
        if x is None or y is None:
            continue
        pts.append((x, y))
    return pts


def _geometry_from_jsonish(row: dict[str, str]) -> tuple[list[tuple[float, float]], str]:
    preferred_terms = [
        "quad",
        "corner",
        "polygon",
        "poly",
        "points",
        "geometry",
        "project",
        "image",
        "table",
        "contour",
        "mask",
    ]

    scored_cols = []
    for col, value in row.items():
        if not isinstance(value, str) or not value.strip():
            continue
        cl = col.lower()
        score = 0
        for term in preferred_terms:
            if term in cl:
                score += 2
        if any(ch in value for ch in ["[", "{", "("]):
            score += 1
        if score > 0:
            scored_cols.append((score, col, value))

    scored_cols.sort(reverse=True)

    for _, col, value in scored_cols:
        obj = _safe_parse_obj(value)
        pts = _collect_points(obj)
        if len(pts) >= 3:
            return pts, col

    return [], ""


def detect_homography_cols(columns: list[str]) -> list[str]:
    low = {c.lower(): c for c in columns}

    patterns = [
        ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"],
        ["h_00", "h_01", "h_02", "h_10", "h_11", "h_12", "h_20", "h_21", "h_22"],
        ["homography_00", "homography_01", "homography_02", "homography_10", "homography_11", "homography_12", "homography_20", "homography_21", "homography_22"],
        ["H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21", "H22"],
    ]

    for names in patterns:
        found = []
        for name in names:
            key = name.lower()
            if key not in low:
                break
            found.append(low[key])
        if len(found) == 9:
            return found

    loose = []
    for c in columns:
        cl = c.lower()
        if re.match(r"^(h|homography)[_ ]?[0-2][_ ]?[0-2]$", cl):
            loose.append(c)

    if len(loose) >= 9:
        return sorted(loose)[:9]

    return []



def _geometry_from_homography(row: dict[str, str], homography_cols: list[str]) -> list[tuple[float, float]]:
    H = None

    # Format 1 : 9 colonnes h00..h22.
    if len(homography_cols) == 9:
        vals = [_num(row.get(c)) for c in homography_cols]
        if not any(v is None for v in vals):
            H = np.array(vals, dtype=np.float64).reshape(3, 3)

    # Format 2 : une colonne JSON/list string, ex H_table_to_img_005C9B.
    if H is None:
        preferred = [
            "H_table_to_img_005C9B",
            "H_table_to_img_005C1",
            "H_table_to_img_005C5B",
            "H_table_to_img",
            "h_table_to_img",
        ]
        candidates = []

        for name in preferred:
            if name in row:
                candidates.append((name, row.get(name)))

        for k, v in row.items():
            kl = k.lower()
            if "table_to_img" in kl or "homography" in kl:
                candidates.append((k, v))

        for _, value in candidates:
            obj = _safe_parse_obj(value)
            if isinstance(obj, (list, tuple)) and len(obj) == 9:
                vals = [_num(v) for v in obj]
                if not any(v is None for v in vals):
                    H = np.array(vals, dtype=np.float64).reshape(3, 3)
                    break

    if H is None:
        return []

    # Table officielle : 2.74m x 1.525m.
    world_sets = [
        np.array([[0, 0, 1], [2.74, 0, 1], [2.74, 1.525, 1], [0, 1.525, 1]], dtype=np.float64).T,
        np.array([[-1.37, -0.7625, 1], [1.37, -0.7625, 1], [1.37, 0.7625, 1], [-1.37, 0.7625, 1]], dtype=np.float64).T,
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64).T,
    ]

    best: list[tuple[float, float]] = []
    best_score = -1

    for W in world_sets:
        try:
            P = H @ W
            P = P[:2, :] / P[2:3, :]
            pts = [(float(P[0, i]), float(P[1, i])) for i in range(P.shape[1])]
            sane = sum(1 for x, y in pts if -3000 <= x <= 5000 and -3000 <= y <= 4000)
            span_x = max(x for x, _ in pts) - min(x for x, _ in pts)
            span_y = max(y for _, y in pts) - min(y for _, y in pts)
            score = sane + (1 if 50 <= span_x <= 3000 else 0) + (1 if 20 <= span_y <= 2000 else 0)
            if score > best_score:
                best_score = score
                best = pts
        except Exception:
            continue

    return best if best_score >= 4 else []


def extract_row_geometry(
    row: dict[str, str],
    xy_pairs: list[tuple[str, str, str]] | None = None,
    homography_cols: list[str] | None = None,
) -> tuple[list[tuple[float, float]], str]:
    if xy_pairs is None:
        xy_pairs = detect_xy_pairs(list(row.keys()))
    if homography_cols is None:
        homography_cols = detect_homography_cols(list(row.keys()))

    pts = _geometry_from_explicit_pairs(row, xy_pairs)
    if len(pts) >= 3:
        return pts, "explicit_xy_pairs"

    pts, source_col = _geometry_from_jsonish(row)
    if len(pts) >= 3:
        return pts, f"jsonish:{source_col}"

    pts = _geometry_from_homography(row, homography_cols)
    if len(pts) >= 3:
        return pts, "homography"

    return [], ""


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    columns = list(rows[0].keys()) if rows else []
    frame_col = detect_frame_col(columns)
    interval_cols = detect_interval_cols(columns)
    xy_pairs = detect_xy_pairs(columns)
    role_cols = detect_role_cols(columns)
    homography_cols = detect_homography_cols(columns)

    frame_values: list[int] = []
    for r in rows:
        if frame_col:
            v = _num(r.get(frame_col))
            if v is not None:
                frame_values.append(int(round(v)))

    interval_values: list[tuple[int, int]] = []
    for r in rows:
        for _, a, b in interval_cols:
            va = _num(r.get(a))
            vb = _num(r.get(b))
            if va is not None and vb is not None:
                interval_values.append((int(round(va)), int(round(vb))))

    role_counts: dict[str, dict[str, int]] = {}
    for col in role_cols:
        c = Counter()
        for r in rows:
            value = str(r.get(col, "")).strip()
            if value:
                c[value] += 1
        role_counts[col] = dict(c.most_common(30))

    geometry_rows = 0
    geometry_sources = Counter()
    geometry_samples = []

    for r in rows:
        pts, source = extract_row_geometry(r, xy_pairs=xy_pairs, homography_cols=homography_cols)
        if len(pts) >= 3:
            geometry_rows += 1
            geometry_sources[source] += 1
            if len(geometry_samples) < 5:
                geometry_samples.append({
                    "source": source,
                    "points": [[round(x, 3), round(y, 3)] for x, y in pts[:8]],
                    "row_keys": {k: r.get(k, "") for k in columns[:16]},
                })

    return {
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "columns": columns,
        "frame_col": frame_col,
        "frame_min": min(frame_values) if frame_values else None,
        "frame_max": max(frame_values) if frame_values else None,
        "frame_unique_count": len(set(frame_values)) if frame_values else 0,
        "interval_cols": [{"label": label, "start": a, "end": b} for label, a, b in interval_cols],
        "interval_min": min([a for a, b in interval_values] + [b for a, b in interval_values]) if interval_values else None,
        "interval_max": max([a for a, b in interval_values] + [b for a, b in interval_values]) if interval_values else None,
        "xy_pairs": [{"label": label, "x": x, "y": y} for label, x, y in xy_pairs],
        "homography_cols": homography_cols,
        "role_cols": role_cols,
        "role_counts": role_counts,
        "geometry_rows": geometry_rows,
        "geometry_sources": dict(geometry_sources.most_common()),
        "geometry_samples": geometry_samples,
        "sample": rows[:5],
    }


def build_scene_table_report(output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = project_root() / "runs" / "007E_scene_table_import"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_summary = summarize_csv(scene_table_csv_path())
    projected_ball_summary = summarize_csv(projected_ball_csv_path())
    reference_fit_summary = summarize_csv(reference_fit_csv_path())
    projection_votes_summary = summarize_csv(projection_votes_csv_path())

    payload = {
        "patch": "007E4_TTFluxV2_fix_suffixed_quad_table_overlay",
        "created_at": time.time(),
        "project_root": str(project_root()),
        "legacy_root": str(legacy_root()),
        "scene_table_csv": scene_summary,
        "projected_ball_csv": projected_ball_summary,
        "reference_fit_csv": reference_fit_summary,
        "projection_votes_csv": projection_votes_summary,
    }

    json_path = output_dir / "scene_table_import_summary_007E.json"
    schema_path = output_dir / "scene_table_schema_probe_007E2.json"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    schema_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    payload["outputs"] = {
        "summary_json": str(json_path),
        "schema_probe_json": str(schema_path),
    }

    return payload


def _row_matches_frame(row: dict[str, str], frame_idx: int, frame_col: str | None, interval_cols: list[tuple[str, str, str]], tol: int) -> bool:
    if frame_col:
        v = _num(row.get(frame_col))
        if v is not None and abs(int(round(v)) - frame_idx) <= tol:
            return True

    for _, a, b in interval_cols:
        va = _num(row.get(a))
        vb = _num(row.get(b))
        if va is None or vb is None:
            continue
        start = min(int(round(va)), int(round(vb)))
        end = max(int(round(va)), int(round(vb)))
        if start - tol <= frame_idx <= end + tol:
            return True

    return False



def _path_match(row: dict[str, str], video_path: str | None) -> bool:
    if not video_path:
        return True

    vp = str(video_path).strip().lower().replace("/", "\\")
    if not vp:
        return True

    vp_path = Path(video_path)
    vp_name = vp_path.name.lower()
    vp_stem = vp_path.stem.lower()

    for key in ["clip_path", "clip_path_vote", "video_path", "path"]:
        rv = str(row.get(key, "")).strip()
        if not rv:
            continue

        rp = rv.lower().replace("/", "\\")
        rp_path = Path(rv)
        rp_name = rp_path.name.lower()
        rp_stem = rp_path.stem.lower()

        if rp == vp:
            return True
        if rp_name and rp_name == vp_name:
            return True
        if rp_stem and rp_stem == vp_stem:
            return True
        if rp.endswith("\\" + vp_name) or vp.endswith("\\" + rp_name):
            return True

    return False


def load_scene_rows_near_frame(
    frame_idx: int,
    tol: int = 0,
    limit: int = 80,
    video_path: str | None = None,
) -> dict[str, Any]:
    path = scene_table_csv_path()
    rows = _read_csv_rows(path)
    columns = list(rows[0].keys()) if rows else []
    frame_col = detect_frame_col(columns)
    interval_cols = detect_interval_cols(columns)
    xy_pairs = detect_xy_pairs(columns)
    homography_cols = detect_homography_cols(columns)

    frame_idx = int(frame_idx)
    tol = max(0, int(tol))

    path_filtered = [r for r in rows if _path_match(r, video_path)]
    if not path_filtered and video_path:
        path_filtered = rows

    matched = []

    for r in path_filtered:
        if frame_col or interval_cols:
            if not _row_matches_frame(r, frame_idx, frame_col, interval_cols, tol):
                continue

        pts, source = extract_row_geometry(r, xy_pairs=xy_pairs, homography_cols=homography_cols)
        rr = dict(r)
        rr["_geometry_source"] = source
        rr["_geometry_points"] = [[float(x), float(y)] for x, y in pts]
        rr["_has_geometry"] = len(pts) >= 3
        rr["_path_match"] = _path_match(r, video_path)
        matched.append(rr)

        if len(matched) >= limit:
            break

    # Si aucune colonne frame/intervalle, une ligne par clip suffit.
    if not matched and path_filtered:
        for r in path_filtered:
            pts, source = extract_row_geometry(r, xy_pairs=xy_pairs, homography_cols=homography_cols)
            if len(pts) < 3:
                continue

            rr = dict(r)
            rr["_geometry_source"] = source
            rr["_geometry_points"] = [[float(x), float(y)] for x, y in pts]
            rr["_has_geometry"] = True
            rr["_path_match"] = _path_match(r, video_path)
            rr["_fallback_path_match_no_frame"] = True
            matched.append(rr)

            if video_path and rr["_path_match"]:
                break

            if len(matched) >= min(limit, 12):
                break

    return {
        "path": str(path),
        "video_path": video_path or "",
        "frame_col": frame_col,
        "interval_cols": [{"label": label, "start": a, "end": b} for label, a, b in interval_cols],
        "target_frame": frame_idx,
        "tol": tol,
        "xy_pairs": [{"label": label, "x": x, "y": y} for label, x, y in xy_pairs],
        "homography_cols": homography_cols,
        "rows": matched,
        "row_count": len(matched),
        "geometry_count": sum(1 for r in matched if r.get("_has_geometry")),
        "path_filtered_count": len(path_filtered),
    }


def _valid_image_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int]]:
    out = []
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if not (-width * 0.6 <= x <= width * 1.6 and -height * 0.6 <= y <= height * 1.6):
            continue
        out.append((int(round(x)), int(round(y))))
    return out


def draw_scene_rows(frame: np.ndarray, scene_payload: dict[str, Any]) -> np.ndarray:
    out = frame.copy()
    rows = scene_payload.get("rows") or []

    h, w = out.shape[:2]
    drawn = 0

    for row_idx, row in enumerate(rows[:40]):
        raw_pts = row.get("_geometry_points") or []
        points = _valid_image_points([(float(x), float(y)) for x, y in raw_pts], w, h)

        if len(points) < 3:
            continue

        drawn += 1

        # Si trop de points, on dessine les 4 premiers comme contour principal.
        contour = points[:4] if len(points) >= 4 else points

        if len(contour) >= 2:
            for i in range(len(contour) - 1):
                cv2.line(out, contour[i], contour[i + 1], (0, 215, 255), 2, cv2.LINE_AA)
            if len(contour) >= 3:
                cv2.line(out, contour[-1], contour[0], (0, 215, 255), 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(points[:8]):
            cv2.circle(out, (x, y), 6, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(out, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 245, 180), 1, cv2.LINE_AA)

        if row_idx < 8 and points:
            x, y = points[0]
            text = str(row.get("_geometry_source") or "legacy_table")
            for k in ["status", "quality_label", "object_type", "kind", "label", "class", "trust"]:
                if row.get(k):
                    text = f"{row.get(k)} | {text}"
                    break
            cv2.putText(out, text[:42], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text[:42], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 245, 180), 1, cv2.LINE_AA)

    if rows:
        cv2.rectangle(out, (8, 8), (430, 34), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"legacy scene table rows: {len(rows)} | drawn geometry: {drawn}",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 245, 180),
            1,
            cv2.LINE_AA,
        )

    return out


if __name__ == "__main__":
    report = build_scene_table_report()
    scene = report["scene_table_csv"]
    ball = report["projected_ball_csv"]
    ref = report["reference_fit_csv"]
    votes = report["projection_votes_csv"]

    print("OK 007E4")
    print(f"scene_csv={scene['path']}")
    print(f"scene_exists={scene['exists']}")
    print(f"scene_rows={scene['row_count']}")
    print(f"scene_frame_col={scene['frame_col']}")
    print(f"scene_interval_cols={scene['interval_cols']}")
    print(f"scene_xy_pairs={scene['xy_pairs']}")
    print(f"scene_homography_cols={scene['homography_cols']}")
    print(f"scene_geometry_rows={scene['geometry_rows']}")
    print(f"scene_geometry_sources={scene['geometry_sources']}")
    print(f"projected_ball_rows={ball['row_count']}")
    print(f"reference_fit_rows={ref['row_count']} geometry_rows={ref['geometry_rows']} sources={ref['geometry_sources']}")
    print(f"projection_votes_rows={votes['row_count']} geometry_rows={votes['geometry_rows']} sources={votes['geometry_sources']}")
    print(f"summary_json={report['outputs']['summary_json']}")
    print(f"schema_probe_json={report['outputs']['schema_probe_json']}")


def _scene_row_score(row: dict[str, str]) -> tuple[int, float, float]:
    status = str(row.get("scene_table_status_005C9B") or "")
    metric = int(_num(row.get("scene_table_metric_usable_005C9B")) or 0)
    review = int(_num(row.get("scene_table_review_usable_005C9B")) or 0)
    partial = int(_num(row.get("scene_table_partial_mask_usable_005C9B")) or 0)
    rating = float(_num(row.get("human_rating_005C8A")) or 0)
    opt = float(_num(row.get("opt_score_005C9A")) or 0)

    status_score = {
        "METRIC_TABLE_TRUSTED": 4,
        "METRIC_TABLE_REVIEW": 3,
        "PARTIAL_TABLE_MASK": 2,
        "NO_TABLE_METRIC": 1,
    }.get(status, 0)

    return (metric * 100 + review * 50 + partial * 20 + status_score * 5, rating, opt)


def list_scene_table_clips(
    status_filter: str = "",
    only_existing: bool = True,
    limit: int = 250,
) -> dict[str, Any]:
    rows = _read_csv_rows(scene_table_csv_path())
    columns = list(rows[0].keys()) if rows else []
    xy_pairs = detect_xy_pairs(columns)
    homography_cols = detect_homography_cols(columns)

    out = []
    status_filter = str(status_filter or "").strip()

    for row in rows:
        clip_path = str(row.get("clip_path") or "").strip()
        if not clip_path:
            continue

        if status_filter and str(row.get("scene_table_status_005C9B") or "") != status_filter:
            continue

        p = Path(clip_path)
        exists = p.exists()

        if only_existing and not exists:
            continue

        pts, source = extract_row_geometry(row, xy_pairs=xy_pairs, homography_cols=homography_cols)

        record = {
            "review_id": row.get("review_id", ""),
            "rally_id": row.get("rally_id", ""),
            "video_id": row.get("video_id", ""),
            "camera_segment_id": row.get("camera_segment_id", ""),
            "clip_path": clip_path,
            "clip_name": p.name,
            "exists": exists,
            "scene_table_status_005C9B": row.get("scene_table_status_005C9B", ""),
            "scene_table_source_005C9B": row.get("scene_table_source_005C9B", ""),
            "metric_quality_005C7B": row.get("metric_quality_005C7B", ""),
            "metric_reason_005C7B": row.get("metric_reason_005C7B", ""),
            "human_rating_005C8A": row.get("human_rating_005C8A", ""),
            "scene_table_metric_usable_005C9B": row.get("scene_table_metric_usable_005C9B", ""),
            "scene_table_review_usable_005C9B": row.get("scene_table_review_usable_005C9B", ""),
            "scene_table_partial_mask_usable_005C9B": row.get("scene_table_partial_mask_usable_005C9B", ""),
            "quad_area_px_005C9B": row.get("quad_area_px_005C9B", ""),
            "geometry_source": source,
            "geometry_points": [[round(float(x), 3), round(float(y), 3)] for x, y in pts],
            "has_geometry": len(pts) >= 3,
            "_score_tuple": _scene_row_score(row),
        }
        out.append(record)

    out.sort(key=lambda r: r["_score_tuple"], reverse=True)

    for r in out:
        r.pop("_score_tuple", None)

    out = out[: max(1, min(int(limit), 1000))]

    return {
        "path": str(scene_table_csv_path()),
        "total": len(out),
        "status_filter": status_filter,
        "only_existing": only_existing,
        "clips": out,
        "status_counts": dict(Counter(str(r.get("scene_table_status_005C9B") or "") for r in out)),
    }


def find_scene_row_for_video(video_path: str) -> dict[str, str] | None:
    rows = _read_csv_rows(scene_table_csv_path())
    if not rows:
        return None

    exact = [r for r in rows if _path_match(r, video_path)]
    candidates = exact if exact else rows

    # Si path non trouvé, on évite de retourner n'importe quoi sauf si une seule ligne plausible.
    if not exact and video_path:
        return None

    candidates.sort(key=_scene_row_score, reverse=True)
    return candidates[0] if candidates else None


def _projected_ball_rows_for_scene(scene_row: dict[str, str]) -> list[dict[str, str]]:
    rows = _read_csv_rows(projected_ball_csv_path())

    review_id = str(scene_row.get("review_id") or "")
    rally_id = str(scene_row.get("rally_id") or "")
    video_id = str(scene_row.get("video_id") or "")

    out = []
    for r in rows:
        if review_id and str(r.get("review_id") or "") != review_id:
            continue
        if rally_id and str(r.get("rally_id") or "") != rally_id:
            continue
        if video_id and str(r.get("video_id") or "") != video_id:
            continue
        if str(r.get("table_project_ok_005C9B") or "") not in {"1", "1.0", "True", "true"}:
            continue

        x = _num(r.get("ball_table_x_m_005C9B"))
        y = _num(r.get("ball_table_y_m_005C9B"))
        f = _num(r.get("frame"))
        if x is None or y is None or f is None:
            continue

        rr = dict(r)
        rr["_table_x"] = x
        rr["_table_y"] = y
        rr["_frame"] = int(round(f))
        out.append(rr)

    out.sort(key=lambda r: int(r.get("_frame") or 0))
    return out


def topdown_payload_for_video(video_path: str) -> dict[str, Any]:
    scene_row = find_scene_row_for_video(video_path)
    if not scene_row:
        return {
            "ok": False,
            "error": "scene_row_not_found_for_video",
            "video_path": video_path,
            "ball_count": 0,
            "scene": None,
            "ball_points": [],
        }

    ball_rows = _projected_ball_rows_for_scene(scene_row)

    pts = []
    for r in ball_rows:
        pts.append({
            "frame": int(r.get("_frame") or 0),
            "x_m": float(r.get("_table_x") or 0),
            "y_m": float(r.get("_table_y") or 0),
            "inside": str(r.get("ball_inside_table_005C9B") or ""),
            "side": str(r.get("ball_table_side_005C9B") or ""),
            "state": str(r.get("state_005A") or r.get("state_004Y") or ""),
        })

    return {
        "ok": True,
        "video_path": video_path,
        "scene": {
            "review_id": scene_row.get("review_id", ""),
            "rally_id": scene_row.get("rally_id", ""),
            "video_id": scene_row.get("video_id", ""),
            "status": scene_row.get("scene_table_status_005C9B", ""),
            "metric_quality": scene_row.get("metric_quality_005C7B", ""),
            "human_rating": scene_row.get("human_rating_005C8A", ""),
            "clip_path": scene_row.get("clip_path", ""),
        },
        "ball_count": len(pts),
        "ball_points": pts,
    }


def _map_table_point_to_canvas(x_m: float, y_m: float, width: int, height: int, margin: int) -> tuple[int, int]:
    table_l = 2.74
    table_w = 1.525

    # Support deux conventions :
    # 1) coords coin : x in [0, 2.74], y in [0, 1.525]
    # 2) coords centre : x in [-1.37, 1.37], y in [-0.7625, 0.7625]
    x = float(x_m)
    y = float(y_m)

    if -1.6 <= x <= 1.6 and -1.0 <= y <= 1.0:
        x = x + table_l / 2.0
        y = y + table_w / 2.0

    sx = (width - 2 * margin) / table_l
    sy = (height - 2 * margin) / table_w
    scale = min(sx, sy)

    draw_w = table_l * scale
    draw_h = table_w * scale
    x0 = (width - draw_w) / 2.0
    y0 = (height - draw_h) / 2.0

    px = int(round(x0 + x * scale))
    py = int(round(y0 + y * scale))
    return px, py


def render_topdown_for_video(video_path: str, frame: int | None = None, width: int = 900, height: int = 560) -> bytes:
    payload = topdown_payload_for_video(video_path)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (12, 18, 32)

    margin = 55
    table_l = 2.74
    table_w = 1.525

    tl = _map_table_point_to_canvas(0, 0, width, height, margin)
    tr = _map_table_point_to_canvas(table_l, 0, width, height, margin)
    br = _map_table_point_to_canvas(table_l, table_w, width, height, margin)
    bl = _map_table_point_to_canvas(0, table_w, width, height, margin)

    cv2.fillConvexPoly(img, np.array([tl, tr, br, bl], dtype=np.int32), (30, 75, 105))
    cv2.polylines(img, [np.array([tl, tr, br, bl], dtype=np.int32)], True, (225, 235, 245), 2, cv2.LINE_AA)

    # Ligne centrale et filet.
    mid_y1 = _map_table_point_to_canvas(0, table_w / 2, width, height, margin)
    mid_y2 = _map_table_point_to_canvas(table_l, table_w / 2, width, height, margin)
    cv2.line(img, mid_y1, mid_y2, (210, 225, 235), 1, cv2.LINE_AA)

    net_a = _map_table_point_to_canvas(table_l / 2, 0, width, height, margin)
    net_b = _map_table_point_to_canvas(table_l / 2, table_w, width, height, margin)
    cv2.line(img, net_a, net_b, (80, 220, 255), 3, cv2.LINE_AA)

    title = "TTFlux V2 top-down table"
    if payload.get("ok"):
        scene = payload.get("scene") or {}
        title = f"{scene.get('review_id','')} | {scene.get('status','')} | ball={payload.get('ball_count',0)}"
    else:
        title = f"no topdown data: {payload.get('error','unknown')}"

    cv2.putText(img, title[:95], (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

    pts = payload.get("ball_points") or []
    mapped = []
    for p in pts:
        x = _num(p.get("x_m"))
        y = _num(p.get("y_m"))
        if x is None or y is None:
            continue

        # Tolérance hors table pour voir trajectoire qui sort.
        if not (-2.5 <= x <= 4.5 and -2.5 <= y <= 3.5):
            continue

        px, py = _map_table_point_to_canvas(x, y, width, height, margin)
        mapped.append((px, py, int(p.get("frame") or 0), str(p.get("inside") or "")))

    for i in range(1, len(mapped)):
        cv2.line(img, mapped[i - 1][:2], mapped[i][:2], (120, 180, 255), 1, cv2.LINE_AA)

    for i, (px, py, fr, inside) in enumerate(mapped):
        if i % max(1, len(mapped) // 180 + 1) != 0:
            continue
        color = (80, 220, 255) if inside in {"1", "1.0", "True", "true"} else (130, 130, 255)
        cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)

    if frame is not None:
        nearest = None
        best_d = 10**9
        for px, py, fr, inside in mapped:
            d = abs(fr - int(frame))
            if d < best_d:
                nearest = (px, py, fr, inside)
                best_d = d

        if nearest and best_d <= 15:
            px, py, fr, inside = nearest
            cv2.circle(img, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), 5, (255, 120, 30), -1, cv2.LINE_AA)
            cv2.putText(img, f"f{fr}", (px + 12, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Légende coords.
    cv2.putText(img, "table 2.74m x 1.525m | projected ball points 005C9B", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 195), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")

    return buf.tobytes()

