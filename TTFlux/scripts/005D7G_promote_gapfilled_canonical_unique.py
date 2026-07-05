from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


PATCH_ID = "005D7G_promote_gapfilled_canonical_unique"


def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=-1):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except Exception:
        return default


def find_col(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def select_sequence_rows(rows, fields, sequence_key):
    seq_col = find_col(fields, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])
    if seq_col and sequence_key:
        return [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key], seq_col
    return rows, seq_col


def choose_frame_representatives(rows, fields):
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    grouped = {}
    invalid_rows = 0

    for idx, r in enumerate(rows):
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            invalid_rows += 1
            continue

        rr = dict(r)
        rr["_input_order"] = idx
        rr["_frame"] = fr
        rr["_x"] = x
        rr["_y"] = y
        rr["_score"] = safe_float(r.get(score_col), 0.0) if score_col else 0.0
        grouped.setdefault(fr, []).append(rr)

    chosen = {}
    duplicate_frames = []

    for fr, items in grouped.items():
        if len(items) > 1:
            duplicate_frames.append(fr)

        # Priorité 1 : point gapfill 005D7E explicite.
        fills = [
            r for r in items
            if str(r.get("point_source", "")).startswith("gapfill_005D7E")
            or str(r.get("fill_patch", "")) == "005D7E_export_gapfilled_preview_track"
        ]

        if fills:
            # Si plusieurs fills improbables, garder le meilleur score.
            best = sorted(fills, key=lambda r: (-safe_float(r.get("fill_strict_score"), r["_score"]), r["_input_order"]))[0]
        else:
            # Important : garder le premier représentant source, pour ne pas modifier le comportement 005D4.
            best = sorted(items, key=lambda r: r["_input_order"])[0]

        chosen[fr] = best

    out = []
    for fr in sorted(chosen):
        r = dict(chosen[fr])
        for k in ["_input_order", "_frame", "_x", "_y", "_score"]:
            r.pop(k, None)
        out.append(r)

    meta = {
        "input_rows": len(rows),
        "valid_rows": sum(len(v) for v in grouped.values()),
        "invalid_rows": invalid_rows,
        "unique_frames": len(out),
        "duplicate_frame_count": len(duplicate_frames),
        "duplicate_frames_sample": sorted(duplicate_frames)[:80],
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "score_col": score_col,
    }

    return out, meta


def frame_map(rows, fields):
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    out = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            out[fr] = (x, y, r)

    return out


def continuity_stats(points):
    frames = sorted(points)
    speeds = []
    large = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue

        ax, ay, _ = points[a]
        bx, by, _ = points[b]
        dist = math.hypot(bx - ax, by - ay)
        speed = dist / dt
        speeds.append(speed)

        if speed > 180 or dist > 260:
            large.append({
                "from": a,
                "to": b,
                "dt": dt,
                "dist": dist,
                "speed": speed,
            })

    if not speeds:
        return {
            "step_count": 0,
            "speed_median": None,
            "speed_p90": None,
            "speed_p95": None,
            "speed_max": None,
            "large_step_count": 0,
        }

    return {
        "step_count": len(speeds),
        "speed_median": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "large_step_count": len(large),
        "largest_steps": sorted(large, key=lambda x: x["speed"], reverse=True)[:20],
    }


def compute_missing_in_ranges(points, ranges):
    present = set(points)
    missing = []
    for g in ranges:
        for fr in range(int(g["start"]), int(g["end"]) + 1):
            if fr not in present:
                missing.append(fr)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-summary", default="runs/005D7F_target_gap_validation/005D7F_target_gap_validation_summary.json")
    ap.add_argument("--out-dir", default="runs/005D7G_gapfilled_canonical")
    args = ap.parse_args()

    validation_summary_path = Path(args.validation_summary)
    if not validation_summary_path.exists():
        raise FileNotFoundError(f"validation_summary introuvable: {validation_summary_path}")

    validation = json.loads(validation_summary_path.read_text(encoding="utf-8"))

    if not validation.get("ok_for_preview_promotion", False):
        raise RuntimeError("005D7F n'autorise pas la promotion : ok_for_preview_promotion != true")

    after_track = Path(validation["after_track"])
    before_track = Path(validation["before_track"])
    sequence_key = str(validation["sequence_key"])
    gap_ranges = [
        {"gap_id": g["gap_id"], "start": g["start"], "end": g["end"], "len": g["len"]}
        for g in validation.get("per_gap", [])
    ]

    if not after_track.exists():
        raise FileNotFoundError(f"after_track introuvable: {after_track}")
    if not before_track.exists():
        raise FileNotFoundError(f"before_track introuvable: {before_track}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    after_rows_all, after_fields = read_csv(after_track)
    before_rows_all, before_fields = read_csv(before_track)

    after_seq_rows, after_seq_col = select_sequence_rows(after_rows_all, after_fields, sequence_key)
    before_seq_rows, before_seq_col = select_sequence_rows(before_rows_all, before_fields, sequence_key)

    canonical_rows, canonical_meta = choose_frame_representatives(after_seq_rows, after_fields)
    before_unique_rows, before_meta = choose_frame_representatives(before_seq_rows, before_fields)

    # Ajouter / stabiliser les colonnes de provenance.
    extra_fields = [
        "canonical_patch",
        "canonical_from_validation",
        "canonical_sequence_key",
        "canonical_note",
    ]
    out_fields = list(after_fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    for r in canonical_rows:
        r["canonical_patch"] = PATCH_ID
        r["canonical_from_validation"] = str(validation_summary_path)
        r["canonical_sequence_key"] = sequence_key
        if str(r.get("point_source", "")).startswith("gapfill"):
            r["canonical_note"] = "promoted_gapfill"
        else:
            r["canonical_note"] = "source_representative"

    canonical_map = frame_map(canonical_rows, out_fields)
    before_map = frame_map(before_unique_rows, before_fields)

    filled_frames_expected = sorted(int(x) for x in validation.get("filled_frames", []))
    still_missing_expected = sorted(int(x) for x in validation.get("still_missing_frames", []))

    filled_present = [fr for fr in filled_frames_expected if fr in canonical_map]
    filled_missing = [fr for fr in filled_frames_expected if fr not in canonical_map]

    target_missing_after = compute_missing_in_ranges(canonical_map, gap_ranges)

    source_count = sum(
        1 for r in canonical_rows
        if not str(r.get("point_source", "")).startswith("gapfill")
    )
    gapfill_count = sum(
        1 for r in canonical_rows
        if str(r.get("point_source", "")).startswith("gapfill")
    )

    duplicate_check = {}
    for r in canonical_rows:
        fr = safe_int(r.get(canonical_meta["frame_col"]))
        duplicate_check[fr] = duplicate_check.get(fr, 0) + 1
    canonical_duplicate_frames = sorted([fr for fr, n in duplicate_check.items() if n > 1])

    continuity = continuity_stats(canonical_map)

    output_csv = out_dir / "005D7G_ball_points_gapfilled_canonical_unique.csv"
    manifest_path = out_dir / "005D7G_gapfilled_canonical_manifest.json"
    next_path_txt = out_dir / "NEXT_TRACK_CSV.txt"
    validation_copy = out_dir / "005D7F_target_gap_validation_summary.copy.json"

    write_csv(output_csv, canonical_rows, out_fields)
    shutil.copy2(validation_summary_path, validation_copy)

    promotion_ok = (
        len(canonical_duplicate_frames) == 0
        and gapfill_count == len(filled_frames_expected)
        and filled_missing == []
        and sorted(target_missing_after) == still_missing_expected
        and len(canonical_rows) == validation.get("target_gap_missing_before", 0) + validation.get("before_meta", {}).get("rows_after_sequence_filter", 0) # informational only, not used below
    )

    # Correction : la condition de taille utile est preview_sequence_points=1129 depuis 005D7E/005D7F logique.
    expected_unique_points = validation.get("target_gap_missing_before", 47)
    # Ici on ne connaît pas source_sequence_points depuis 005D7F, donc on valide sur before_unique + filled.
    expected_unique_points = len(before_map) + len(filled_frames_expected)

    promotion_ok = (
        len(canonical_duplicate_frames) == 0
        and len(canonical_rows) == expected_unique_points
        and gapfill_count == len(filled_frames_expected)
        and filled_missing == []
        and sorted(target_missing_after) == still_missing_expected
    )

    manifest = {
        "patch": PATCH_ID,
        "promotion_ok": promotion_ok,
        "validation_summary": str(validation_summary_path),
        "validation_copy": str(validation_copy),
        "before_track": str(before_track),
        "after_preview_track": str(after_track),
        "sequence_key": sequence_key,
        "output_csv": str(output_csv),
        "canonical_rows": len(canonical_rows),
        "before_unique_rows": len(before_map),
        "expected_unique_rows": expected_unique_points,
        "source_representative_count": source_count,
        "gapfill_promoted_count": gapfill_count,
        "filled_frames_expected": filled_frames_expected,
        "filled_frames_present": filled_present,
        "filled_frames_missing": filled_missing,
        "still_missing_expected": still_missing_expected,
        "target_missing_after_canonical": target_missing_after,
        "canonical_duplicate_frames": canonical_duplicate_frames,
        "canonical_meta": canonical_meta,
        "before_meta": before_meta,
        "continuity_canonical": continuity,
        "note": "Track canonique unique par frame pour la séquence cible. Le 005D4 original n'est pas modifié.",
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    next_path_txt.write_text(str(output_csv), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"validation_summary = {validation_summary_path}")
    print(f"before_track       = {before_track}")
    print(f"after_preview      = {after_track}")
    print(f"sequence           = {sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005D7G")
    print(f"canonical_csv = {output_csv}")
    print(f"manifest      = {manifest_path}")
    print(f"next_track    = {next_path_txt}")
    print("")
    print(f"promotion_ok={promotion_ok}")
    print(f"canonical_rows={len(canonical_rows)}")
    print(f"before_unique_rows={len(before_map)}")
    print(f"expected_unique_rows={expected_unique_points}")
    print(f"gapfill_promoted_count={gapfill_count}")
    print(f"canonical_duplicate_frames={canonical_duplicate_frames}")
    print(f"target_missing_after_canonical={target_missing_after}")
    print(f"continuity_speed_p95={continuity.get('speed_p95')}")
    print(f"continuity_large_step_count={continuity.get('large_step_count')}")


if __name__ == "__main__":
    main()
