from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


PATCH_ID = "005D7K_promote_secondpass_canonical"


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


def frame_map(rows, fields):
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    out = {}
    dupes = []

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        if fr in out:
            dupes.append(fr)

        out[fr] = (x, y, r)

    return out, sorted(set(dupes)), frame_col, x_col, y_col


def canonicalize(rows, fields):
    points_by_frame = {}
    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    invalid = 0
    duplicates = []

    for idx, r in enumerate(rows):
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            invalid += 1
            continue

        rr = dict(r)
        rr["_input_order"] = idx

        if fr in points_by_frame:
            duplicates.append(fr)
            old = points_by_frame[fr]
            old_src = str(old.get("point_source", ""))
            new_src = str(rr.get("point_source", ""))

            # Priorité au point second-pass, puis au premier-pass, puis à l'existant.
            if new_src.startswith("gapfill_005D7J"):
                points_by_frame[fr] = rr
            elif old_src.startswith("gapfill_005D7J"):
                pass
            elif new_src.startswith("gapfill_005D7E"):
                points_by_frame[fr] = rr
            else:
                pass
        else:
            points_by_frame[fr] = rr

    out = []
    for fr in sorted(points_by_frame):
        r = dict(points_by_frame[fr])
        r.pop("_input_order", None)
        out.append(r)

    return out, {
        "input_rows": len(rows),
        "invalid_rows": invalid,
        "canonical_rows": len(out),
        "duplicate_frame_count_before_resolution": len(set(duplicates)),
        "duplicate_frames_before_resolution_sample": sorted(set(duplicates))[:80],
    }


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
            "largest_steps": [],
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


def compute_missing(per_gap, points):
    present = set(points)
    missing = []
    for g in per_gap:
        for fr in range(int(g["start"]), int(g["end"]) + 1):
            if fr not in present:
                missing.append(fr)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-summary", default="runs/005D7J_secondpass_audit/005D7J_secondpass_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005D7K_gapfilled_canonical_v2")
    args = ap.parse_args()

    audit_summary_path = Path(args.audit_summary)
    if not audit_summary_path.exists():
        raise FileNotFoundError(f"audit_summary introuvable: {audit_summary_path}")

    audit = json.loads(audit_summary_path.read_text(encoding="utf-8"))

    if not audit.get("ok_for_next_promotion", False):
        raise RuntimeError("005D7J n'autorise pas la promotion : ok_for_next_promotion != true")

    preview_csv = Path(audit["preview_csv"])
    track_csv = Path(audit["track_csv"])
    validation_summary = Path(audit["validation_summary"])

    if not preview_csv.exists():
        raise FileNotFoundError(f"preview_csv introuvable: {preview_csv}")
    if not track_csv.exists():
        raise FileNotFoundError(f"track_csv introuvable: {track_csv}")
    if not validation_summary.exists():
        raise FileNotFoundError(f"validation_summary introuvable: {validation_summary}")

    validation = json.loads(validation_summary.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_rows, preview_fields = read_csv(preview_csv)
    before_rows, before_fields = read_csv(track_csv)

    canonical_rows, canon_meta = canonicalize(preview_rows, preview_fields)

    extra_fields = [
        "canonical_patch",
        "canonical_from_audit",
        "canonical_note",
    ]

    out_fields = list(preview_fields)
    for f in extra_fields:
        if f not in out_fields:
            out_fields.append(f)

    for r in canonical_rows:
        r["canonical_patch"] = PATCH_ID
        r["canonical_from_audit"] = str(audit_summary_path)
        src = str(r.get("point_source", ""))
        if src.startswith("gapfill_005D7J"):
            r["canonical_note"] = "promoted_secondpass_gapfill"
        elif src.startswith("gapfill_005D7E"):
            r["canonical_note"] = "promoted_firstpass_gapfill"
        else:
            r["canonical_note"] = "source_representative"

    before_map, before_dupes, *_ = frame_map(before_rows, before_fields)
    after_map, after_dupes, *_ = frame_map(canonical_rows, out_fields)

    per_gap = validation.get("per_gap", [])
    target_missing_after = compute_missing(per_gap, after_map)

    firstpass_frames = [
        safe_int(r.get("frame"))
        for r in canonical_rows
        if str(r.get("point_source", "")).startswith("gapfill_005D7E")
    ]

    secondpass_frames = [
        safe_int(r.get("frame"))
        for r in canonical_rows
        if str(r.get("point_source", "")).startswith("gapfill_005D7J")
    ]

    continuity_before = continuity_stats(before_map)
    continuity_after = continuity_stats(after_map)

    expected_safe_frames = sorted(int(x) for x in audit.get("safe_frames", []))
    expected_missing = sorted(int(x) for x in audit.get("target_missing_after_frames", []))

    promotion_ok = (
        after_dupes == []
        and sorted(secondpass_frames) == expected_safe_frames
        and sorted(target_missing_after) == expected_missing
        and len(canonical_rows) == len(before_map) + len(expected_safe_frames)
        and continuity_after["large_step_count"] <= continuity_before["large_step_count"] + 2
    )

    output_csv = out_dir / "005D7K_ball_points_gapfilled_canonical_unique.csv"
    manifest_path = out_dir / "005D7K_gapfilled_canonical_manifest.json"
    next_path = out_dir / "NEXT_TRACK_CSV.txt"
    audit_copy = out_dir / "005D7J_secondpass_audit_summary.copy.json"
    validation_copy = out_dir / "005D7F_target_gap_validation_summary.copy.json"

    write_csv(output_csv, canonical_rows, out_fields)
    shutil.copy2(audit_summary_path, audit_copy)
    shutil.copy2(validation_summary, validation_copy)
    next_path.write_text(str(output_csv), encoding="utf-8")

    manifest = {
        "patch": PATCH_ID,
        "promotion_ok": promotion_ok,
        "audit_summary": str(audit_summary_path),
        "audit_copy": str(audit_copy),
        "validation_summary": str(validation_summary),
        "validation_copy": str(validation_copy),
        "input_track_csv": str(track_csv),
        "preview_csv": str(preview_csv),
        "output_csv": str(output_csv),
        "canonical_rows": len(canonical_rows),
        "before_rows_unique": len(before_map),
        "expected_rows": len(before_map) + len(expected_safe_frames),
        "firstpass_gapfill_count": len(firstpass_frames),
        "secondpass_gapfill_count": len(secondpass_frames),
        "firstpass_frames": sorted(firstpass_frames),
        "secondpass_frames": sorted(secondpass_frames),
        "expected_secondpass_frames": expected_safe_frames,
        "target_missing_after": target_missing_after,
        "expected_target_missing_after": expected_missing,
        "before_duplicate_frames": before_dupes,
        "after_duplicate_frames": after_dupes,
        "canonicalize_meta": canon_meta,
        "continuity_before": continuity_before,
        "continuity_after": continuity_after,
        "note": "Track canonique v2 : 005D7G + second-pass 005D7J. Ne modifie pas les tracks précédents.",
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"audit_summary = {audit_summary_path}")
    print(f"input_track   = {track_csv}")
    print(f"preview_csv   = {preview_csv}")
    print("=" * 72)
    print("")
    print("OK 005D7K")
    print(f"canonical_csv = {output_csv}")
    print(f"manifest      = {manifest_path}")
    print(f"next_track    = {next_path}")
    print("")
    print(f"promotion_ok={promotion_ok}")
    print(f"canonical_rows={len(canonical_rows)}")
    print(f"before_rows_unique={len(before_map)}")
    print(f"expected_rows={len(before_map) + len(expected_safe_frames)}")
    print(f"firstpass_gapfill_count={len(firstpass_frames)}")
    print(f"secondpass_gapfill_count={len(secondpass_frames)}")
    print(f"secondpass_frames={sorted(secondpass_frames)}")
    print(f"target_missing_after={target_missing_after}")
    print(f"after_duplicate_frames={after_dupes}")
    print(f"continuity_before_large_step_count={continuity_before['large_step_count']}")
    print(f"continuity_after_large_step_count={continuity_after['large_step_count']}")
    print(f"continuity_after_speed_p95={continuity_after['speed_p95']}")


if __name__ == "__main__":
    main()
