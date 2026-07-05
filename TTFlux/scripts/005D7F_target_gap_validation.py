from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7F_target_gap_validation"


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
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def read_track_map(path: Path, sequence_key: str | None):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    seq_col = find_col(fields, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError(f"Colonnes frame/x/y introuvables dans {path}")

    if seq_col and sequence_key:
        rows = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]

    out = {}
    dupes = {}

    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue

        src = str(r.get("point_source", "base"))
        item = {
            "frame": fr,
            "x": x,
            "y": y,
            "score": sc,
            "point_source": src,
            "row": r,
        }

        if fr in out:
            dupes.setdefault(fr, [out[fr]]).append(item)
            # Si doublon, préférer le gapfill explicite, sinon garder le premier.
            if src.startswith("gapfill"):
                out[fr] = item
        else:
            out[fr] = item

    return out, {
        "rows_after_sequence_filter": len(rows),
        "field_count": len(fields),
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "seq_col": seq_col,
        "duplicate_frame_count": len(dupes),
        "duplicate_frames": sorted(dupes.keys())[:50],
    }


def load_gap_summary(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    gaps = data.get("gaps", [])
    out = []
    for g in gaps:
        gid = safe_int(g.get("gap_id"))
        start = safe_int(g.get("start_frame", g.get("start")))
        end = safe_int(g.get("end_frame", g.get("end")))
        if gid >= 0 and start >= 0 and end >= start:
            out.append({
                "gap_id": gid,
                "start": start,
                "end": end,
                "len": end - start + 1,
            })

    return data, out


def build_status_rows(gaps, before_map, after_map):
    rows = []
    filled = []
    still_missing = []
    already_present = []
    unexpected_lost = []
    drifted_existing = []

    for g in gaps:
        for fr in range(g["start"], g["end"] + 1):
            b = before_map.get(fr)
            a = after_map.get(fr)

            before_present = b is not None
            after_present = a is not None

            status = ""
            dx = dy = dist = ""

            if not before_present and after_present:
                status = "filled_by_preview"
                filled.append(fr)
            elif not before_present and not after_present:
                status = "still_missing"
                still_missing.append(fr)
            elif before_present and after_present:
                status = "already_present"
                already_present.append(fr)

                dxx = a["x"] - b["x"]
                dyy = a["y"] - b["y"]
                dd = math.hypot(dxx, dyy)
                dx = f"{dxx:.6f}"
                dy = f"{dyy:.6f}"
                dist = f"{dd:.6f}"

                if dd > 0.01:
                    drifted_existing.append(fr)
            elif before_present and not after_present:
                status = "unexpected_lost"
                unexpected_lost.append(fr)

            rows.append({
                "gap_id": g["gap_id"],
                "gap_start": g["start"],
                "gap_end": g["end"],
                "gap_len": g["len"],
                "frame": fr,
                "before_present": int(before_present),
                "after_present": int(after_present),
                "target_status": status,
                "before_x": "" if b is None else f"{b['x']:.3f}",
                "before_y": "" if b is None else f"{b['y']:.3f}",
                "after_x": "" if a is None else f"{a['x']:.3f}",
                "after_y": "" if a is None else f"{a['y']:.3f}",
                "after_source": "" if a is None else a.get("point_source", ""),
                "existing_dx": dx,
                "existing_dy": dy,
                "existing_dist": dist,
            })

    return rows, {
        "target_gap_frames": len(rows),
        "target_gap_missing_before": sum(1 for r in rows if r["before_present"] == 0),
        "target_gap_missing_after": sum(1 for r in rows if r["after_present"] == 0),
        "filled_in_target_gaps": len(filled),
        "still_missing_in_target_gaps": len(still_missing),
        "already_present_in_target_gaps": len(already_present),
        "unexpected_lost_count": len(unexpected_lost),
        "drifted_existing_count": len(drifted_existing),
        "filled_frames": filled,
        "still_missing_frames": still_missing,
        "unexpected_lost_frames": unexpected_lost,
        "drifted_existing_frames": drifted_existing,
    }


def gap_level_summary(gaps, status_rows):
    by_gap = {}
    for g in gaps:
        by_gap[g["gap_id"]] = {
            "gap_id": g["gap_id"],
            "start": g["start"],
            "end": g["end"],
            "len": g["len"],
            "missing_before": 0,
            "missing_after": 0,
            "filled": 0,
            "still_missing": 0,
            "after_sources": {},
        }

    for r in status_rows:
        g = by_gap[r["gap_id"]]
        if r["before_present"] == 0:
            g["missing_before"] += 1
        if r["after_present"] == 0:
            g["missing_after"] += 1
        if r["target_status"] == "filled_by_preview":
            g["filled"] += 1
        if r["target_status"] == "still_missing":
            g["still_missing"] += 1
        src = r.get("after_source", "")
        if src:
            g["after_sources"][src] = g["after_sources"].get(src, 0) + 1

    return [by_gap[k] for k in sorted(by_gap)]


def draw_overlay(video_path: Path, out_path: Path, status_rows, before_map, after_map):
    if not video_path.exists():
        print(f"WARN: vidéo introuvable pour overlay: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARN: impossible d'ouvrir vidéo: {video_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        print(f"WARN: impossible d'écrire overlay: {out_path}")
        return False

    status_by_frame = {safe_int(r["frame"]): r for r in status_rows}
    target_frames = set(status_by_frame)

    frame_idx = -1
    trail = 16

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx not in target_frames:
            continue

        out = frame.copy()

        # Trail before.
        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = before_map.get(fr)
            if p:
                cv2.circle(out, (int(round(p["x"])), int(round(p["y"]))), 3, (150, 150, 150), -1)

        # Trail after.
        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = after_map.get(fr)
            if p:
                src = p.get("point_source", "")
                if src.startswith("gapfill"):
                    cv2.circle(out, (int(round(p["x"])), int(round(p["y"]))), 8, (255, 0, 255), 2)
                else:
                    cv2.circle(out, (int(round(p["x"])), int(round(p["y"]))), 3, (0, 220, 255), -1)

        r = status_by_frame[frame_idx]
        status = r["target_status"]

        if status == "filled_by_preview":
            color = (255, 0, 255)
            label_status = "FILLED"
        elif status == "still_missing":
            color = (0, 0, 255)
            label_status = "STILL_MISSING"
        elif status == "already_present":
            color = (0, 220, 255)
            label_status = "ALREADY_PRESENT"
        else:
            color = (90, 90, 90)
            label_status = status

        ax = safe_float(r.get("after_x"))
        ay = safe_float(r.get("after_y"))
        if math.isfinite(ax) and math.isfinite(ay):
            cv2.circle(out, (int(round(ax)), int(round(ay))), 15, color, 2)
            cv2.putText(out, label_status, (int(round(ax)) + 14, int(round(ay)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        label = f"{PATCH_ID} | frame={frame_idx} | gap={r['gap_id']} | {label_status} | source={r.get('after_source','')}"
        cv2.rectangle(out, (8, 8), (min(w - 8, 1180), 42), (0, 0, 0), -1)
        cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-summary", default="runs/005D7A_gap_candidates/005D7A_gap_summary.json")
    ap.add_argument("--before-track", default="runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv")
    ap.add_argument("--after-track", default="runs/005D7E_gapfilled_preview/005D7E_ball_points_gapfilled_preview.csv")
    ap.add_argument("--video", default="")
    ap.add_argument("--sequence-key", default="")
    ap.add_argument("--out-dir", default="runs/005D7F_target_gap_validation")
    args = ap.parse_args()

    gap_summary_path = Path(args.gap_summary)
    before_track = Path(args.before_track)
    after_track = Path(args.after_track)

    if not gap_summary_path.exists():
        raise FileNotFoundError(f"gap_summary introuvable: {gap_summary_path}")
    if not before_track.exists():
        raise FileNotFoundError(f"before_track introuvable: {before_track}")
    if not after_track.exists():
        raise FileNotFoundError(f"after_track introuvable: {after_track}")

    gap_summary, gaps = load_gap_summary(gap_summary_path)
    sequence_key = args.sequence_key or str(gap_summary.get("chosen_sequence") or "")

    video_path = Path(args.video) if args.video else None
    if video_path is None and gap_summary.get("video"):
        video_path = Path(str(gap_summary.get("video")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_map, before_meta = read_track_map(before_track, sequence_key)
    after_map, after_meta = read_track_map(after_track, sequence_key)

    status_rows, status_summary = build_status_rows(gaps, before_map, after_map)
    per_gap = gap_level_summary(gaps, status_rows)

    status_csv = out_dir / "005D7F_target_gap_frame_status.csv"
    per_gap_csv = out_dir / "005D7F_target_gap_per_gap_summary.csv"
    overlay_path = out_dir / "005D7F_target_gap_validation_overlay.mp4"
    summary_path = out_dir / "005D7F_target_gap_validation_summary.json"

    status_fields = [
        "gap_id", "gap_start", "gap_end", "gap_len", "frame",
        "before_present", "after_present", "target_status",
        "before_x", "before_y", "after_x", "after_y", "after_source",
        "existing_dx", "existing_dy", "existing_dist",
    ]
    write_csv(status_csv, status_rows, status_fields)

    per_gap_fields = [
        "gap_id", "start", "end", "len",
        "missing_before", "missing_after", "filled", "still_missing", "after_sources",
    ]
    write_csv(per_gap_csv, per_gap, per_gap_fields)

    overlay_ok = False
    if video_path is not None:
        overlay_ok = draw_overlay(video_path, overlay_path, status_rows, before_map, after_map)

    ok_for_promotion = (
        status_summary["target_gap_missing_before"] == 47
        and status_summary["filled_in_target_gaps"] == 13
        and status_summary["target_gap_missing_after"] == 34
        and status_summary["unexpected_lost_count"] == 0
        and status_summary["drifted_existing_count"] == 0
    )

    summary = {
        "patch": PATCH_ID,
        "gap_summary": str(gap_summary_path),
        "before_track": str(before_track),
        "after_track": str(after_track),
        "video": str(video_path) if video_path else None,
        "sequence_key": sequence_key,
        "gap_count": len(gaps),
        **status_summary,
        "ok_for_preview_promotion": ok_for_promotion,
        "before_meta": before_meta,
        "after_meta": after_meta,
        "per_gap": per_gap,
        "status_csv": str(status_csv),
        "per_gap_csv": str(per_gap_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "note": "Validation ciblée sur les gaps issus de 005D7A. Pas de remplacement source.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"gap_summary  = {gap_summary_path}")
    print(f"before_track = {before_track}")
    print(f"after_track  = {after_track}")
    print(f"video        = {video_path}")
    print(f"sequence     = {sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005D7F")
    print(f"status_csv  = {status_csv}")
    print(f"per_gap_csv = {per_gap_csv}")
    print(f"summary     = {summary_path}")
    print(f"overlay     = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"target_gap_frames={status_summary['target_gap_frames']}")
    print(f"target_gap_missing_before={status_summary['target_gap_missing_before']}")
    print(f"target_gap_missing_after={status_summary['target_gap_missing_after']}")
    print(f"filled_in_target_gaps={status_summary['filled_in_target_gaps']}")
    print(f"still_missing_in_target_gaps={status_summary['still_missing_in_target_gaps']}")
    print(f"unexpected_lost_count={status_summary['unexpected_lost_count']}")
    print(f"drifted_existing_count={status_summary['drifted_existing_count']}")
    print(f"ok_for_preview_promotion={ok_for_promotion}")


if __name__ == "__main__":
    main()
