from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005F0_rollback_baseline_005D4_policy_compare"


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


def load_grouped_points(track_csv: Path, sequence_key: str):
    rows, fields = read_csv(track_csv)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    seq_col = find_col(fields, ["sequence_key", "video_id", "clip_id", "segment_id", "source_id"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables dans le CSV 005D4.")

    if seq_col and sequence_key:
        rows = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]

    grouped = {}
    invalid = 0

    for idx, r in enumerate(rows):
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        score = safe_float(r.get(score_col), 0.0) if score_col else 0.0

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            invalid += 1
            continue

        rr = dict(r)
        rr["_idx"] = idx
        rr["_frame"] = fr
        rr["_x"] = x
        rr["_y"] = y
        rr["_score"] = score
        grouped.setdefault(fr, []).append(rr)

    meta = {
        "rows_total_after_sequence_filter": len(rows),
        "valid_rows": sum(len(v) for v in grouped.values()),
        "invalid_rows": invalid,
        "unique_frames": len(grouped),
        "duplicate_frame_count": sum(1 for v in grouped.values() if len(v) > 1),
        "max_candidates_same_frame": max((len(v) for v in grouped.values()), default=0),
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "seq_col": seq_col,
        "score_col": score_col,
        "fields": fields,
    }

    return grouped, meta


def choose_policy(grouped, policy):
    chosen = {}

    if policy == "first":
        for fr, items in grouped.items():
            chosen[fr] = sorted(items, key=lambda r: r["_idx"])[0]

    elif policy == "last":
        for fr, items in grouped.items():
            chosen[fr] = sorted(items, key=lambda r: r["_idx"])[-1]

    elif policy == "max_score":
        for fr, items in grouped.items():
            chosen[fr] = sorted(items, key=lambda r: (-r["_score"], r["_idx"]))[0]

    elif policy == "min_jump_greedy":
        frames = sorted(grouped)
        prev = None
        for fr in frames:
            items = grouped[fr]
            if prev is None:
                # Démarrage : score max si possible, sinon premier.
                best = sorted(items, key=lambda r: (-r["_score"], r["_idx"]))[0]
            else:
                px, py = prev["_x"], prev["_y"]

                def cost(r):
                    dist = math.hypot(r["_x"] - px, r["_y"] - py)
                    # léger bonus score, mais priorité à la continuité
                    return dist - 18.0 * r["_score"]

                best = sorted(items, key=cost)[0]

            chosen[fr] = best
            prev = best

    else:
        raise ValueError(policy)

    return chosen


def clean_rows_from_chosen(chosen, fields, policy):
    out = []
    for fr in sorted(chosen):
        r = dict(chosen[fr])
        for k in ["_idx", "_frame", "_x", "_y", "_score"]:
            r.pop(k, None)
        r["baseline_policy"] = policy
        r["baseline_patch"] = PATCH_ID
        out.append(r)

    out_fields = list(fields)
    for f in ["baseline_policy", "baseline_patch"]:
        if f not in out_fields:
            out_fields.append(f)

    return out, out_fields


def stats(chosen):
    frames = sorted(chosen)
    speeds = []
    large = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue

        p0 = chosen[a]
        p1 = chosen[b]
        dx = p1["_x"] - p0["_x"]
        dy = p1["_y"] - p0["_y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt
        speeds.append(speed)

        if speed >= 180.0 or dist >= 260.0 or abs(dx) >= 300.0 or abs(dy) >= 220.0:
            large.append({
                "from": a,
                "to": b,
                "dt": dt,
                "dist": dist,
                "speed": speed,
                "dx": dx,
                "dy": dy,
            })

    if not speeds:
        return {
            "point_count": len(chosen),
            "step_count": 0,
            "speed_median": None,
            "speed_p90": None,
            "speed_p95": None,
            "speed_max": None,
            "large_step_count": 0,
        }

    return {
        "point_count": len(chosen),
        "step_count": len(speeds),
        "speed_median": float(np.median(speeds)),
        "speed_p90": float(np.percentile(speeds, 90)),
        "speed_p95": float(np.percentile(speeds, 95)),
        "speed_max": float(np.max(speeds)),
        "large_step_count": len(large),
        "largest_steps": sorted(large, key=lambda x: x["speed"], reverse=True)[:20],
    }


def draw_simple(frame, frame_idx, chosen, trail=22, label_prefix=""):
    out = frame.copy()
    h, w = out.shape[:2]

    prev = None
    for fr in range(frame_idx - trail, frame_idx + 1):
        p = chosen.get(fr)
        if not p:
            prev = None
            continue

        x = int(round(p["_x"]))
        y = int(round(p["_y"]))

        if not (0 <= x < w and 0 <= y < h):
            prev = None
            continue

        age = frame_idx - fr
        if age == 0:
            color = (255, 255, 255)
            radius = 11
            thick = 2
        elif age <= 5:
            color = (0, 255, 255)
            radius = 5
            thick = -1
        else:
            color = (0, 155, 155)
            radius = 3
            thick = -1

        if prev is not None:
            cv2.line(out, prev, (x, y), (0, 150, 150), 2)

        cv2.circle(out, (x, y), radius, color, thick)
        prev = (x, y)

    cv2.rectangle(out, (8, 8), (420, 36), (0, 0, 0), -1)
    cv2.putText(out, f"{label_prefix} f{frame_idx}", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)

    return out


def write_video(video_path: Path, out_path: Path, chosen, label_prefix):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'écrire vidéo: {out_path}")

    frame_idx = -1
    present = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx in chosen:
            present += 1

        out = draw_simple(frame, frame_idx, chosen, label_prefix=label_prefix)
        writer.write(out)
        written += 1

    cap.release()
    writer.release()

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "video_total_frames": total,
        "written_frames": written,
        "present_frames": present,
        "missing_frames": written - present,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv")
    ap.add_argument("--video", default="runs/rally_ball_gap_audit_005D5/gap_focus/RLY0074_005D5_gap_focus.mp4")
    ap.add_argument("--sequence-key", default="-0bM0t0qS8Q")
    ap.add_argument("--out-dir", default="runs/005F0_rollback_baseline_005D4")
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    grouped, meta = load_grouped_points(track_csv, args.sequence_key)

    policies = ["first", "last", "max_score", "min_jump_greedy"]
    summaries = {}

    for policy in policies:
        chosen = choose_policy(grouped, policy)
        rows, fields = clean_rows_from_chosen(chosen, meta["fields"], policy)

        csv_out = out_dir / f"005F0_baseline_005D4_{policy}.csv"
        video_out = out_dir / f"005F0_baseline_005D4_{policy}_trajectory.mp4"

        write_csv(csv_out, rows, fields)
        video_meta = write_video(video_path, video_out, chosen, label_prefix=f"005D4 {policy}")

        summaries[policy] = {
            "policy": policy,
            "csv": str(csv_out),
            "video": str(video_out),
            "stats": stats(chosen),
            "video_meta": video_meta,
        }

    # Recommandation automatique grossière : minimiser les grands sauts,
    # mais sans prétendre que c'est visuellement vrai.
    recommended_policy = sorted(
        policies,
        key=lambda p: (
            summaries[p]["stats"]["large_step_count"],
            summaries[p]["stats"]["speed_p95"] if summaries[p]["stats"]["speed_p95"] is not None else 99999,
        )
    )[0]

    next_track = out_dir / "NEXT_BASELINE_TRACK_CSV.txt"
    next_track.write_text(summaries[recommended_policy]["csv"], encoding="utf-8")

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "video": str(video_path),
        "sequence_key": args.sequence_key,
        "grouped_meta": {k: v for k, v in meta.items() if k != "fields"},
        "policies": summaries,
        "recommended_policy_by_motion_stats": recommended_policy,
        "next_baseline_track_csv": summaries[recommended_policy]["csv"],
        "note": "Rollback comparatif. Choisir visuellement la meilleure politique avant toute nouvelle étape.",
    }

    summary_path = out_dir / "005F0_rollback_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"video     = {video_path}")
    print(f"sequence  = {args.sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005F0")
    print(f"summary = {summary_path}")
    print(f"next    = {next_track}")
    print("")
    print(f"rows_total_after_sequence_filter={meta['rows_total_after_sequence_filter']}")
    print(f"valid_rows={meta['valid_rows']}")
    print(f"unique_frames={meta['unique_frames']}")
    print(f"duplicate_frame_count={meta['duplicate_frame_count']}")
    print(f"max_candidates_same_frame={meta['max_candidates_same_frame']}")
    print("")
    for policy in policies:
        s = summaries[policy]["stats"]
        print(f"{policy}: video={summaries[policy]['video']}")
        print(f"  points={s['point_count']} large_steps={s['large_step_count']} speed_p95={s['speed_p95']} speed_max={s['speed_max']}")
    print("")
    print(f"recommended_policy_by_motion_stats={recommended_policy}")
    print(f"recommended_csv={summaries[recommended_policy]['csv']}")


if __name__ == "__main__":
    main()
