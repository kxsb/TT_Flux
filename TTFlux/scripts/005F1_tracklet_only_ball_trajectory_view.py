from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005F1_tracklet_only_ball_trajectory_view"


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


def load_candidates(track_csv: Path, sequence_key: str):
    rows, fields = read_csv(track_csv)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    seq_col = find_col(fields, ["sequence_key", "video_id", "clip_id", "segment_id", "source_id"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    if seq_col and sequence_key:
        rows = [r for r in rows if str(r.get(seq_col, "")).strip() == sequence_key]

    by_frame = {}
    invalid = 0

    for idx, r in enumerate(rows):
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        score = safe_float(r.get(score_col), 0.0) if score_col else 0.0

        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            invalid += 1
            continue

        cand = {
            "idx": idx,
            "frame": fr,
            "x": x,
            "y": y,
            "score": score,
            "row": r,
        }
        by_frame.setdefault(fr, []).append(cand)

    for fr in by_frame:
        by_frame[fr].sort(key=lambda c: c["score"], reverse=True)

    meta = {
        "rows_after_sequence_filter": len(rows),
        "candidate_count": sum(len(v) for v in by_frame.values()),
        "invalid_rows": invalid,
        "candidate_frames": len(by_frame),
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "seq_col": seq_col,
        "score_col": score_col,
        "fields": fields,
    }

    return by_frame, meta


class Tracklet:
    def __init__(self, tid: int, cand: dict):
        self.id = tid
        self.items = [cand]
        self.last_frame = cand["frame"]
        self.last_x = cand["x"]
        self.last_y = cand["y"]
        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0

    def predict(self, frame: int):
        dt = max(1, frame - self.last_frame)
        return self.last_x + self.vx * dt, self.last_y + self.vy * dt, dt

    def add(self, cand: dict):
        dt = max(1, cand["frame"] - self.last_frame)
        nvx = (cand["x"] - self.last_x) / dt
        nvy = (cand["y"] - self.last_y) / dt

        # Lissage léger : éviter les zigzags absurdes.
        self.vx = 0.55 * self.vx + 0.45 * nvx
        self.vy = 0.55 * self.vy + 0.45 * nvy

        self.items.append(cand)
        self.last_frame = cand["frame"]
        self.last_x = cand["x"]
        self.last_y = cand["y"]
        self.missed = 0


def link_tracklets(by_frame, max_gap=2, max_link_px_f=48.0, max_link_abs=95.0):
    frames = sorted(by_frame)
    active = []
    finished = []
    tid = 0

    for fr in frames:
        cands = by_frame[fr]
        assigned_cands = set()
        assigned_tracks = set()

        # Expire les trop vieux.
        still_active = []
        for t in active:
            if fr - t.last_frame > max_gap:
                finished.append(t)
            else:
                still_active.append(t)
        active = still_active

        pairs = []

        for ti, t in enumerate(active):
            px, py, dt = t.predict(fr)
            max_link = min(max_link_abs, max_link_px_f * dt)

            for ci, c in enumerate(cands):
                dist_pred = math.hypot(c["x"] - px, c["y"] - py)

                if dist_pred > max_link:
                    continue

                # coût : priorité à la continuité, score seulement secondaire.
                vx = (c["x"] - t.last_x) / max(1, fr - t.last_frame)
                vy = (c["y"] - t.last_y) / max(1, fr - t.last_frame)
                vel_residual = math.hypot(vx - t.vx, vy - t.vy)

                cost = dist_pred + 0.35 * vel_residual - 10.0 * c["score"]
                pairs.append((cost, ti, ci))

        pairs.sort(key=lambda x: x[0])

        for cost, ti, ci in pairs:
            if ti in assigned_tracks or ci in assigned_cands:
                continue
            t = active[ti]
            c = cands[ci]
            t.add(c)
            assigned_tracks.add(ti)
            assigned_cands.add(ci)

        # Les candidats non assignés démarrent des tracklets invisibles tant qu'ils ne passent pas les critères.
        for ci, c in enumerate(cands):
            if ci in assigned_cands:
                continue
            active.append(Tracklet(tid, c))
            tid += 1

    finished.extend(active)
    return finished


def tracklet_stats(t: Tracklet):
    items = sorted(t.items, key=lambda c: c["frame"])
    frames = [c["frame"] for c in items]
    xs = [c["x"] for c in items]
    ys = [c["y"] for c in items]
    scores = [c["score"] for c in items]

    speeds = []
    travel = 0.0
    large_steps = 0

    for a, b in zip(items[:-1], items[1:]):
        dt = max(1, b["frame"] - a["frame"])
        dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        spd = dist / dt
        speeds.append(spd)
        travel += dist
        if spd > 130 or dist > 150:
            large_steps += 1

    displacement = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) if len(xs) >= 2 else 0.0
    x_range = max(xs) - min(xs) if xs else 0.0
    y_range = max(ys) - min(ys) if ys else 0.0

    if speeds:
        speed_median = float(np.median(speeds))
        speed_p90 = float(np.percentile(speeds, 90))
        speed_max = float(np.max(speeds))
    else:
        speed_median = speed_p90 = speed_max = 0.0

    # Score volontairement conservateur :
    # - favorise longueur + mouvement réel
    # - pénalise statique/logo
    # - pénalise trajectoires trop violentes
    length = len(items)
    avg_score = float(np.mean(scores)) if scores else 0.0
    static_penalty = 50.0 if (x_range + y_range) < 25.0 else 0.0
    violent_penalty = 4.0 * large_steps + 0.025 * max(0.0, speed_p90 - 90.0)

    quality = (
        3.0 * length
        + 0.05 * travel
        + 0.04 * displacement
        + 10.0 * avg_score
        - static_penalty
        - violent_penalty
    )

    return {
        "tracklet_id": t.id,
        "start_frame": frames[0],
        "end_frame": frames[-1],
        "len": length,
        "duration": frames[-1] - frames[0] + 1,
        "frame_density": length / max(1, frames[-1] - frames[0] + 1),
        "travel": travel,
        "displacement": displacement,
        "x_range": x_range,
        "y_range": y_range,
        "score_mean": avg_score,
        "speed_median": speed_median,
        "speed_p90": speed_p90,
        "speed_max": speed_max,
        "large_steps": large_steps,
        "quality": quality,
    }


def select_tracklets(tracklets, min_len, min_travel, max_speed_p90, max_tracklets):
    stats_rows = []
    by_id = {}

    for t in tracklets:
        s = tracklet_stats(t)
        by_id[t.id] = t
        stats_rows.append(s)

    candidates = []
    for s in stats_rows:
        if s["len"] < min_len:
            continue
        if s["travel"] < min_travel:
            continue
        if s["speed_p90"] > max_speed_p90:
            continue
        if s["x_range"] + s["y_range"] < 25:
            continue
        if s["frame_density"] < 0.45:
            continue
        candidates.append(s)

    candidates.sort(key=lambda r: r["quality"], reverse=True)

    selected = []
    occupied = set()

    for s in candidates:
        tid = s["tracklet_id"]
        t = by_id[tid]
        frames = set(c["frame"] for c in t.items)

        overlap = len(frames & occupied) / max(1, len(frames))
        if overlap > 0.25:
            continue

        selected.append(s)
        occupied |= frames

        if len(selected) >= max_tracklets:
            break

    return stats_rows, selected, by_id


def selected_point_map(selected_stats, by_id):
    points = {}
    selected_ids = set(int(s["tracklet_id"]) for s in selected_stats)

    for tid in selected_ids:
        t = by_id[tid]
        for c in t.items:
            fr = c["frame"]
            # Si conflit rare, garder le candidat du tracklet de meilleure qualité déjà sélectionné avant.
            if fr not in points:
                points[fr] = {
                    "frame": fr,
                    "x": c["x"],
                    "y": c["y"],
                    "score": c["score"],
                    "tracklet_id": tid,
                    "row": c["row"],
                }

    return points


def draw_tracklet_only(frame, frame_idx, points, trail=22, show_frame=True):
    out = frame.copy()
    h, w = out.shape[:2]

    # Uniquement trajectoire sélectionnée. Aucun candidat brut.
    prev = None
    prev_tid = None

    for fr in range(frame_idx - trail, frame_idx + 1):
        p = points.get(fr)

        if not p:
            prev = None
            prev_tid = None
            continue

        x = int(round(p["x"]))
        y = int(round(p["y"]))
        tid = p["tracklet_id"]

        if not (0 <= x < w and 0 <= y < h):
            prev = None
            prev_tid = None
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
            color = (0, 150, 150)
            radius = 3
            thick = -1

        if prev is not None and prev_tid == tid:
            cv2.line(out, prev, (x, y), (0, 150, 150), 2)

        cv2.circle(out, (x, y), radius, color, thick)

        prev = (x, y)
        prev_tid = tid

    if show_frame:
        cv2.rectangle(out, (8, 8), (112, 34), (0, 0, 0), -1)
        cv2.putText(out, f"f{frame_idx}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)

    return out


def write_video(video_path: Path, out_path: Path, points):
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
    written = 0
    present = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx in points:
            present += 1

        out = draw_tracklet_only(frame, frame_idx, points)
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
        "visible_track_frames": present,
        "blank_frames": written - present,
    }


def export_selected_points(points, source_fields, out_csv: Path):
    fields = list(source_fields)
    for f in ["tracklet_id", "tracklet_patch", "tracklet_visible"]:
        if f not in fields:
            fields.append(f)

    rows = []
    for fr in sorted(points):
        p = points[fr]
        r = dict(p["row"])
        r["tracklet_id"] = p["tracklet_id"]
        r["tracklet_patch"] = PATCH_ID
        r["tracklet_visible"] = "1"
        rows.append(r)

    write_csv(out_csv, rows, fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/rally_ball_table_relink_005D4/ball_points_table_relinked_005D4.csv")
    ap.add_argument("--video", default="runs/rally_ball_gap_audit_005D5/gap_focus/RLY0074_005D5_gap_focus.mp4")
    ap.add_argument("--sequence-key", default="-0bM0t0qS8Q")
    ap.add_argument("--out-dir", default="runs/005F1_tracklet_only_view")
    ap.add_argument("--max-gap", type=int, default=2)
    ap.add_argument("--max-link-px-f", type=float, default=48.0)
    ap.add_argument("--max-link-abs", type=float, default=95.0)
    ap.add_argument("--min-len", type=int, default=7)
    ap.add_argument("--min-travel", type=float, default=35.0)
    ap.add_argument("--max-speed-p90", type=float, default=115.0)
    ap.add_argument("--max-tracklets", type=int, default=8)
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_frame, meta = load_candidates(track_csv, args.sequence_key)

    tracklets = link_tracklets(
        by_frame,
        max_gap=args.max_gap,
        max_link_px_f=args.max_link_px_f,
        max_link_abs=args.max_link_abs,
    )

    all_stats, selected_stats, by_id = select_tracklets(
        tracklets,
        min_len=args.min_len,
        min_travel=args.min_travel,
        max_speed_p90=args.max_speed_p90,
        max_tracklets=args.max_tracklets,
    )

    points = selected_point_map(selected_stats, by_id)

    all_tracklets_csv = out_dir / "005F1_all_tracklets.csv"
    selected_tracklets_csv = out_dir / "005F1_selected_tracklets.csv"
    selected_points_csv = out_dir / "005F1_selected_tracklet_points.csv"
    video_out = out_dir / "005F1_tracklet_only_trajectory.mp4"
    summary_path = out_dir / "005F1_tracklet_only_summary.json"

    stat_fields = [
        "tracklet_id",
        "start_frame",
        "end_frame",
        "len",
        "duration",
        "frame_density",
        "travel",
        "displacement",
        "x_range",
        "y_range",
        "score_mean",
        "speed_median",
        "speed_p90",
        "speed_max",
        "large_steps",
        "quality",
    ]

    write_csv(all_tracklets_csv, all_stats, stat_fields)
    write_csv(selected_tracklets_csv, selected_stats, stat_fields)
    export_selected_points(points, meta["fields"], selected_points_csv)

    video_meta = write_video(video_path, video_out, points)

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "video": str(video_path),
        "sequence_key": args.sequence_key,
        "candidate_meta": {k: v for k, v in meta.items() if k != "fields"},
        "params": vars(args),
        "raw_tracklet_count": len(tracklets),
        "selected_tracklet_count": len(selected_stats),
        "visible_point_count": len(points),
        "all_tracklets_csv": str(all_tracklets_csv),
        "selected_tracklets_csv": str(selected_tracklets_csv),
        "selected_points_csv": str(selected_points_csv),
        "video_out": str(video_out),
        "video_meta": video_meta,
        "selected_tracklets": selected_stats,
        "note": "Candidats/probas invisibles. Seuls les tracklets sélectionnés sont dessinés.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"video     = {video_path}")
    print(f"sequence  = {args.sequence_key}")
    print("=" * 72)
    print("")
    print("OK 005F1")
    print(f"video                = {video_out}")
    print(f"summary              = {summary_path}")
    print(f"all_tracklets_csv    = {all_tracklets_csv}")
    print(f"selected_tracklets   = {selected_tracklets_csv}")
    print(f"selected_points_csv  = {selected_points_csv}")
    print("")
    print(f"candidate_count={meta['candidate_count']}")
    print(f"candidate_frames={meta['candidate_frames']}")
    print(f"raw_tracklet_count={len(tracklets)}")
    print(f"selected_tracklet_count={len(selected_stats)}")
    print(f"visible_point_count={len(points)}")
    print(f"visible_track_frames={video_meta['visible_track_frames']}")
    print(f"blank_frames={video_meta['blank_frames']}")
    print("")
    print("selected tracklets:")
    for s in selected_stats:
        print(
            f"  id={s['tracklet_id']} "
            f"f={s['start_frame']}->{s['end_frame']} "
            f"len={s['len']} "
            f"travel={s['travel']:.1f} "
            f"p90={s['speed_p90']:.1f} "
            f"q={s['quality']:.1f}"
        )


if __name__ == "__main__":
    main()
