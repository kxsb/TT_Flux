from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005E0_canonical_jump_audit"


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


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def read_track(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(fields, ["score", "conf", "confidence", "prob"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables.")

    points = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            points[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "score": sc,
                "source": str(r.get("point_source", "")) or "source",
                "row": r,
            }

    return points, {
        "rows": len(rows),
        "valid_points": len(points),
        "fields": fields,
        "frame_col": frame_col,
        "x_col": x_col,
        "y_col": y_col,
        "score_col": score_col,
    }


def build_steps(points):
    frames = sorted(points)
    steps = []

    for a, b in zip(frames[:-1], frames[1:]):
        dt = b - a
        if dt <= 0:
            continue

        p0 = points[a]
        p1 = points[b]

        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt

        steps.append({
            "from_frame": a,
            "to_frame": b,
            "dt": dt,
            "from_x": p0["x"],
            "from_y": p0["y"],
            "to_x": p1["x"],
            "to_y": p1["y"],
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "speed_px_f": speed,
            "from_source": p0["source"],
            "to_source": p1["source"],
            "from_score": p0["score"],
            "to_score": p1["score"],
        })

    return steps


def classify_step(s, speed_hard, dist_hard):
    reasons = []

    if s["dt"] > 1:
        reasons.append("cross_gap")

    if s["speed_px_f"] >= speed_hard:
        reasons.append("speed_high")

    if s["dist"] >= dist_hard:
        reasons.append("dist_high")

    if str(s["from_source"]).startswith("gapfill") or str(s["to_source"]).startswith("gapfill"):
        reasons.append("near_gapfill")

    if abs(s["dx"]) > 300:
        reasons.append("x_jump")

    if abs(s["dy"]) > 220:
        reasons.append("y_jump")

    if not reasons:
        return "normal", ""

    if "speed_high" in reasons or "dist_high" in reasons:
        return "jump", "|".join(reasons)

    return "watch", "|".join(reasons)


def cluster_jumps(jump_rows):
    if not jump_rows:
        return []

    rows = sorted(jump_rows, key=lambda r: safe_int(r["from_frame"]))
    clusters = []
    cur = [rows[0]]

    for r in rows[1:]:
        prev = cur[-1]
        if safe_int(r["from_frame"]) <= safe_int(prev["to_frame"]) + 3:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]

    clusters.append(cur)

    out = []
    for idx, cl in enumerate(clusters):
        start = min(safe_int(r["from_frame"]) for r in cl)
        end = max(safe_int(r["to_frame"]) for r in cl)
        max_speed = max(safe_float(r["speed_px_f"]) for r in cl)
        max_dist = max(safe_float(r["dist"]) for r in cl)
        reasons_set = set()
        for r in cl:
            for part in str(r.get("jump_reason", "")).split("|"):
                if part:
                    reasons_set.add(part)
        reasons = sorted(reasons_set)

        out.append({
            "cluster_id": idx,
            "start_frame": start,
            "end_frame": end,
            "len_frames": end - start + 1,
            "step_count": len(cl),
            "max_speed_px_f": max_speed,
            "max_dist": max_dist,
            "reasons": "|".join([x for x in reasons if x]),
        })

    return out


def draw_overlay(video_path: Path, out_path: Path, points, jump_rows, max_items=220):
    if not video_path.exists():
        print(f"WARN: vidéo introuvable: {video_path}")
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

    selected = sorted(jump_rows, key=lambda r: safe_float(r["speed_px_f"]), reverse=True)[:max_items]
    target = set()
    jump_by_frame = {}

    for r in selected:
        a = safe_int(r["from_frame"])
        b = safe_int(r["to_frame"])
        for fr in range(max(0, a - 4), b + 5):
            target.add(fr)
        jump_by_frame.setdefault(a, []).append(r)
        jump_by_frame.setdefault(b, []).append(r)

    trail = 14
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        if frame_idx not in target:
            continue

        out = frame.copy()

        # Trajectoire locale.
        prev_xy = None
        for fr in range(frame_idx - trail, frame_idx + trail + 1):
            p = points.get(fr)
            if not p:
                prev_xy = None
                continue

            x = int(round(p["x"]))
            y = int(round(p["y"]))
            src = p.get("source", "")

            if src.startswith("gapfill_005D7J"):
                color = (255, 0, 255)
                rad = 7
            elif src.startswith("gapfill_005D7E"):
                color = (180, 0, 255)
                rad = 6
            else:
                color = (0, 220, 255)
                rad = 3

            cv2.circle(out, (x, y), rad, color, 2 if rad >= 6 else -1)

            if prev_xy is not None:
                cv2.line(out, prev_xy, (x, y), (90, 90, 90), 1)

            prev_xy = (x, y)

        # Jump courant.
        rows = jump_by_frame.get(frame_idx, [])
        if rows:
            r = sorted(rows, key=lambda x: safe_float(x["speed_px_f"]), reverse=True)[0]
            x0 = int(round(safe_float(r["from_x"])))
            y0 = int(round(safe_float(r["from_y"])))
            x1 = int(round(safe_float(r["to_x"])))
            y1 = int(round(safe_float(r["to_y"])))

            cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.circle(out, (x0, y0), 12, (0, 0, 255), 2)
            cv2.circle(out, (x1, y1), 12, (0, 0, 255), 2)

            label = (
                f"{PATCH_ID} | f={frame_idx} | jump {r['from_frame']}->{r['to_frame']} "
                f"dt={r['dt']} speed={safe_float(r['speed_px_f']):.1f} reason={r['jump_reason']}"
            )
        else:
            label = f"{PATCH_ID} | f={frame_idx} | context"

        cv2.rectangle(out, (8, 8), (min(w - 8, 1260), 42), (0, 0, 0), -1)
        cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(out)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--final-summary", default="runs/005D7L_final_canonical_audit/005D7L_final_canonical_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005E0_canonical_jump_audit")
    ap.add_argument("--speed-hard", type=float, default=180.0)
    ap.add_argument("--dist-hard", type=float, default=260.0)
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    final_summary_path = Path(args.final_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not track_csv.exists():
        raise FileNotFoundError(track_csv)
    if not final_summary_path.exists():
        raise FileNotFoundError(final_summary_path)

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    video_path = Path(final_summary["video"])

    points, meta = read_track(track_csv)
    steps = build_steps(points)

    rows = []
    for s in steps:
        status, reason = classify_step(s, args.speed_hard, args.dist_hard)
        rows.append({
            "step_status": status,
            "jump_reason": reason,
            "from_frame": s["from_frame"],
            "to_frame": s["to_frame"],
            "dt": s["dt"],
            "from_x": f"{s['from_x']:.3f}",
            "from_y": f"{s['from_y']:.3f}",
            "to_x": f"{s['to_x']:.3f}",
            "to_y": f"{s['to_y']:.3f}",
            "dx": f"{s['dx']:.3f}",
            "dy": f"{s['dy']:.3f}",
            "dist": f"{s['dist']:.3f}",
            "speed_px_f": f"{s['speed_px_f']:.3f}",
            "from_source": s["from_source"],
            "to_source": s["to_source"],
            "from_score": f"{s['from_score']:.6f}",
            "to_score": f"{s['to_score']:.6f}",
        })

    jump_rows = [r for r in rows if r["step_status"] == "jump"]
    watch_rows = [r for r in rows if r["step_status"] == "watch"]
    clusters = cluster_jumps(jump_rows)

    step_csv = out_dir / "005E0_step_audit_all.csv"
    jump_csv = out_dir / "005E0_jump_steps_only.csv"
    cluster_csv = out_dir / "005E0_jump_clusters.csv"
    overlay_path = out_dir / "005E0_jump_audit_overlay.mp4"
    summary_path = out_dir / "005E0_jump_audit_summary.json"

    fields = [
        "step_status", "jump_reason",
        "from_frame", "to_frame", "dt",
        "from_x", "from_y", "to_x", "to_y",
        "dx", "dy", "dist", "speed_px_f",
        "from_source", "to_source", "from_score", "to_score",
    ]

    write_csv(step_csv, rows, fields)
    write_csv(jump_csv, jump_rows, fields)
    write_csv(cluster_csv, clusters, [
        "cluster_id", "start_frame", "end_frame", "len_frames",
        "step_count", "max_speed_px_f", "max_dist", "reasons",
    ])

    overlay_ok = draw_overlay(video_path, overlay_path, points, jump_rows)

    speeds = [safe_float(r["speed_px_f"]) for r in rows]
    jump_speeds = [safe_float(r["speed_px_f"]) for r in jump_rows]

    reason_counts = {}
    for r in jump_rows:
        for part in str(r["jump_reason"]).split("|"):
            if part:
                reason_counts[part] = reason_counts.get(part, 0) + 1

    source_transition_counts = {}
    for r in jump_rows:
        k = f"{r['from_source']} -> {r['to_source']}"
        source_transition_counts[k] = source_transition_counts.get(k, 0) + 1

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "final_summary": str(final_summary_path),
        "video": str(video_path),
        "point_count": len(points),
        "step_count": len(rows),
        "jump_count": len(jump_rows),
        "watch_count": len(watch_rows),
        "cluster_count": len(clusters),
        "speed_median": float(np.median(speeds)) if speeds else None,
        "speed_p90": float(np.percentile(speeds, 90)) if speeds else None,
        "speed_p95": float(np.percentile(speeds, 95)) if speeds else None,
        "speed_max": float(np.max(speeds)) if speeds else None,
        "jump_speed_median": float(np.median(jump_speeds)) if jump_speeds else None,
        "jump_speed_p90": float(np.percentile(jump_speeds, 90)) if jump_speeds else None,
        "reason_counts": reason_counts,
        "source_transition_counts": source_transition_counts,
        "top_jump_steps": sorted(jump_rows, key=lambda r: safe_float(r["speed_px_f"]), reverse=True)[:25],
        "track_meta": meta,
        "step_csv": str(step_csv),
        "jump_csv": str(jump_csv),
        "cluster_csv": str(cluster_csv),
        "overlay": str(overlay_path) if overlay_ok else None,
        "note": "Audit seulement. Ne modifie pas le track canonique.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv = {track_csv}")
    print(f"video     = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005E0")
    print(f"summary     = {summary_path}")
    print(f"step_csv    = {step_csv}")
    print(f"jump_csv    = {jump_csv}")
    print(f"cluster_csv = {cluster_csv}")
    print(f"overlay     = {overlay_path if overlay_ok else 'NON_ECRIT'}")
    print("")
    print(f"point_count={len(points)}")
    print(f"step_count={len(rows)}")
    print(f"jump_count={len(jump_rows)}")
    print(f"watch_count={len(watch_rows)}")
    print(f"cluster_count={len(clusters)}")
    print(f"speed_median={summary['speed_median']:.2f}")
    print(f"speed_p90={summary['speed_p90']:.2f}")
    print(f"speed_p95={summary['speed_p95']:.2f}")
    print(f"speed_max={summary['speed_max']:.2f}")
    print(f"reason_counts={reason_counts}")
    print(f"source_transition_counts={source_transition_counts}")


if __name__ == "__main__":
    main()
