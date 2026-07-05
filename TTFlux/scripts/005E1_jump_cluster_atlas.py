from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005E1_jump_cluster_atlas"


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


def imwrite_unicode(path: Path, img) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_track_points(path: Path):
    rows, fields = read_csv(path)

    frame_col = find_col(fields, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(fields, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(fields, ["y", "cy", "ball_y", "center_y"])

    if frame_col is None or x_col is None or y_col is None:
        raise RuntimeError("Colonnes frame/x/y introuvables dans le track.")

    points = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        if fr >= 0 and math.isfinite(x) and math.isfinite(y):
            points[fr] = {
                "frame": fr,
                "x": x,
                "y": y,
                "source": str(r.get("point_source", "")) or "source",
            }

    return points


def video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def read_frame(cap, fr: int, w: int, h: int):
    if fr < 0:
        return np.zeros((h, w, 3), dtype=np.uint8), False

    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, img = cap.read()
    if not ok or img is None:
        return np.zeros((h, w, 3), dtype=np.uint8), False
    return img, True


def draw_local_track(img, frame_idx, points, jumps, label):
    out = img.copy()
    h, w = out.shape[:2]

    prev = None
    for fr in range(frame_idx - 14, frame_idx + 15):
        p = points.get(fr)
        if not p:
            prev = None
            continue

        x = int(round(p["x"]))
        y = int(round(p["y"]))
        src = p.get("source", "")

        if src.startswith("gapfill_005D7J"):
            color = (255, 0, 255)
            radius = 7
        elif src.startswith("gapfill_005D7E"):
            color = (180, 0, 255)
            radius = 6
        else:
            color = (0, 220, 255)
            radius = 3

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, 2 if radius >= 6 else -1)

        if prev is not None:
            cv2.line(out, prev, (x, y), (90, 90, 90), 1)

        prev = (x, y)

    # Lignes rouges des jumps du cluster.
    for j in jumps:
        x0 = int(round(safe_float(j.get("from_x"))))
        y0 = int(round(safe_float(j.get("from_y"))))
        x1 = int(round(safe_float(j.get("to_x"))))
        y1 = int(round(safe_float(j.get("to_y"))))

        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.circle(out, (x0, y0), 11, (0, 0, 255), 2)
        cv2.circle(out, (x1, y1), 11, (0, 0, 255), 2)

    cv2.rectangle(out, (0, 0), (min(w - 1, 1260), 38), (0, 0, 0), -1)
    cv2.putText(out, label[:150], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def make_panel(cap, cluster, jumps, points, vi):
    cid = safe_int(cluster.get("cluster_id"))
    start = safe_int(cluster.get("start_frame"))
    end = safe_int(cluster.get("end_frame"))

    jumps_sorted = sorted(jumps, key=lambda r: safe_float(r.get("speed_px_f")), reverse=True)
    max_jump = jumps_sorted[0] if jumps_sorted else None

    if max_jump:
        a = safe_int(max_jump.get("from_frame"))
        b = safe_int(max_jump.get("to_frame"))
        max_speed = safe_float(max_jump.get("speed_px_f"))
        reason = str(max_jump.get("jump_reason", ""))
    else:
        a = start
        b = end
        max_speed = safe_float(cluster.get("max_speed_px_f"))
        reason = str(cluster.get("reasons", ""))

    frames = []
    for fr in [a - 2, a - 1, a, b, b + 1, b + 2]:
        if fr not in frames:
            frames.append(fr)

    thumbs = []
    labels = []

    for fr in frames:
        raw, ok = read_frame(cap, fr, vi["width"], vi["height"])
        label = f"c{cid} f{fr} {'OK' if ok else 'NO_VIDEO'}"

        if ok:
            drawn = draw_local_track(
                raw,
                fr,
                points,
                jumps_sorted[:4],
                f"{PATCH_ID} | cluster={cid} | f={fr} | max {a}->{b} speed={max_speed:.1f} | {reason}",
            )
        else:
            drawn = raw
            cv2.putText(drawn, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        thumb = cv2.resize(drawn, (320, 180))
        thumbs.append(thumb)
        labels.append(label)

    cols = 3
    rows = int(math.ceil(len(thumbs) / cols))
    panel = np.zeros((rows * 210, cols * 320, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = c * 320
        y = r * 210
        panel[y:y+180, x:x+320] = thumb
        cv2.putText(panel, labels[i], (x + 8, y + 202), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return panel, {
        "cluster_id": cid,
        "start_frame": start,
        "end_frame": end,
        "len_frames": safe_int(cluster.get("len_frames")),
        "step_count": safe_int(cluster.get("step_count")),
        "max_jump_from": a,
        "max_jump_to": b,
        "max_speed_px_f": max_speed,
        "max_dist": safe_float(cluster.get("max_dist")),
        "reasons": reason,
        "inside_video": int(0 <= a < vi["frames"] and 0 <= b < vi["frames"]),
        "frame_window": ",".join(str(x) for x in frames),
    }


def make_contact_sheet(atlas_rows, out_path: Path, max_items=80):
    imgs = []
    labels = []

    for r in atlas_rows[:max_items]:
        img = imread_unicode(Path(r["panel_path"]))
        if img is None:
            continue
        thumb = cv2.resize(img, (320, 210))
        imgs.append(thumb)
        labels.append(f"c{r['cluster_id']} {r['start_frame']}-{r['end_frame']} spd={float(r['max_speed_px_f']):.0f}")

    if not imgs:
        return False

    cols = 4
    rows = int(math.ceil(len(imgs) / cols))
    sheet = np.zeros((rows * 240, cols * 320, 3), dtype=np.uint8)

    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = c * 320
        y = r * 240
        sheet[y:y+210, x:x+320] = img
        cv2.putText(sheet, labels[i][:42], (x + 6, y + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return imwrite_unicode(out_path, sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-csv", default="runs/005D7K_gapfilled_canonical_v2/005D7K_ball_points_gapfilled_canonical_unique.csv")
    ap.add_argument("--jump-csv", default="runs/005E0_canonical_jump_audit/005E0_jump_steps_only.csv")
    ap.add_argument("--cluster-csv", default="runs/005E0_canonical_jump_audit/005E0_jump_clusters.csv")
    ap.add_argument("--final-summary", default="runs/005D7L_final_canonical_audit/005D7L_final_canonical_audit_summary.json")
    ap.add_argument("--out-dir", default="runs/005E1_jump_cluster_atlas")
    ap.add_argument("--max-clusters", type=int, default=80)
    args = ap.parse_args()

    track_csv = Path(args.track_csv)
    jump_csv = Path(args.jump_csv)
    cluster_csv = Path(args.cluster_csv)
    final_summary_path = Path(args.final_summary)
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"

    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    video_path = Path(final_summary["video"])

    vi = video_info(video_path)
    points = read_track_points(track_csv)
    jumps, _ = read_csv(jump_csv)
    clusters, _ = read_csv(cluster_csv)

    clusters = sorted(clusters, key=lambda r: safe_float(r.get("max_speed_px_f")), reverse=True)[:args.max_clusters]

    jumps_by_cluster = {}
    for c in clusters:
        cid = safe_int(c.get("cluster_id"))
        start = safe_int(c.get("start_frame"))
        end = safe_int(c.get("end_frame"))

        related = []
        for j in jumps:
            a = safe_int(j.get("from_frame"))
            b = safe_int(j.get("to_frame"))
            if a <= end and b >= start:
                related.append(j)

        jumps_by_cluster[cid] = related

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir vidéo: {video_path}")

    atlas_rows = []

    for c in clusters:
        cid = safe_int(c.get("cluster_id"))
        related = jumps_by_cluster.get(cid, [])

        panel, row = make_panel(cap, c, related, points, vi)
        panel_path = panels_dir / f"cluster_{cid:03d}_f{row['start_frame']:04d}_{row['end_frame']:04d}.jpg"
        imwrite_unicode(panel_path, panel)

        row["panel_path"] = str(panel_path)
        row["jump_count_in_panel"] = len(related)
        atlas_rows.append(row)

    cap.release()

    atlas_csv = out_dir / "005E1_jump_cluster_atlas.csv"
    contact_sheet = out_dir / "005E1_jump_cluster_contact_sheet.jpg"
    html_path = out_dir / "005E1_jump_cluster_atlas.html"
    summary_path = out_dir / "005E1_jump_cluster_atlas_summary.json"

    write_csv(atlas_csv, atlas_rows, [
        "cluster_id",
        "start_frame",
        "end_frame",
        "len_frames",
        "step_count",
        "jump_count_in_panel",
        "max_jump_from",
        "max_jump_to",
        "max_speed_px_f",
        "max_dist",
        "reasons",
        "inside_video",
        "frame_window",
        "panel_path",
    ])

    make_contact_sheet(atlas_rows, contact_sheet)

    cards = []
    for r in atlas_rows:
        rel = Path(r["panel_path"]).relative_to(out_dir).as_posix()
        cls = "inside" if int(r["inside_video"]) else "outside"
        cards.append(f"""
        <div class="card {cls}">
          <h3>Cluster {r['cluster_id']} — frames {r['start_frame']}→{r['end_frame']}</h3>
          <p>
            max jump {r['max_jump_from']}→{r['max_jump_to']} |
            speed={float(r['max_speed_px_f']):.1f} |
            dist={float(r['max_dist']):.1f} |
            steps={r['step_count']} |
            reasons={html.escape(str(r['reasons']))} |
            inside_video={r['inside_video']}
          </p>
          <a href="{rel}"><img src="{rel}"></a>
        </div>
        """)

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{PATCH_ID}</title>
<style>
body {{
  font-family: Arial, sans-serif;
  background:#111;
  color:#eee;
  margin:24px;
}}
.meta {{ color:#bbb; margin-bottom:20px; }}
.grid {{
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
  gap:16px;
}}
.card {{
  background:#1b1b1b;
  border:1px solid #333;
  border-radius:10px;
  padding:12px;
}}
.card h3 {{ margin:0 0 8px 0; }}
.card p {{ color:#bbb; font-size:12px; }}
.card img {{ width:100%; border-radius:8px; border:1px solid #444; }}
.inside {{ border-color:#555; }}
.outside {{ border-color:#884444; }}
a {{ color:#9cf; }}
</style>
</head>
<body>
<h1>{PATCH_ID}</h1>
<div class="meta">
track={html.escape(str(track_csv))}<br>
video={html.escape(str(video_path))} |
video_frames={vi['frames']} |
clusters={len(atlas_rows)} |
<a href="{contact_sheet.name}">contact sheet</a> |
<a href="{atlas_csv.name}">atlas csv</a>
</div>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""

    html_path.write_text(html_doc, encoding="utf-8")

    inside_count = sum(1 for r in atlas_rows if int(r["inside_video"]) == 1)
    outside_count = len(atlas_rows) - inside_count

    summary = {
        "patch": PATCH_ID,
        "track_csv": str(track_csv),
        "jump_csv": str(jump_csv),
        "cluster_csv": str(cluster_csv),
        "final_summary": str(final_summary_path),
        "video": str(video_path),
        "video_info": vi,
        "point_count": len(points),
        "input_jump_count": len(jumps),
        "input_cluster_count": len(clusters),
        "atlas_cluster_count": len(atlas_rows),
        "inside_video_cluster_count": inside_count,
        "outside_video_cluster_count": outside_count,
        "atlas_csv": str(atlas_csv),
        "contact_sheet": str(contact_sheet),
        "html": str(html_path),
        "note": "Atlas visuel uniquement. Ne modifie pas le track.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print(f"track_csv   = {track_csv}")
    print(f"jump_csv    = {jump_csv}")
    print(f"cluster_csv = {cluster_csv}")
    print(f"video       = {video_path}")
    print("=" * 72)
    print("")
    print("OK 005E1")
    print(f"summary       = {summary_path}")
    print(f"html          = {html_path}")
    print(f"contact_sheet = {contact_sheet}")
    print(f"atlas_csv     = {atlas_csv}")
    print("")
    print(f"video_frames={vi['frames']}")
    print(f"input_jump_count={len(jumps)}")
    print(f"atlas_cluster_count={len(atlas_rows)}")
    print(f"inside_video_cluster_count={inside_count}")
    print(f"outside_video_cluster_count={outside_count}")


if __name__ == "__main__":
    main()
