from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C4"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def read_frame(clip_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def quad_src_from_row(r) -> np.ndarray:
    return np.asarray([
        [float(r["quad_tl_x_005C3"]), float(r["quad_tl_y_005C3"])],
        [float(r["quad_tr_x_005C3"]), float(r["quad_tr_y_005C3"])],
        [float(r["quad_br_x_005C3"]), float(r["quad_br_y_005C3"])],
        [float(r["quad_bl_x_005C3"]), float(r["quad_bl_y_005C3"])],
    ], dtype=np.float32)


def table_xy_to_px(x_m: float, y_m: float, w_px: int, h_px: int):
    x = (x_m + TABLE_HALF_W) / TABLE_WIDTH_M * (w_px - 1)
    y = (y_m + TABLE_HALF_L) / TABLE_LENGTH_M * (h_px - 1)
    return int(round(x)), int(round(y))


def draw_quad_on_frame(frame, quad, color=(0, 255, 255)):
    q = np.round(quad).astype(int)
    for a, b in zip(q, np.vstack([q[1:], q[:1]])):
        cv2.line(frame, tuple(a), tuple(b), color, 3, cv2.LINE_AA)

    labels = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(q):
        cv2.circle(frame, tuple(p), 8, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            labels[i],
            (int(p[0]) + 8, int(p[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_canonical_grid(img):
    h, w = img.shape[:2]

    # Border
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 255, 255), 3, cv2.LINE_AA)

    # Net y=0
    y_net = table_xy_to_px(0, 0, w, h)[1]
    cv2.line(img, (0, y_net), (w - 1, y_net), (0, 180, 255), 3, cv2.LINE_AA)

    # Center line x=0
    x_mid = table_xy_to_px(0, 0, w, h)[0]
    cv2.line(img, (x_mid, 0), (x_mid, h - 1), (255, 255, 0), 2, cv2.LINE_AA)

    # Metric grid
    for xm in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        x = table_xy_to_px(float(xm), 0, w, h)[0]
        cv2.line(img, (x, 0), (x, h - 1), (80, 180, 180), 1, cv2.LINE_AA)

    for ym in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        y = table_xy_to_px(0, float(ym), w, h)[1]
        cv2.line(img, (0, y), (w - 1, y), (80, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(
        img,
        "CANONICAL TABLE 2.74m x 1.525m",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_projected_points_on_canonical(img, pts):
    if pts.empty:
        return

    h, w = img.shape[:2]

    for _, p in pts.iterrows():
        ok = int(to_num(pd.Series([p.get("table_project_ok_005C3", 0)])).fillna(0).iloc[0])
        if ok != 1:
            continue

        try:
            x_m = float(p["ball_table_x_m_005C3"])
            y_m = float(p["ball_table_y_m_005C3"])
        except Exception:
            continue

        x, y = table_xy_to_px(x_m, y_m, w, h)

        inside = int(to_num(pd.Series([p.get("ball_inside_table_005C3", 0)])).fillna(0).iloc[0])
        color = (0, 255, 0) if inside else (0, 0, 255)

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)


def resize_to_width(img, target_w: int):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / max(1, w)
    return cv2.resize(img, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def make_side_by_side(original, canonical, target_w=720):
    original = resize_to_width(original, target_w)
    canonical = resize_to_width(canonical, min(420, target_w))

    h = max(original.shape[0], canonical.shape[0])

    def pad(img):
        dh = h - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    return cv2.hconcat([pad(original), pad(canonical)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C3_corrected/table_objects_corrected_005C3.csv")
    ap.add_argument("--projected-points", default="runs/rally_table_object_005C3_corrected/ball_points_table_projected_005C3.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_object_005C4_canonical_audit")
    ap.add_argument("--canonical-w", type=int, default=520)
    ap.add_argument("--canonical-h", type=int, default=936)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    root = Path.cwd()

    table_path = Path(args.table_objects)
    points_path = Path(args.projected_points)
    out_dir = Path(args.out_dir)

    if not table_path.is_absolute():
        table_path = root / table_path
    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path).fillna("")
    points = pd.read_csv(points_path).fillna("") if points_path.is_file() else pd.DataFrame()

    table = table[to_num(table["table_ok_005C3"]).fillna(0).astype(int).eq(1)].copy()

    # Priorité aux objets humains puis faibles conf auto.
    table["_human"] = table["table_source_005C3"].astype(str).eq("005C2_human").astype(int)
    table["_conf"] = to_num(table.get("table_confidence_005C1", 0)).fillna(0)

    table = table.sort_values(["_human", "_conf"], ascending=[False, True]).copy()

    if args.max_items > 0:
        table = table.head(args.max_items).copy()

    rows = []
    html_items = []

    dst = np.asarray([
        [0, 0],
        [args.canonical_w - 1, 0],
        [args.canonical_w - 1, args.canonical_h - 1],
        [0, args.canonical_h - 1],
    ], dtype=np.float32)

    for i, (_, r) in enumerate(table.iterrows(), start=1):
        review_id = str(r["review_id"])
        seg = int(to_num(pd.Series([r["camera_segment_id_005B2"]])).fillna(1).iloc[0])
        frame_idx = int(to_num(pd.Series([r.get("best_frame_005C1", r.get("first_frame", 0))])).fillna(0).iloc[0])

        clip_path = Path(str(r["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame = read_frame(clip_path, frame_idx)

        if frame is None:
            rows.append({
                "review_id": review_id,
                "camera_segment_id": seg,
                "status_005C4": "missing_frame",
            })
            continue

        quad = quad_src_from_row(r)

        H = cv2.getPerspectiveTransform(quad, dst)
        canonical = cv2.warpPerspective(frame, H, (args.canonical_w, args.canonical_h))
        draw_canonical_grid(canonical)

        pseg = pd.DataFrame()
        if not points.empty:
            pseg = points[
                points["review_id"].astype(str).eq(review_id)
                & to_num(points["camera_segment_id_005B2"]).fillna(-1).astype(int).eq(seg)
            ].copy()

        draw_projected_points_on_canonical(canonical, pseg)

        original = frame.copy()
        draw_quad_on_frame(original, quad)
        cv2.putText(
            original,
            f"{review_id} seg={seg} source={r.get('table_source_005C3','')} status={r.get('table_status_005C3','')}",
            (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combo = make_side_by_side(original, canonical, target_w=760)

        fname = f"{i:03d}_{review_id}_seg{seg}_005C4_table_audit.jpg"
        out_img = out_dir / "images" / fname
        imwrite_unicode(out_img, combo)

        proj_ok = 0
        inside = 0
        if not pseg.empty and "table_project_ok_005C3" in pseg.columns:
            proj_ok = int(to_num(pseg["table_project_ok_005C3"]).fillna(0).sum())
            inside = int(to_num(pseg["ball_inside_table_005C3"]).fillna(0).sum())

        inside_ratio = round(inside / max(1, proj_ok), 4)

        row = {
            "idx": i,
            "review_id": review_id,
            "camera_segment_id": seg,
            "frame": frame_idx,
            "table_source_005C3": str(r.get("table_source_005C3", "")),
            "table_status_005C3": str(r.get("table_status_005C3", "")),
            "table_confidence_005C1": float(to_num(pd.Series([r.get("table_confidence_005C1", 0)])).fillna(0).iloc[0]),
            "points_projected": proj_ok,
            "points_inside": inside,
            "inside_ratio": inside_ratio,
            "image": str(out_img),
            "status_005C4": "audit_image_written",
        }

        rows.append(row)

        html_items.append(f"""
        <section class="card">
          <h2>{html.escape(review_id)} · seg {seg}</h2>
          <p>
            source={html.escape(str(row["table_source_005C3"]))}
            · status={html.escape(str(row["table_status_005C3"]))}
            · conf_auto={row["table_confidence_005C1"]:.3f}
            · points_inside={inside}/{proj_ok} ({inside_ratio})
          </p>
          <img src="images/{html.escape(fname)}">
        </section>
        """)

    audit = pd.DataFrame(rows)

    out_csv = out_dir / "table_object_canonical_audit_005C4.csv"
    out_json = out_dir / "table_object_canonical_audit_summary_005C4.json"
    out_html = out_dir / "table_object_canonical_audit_005C4.html"

    audit.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "canonical_top_down_visual_audit_of_corrected_metric_table_objects",
        "table_objects": str(table_path),
        "projected_points": str(points_path),
        "objects_audited": int(len(audit)),
        "canonical_w": args.canonical_w,
        "canonical_h": args.canonical_h,
        "outputs": {
            "csv": str(out_csv),
            "html": str(out_html),
        },
        "next": "Use audit to decide whether table object is good enough or needs white-line snapping/manual correction."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_html.write_text(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C4 table canonical audit</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005C4 · audit table objet canonique</h1>
<p>Gauche = frame originale + quad. Droite = table warpée top-down officielle 2.74 × 1.525 m + projections balle.</p>
</header>
{''.join(html_items)}
</body>
</html>
""", encoding="utf-8")

    print("005C4 status=OK")
    print("objects_audited=", summary["objects_audited"])
    print("wrote", out_csv)
    print("wrote", out_json)
    print("wrote", out_html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
