from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C9B"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0

CANON_W = 520
CANON_H = 936


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def parse_h(s):
    try:
        arr = json.loads(str(s))
        H = np.asarray(arr, dtype=np.float64).reshape(3, 3)
        return H
    except Exception:
        return None


def table_corners_m():
    return np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)


def homography_img_to_table_from_quad(q_img: np.ndarray):
    return cv2.getPerspectiveTransform(np.asarray(q_img, dtype=np.float32), table_corners_m())


def homography_table_to_img_from_quad(q_img: np.ndarray):
    return cv2.getPerspectiveTransform(table_corners_m(), np.asarray(q_img, dtype=np.float32))


def project_img_to_table(H, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def get_seg(row):
    for c in ["camera_segment_id", "camera_segment_id_005B2"]:
        if c in row and str(row[c]) != "":
            return int(to_num(pd.Series([row[c]])).fillna(1).iloc[0])
    return 1


def metric_key(review_id, seg):
    return (str(review_id), int(seg))


def row_quad_opt(row):
    return np.asarray([
        [float(row["opt_tl_x_005C9A"]), float(row["opt_tl_y_005C9A"])],
        [float(row["opt_tr_x_005C9A"]), float(row["opt_tr_y_005C9A"])],
        [float(row["opt_br_x_005C9A"]), float(row["opt_br_y_005C9A"])],
        [float(row["opt_bl_x_005C9A"]), float(row["opt_bl_y_005C9A"])],
    ], dtype=np.float32)


def row_quad_snap(row):
    return np.asarray([
        [float(row["snap_tl_x_005C7A"]), float(row["snap_tl_y_005C7A"])],
        [float(row["snap_tr_x_005C7A"]), float(row["snap_tr_y_005C7A"])],
        [float(row["snap_br_x_005C7A"]), float(row["snap_br_y_005C7A"])],
        [float(row["snap_bl_x_005C7A"]), float(row["snap_bl_y_005C7A"])],
    ], dtype=np.float32)


def quad_area(q):
    return float(abs(cv2.contourArea(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2))))


def quad_signature(q, H):
    vals = []

    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    vals.extend(np.round(q.reshape(-1), 2).tolist())

    if H is not None:
        hh = H.astype(np.float64).copy()
        denom = hh[2, 2] if abs(hh[2, 2]) > 1e-9 else 1.0
        hh = hh / denom
        vals.extend(np.round(hh.reshape(-1), 5).tolist())

    raw = ",".join(map(str, vals)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def xy_m_to_px(x, y, w=CANON_W, h=CANON_H):
    px = int(round((float(x) + TABLE_HALF_W) / TABLE_WIDTH_M * (w - 1)))
    py = int(round((float(y) + TABLE_HALF_L) / TABLE_LENGTH_M * (h - 1)))
    return px, py


def draw_topdown_base(w=CANON_W, h=CANON_H):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (24, 74, 58)

    # Table rectangle.
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (235, 235, 235), 3, cv2.LINE_AA)

    # Filet au centre : y = 0.
    net_y = xy_m_to_px(0, 0, w, h)[1]
    cv2.line(img, (0, net_y), (w - 1, net_y), (40, 210, 255), 3, cv2.LINE_AA)

    # Ligne centrale : x = 0.
    center_x = xy_m_to_px(0, 0, w, h)[0]
    cv2.line(img, (center_x, 0), (center_x, h - 1), (255, 255, 80), 2, cv2.LINE_AA)

    # Grille métrique légère.
    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        px, _ = xy_m_to_px(x, 0, w, h)
        cv2.line(img, (px, 0), (px, h - 1), (72, 118, 100), 1, cv2.LINE_AA)

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        _, py = xy_m_to_px(0, y, w, h)
        cv2.line(img, (0, py), (w - 1, py), (72, 118, 100), 1, cv2.LINE_AA)

    cv2.putText(img, "FAR SIDE", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "NEAR SIDE", (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 2, cv2.LINE_AA)

    return img


def draw_ball_points_topdown(base, points_table, title):
    img = base.copy()

    if len(points_table):
        pts = points_table.copy()
        pts["frame_num"] = to_num(pts["frame_num"]).fillna(-1).astype(int)
        pts = pts.sort_values("frame_num").copy()

        last = None

        for _, r in pts.iterrows():
            x = float(r["ball_table_x_m_005C9B"])
            y = float(r["ball_table_y_m_005C9B"])
            px, py = xy_m_to_px(x, y)

            inside = int(abs(x) <= TABLE_HALF_W and abs(y) <= TABLE_HALF_L)

            color = (0, 255, 0) if inside else (0, 0, 255)

            if last is not None:
                cv2.line(img, last, (px, py), (130, 170, 255), 1, cv2.LINE_AA)

            cv2.circle(img, (px, py), 3, color, -1, cv2.LINE_AA)
            last = (px, py)

    cv2.putText(img, title, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def build_scene_objects(metric_df: pd.DataFrame, fit_df: pd.DataFrame):
    fit_map = {}

    for _, r in fit_df.iterrows():
        review_id = str(r["review_id"])
        seg = get_seg(r)
        fit_map[metric_key(review_id, seg)] = r.to_dict()

    rows = []

    for _, r in metric_df.iterrows():
        row = r.to_dict()
        review_id = str(row["review_id"])
        seg = get_seg(row)

        key = metric_key(review_id, seg)
        fit = fit_map.get(key)

        source = ""
        status = ""
        H_img_to_table = None
        H_table_to_img = None
        quad = None
        human_rating = ""

        if fit is not None and str(fit.get("fit_status_005C9A", "")) == "OK":
            status = "METRIC_TABLE_TRUSTED"
            source = "005C9A_human_accepted_optimized"
            human_rating = fit.get("rating_1_10", "")
            quad = row_quad_opt(fit)
            H_img_to_table = parse_h(fit.get("H_img_to_table_005C9A", ""))
            H_table_to_img = parse_h(fit.get("H_table_to_img_005C9A", ""))

            if H_img_to_table is None or H_table_to_img is None:
                H_img_to_table = homography_img_to_table_from_quad(quad)
                H_table_to_img = homography_table_to_img_from_quad(quad)

        else:
            q = str(row.get("metric_quality_005C7B", ""))

            if q in {"METRIC_TABLE_STRICT", "METRIC_TABLE_REVIEW"}:
                status = "METRIC_TABLE_REVIEW"
                source = "005C7A_snap_unvalidated"
                quad = row_quad_snap(row)
                H_img_to_table = parse_h(row.get("H_img_to_table_005C7A", ""))
                H_table_to_img = parse_h(row.get("H_table_to_img_005C7A", ""))

                if H_img_to_table is None or H_table_to_img is None:
                    H_img_to_table = homography_img_to_table_from_quad(quad)
                    H_table_to_img = homography_table_to_img_from_quad(quad)

            elif q == "PARTIAL_TABLE_OBJECT":
                status = "PARTIAL_TABLE_MASK"
                source = "005C7B_partial_mask_only"
                try:
                    quad = row_quad_snap(row)
                except Exception:
                    quad = np.zeros((4, 2), dtype=np.float32)

            else:
                status = "NO_TABLE_METRIC"
                source = "005C7B_rejected_or_missing"
                try:
                    quad = row_quad_snap(row)
                except Exception:
                    quad = np.zeros((4, 2), dtype=np.float32)

        if quad is None:
            quad = np.zeros((4, 2), dtype=np.float32)

        sig = quad_signature(quad, H_img_to_table)

        out = {
            "version": VERSION,
            "review_id": review_id,
            "camera_segment_id": seg,
            "video_id": str(row.get("video_id", "")),
            "rally_id": str(row.get("rally_id", "")),
            "clip_path": str(row.get("clip_path", "")),
            "scene_table_status_005C9B": status,
            "scene_table_source_005C9B": source,
            "scene_table_metric_usable_005C9B": int(status == "METRIC_TABLE_TRUSTED"),
            "scene_table_review_usable_005C9B": int(status == "METRIC_TABLE_REVIEW"),
            "scene_table_partial_mask_usable_005C9B": int(status == "PARTIAL_TABLE_MASK"),
            "human_rating_005C8A": human_rating,

            "metric_quality_005C7B": str(row.get("metric_quality_005C7B", "")),
            "metric_reason_005C7B": str(row.get("metric_reason_005C7B", "")),
            "snap_status_005C7A": str(row.get("snap_status_005C7A", "")),
            "snap_confidence_005C7A": row.get("snap_confidence_005C7A", ""),

            "camera_signature_005C9B": sig,
            "quad_area_px_005C9B": round(float(quad_area(quad)), 3),

            "quad_tl_x_005C9B": round(float(quad[0, 0]), 3),
            "quad_tl_y_005C9B": round(float(quad[0, 1]), 3),
            "quad_tr_x_005C9B": round(float(quad[1, 0]), 3),
            "quad_tr_y_005C9B": round(float(quad[1, 1]), 3),
            "quad_br_x_005C9B": round(float(quad[2, 0]), 3),
            "quad_br_y_005C9B": round(float(quad[2, 1]), 3),
            "quad_bl_x_005C9B": round(float(quad[3, 0]), 3),
            "quad_bl_y_005C9B": round(float(quad[3, 1]), 3),

            "H_img_to_table_005C9B": "" if H_img_to_table is None else json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C9B": "" if H_table_to_img is None else json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),

            "table_length_m": TABLE_LENGTH_M,
            "table_width_m": TABLE_WIDTH_M,
        }

        # métriques 005C9A si dispo.
        if fit is not None:
            for c in [
                "init_score_005C9A",
                "opt_score_005C9A",
                "opt_gain_005C9A",
                "opt_inside_prob_005C9A",
                "opt_poly_support_005C9A",
                "opt_mask_coverage_005C9A",
                "opt_line_score_005C9A",
            ]:
                out[c] = fit.get(c, "")

        rows.append(out)

    return pd.DataFrame(rows)


def project_ball_points(scene_df: pd.DataFrame, points_df: pd.DataFrame):
    if points_df.empty:
        return pd.DataFrame()

    pts = points_df.copy()

    pts["camera_segment_id_005B2"] = to_num(pts["camera_segment_id_005B2"]).fillna(1).astype(int)
    pts["x_num"] = to_num(pts["x_num"])
    pts["y_num"] = to_num(pts["y_num"])

    scene_map = {}

    for _, r in scene_df.iterrows():
        key = metric_key(r["review_id"], r["camera_segment_id"])
        scene_map[key] = r.to_dict()

    rows = []

    for _, p in pts.iterrows():
        review_id = str(p["review_id"])
        seg = int(p["camera_segment_id_005B2"])
        key = metric_key(review_id, seg)

        scene = scene_map.get(key)
        out = p.to_dict()

        if scene is None or int(scene.get("scene_table_metric_usable_005C9B", 0)) != 1:
            out.update({
                "scene_table_status_005C9B": "" if scene is None else scene.get("scene_table_status_005C9B", ""),
                "table_project_ok_005C9B": 0,
                "ball_table_x_m_005C9B": "",
                "ball_table_y_m_005C9B": "",
                "ball_inside_table_005C9B": 0,
                "ball_table_side_005C9B": "",
                "camera_signature_005C9B": "" if scene is None else scene.get("camera_signature_005C9B", ""),
            })
            rows.append(out)
            continue

        H = parse_h(scene.get("H_img_to_table_005C9B", ""))
        if H is None:
            out.update({
                "scene_table_status_005C9B": scene.get("scene_table_status_005C9B", ""),
                "table_project_ok_005C9B": 0,
                "ball_table_x_m_005C9B": "",
                "ball_table_y_m_005C9B": "",
                "ball_inside_table_005C9B": 0,
                "ball_table_side_005C9B": "",
                "camera_signature_005C9B": scene.get("camera_signature_005C9B", ""),
            })
            rows.append(out)
            continue

        xy = project_img_to_table(H, np.asarray([[float(p["x_num"]), float(p["y_num"])]], dtype=np.float32))[0]
        x = float(xy[0])
        y = float(xy[1])

        inside = int(abs(x) <= TABLE_HALF_W and abs(y) <= TABLE_HALF_L)
        side = "near" if y >= 0 else "far"

        out.update({
            "scene_table_status_005C9B": scene.get("scene_table_status_005C9B", ""),
            "table_project_ok_005C9B": 1,
            "ball_table_x_m_005C9B": round(x, 4),
            "ball_table_y_m_005C9B": round(y, 4),
            "ball_inside_table_005C9B": inside,
            "ball_table_side_005C9B": side,
            "camera_signature_005C9B": scene.get("camera_signature_005C9B", ""),
        })

        rows.append(out)

    return pd.DataFrame(rows)


def make_topdown_report(scene_df, projected_df, out_dir: Path, max_images: int):
    cards = []
    img_dir = out_dir / "topdown_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    trusted = scene_df[scene_df["scene_table_status_005C9B"].astype(str).eq("METRIC_TABLE_TRUSTED")].copy()

    for i, (_, s) in enumerate(trusted.iterrows(), start=1):
        if i > max_images:
            break

        review_id = str(s["review_id"])
        seg = int(s["camera_segment_id"])

        g = projected_df[
            projected_df["review_id"].astype(str).eq(review_id)
            & to_num(projected_df["camera_segment_id_005B2"]).fillna(1).astype(int).eq(seg)
            & to_num(projected_df["table_project_ok_005C9B"]).fillna(0).astype(int).eq(1)
        ].copy()

        base = draw_topdown_base()
        title = f"{review_id} seg={seg} rating={s.get('human_rating_005C8A','')} pts={len(g)}"
        img = draw_ball_points_topdown(base, g, title)

        fname = f"{i:03d}_{review_id}_seg{seg}_005C9B_topdown.jpg"
        path = img_dir / fname
        imwrite_unicode(path, img)

        inside = int(to_num(g["ball_inside_table_005C9B"]).fillna(0).sum()) if len(g) else 0
        ratio = round(float(inside / max(1, len(g))), 4)

        cards.append({
            "review_id": review_id,
            "seg": seg,
            "points": int(len(g)),
            "inside": inside,
            "inside_ratio": ratio,
            "image": str(path),
        })

    html_text = build_topdown_html(cards)
    html_path = out_dir / "scene_topdown_report_005C9B.html"
    html_path.write_text(html_text, encoding="utf-8")

    return html_path, cards


def build_topdown_html(cards):
    sections = []

    for c in cards:
        sections.append(f"""
<section class="card">
  <h2>{html.escape(c["review_id"])} · seg {c["seg"]}</h2>
  <p>points={c["points"]} · inside={c["inside"]} · inside_ratio={c["inside_ratio"]}</p>
  <img src="topdown_images/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C9B scene topdown</title>
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
<h1>TTFlux 005C9B · scène table top-down canonique</h1>
<p>Table fixe vue de dessus. Haut = côté fond. Bas = côté proche. Ligne orange = filet. Points verts = balle projetée dans les limites de la table.</p>
</header>
{''.join(sections)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-csv", default="runs/rally_table_metric_qa_005C7B/table_metric_quality_005C7B.csv")
    ap.add_argument("--fit-csv", default="runs/rally_table_reference_fit_005C9A/table_reference_fit_005C9A.csv")
    ap.add_argument("--display-points", default="runs/rally_table_camera_005B2_pass33/display_points_table_camera_005B2.csv")
    ap.add_argument("--out-dir", default="runs/rally_scene_table_objects_005C9B")
    ap.add_argument("--make-topdown", action="store_true")
    ap.add_argument("--max-topdown", type=int, default=60)
    args = ap.parse_args()

    root = Path.cwd()

    metric_path = Path(args.metric_csv)
    fit_path = Path(args.fit_csv)
    points_path = Path(args.display_points)
    out_dir = Path(args.out_dir)

    if not metric_path.is_absolute():
        metric_path = root / metric_path
    if not fit_path.is_absolute():
        fit_path = root / fit_path
    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    metric = pd.read_csv(metric_path).fillna("")
    fit = pd.read_csv(fit_path).fillna("")
    points = pd.read_csv(points_path).fillna("") if points_path.is_file() else pd.DataFrame()

    scene = build_scene_objects(metric, fit)
    projected = project_ball_points(scene, points)

    out_scene = out_dir / "scene_table_objects_005C9B.csv"
    out_points = out_dir / "ball_points_scene_table_projected_005C9B.csv"
    out_review = out_dir / "scene_table_review_summary_005C9B.csv"
    out_json = out_dir / "scene_table_objects_summary_005C9B.json"

    scene.to_csv(out_scene, index=False, encoding="utf-8")
    projected.to_csv(out_points, index=False, encoding="utf-8")

    review_rows = []

    if not projected.empty:
        for review_id, g in projected.groupby("review_id", dropna=False):
            ok = to_num(g["table_project_ok_005C9B"]).fillna(0).astype(int)
            inside = to_num(g["ball_inside_table_005C9B"]).fillna(0).astype(int)

            review_rows.append({
                "review_id": str(review_id),
                "points_total": int(len(g)),
                "projected_ok": int(ok.sum()),
                "inside_table": int(inside.sum()),
                "inside_ratio": round(float(inside.sum() / max(1, ok.sum())), 4),
                "trusted_segments": int(scene[
                    scene["review_id"].astype(str).eq(str(review_id))
                    & scene["scene_table_status_005C9B"].astype(str).eq("METRIC_TABLE_TRUSTED")
                ].shape[0]),
            })

    review = pd.DataFrame(review_rows)
    review.to_csv(out_review, index=False, encoding="utf-8")

    topdown_html = ""
    topdown_cards = []

    if args.make_topdown:
        html_path, topdown_cards = make_topdown_report(scene, projected, out_dir, max_images=args.max_topdown)
        topdown_html = str(html_path)

    status_counts = scene["scene_table_status_005C9B"].astype(str).value_counts().to_dict()

    projected_ok = int(to_num(projected["table_project_ok_005C9B"]).fillna(0).sum()) if not projected.empty else 0
    projected_inside = int(to_num(projected["ball_inside_table_005C9B"]).fillna(0).sum()) if not projected.empty else 0

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "promote_human_accepted_optimized_reference_table_to_scene_table_objects_and_project_ball_topdown",
        "metric_csv": str(metric_path),
        "fit_csv": str(fit_path),
        "display_points": str(points_path),
        "objects_total": int(len(scene)),
        "scene_table_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "trusted_metric_tables": int((scene["scene_table_status_005C9B"].astype(str).eq("METRIC_TABLE_TRUSTED")).sum()),
        "review_metric_tables": int((scene["scene_table_status_005C9B"].astype(str).eq("METRIC_TABLE_REVIEW")).sum()),
        "partial_table_masks": int((scene["scene_table_status_005C9B"].astype(str).eq("PARTIAL_TABLE_MASK")).sum()),
        "ball_points_total": int(len(projected)),
        "ball_points_projected_ok": projected_ok,
        "ball_points_inside_table": projected_inside,
        "ball_points_inside_ratio": round(float(projected_inside / max(1, projected_ok)), 4),
        "outputs": {
            "scene_table_objects": str(out_scene),
            "ball_points_projected": str(out_points),
            "review_summary": str(out_review),
            "topdown_html": topdown_html,
        },
        "next": "Use METRIC_TABLE_TRUSTED camera signatures and topdown ball coordinates in 005D table/camera-aware tracking."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C9B status=OK")
    print("objects_total=", summary["objects_total"])
    print("scene_table_status_counts=", json.dumps(summary["scene_table_status_counts"], ensure_ascii=False))
    print("trusted_metric_tables=", summary["trusted_metric_tables"])
    print("ball_points_total=", summary["ball_points_total"])
    print("ball_points_projected_ok=", summary["ball_points_projected_ok"])
    print("ball_points_inside_ratio=", summary["ball_points_inside_ratio"])
    print("wrote", out_scene)
    print("wrote", out_points)
    print("wrote", out_review)
    if topdown_html:
        print("wrote", topdown_html)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
