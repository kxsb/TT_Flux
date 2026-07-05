from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C5C2"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0
TABLE_HEIGHT_M = 0.760
NET_HEIGHT_M = 0.1525


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def table_pts_m():
    return np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)


def homographies_from_quad_src(quad_src: np.ndarray):
    quad_src = np.asarray(quad_src, dtype=np.float32).reshape(4, 2)
    tpts = table_pts_m()
    h_img_to_table = cv2.getPerspectiveTransform(quad_src, tpts)
    h_table_to_img = cv2.getPerspectiveTransform(tpts, quad_src)
    return h_img_to_table, h_table_to_img


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def original_quad_src(row: dict) -> np.ndarray:
    src_w = float(row["src_w"])
    src_h = float(row["src_h"])
    resize_w = float(row["resize_w"])
    resize_h = float(row["resize_h"])

    sx = src_w / max(1.0, resize_w)
    sy = src_h / max(1.0, resize_h)

    quad_small = np.asarray([
        [float(row["quad_tl_x_005C1"]), float(row["quad_tl_y_005C1"])],
        [float(row["quad_tr_x_005C1"]), float(row["quad_tr_y_005C1"])],
        [float(row["quad_br_x_005C1"]), float(row["quad_br_y_005C1"])],
        [float(row["quad_bl_x_005C1"]), float(row["quad_bl_y_005C1"])],
    ], dtype=np.float32)

    quad_src = quad_small.copy()
    quad_src[:, 0] *= sx
    quad_src[:, 1] *= sy
    return quad_src


def correction_quad_src(corr: dict) -> np.ndarray:
    return np.asarray([
        [float(corr["tl_x"]), float(corr["tl_y"])],
        [float(corr["tr_x"]), float(corr["tr_y"])],
        [float(corr["br_x"]), float(corr["br_y"])],
        [float(corr["bl_x"]), float(corr["bl_y"])],
    ], dtype=np.float32)


def load_corrections(path: Path, seg_col: str) -> dict:
    if not path.is_file():
        return {}

    df = pd.read_csv(path).fillna("")
    out = {}

    for _, r in df.iterrows():
        review_id = str(r["review_id"])
        seg = int(to_num(pd.Series([r[seg_col]])).fillna(0).iloc[0])
        out[(review_id, seg)] = r.to_dict()

    return out


def build_merged_objects(table_c1: pd.DataFrame, c2: dict, c5b: dict) -> pd.DataFrame:
    rows = []

    for _, r in table_c1.iterrows():
        base = r.to_dict()
        review_id = str(base["review_id"])
        seg = int(to_num(pd.Series([base["camera_segment_id_005B2"]])).fillna(1).iloc[0])
        key = (review_id, seg)

        quad_src = original_quad_src(base)
        source = "005C1_auto"
        status = "auto"
        table_ok = 1

        if key in c2:
            corr = c2[key]
            status = str(corr.get("status", "corrected"))

            if status == "bad_frame":
                source = "005C2_bad_frame_auto_fallback"
                table_ok = 0
            else:
                quad_src = correction_quad_src(corr)
                source = "005C2_human"
                table_ok = 1

        # 005C5B override sémantique prioritaire.
        if key in c5b:
            corr = c5b[key]
            status = str(corr.get("status", "semantic_corrected"))

            if status in {"reject_table_not_visible", "bad_frame"}:
                source = "005C5B_rejected"
                table_ok = 0
            else:
                quad_src = correction_quad_src(corr)
                source = "005C5B_semantic"
                table_ok = 1

        try:
            H_img_to_table, H_table_to_img = homographies_from_quad_src(quad_src)
            h_ok = 1
        except Exception:
            H_img_to_table = np.eye(3, dtype=np.float64)
            H_table_to_img = np.eye(3, dtype=np.float64)
            h_ok = 0
            table_ok = 0

        out = dict(base)

        # Compatibilité avec 005C4 : colonnes 005C3.
        out.update({
            "table_ok_005C3": int(table_ok and h_ok),
            "table_source_005C3": source,
            "table_status_005C3": status,

            "table_ok_005C5C2": int(table_ok and h_ok),
            "table_source_005C5C2": source,
            "table_status_005C5C2": status,

            "quad_tl_x_005C3": round(float(quad_src[0, 0]), 3),
            "quad_tl_y_005C3": round(float(quad_src[0, 1]), 3),
            "quad_tr_x_005C3": round(float(quad_src[1, 0]), 3),
            "quad_tr_y_005C3": round(float(quad_src[1, 1]), 3),
            "quad_br_x_005C3": round(float(quad_src[2, 0]), 3),
            "quad_br_y_005C3": round(float(quad_src[2, 1]), 3),
            "quad_bl_x_005C3": round(float(quad_src[3, 0]), 3),
            "quad_bl_y_005C3": round(float(quad_src[3, 1]), 3),

            "quad_tl_x_005C5C2": round(float(quad_src[0, 0]), 3),
            "quad_tl_y_005C5C2": round(float(quad_src[0, 1]), 3),
            "quad_tr_x_005C5C2": round(float(quad_src[1, 0]), 3),
            "quad_tr_y_005C5C2": round(float(quad_src[1, 1]), 3),
            "quad_br_x_005C5C2": round(float(quad_src[2, 0]), 3),
            "quad_br_y_005C5C2": round(float(quad_src[2, 1]), 3),
            "quad_bl_x_005C5C2": round(float(quad_src[3, 0]), 3),
            "quad_bl_y_005C5C2": round(float(quad_src[3, 1]), 3),

            "H_img_to_table_005C3": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C3": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),
            "H_img_to_table_005C5C2": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C5C2": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),

            "table_length_m_005C3": TABLE_LENGTH_M,
            "table_width_m_005C3": TABLE_WIDTH_M,
            "table_height_m_005C3": TABLE_HEIGHT_M,
            "net_height_m_005C3": NET_HEIGHT_M,
        })

        rows.append(out)

    return pd.DataFrame(rows)


def annotate_points(points: pd.DataFrame, table_objects: pd.DataFrame) -> pd.DataFrame:
    pts = points.copy()

    if pts.empty or table_objects.empty:
        return pts

    pts["frame_num"] = to_num(pts["frame_num"]).fillna(-1).astype(int)
    pts["camera_segment_id_005B2"] = to_num(pts["camera_segment_id_005B2"]).fillna(1).astype(int)
    pts["x_num"] = to_num(pts["x_num"])
    pts["y_num"] = to_num(pts["y_num"])

    obj_map = {}

    for _, o in table_objects.iterrows():
        key = (str(o["review_id"]), int(o["camera_segment_id_005B2"]))
        obj_map[key] = o.to_dict()

    rows = []

    for _, p in pts.iterrows():
        row = p.to_dict()
        key = (str(p["review_id"]), int(p["camera_segment_id_005B2"]))
        obj = obj_map.get(key)

        if obj is None or int(obj.get("table_ok_005C3", 0)) != 1:
            row.update({
                "table_project_ok_005C3": 0,
                "ball_table_x_m_005C3": "",
                "ball_table_y_m_005C3": "",
                "ball_inside_table_005C3": 0,
                "ball_table_edge_margin_m_005C3": "",
                "ball_dist_to_net_m_005C3": "",
                "ball_table_side_005C3": "",
                "ball_table_zone_005C3": "",
            })
            rows.append(row)
            continue

        H = np.asarray(json.loads(obj["H_img_to_table_005C3"]), dtype=np.float64).reshape(3, 3)
        xy = project_points(H, np.asarray([[float(p["x_num"]), float(p["y_num"])]], dtype=np.float32))[0]

        xt = float(xy[0])
        yt = float(xy[1])

        inside = int(abs(xt) <= TABLE_HALF_W and abs(yt) <= TABLE_HALF_L)
        edge_margin = min(TABLE_HALF_W - abs(xt), TABLE_HALF_L - abs(yt))
        dist_net = abs(yt)
        side = "near" if yt >= 0 else "far"

        if inside:
            x_zone = "L" if xt < -TABLE_HALF_W / 3 else ("R" if xt > TABLE_HALF_W / 3 else "C")
            y_zone = "N" if abs(yt) < TABLE_HALF_L / 3 else ("D" if yt > 0 else "F")
            zone = f"{side}_{y_zone}_{x_zone}"
        else:
            zone = "outside_projection"

        row.update({
            "table_project_ok_005C3": 1,
            "ball_table_x_m_005C3": round(xt, 4),
            "ball_table_y_m_005C3": round(yt, 4),
            "ball_inside_table_005C3": inside,
            "ball_table_edge_margin_m_005C3": round(edge_margin, 4),
            "ball_dist_to_net_m_005C3": round(dist_net, 4),
            "ball_table_side_005C3": side,
            "ball_table_zone_005C3": zone,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-objects-c1", default="runs/rally_table_object_005C1_pass33/table_objects_005C1.csv")
    ap.add_argument("--corrections-005c2", default="runs/rally_table_object_005C2_corner_review/table_corner_corrections_005C2.csv")
    ap.add_argument("--corrections-005c5b", default="runs/rally_table_object_005C5_semantic_rework_review/table_corner_semantic_corrections_005C5B.csv")
    ap.add_argument("--display-points", default="runs/rally_table_camera_005B2_pass33/display_points_table_camera_005B2.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_object_005C5C2_merged")
    args = ap.parse_args()

    root = Path.cwd()

    table_path = Path(args.table_objects_c1)
    c2_path = Path(args.corrections_005c2)
    c5b_path = Path(args.corrections_005c5b)
    points_path = Path(args.display_points)
    out_dir = Path(args.out_dir)

    if not table_path.is_absolute():
        table_path = root / table_path
    if not c2_path.is_absolute():
        c2_path = root / c2_path
    if not c5b_path.is_absolute():
        c5b_path = root / c5b_path
    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table_c1 = pd.read_csv(table_path).fillna("")
    points = pd.read_csv(points_path).fillna("")

    c2 = load_corrections(c2_path, "camera_segment_id")
    c5b = load_corrections(c5b_path, "camera_segment_id")

    merged = build_merged_objects(table_c1, c2, c5b)
    projected = annotate_points(points, merged)

    out_objects = out_dir / "table_objects_merged_005C5C2.csv"
    out_points = out_dir / "ball_points_table_projected_005C5C2.csv"
    out_reviews = out_dir / "table_projection_review_summary_005C5C2.csv"
    out_json = out_dir / "table_object_merged_summary_005C5C2.json"

    merged.to_csv(out_objects, index=False, encoding="utf-8")
    projected.to_csv(out_points, index=False, encoding="utf-8")

    review_rows = []

    for review_id, g in projected.groupby("review_id", dropna=False):
        proj_ok = to_num(g["table_project_ok_005C3"]).fillna(0).astype(int)
        inside = to_num(g["ball_inside_table_005C3"]).fillna(0).astype(int)

        objs = merged[merged["review_id"].astype(str).eq(str(review_id))]

        review_rows.append({
            "review_id": str(review_id),
            "points": int(len(g)),
            "project_ok": int(proj_ok.sum()),
            "inside_table": int(inside.sum()),
            "inside_ratio": round(float(inside.sum() / max(1, proj_ok.sum())), 4),
            "table_segments": int(len(objs)),
            "segments_005C1_auto": int((objs["table_source_005C5C2"].astype(str).eq("005C1_auto")).sum()),
            "segments_005C2_human": int((objs["table_source_005C5C2"].astype(str).eq("005C2_human")).sum()),
            "segments_005C5B_semantic": int((objs["table_source_005C5C2"].astype(str).eq("005C5B_semantic")).sum()),
            "segments_rejected": int((objs["table_ok_005C5C2"].astype(int).eq(0)).sum()),
        })

    reviews = pd.DataFrame(review_rows)
    reviews.to_csv(out_reviews, index=False, encoding="utf-8")

    table_total = int(len(merged))
    table_ok = int(to_num(merged["table_ok_005C5C2"]).fillna(0).sum())

    source_counts = merged["table_source_005C5C2"].astype(str).value_counts().to_dict()

    proj_total = int(len(projected))
    proj_ok = int(to_num(projected["table_project_ok_005C3"]).fillna(0).sum())
    inside = int(to_num(projected["ball_inside_table_005C3"]).fillna(0).sum())

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "merge_005C1_auto_005C2_human_and_005C5B_semantic_table_corrections_without_losing_previous_human_edits",
        "table_objects_c1": str(table_path),
        "corrections_005C2": str(c2_path),
        "corrections_005C5B": str(c5b_path),
        "display_points": str(points_path),
        "table_objects_total": table_total,
        "table_objects_ok": table_ok,
        "table_objects_ok_ratio": round(table_ok / max(1, table_total), 4),
        "corrections_005C2_count": int(len(c2)),
        "corrections_005C5B_count": int(len(c5b)),
        "manual_unique_segments": int(len(set(c2.keys()) | set(c5b.keys()))),
        "source_counts": source_counts,
        "projected_points_total": proj_total,
        "projected_points_ok": proj_ok,
        "projected_points_inside_table": inside,
        "projected_points_inside_ratio": round(inside / max(1, proj_ok), 4),
        "reviews": int(projected["review_id"].nunique()) if len(projected) else 0,
        "outputs": {
            "table_objects": str(out_objects),
            "ball_points_projected": str(out_points),
            "review_summary": str(out_reviews),
        },
        "next": "Run 005C4 canonical audit on merged output, then classify table objects as METRIC/PARTIAL/BAD."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C5C2 status=OK")
    print("table_objects_total=", summary["table_objects_total"])
    print("table_objects_ok=", summary["table_objects_ok"])
    print("corrections_005C2_count=", summary["corrections_005C2_count"])
    print("corrections_005C5B_count=", summary["corrections_005C5B_count"])
    print("manual_unique_segments=", summary["manual_unique_segments"])
    print("source_counts=", json.dumps(source_counts, ensure_ascii=False))
    print("projected_points_total=", summary["projected_points_total"])
    print("projected_points_ok=", summary["projected_points_ok"])
    print("projected_points_inside_table=", summary["projected_points_inside_table"])
    print("projected_points_inside_ratio=", summary["projected_points_inside_ratio"])
    print("wrote", out_objects)
    print("wrote", out_points)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
