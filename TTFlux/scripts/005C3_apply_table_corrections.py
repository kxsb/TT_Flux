from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C3"

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


def table_to_img(H_table_to_img: np.ndarray, pts_table: list[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(pts_table, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_table_to_img.astype(np.float64))
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


def corrected_quad_src(corr: dict) -> np.ndarray:
    return np.asarray([
        [float(corr["tl_x"]), float(corr["tl_y"])],
        [float(corr["tr_x"]), float(corr["tr_y"])],
        [float(corr["br_x"]), float(corr["br_y"])],
        [float(corr["bl_x"]), float(corr["bl_y"])],
    ], dtype=np.float32)


def build_corrected_objects(table: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    corr_map = {}

    if not corrections.empty:
        for _, r in corrections.iterrows():
            key = (str(r["review_id"]), int(to_num(pd.Series([r["camera_segment_id"]])).fillna(0).iloc[0]))
            corr_map[key] = r.to_dict()

    rows = []

    for _, r in table.iterrows():
        row = r.to_dict()

        review_id = str(row["review_id"])
        seg = int(to_num(pd.Series([row["camera_segment_id_005B2"]])).fillna(1).iloc[0])
        key = (review_id, seg)

        corr = corr_map.get(key)
        source = "005C1_auto"
        status = "auto"
        bad = False

        if corr is not None:
            status = str(corr.get("status", "corrected"))
            if status == "bad_frame":
                bad = True
                quad_src = original_quad_src(row)
                source = "005C2_bad_frame_auto_fallback"
            else:
                quad_src = corrected_quad_src(corr)
                source = "005C2_human"
        else:
            quad_src = original_quad_src(row)

        try:
            H_img_to_table, H_table_to_img = homographies_from_quad_src(quad_src)
            h_ok = 1
        except Exception:
            H_img_to_table = np.eye(3, dtype=np.float64)
            H_table_to_img = np.eye(3, dtype=np.float64)
            h_ok = 0

        table_ok = int(h_ok == 1 and not bad)

        out = dict(row)
        out.update({
            "table_ok_005C3": table_ok,
            "table_source_005C3": source,
            "table_status_005C3": status,
            "quad_tl_x_005C3": round(float(quad_src[0, 0]), 3),
            "quad_tl_y_005C3": round(float(quad_src[0, 1]), 3),
            "quad_tr_x_005C3": round(float(quad_src[1, 0]), 3),
            "quad_tr_y_005C3": round(float(quad_src[1, 1]), 3),
            "quad_br_x_005C3": round(float(quad_src[2, 0]), 3),
            "quad_br_y_005C3": round(float(quad_src[2, 1]), 3),
            "quad_bl_x_005C3": round(float(quad_src[3, 0]), 3),
            "quad_bl_y_005C3": round(float(quad_src[3, 1]), 3),
            "H_img_to_table_005C3": json.dumps(H_img_to_table.reshape(-1).round(10).tolist()),
            "H_table_to_img_005C3": json.dumps(H_table_to_img.reshape(-1).round(10).tolist()),
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

    if "frame_num" not in pts.columns:
        if "frame" in pts.columns:
            pts["frame_num"] = to_num(pts["frame"])
        else:
            raise SystemExit("points: frame_num/frame absent")

    if "camera_segment_id_005B2" not in pts.columns:
        raise SystemExit("points: camera_segment_id_005B2 absent. Utilise le fichier display_points_table_camera_005B2.csv.")

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


def draw_table_model(frame, obj):
    H = np.asarray(json.loads(obj["H_table_to_img_005C3"]), dtype=np.float64).reshape(3, 3)

    def pt(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    perimeter = [
        (-TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W, -TABLE_HALF_L),
    ]

    pp = table_to_img(H, perimeter)
    for a, b in zip(pp[:-1], pp[1:]):
        cv2.line(frame, pt(a), pt(b), (0, 255, 255), 3, cv2.LINE_AA)

    # Net y=0.
    net = [(-TABLE_HALF_W, 0.0), (TABLE_HALF_W, 0.0)]
    pp = table_to_img(H, net)
    cv2.line(frame, pt(pp[0]), pt(pp[1]), (0, 180, 255), 3, cv2.LINE_AA)

    # Center x=0.
    center = [(0.0, -TABLE_HALF_L), (0.0, TABLE_HALF_L)]
    pp = table_to_img(H, center)
    cv2.line(frame, pt(pp[0]), pt(pp[1]), (255, 255, 0), 2, cv2.LINE_AA)

    # Grid.
    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        pp = table_to_img(H, [(float(x), -TABLE_HALF_L), (float(x), TABLE_HALF_L)])
        cv2.line(frame, pt(pp[0]), pt(pp[1]), (80, 180, 180), 1, cv2.LINE_AA)

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        pp = table_to_img(H, [(-TABLE_HALF_W, float(y)), (TABLE_HALF_W, float(y))])
        cv2.line(frame, pt(pp[0]), pt(pp[1]), (80, 180, 180), 1, cv2.LINE_AA)


def make_overlay(review_id: str, clip_path: Path, points: pd.DataFrame, objects: pd.DataFrame, out_path: Path, max_video_frames: int) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("WARN open failed", review_id, clip_path)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = n if max_video_frames <= 0 else min(n, max_video_frames)

    obj_by_seg = {
        int(o["camera_segment_id_005B2"]): o.to_dict()
        for _, o in objects.iterrows()
        if int(o.get("table_ok_005C3", 0)) == 1
    }

    pts_by_frame = {
        int(p["frame_num"]): p.to_dict()
        for _, p in points.iterrows()
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        p = pts_by_frame.get(fidx)
        seg = 1

        if p is not None:
            seg = int(p.get("camera_segment_id_005B2", 1))

        obj = obj_by_seg.get(seg)
        if obj is not None:
            draw_table_model(frame, obj)
            cv2.putText(
                frame,
                f"{review_id} f={fidx} seg={seg} table={obj.get('table_source_005C3','')}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if p is not None:
            x = int(round(float(p["x_num"])))
            y = int(round(float(p["y_num"])))
            cv2.circle(frame, (x, y), 7, (0, 255, 0), 2, cv2.LINE_AA)

            if int(p.get("table_project_ok_005C3", 0)) == 1:
                tx = float(p["ball_table_x_m_005C3"])
                ty = float(p["ball_table_y_m_005C3"])
                inside = int(p["ball_inside_table_005C3"])
                zone = str(p.get("ball_table_zone_005C3", ""))
                txt = f"({tx:.2f},{ty:.2f}) inside={inside} {zone}"
                cv2.putText(frame, txt, (x + 10, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            "005C3 corrected table object + ball table coordinates",
            (24, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C1_pass33/table_objects_005C1.csv")
    ap.add_argument("--corrections", default="runs/rally_table_object_005C2_corner_review/table_corner_corrections_005C2.csv")
    ap.add_argument("--display-points", default="runs/rally_table_camera_005B2_pass33/display_points_table_camera_005B2.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_object_005C3_corrected")
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-video-frames", type=int, default=3000)
    args = ap.parse_args()

    root = Path.cwd()

    table_path = Path(args.table_objects)
    corrections_path = Path(args.corrections)
    points_path = Path(args.display_points)
    out_dir = Path(args.out_dir)

    for p_name, p in [("table_objects", table_path), ("display_points", points_path)]:
        if not p.is_absolute():
            locals()[p_name + "_path"] = root / p

    if not table_path.is_absolute():
        table_path = root / table_path
    if not corrections_path.is_absolute():
        corrections_path = root / corrections_path
    if not points_path.is_absolute():
        points_path = root / points_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path).fillna("")

    if corrections_path.is_file():
        corrections = pd.read_csv(corrections_path).fillna("")
    else:
        corrections = pd.DataFrame()

    points = pd.read_csv(points_path).fillna("")

    corrected = build_corrected_objects(table, corrections)
    projected = annotate_points(points, corrected)

    out_objects = out_dir / "table_objects_corrected_005C3.csv"
    out_points = out_dir / "ball_points_table_projected_005C3.csv"
    out_reviews = out_dir / "table_projection_review_summary_005C3.csv"
    out_json = out_dir / "table_object_corrected_summary_005C3.json"

    corrected.to_csv(out_objects, index=False, encoding="utf-8")
    projected.to_csv(out_points, index=False, encoding="utf-8")

    review_rows = []

    for review_id, g in projected.groupby("review_id", dropna=False):
        review_id = str(review_id)
        proj_ok = to_num(g["table_project_ok_005C3"]).fillna(0).astype(int)
        inside = to_num(g["ball_inside_table_005C3"]).fillna(0).astype(int)

        objs = corrected[corrected["review_id"].astype(str).eq(review_id)]

        review_rows.append({
            "review_id": review_id,
            "points": int(len(g)),
            "project_ok": int(proj_ok.sum()),
            "inside_table": int(inside.sum()),
            "inside_ratio": round(float(inside.sum() / max(1, proj_ok.sum())), 4),
            "table_segments": int(len(objs)),
            "human_segments": int((objs["table_source_005C3"].astype(str).eq("005C2_human")).sum()),
            "bad_segments": int((objs["table_status_005C3"].astype(str).eq("bad_frame")).sum()),
            "clip_path": str(g.iloc[0].get("clip_path", "")) if len(g) else "",
            "overlay_005C3": "",
        })

    review_summary = pd.DataFrame(review_rows)
    review_summary.to_csv(out_reviews, index=False, encoding="utf-8")

    overlays = []

    if args.make_video:
        viz = review_summary.sort_values(
            ["human_segments", "inside_ratio", "points"],
            ascending=[False, True, False],
        ).head(args.max_videos).copy()

        for i, (_, r) in enumerate(viz.iterrows(), start=1):
            review_id = str(r["review_id"])
            pts = projected[projected["review_id"].astype(str).eq(review_id)].copy()
            objs = corrected[corrected["review_id"].astype(str).eq(review_id)].copy()

            if pts.empty:
                continue

            clip_path = Path(str(pts.iloc[0].get("clip_path", "")))
            if not clip_path.is_absolute():
                clip_path = root / clip_path

            if not clip_path.is_file():
                # fallback depuis objects
                clip_path = Path(str(objs.iloc[0].get("clip_path", "")))
                if not clip_path.is_absolute():
                    clip_path = root / clip_path

            if not clip_path.is_file():
                print("WARN missing clip", review_id, clip_path)
                continue

            out_video = out_dir / "overlays" / f"{i:03d}_{review_id}_005C3_corrected_table_overlay.mp4"

            ok = make_overlay(
                review_id=review_id,
                clip_path=clip_path,
                points=pts,
                objects=objs,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})
                review_summary.loc[
                    review_summary["review_id"].astype(str).eq(review_id),
                    "overlay_005C3",
                ] = str(out_video)

        review_summary.to_csv(out_reviews, index=False, encoding="utf-8")

    table_total = int(len(corrected))
    table_ok = int(to_num(corrected["table_ok_005C3"]).fillna(0).sum())
    human = int((corrected["table_source_005C3"].astype(str).eq("005C2_human")).sum())
    bad = int((corrected["table_status_005C3"].astype(str).eq("bad_frame")).sum())

    proj_total = int(len(projected))
    proj_ok = int(to_num(projected["table_project_ok_005C3"]).fillna(0).sum())
    inside = int(to_num(projected["ball_inside_table_005C3"]).fillna(0).sum())

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "apply_human_table_corner_corrections_recompute_metric_homographies_project_ball",
        "table_objects": str(table_path),
        "corrections": str(corrections_path),
        "display_points": str(points_path),
        "official_table_model": {
            "length_m": TABLE_LENGTH_M,
            "width_m": TABLE_WIDTH_M,
            "height_m": TABLE_HEIGHT_M,
            "net_height_m": NET_HEIGHT_M,
        },
        "table_objects_total": table_total,
        "table_objects_ok": table_ok,
        "table_objects_ok_ratio": round(table_ok / max(1, table_total), 4),
        "human_corrected_segments": human,
        "bad_frame_segments": bad,
        "projected_points_total": proj_total,
        "projected_points_ok": proj_ok,
        "projected_points_inside_table": inside,
        "projected_points_inside_ratio": round(inside / max(1, proj_ok), 4),
        "reviews": int(projected["review_id"].nunique()) if len(projected) else 0,
        "overlays": overlays,
        "outputs": {
            "table_objects_corrected": str(out_objects),
            "ball_points_projected": str(out_points),
            "review_summary": str(out_reviews),
        },
        "next": "Use corrected table projection for plausibility filtering and bounce candidate detection."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C3 status=OK")
    print("table_objects_total=", summary["table_objects_total"])
    print("table_objects_ok=", summary["table_objects_ok"])
    print("table_objects_ok_ratio=", summary["table_objects_ok_ratio"])
    print("human_corrected_segments=", summary["human_corrected_segments"])
    print("bad_frame_segments=", summary["bad_frame_segments"])
    print("projected_points_total=", summary["projected_points_total"])
    print("projected_points_ok=", summary["projected_points_ok"])
    print("projected_points_inside_table=", summary["projected_points_inside_table"])
    print("projected_points_inside_ratio=", summary["projected_points_inside_ratio"])
    print("overlays=", len(overlays))
    print("wrote", out_objects)
    print("wrote", out_points)
    print("wrote", out_reviews)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
