from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C1"

TABLE_LENGTH_M = 2.740
TABLE_WIDTH_M = 1.525
TABLE_HALF_L = TABLE_LENGTH_M / 2.0
TABLE_HALF_W = TABLE_WIDTH_M / 2.0
TABLE_HEIGHT_M = 0.760
NET_HEIGHT_M = 0.1525


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def hue_dist(h, center):
    d = np.abs(h.astype(np.int16) - int(center))
    return np.minimum(d, 180 - d)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    return inter / max(1.0, union)


def adaptive_table_mask(frame_small: np.ndarray, calib: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hc = int(calib.get("h", 115))
    ht = int(calib.get("h_tol", 16))
    sm = int(calib.get("s", 75))
    vm = int(calib.get("v", 70))

    d = hue_dist(h, hc)

    s_low = max(18, sm - 90)
    s_high = min(255, sm + 125)
    v_low = max(18, vm - 105)
    v_high = min(255, vm + 150)

    mask = (
        (d <= ht) &
        (s >= s_low) & (s <= s_high) &
        (v >= v_low) & (v <= v_high)
    ).astype(np.uint8) * 255

    k1 = np.ones((3, 3), np.uint8)
    k2 = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    return mask


def pick_table_contour(mask: np.ndarray):
    h_img, w_img = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < w_img * h_img * 0.008:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < w_img * 0.18 or h < h_img * 0.04:
            continue

        aspect = w / max(1, h)
        if aspect < 0.85 or aspect > 8.5:
            continue

        fill = area / max(1, w * h)
        cx = x + w / 2
        cy = y + h / 2

        center_penalty = (
            abs(cx - w_img / 2) / max(1, w_img)
            + 0.35 * abs(cy - h_img / 2) / max(1, h_img)
        )

        score = area * (0.5 + fill) * (1.0 - min(0.75, center_penalty))

        if score > best_score:
            best_score = score
            best = cnt

    return best


def white_support(frame_small: np.ndarray, quad: np.ndarray) -> float:
    h_img, w_img = frame_small.shape[:2]
    x, y, w, h = cv2.boundingRect(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2))

    pad = 7
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)

    roi = frame_small[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 72) & (hsv[:, :, 2] > 135)).astype(np.uint8)
    return float(white.mean())


def approx_table_quad(frame_small: np.ndarray, calib: dict) -> dict:
    mask = adaptive_table_mask(frame_small, calib)
    cnt = pick_table_contour(mask)

    h_img, w_img = mask.shape[:2]

    if cnt is None:
        return {"ok": 0, "reason": "no_contour"}

    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))

    if hull_area < 50:
        return {"ok": 0, "reason": "tiny_hull"}

    quad = None
    approx_best = None

    for eps_ratio in [0.015, 0.02, 0.025, 0.035, 0.05, 0.07]:
        eps = eps_ratio * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True)

        if len(approx) == 4:
            approx_best = approx.reshape(4, 2).astype(np.float32)
            break

    if approx_best is not None:
        quad = order_quad_points(approx_best)
        method = "approx4"
    else:
        pts = hull.reshape(-1, 2).astype(np.float32)
        quad = order_quad_points(pts)
        method = "extreme4"

    q_area = quad_area(quad)
    if q_area < w_img * h_img * 0.006:
        return {"ok": 0, "reason": "quad_too_small"}

    quad_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, np.round(quad).astype(np.int32), 255)

    iou = mask_iou(mask, quad_mask)
    fill = area / max(1.0, q_area)
    ws = white_support(frame_small, quad)

    x, y, w, h = cv2.boundingRect(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2))
    aspect_bbox = w / max(1, h)

    # Confidence volontairement simple.
    conf = (
        0.42 * min(1.0, iou / 0.72)
        + 0.28 * min(1.0, fill / 0.82)
        + 0.15 * min(1.0, q_area / max(1.0, w_img * h_img * 0.18))
        + 0.15 * min(1.0, ws / 0.18)
    )

    # Homographie image-small -> table mètres.
    table_pts = np.asarray([
        [-TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W, -TABLE_HALF_L],
        [ TABLE_HALF_W,  TABLE_HALF_L],
        [-TABLE_HALF_W,  TABLE_HALF_L],
    ], dtype=np.float32)

    try:
        H_img_to_table = cv2.getPerspectiveTransform(quad.astype(np.float32), table_pts)
        H_table_to_img = cv2.getPerspectiveTransform(table_pts, quad.astype(np.float32))
        h_ok = 1
    except Exception:
        H_img_to_table = np.eye(3, dtype=np.float64)
        H_table_to_img = np.eye(3, dtype=np.float64)
        h_ok = 0

    return {
        "ok": 1,
        "method": method,
        "quad": quad,
        "mask_area": area,
        "hull_area": hull_area,
        "quad_area": q_area,
        "quad_iou_mask": iou,
        "quad_fill": fill,
        "white_support": ws,
        "aspect_bbox": aspect_bbox,
        "confidence": float(conf),
        "homography_ok": int(h_ok),
        "H_img_to_table": H_img_to_table,
        "H_table_to_img": H_table_to_img,
        "mask": mask,
    }


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2)


def table_to_img(H_table_to_img: np.ndarray, xy_table: list[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(xy_table, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_table_to_img.astype(np.float64))
    return out.reshape(-1, 2)


def draw_table_model(frame: np.ndarray, obj: dict, scale_x: float, scale_y: float):
    if int(obj.get("table_ok_005C1", 0)) != 1:
        return

    H = np.asarray(json.loads(obj["H_table_to_img_005C1"]), dtype=np.float64).reshape(3, 3)

    # H est en coordonnées small-frame. On scale vers source frame.
    def sc(p):
        return (int(round(p[0] * scale_x)), int(round(p[1] * scale_y)))

    # Contour table.
    perimeter = [
        (-TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W, -TABLE_HALF_L),
        ( TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W,  TABLE_HALF_L),
        (-TABLE_HALF_W, -TABLE_HALF_L),
    ]

    pp = table_to_img(H, perimeter)
    for a, b in zip(pp[:-1], pp[1:]):
        cv2.line(frame, sc(a), sc(b), (0, 255, 255), 3, cv2.LINE_AA)

    # Filet : y=0, traverse la largeur.
    net = [(-TABLE_HALF_W, 0.0), (TABLE_HALF_W, 0.0)]
    pp = table_to_img(H, net)
    cv2.line(frame, sc(pp[0]), sc(pp[1]), (0, 180, 255), 3, cv2.LINE_AA)

    # Ligne centrale doubles : x=0, longueur.
    center = [(0.0, -TABLE_HALF_L), (0.0, TABLE_HALF_L)]
    pp = table_to_img(H, center)
    cv2.line(frame, sc(pp[0]), sc(pp[1]), (255, 255, 0), 2, cv2.LINE_AA)

    # Grille légère tous les 0.25 m largeur / 0.5 m longueur.
    for x in np.linspace(-TABLE_HALF_W, TABLE_HALF_W, 5):
        pp = table_to_img(H, [(float(x), -TABLE_HALF_L), (float(x), TABLE_HALF_L)])
        cv2.line(frame, sc(pp[0]), sc(pp[1]), (80, 180, 180), 1, cv2.LINE_AA)

    for y in np.linspace(-TABLE_HALF_L, TABLE_HALF_L, 7):
        pp = table_to_img(H, [(-TABLE_HALF_W, float(y)), (TABLE_HALF_W, float(y))])
        cv2.line(frame, sc(pp[0]), sc(pp[1]), (80, 180, 180), 1, cv2.LINE_AA)

    quad = [
        (obj["quad_tl_x_005C1"], obj["quad_tl_y_005C1"]),
        (obj["quad_tr_x_005C1"], obj["quad_tr_y_005C1"]),
        (obj["quad_br_x_005C1"], obj["quad_br_y_005C1"]),
        (obj["quad_bl_x_005C1"], obj["quad_bl_y_005C1"]),
    ]

    for i, p in enumerate(quad):
        q = sc(p)
        cv2.circle(frame, q, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"C{i+1}", (q[0] + 7, q[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)


def choose_sample_frames(seg_df: pd.DataFrame, max_samples: int) -> list[int]:
    frames = sorted(to_num(seg_df["frame"]).dropna().astype(int).unique().tolist())
    if not frames:
        return []

    if len(frames) <= max_samples:
        return frames

    qs = np.linspace(0.10, 0.90, max_samples)
    idxs = sorted(set(int(round(q * (len(frames) - 1))) for q in qs))
    return [frames[i] for i in idxs]


def build_table_objects(
    reviews: pd.DataFrame,
    timeline: pd.DataFrame,
    out_dir: Path,
    max_samples_per_segment: int,
    min_confidence: float,
) -> tuple[pd.DataFrame, dict]:
    root = Path.cwd()

    objects = []
    diagnostics = {}

    by_review = {str(r["review_id"]): r.to_dict() for _, r in reviews.iterrows()}

    for review_id, tl_review in timeline.groupby("review_id", dropna=False):
        review_id = str(review_id)
        meta = by_review.get(review_id)
        if not meta:
            continue

        clip_path = Path(str(meta["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print("WARN open failed", review_id, clip_path)
            continue

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Calibration depuis 005B2, médiane par review.
        calib = {
            "h": int(to_num(tl_review["calib_h"]).dropna().median()) if "calib_h" in tl_review.columns else 115,
            "s": int(to_num(tl_review["calib_s"]).dropna().median()) if "calib_s" in tl_review.columns else 75,
            "v": int(to_num(tl_review["calib_v"]).dropna().median()) if "calib_v" in tl_review.columns else 70,
            "h_tol": int(to_num(tl_review["calib_h_tol"]).dropna().median()) if "calib_h_tol" in tl_review.columns else 16,
        }

        resize_w = int(to_num(tl_review["resize_w"]).dropna().median())
        resize_h = int(to_num(tl_review["resize_h"]).dropna().median())

        for cam_seg, seg_df in tl_review.groupby("camera_segment_id_005B2", dropna=False):
            cam_seg = int(cam_seg)

            sample_frames = choose_sample_frames(seg_df, max_samples=max_samples_per_segment)
            cand_rows = []

            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
                det = approx_table_quad(small, calib)

                if int(det.get("ok", 0)) != 1:
                    cand_rows.append({
                        "frame": int(frame_idx),
                        "ok": 0,
                        "reason": det.get("reason", "unknown"),
                        "confidence": 0.0,
                    })
                    continue

                cand_rows.append({
                    "frame": int(frame_idx),
                    "ok": 1,
                    "confidence": float(det["confidence"]),
                    "quad_iou_mask": float(det["quad_iou_mask"]),
                    "quad_fill": float(det["quad_fill"]),
                    "white_support": float(det["white_support"]),
                    "quad_area": float(det["quad_area"]),
                    "method": det["method"],
                    "det": det,
                })

            ok_cands = [c for c in cand_rows if int(c.get("ok", 0)) == 1]

            if ok_cands:
                best = sorted(
                    ok_cands,
                    key=lambda c: (
                        float(c["confidence"]),
                        float(c["quad_iou_mask"]),
                        float(c["quad_area"]),
                    ),
                    reverse=True,
                )[0]

                det = best["det"]
                quad = det["quad"]

                table_ok = int(float(best["confidence"]) >= min_confidence and int(det["homography_ok"]) == 1)

                H_img_to_table = det["H_img_to_table"]
                H_table_to_img = det["H_table_to_img"]

                reason = "ok" if table_ok else "low_confidence"
            else:
                best = {}
                quad = np.zeros((4, 2), dtype=np.float32)
                table_ok = 0
                H_img_to_table = np.eye(3, dtype=np.float64)
                H_table_to_img = np.eye(3, dtype=np.float64)
                reason = "no_valid_quad"

            first_f = int(to_num(seg_df["frame"]).min())
            last_f = int(to_num(seg_df["frame"]).max())

            objects.append({
                "review_id": review_id,
                "video_id": str(meta.get("video_id", "")),
                "rally_id": str(meta.get("rally_id", "")),
                "camera_segment_id_005B2": cam_seg,
                "first_frame": first_f,
                "last_frame": last_f,
                "sample_frames": "|".join(str(x) for x in sample_frames),
                "best_frame_005C1": int(best.get("frame", first_f)) if best else first_f,
                "table_ok_005C1": table_ok,
                "table_reason_005C1": reason,
                "table_confidence_005C1": round(float(best.get("confidence", 0.0)), 6),
                "quad_iou_mask_005C1": round(float(best.get("quad_iou_mask", 0.0)), 6),
                "quad_fill_005C1": round(float(best.get("quad_fill", 0.0)), 6),
                "white_support_005C1": round(float(best.get("white_support", 0.0)), 6),
                "quad_area_small_005C1": round(float(best.get("quad_area", 0.0)), 3),
                "quad_method_005C1": str(best.get("method", "")),
                "quad_tl_x_005C1": round(float(quad[0, 0]), 3),
                "quad_tl_y_005C1": round(float(quad[0, 1]), 3),
                "quad_tr_x_005C1": round(float(quad[1, 0]), 3),
                "quad_tr_y_005C1": round(float(quad[1, 1]), 3),
                "quad_br_x_005C1": round(float(quad[2, 0]), 3),
                "quad_br_y_005C1": round(float(quad[2, 1]), 3),
                "quad_bl_x_005C1": round(float(quad[3, 0]), 3),
                "quad_bl_y_005C1": round(float(quad[3, 1]), 3),
                "H_img_to_table_005C1": json.dumps(np.asarray(H_img_to_table).reshape(-1).round(10).tolist()),
                "H_table_to_img_005C1": json.dumps(np.asarray(H_table_to_img).reshape(-1).round(10).tolist()),
                "table_length_m_005C1": TABLE_LENGTH_M,
                "table_width_m_005C1": TABLE_WIDTH_M,
                "table_height_m_005C1": TABLE_HEIGHT_M,
                "net_height_m_005C1": NET_HEIGHT_M,
                "src_w": src_w,
                "src_h": src_h,
                "resize_w": resize_w,
                "resize_h": resize_h,
                "calib_h": calib["h"],
                "calib_s": calib["s"],
                "calib_v": calib["v"],
                "calib_h_tol": calib["h_tol"],
                "clip_path": str(clip_path),
            })

            diagnostics[f"{review_id}_seg{cam_seg}"] = [
                {k: v for k, v in c.items() if k != "det"}
                for c in cand_rows
            ]

        cap.release()

    return pd.DataFrame(objects), diagnostics


def annotate_ball_points(points: pd.DataFrame, table_objects: pd.DataFrame) -> pd.DataFrame:
    if points.empty or table_objects.empty:
        return pd.DataFrame()

    pts = points.copy()

    if "frame_num" not in pts.columns:
        if "frame" in pts.columns:
            pts["frame_num"] = to_num(pts["frame"])
        else:
            raise SystemExit("display points missing frame_num/frame")

    pts["frame_num"] = to_num(pts["frame_num"]).fillna(-1).astype(int)

    if "camera_segment_id_005B2" not in pts.columns:
        raise SystemExit("display points missing camera_segment_id_005B2. Relance depuis 005B2 output.")

    pts["camera_segment_id_005B2"] = to_num(pts["camera_segment_id_005B2"]).fillna(1).astype(int)

    obj_key = {}
    for _, o in table_objects.iterrows():
        obj_key[(str(o["review_id"]), int(o["camera_segment_id_005B2"]))] = o.to_dict()

    out_rows = []

    for _, p in pts.iterrows():
        review_id = str(p["review_id"])
        cam_seg = int(p["camera_segment_id_005B2"])
        obj = obj_key.get((review_id, cam_seg))

        row = p.to_dict()

        if obj is None or int(obj.get("table_ok_005C1", 0)) != 1:
            row.update({
                "table_project_ok_005C1": 0,
                "ball_table_x_m_005C1": "",
                "ball_table_y_m_005C1": "",
                "ball_inside_table_005C1": 0,
                "ball_table_edge_margin_m_005C1": "",
                "ball_dist_to_net_m_005C1": "",
                "ball_table_side_005C1": "",
            })
            out_rows.append(row)
            continue

        H = np.asarray(json.loads(obj["H_img_to_table_005C1"]), dtype=np.float64).reshape(3, 3)

        sx = float(obj["resize_w"]) / max(1.0, float(obj["src_w"]))
        sy = float(obj["resize_h"]) / max(1.0, float(obj["src_h"]))

        x_small = float(p["x_num"]) * sx
        y_small = float(p["y_num"]) * sy

        xy = project_points(H, np.asarray([[x_small, y_small]], dtype=np.float32))[0]
        xt = float(xy[0])
        yt = float(xy[1])

        inside = int(
            abs(xt) <= TABLE_HALF_W
            and abs(yt) <= TABLE_HALF_L
        )

        edge_margin = min(TABLE_HALF_W - abs(xt), TABLE_HALF_L - abs(yt))
        dist_net = abs(yt)
        side = "near" if yt >= 0 else "far"

        row.update({
            "table_project_ok_005C1": 1,
            "ball_table_x_m_005C1": round(xt, 4),
            "ball_table_y_m_005C1": round(yt, 4),
            "ball_inside_table_005C1": inside,
            "ball_table_edge_margin_m_005C1": round(edge_margin, 4),
            "ball_dist_to_net_m_005C1": round(dist_net, 4),
            "ball_table_side_005C1": side,
        })

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def make_overlay_video(review_id: str, clip_path: Path, timeline: pd.DataFrame, objects: pd.DataFrame, points: pd.DataFrame, out_path: Path, max_video_frames: int) -> bool:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = n if max_video_frames <= 0 else min(n, max_video_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (src_w, src_h))

    tl_by_frame = {int(r["frame"]): r for _, r in timeline.iterrows()}
    obj_by_seg = {int(r["camera_segment_id_005B2"]): r.to_dict() for _, r in objects.iterrows()}
    pts_by_frame = {int(r["frame_num"]): r for _, r in points.iterrows()} if not points.empty else {}

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        tl = tl_by_frame.get(fidx)
        cam_seg = 1
        if tl is not None:
            cam_seg = int(tl["camera_segment_id_005B2"])

        obj = obj_by_seg.get(cam_seg)

        if obj:
            scale_x = src_w / max(1.0, float(obj["resize_w"]))
            scale_y = src_h / max(1.0, float(obj["resize_h"]))
            draw_table_model(frame, obj, scale_x=scale_x, scale_y=scale_y)

            conf = float(obj.get("table_confidence_005C1", 0.0))
            ok_txt = int(obj.get("table_ok_005C1", 0))
            cv2.putText(
                frame,
                f"{review_id} f={fidx} cam_seg={cam_seg} table_ok={ok_txt} conf={conf:.2f}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        p = pts_by_frame.get(fidx)
        if p is not None:
            x = int(round(float(p["x_num"])))
            y = int(round(float(p["y_num"])))
            cv2.circle(frame, (x, y), 7, (0, 255, 0), 2, cv2.LINE_AA)

            if int(p.get("table_project_ok_005C1", 0)) == 1:
                tx = float(p["ball_table_x_m_005C1"])
                ty = float(p["ball_table_y_m_005C1"])
                inside = int(p["ball_inside_table_005C1"])
                txt = f"ball_table=({tx:.2f},{ty:.2f}) inside={inside}"
                cv2.putText(frame, txt, (x + 10, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(
            frame,
            "005C1 table object: official metric plane + homography",
            (24, src_h - 24),
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
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--display-points", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples-per-segment", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.42)

    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-video-frames", type=int, default=3000)

    args = ap.parse_args()

    root = Path.cwd()

    review_path = Path(args.review_summary)
    timeline_path = Path(args.timeline)
    out_dir = Path(args.out_dir)

    if not review_path.is_absolute():
        review_path = root / review_path
    if not timeline_path.is_absolute():
        timeline_path = root / timeline_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_csv(review_path).fillna("")
    timeline = pd.read_csv(timeline_path).fillna("")

    if "camera_segment_id_005B2" not in timeline.columns:
        raise SystemExit("timeline doit venir de 005B2 et contenir camera_segment_id_005B2")

    table_objects, diagnostics = build_table_objects(
        reviews=reviews,
        timeline=timeline,
        out_dir=out_dir,
        max_samples_per_segment=args.max_samples_per_segment,
        min_confidence=args.min_confidence,
    )

    projected_points = pd.DataFrame()

    if args.display_points:
        points_path = Path(args.display_points)
        if not points_path.is_absolute():
            points_path = root / points_path

        if points_path.is_file():
            pts = pd.read_csv(points_path).fillna("")
            projected_points = annotate_ball_points(pts, table_objects)

    overlays = []

    if args.make_video and not table_objects.empty:
        # Priorité aux reviews avec plusieurs segments + bonne table.
        obj_summary = table_objects.groupby("review_id").agg(
            segments=("camera_segment_id_005B2", "nunique"),
            ok_segments=("table_ok_005C1", "sum"),
            conf_med=("table_confidence_005C1", "median"),
        ).reset_index()

        obj_summary = obj_summary.sort_values(["segments", "ok_segments", "conf_med"], ascending=[False, False, False]).head(args.max_videos)

        for i, r in enumerate(obj_summary.to_dict(orient="records"), start=1):
            review_id = str(r["review_id"])
            rv = reviews[reviews["review_id"].astype(str).eq(review_id)]
            if rv.empty:
                continue

            clip_path = Path(str(rv.iloc[0]["clip_path"]))
            if not clip_path.is_absolute():
                clip_path = root / clip_path

            tl = timeline[timeline["review_id"].astype(str).eq(review_id)].copy()
            objs = table_objects[table_objects["review_id"].astype(str).eq(review_id)].copy()
            pts = projected_points[projected_points["review_id"].astype(str).eq(review_id)].copy() if not projected_points.empty else pd.DataFrame()

            out_video = out_dir / "overlays" / f"{i:03d}_{review_id}_005C1_table_object_overlay.mp4"

            ok = make_overlay_video(
                review_id=review_id,
                clip_path=clip_path,
                timeline=tl,
                objects=objs,
                points=pts,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})

    out_objects = out_dir / "table_objects_005C1.csv"
    out_projected = out_dir / "ball_points_table_projected_005C1.csv"
    out_diag = out_dir / "table_object_detection_diagnostics_005C1.json"
    out_json = out_dir / "table_object_summary_005C1.json"

    table_objects.to_csv(out_objects, index=False, encoding="utf-8")
    if not projected_points.empty:
        projected_points.to_csv(out_projected, index=False, encoding="utf-8")

    out_diag.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_objects = int(to_num(table_objects["table_ok_005C1"]).fillna(0).sum()) if not table_objects.empty else 0
    total_objects = int(len(table_objects))

    if not projected_points.empty and "table_project_ok_005C1" in projected_points.columns:
        proj_ok = int(to_num(projected_points["table_project_ok_005C1"]).fillna(0).sum())
        inside = int(to_num(projected_points["ball_inside_table_005C1"]).fillna(0).sum())
        proj_total = int(len(projected_points))
    else:
        proj_ok = 0
        inside = 0
        proj_total = 0

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "build_metric_table_object_from_adaptive_color_mask_and_camera_segments",
        "official_table_model": {
            "length_m": TABLE_LENGTH_M,
            "width_m": TABLE_WIDTH_M,
            "height_m": TABLE_HEIGHT_M,
            "net_height_m": NET_HEIGHT_M,
            "coordinate_system": "x=width meters, y=length meters, z=table_plane_0"
        },
        "review_summary": str(review_path),
        "timeline": str(timeline_path),
        "display_points": str(args.display_points),
        "params": {
            "max_samples_per_segment": args.max_samples_per_segment,
            "min_confidence": args.min_confidence,
        },
        "reviews": int(table_objects["review_id"].nunique()) if not table_objects.empty else 0,
        "table_objects_total": total_objects,
        "table_objects_ok": ok_objects,
        "table_objects_ok_ratio": round(ok_objects / max(1, total_objects), 4),
        "projected_points_total": proj_total,
        "projected_points_ok": proj_ok,
        "projected_points_inside_table": inside,
        "projected_points_inside_ratio": round(inside / max(1, proj_ok), 4),
        "overlays": overlays,
        "outputs": {
            "table_objects": str(out_objects),
            "projected_points": str(out_projected) if not projected_points.empty else "",
            "diagnostics": str(out_diag),
        },
        "next": "Review table-object overlays. If quad follows the official table plane, use projected ball coordinates for plausibility filtering and bounce candidates."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C1 status=OK")
    print("reviews=", summary["reviews"])
    print("table_objects_total=", summary["table_objects_total"])
    print("table_objects_ok=", summary["table_objects_ok"])
    print("table_objects_ok_ratio=", summary["table_objects_ok_ratio"])
    print("projected_points_total=", summary["projected_points_total"])
    print("projected_points_ok=", summary["projected_points_ok"])
    print("projected_points_inside_table=", summary["projected_points_inside_table"])
    print("projected_points_inside_ratio=", summary["projected_points_inside_ratio"])
    print("overlays=", len(overlays))
    print("wrote", out_objects)
    if not projected_points.empty:
        print("wrote", out_projected)
    print("wrote", out_diag)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
