from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005B2"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def robust_thr(vals, base_min: float, k: float = 5.0) -> float:
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float32)
    if vals.size < 8:
        return base_min
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-9
    return max(base_min, med + k * mad)


def hue_dist(h, center):
    d = np.abs(h.astype(np.int16) - int(center))
    return np.minimum(d, 180 - d)


def bbox_iou(a, b):
    if not a or not b or int(a.get("ok", 0)) != 1 or int(b.get("ok", 0)) != 1:
        return 0.0

    ax1 = float(a["x"]); ay1 = float(a["y"])
    ax2 = ax1 + float(a["w"]); ay2 = ay1 + float(a["h"])

    bx1 = float(b["x"]); by1 = float(b["y"])
    bx2 = bx1 + float(b["w"]); by2 = by1 + float(b["h"])

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / max(1e-9, aa + ab - inter)


def hist_hs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def coarse_table_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Table bleue/cyan fréquente, volontairement large.
    mask = (
        (h >= 78) & (h <= 135) &
        (s >= 35) &
        (v >= 35)
    ).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask


def pick_table_component(mask):
    h_img, w_img = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    best = None
    best_score = -1

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < w_img * h_img * 0.008:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < w_img * 0.18 or h < h_img * 0.04:
            continue

        aspect = w / max(1, h)
        if aspect < 0.9 or aspect > 8.0:
            continue

        fill = area / max(1, w * h)

        # Favorise grandes surfaces centrales et remplies.
        cx = x + w / 2
        cy = y + h / 2
        center_penalty = abs(cx - w_img / 2) / max(1, w_img) + 0.35 * abs(cy - h_img / 2) / max(1, h_img)

        score = area * (0.65 + fill) * (1.0 - min(0.7, center_penalty))

        if score > best_score:
            best_score = score
            best = cnt

    return best


def calibrate_table_color(clip_path: Path, resize_w: int, sample_count: int = 12):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {"ok": 0, "h": 105, "s": 110, "v": 100, "h_tol": 16, "samples": 0}

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if n <= 0 or w0 <= 0 or h0 <= 0:
        cap.release()
        return {"ok": 0, "h": 105, "s": 110, "v": 100, "h_tol": 16, "samples": 0}

    resize_h = max(1, int(round(h0 * resize_w / max(1, w0))))
    frame_ids = np.linspace(0, max(0, n - 1), min(sample_count, n)).astype(int).tolist()

    hs = []
    ss = []
    vs = []

    for f in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        mask = coarse_table_mask(small)
        cnt = pick_table_component(mask)

        if cnt is None:
            continue

        comp = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(comp, [cnt], -1, 255, -1)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pix = hsv[comp > 0]

        if len(pix) < 200:
            continue

        # Sous-échantillonne pour stabilité.
        if len(pix) > 6000:
            idx = np.linspace(0, len(pix) - 1, 6000).astype(int)
            pix = pix[idx]

        hs.extend(pix[:, 0].astype(int).tolist())
        ss.extend(pix[:, 1].astype(int).tolist())
        vs.extend(pix[:, 2].astype(int).tolist())

    cap.release()

    if len(hs) < 500:
        return {"ok": 0, "h": 105, "s": 110, "v": 100, "h_tol": 18, "samples": len(hs)}

    h_med = int(np.median(hs))
    s_med = int(np.median(ss))
    v_med = int(np.median(vs))

    hd = np.asarray([min(abs(h - h_med), 180 - abs(h - h_med)) for h in hs], dtype=np.float32)
    h_tol = int(max(10, min(28, np.percentile(hd, 85) + 5)))

    return {
        "ok": 1,
        "h": h_med,
        "s": s_med,
        "v": v_med,
        "h_tol": h_tol,
        "samples": len(hs),
    }


def adaptive_table_mask(frame, calib):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hc = int(calib.get("h", 105))
    ht = int(calib.get("h_tol", 16))
    sm = int(calib.get("s", 110))
    vm = int(calib.get("v", 100))

    d = hue_dist(h, hc)

    s_low = max(20, sm - 85)
    s_high = min(255, sm + 115)
    v_low = max(20, vm - 95)
    v_high = min(255, vm + 135)

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


def table_geometry(frame, calib):
    mask = adaptive_table_mask(frame, calib)
    cnt = pick_table_component(mask)

    h_img, w_img = mask.shape[:2]

    if cnt is None:
        return {
            "ok": 0,
            "x": "", "y": "", "w": "", "h": "",
            "cx": "", "cy": "",
            "area_ratio": 0.0,
            "fill": 0.0,
            "white_edge_ratio": 0.0,
        }

    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    fill = area / max(1, w * h)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = hsv[max(0, y - 4):min(h_img, y + h + 4), max(0, x - 4):min(w_img, x + w + 4)]
    if roi.size:
        white = ((roi[:, :, 1] < 75) & (roi[:, :, 2] > 145)).astype(np.uint8)
        white_edge_ratio = float(white.mean())
    else:
        white_edge_ratio = 0.0

    return {
        "ok": 1,
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "cx": round(float(x + w / 2), 3),
        "cy": round(float(y + h / 2), 3),
        "area_ratio": round(float(area / max(1, w_img * h_img)), 6),
        "fill": round(float(fill), 6),
        "white_edge_ratio": round(float(white_edge_ratio), 6),
    }


def scan_clip(review_id: str, clip_path: Path, resize_w: int, max_frames: int):
    calib = calibrate_table_color(clip_path, resize_w=resize_w)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return pd.DataFrame(), calib

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    resize_h = max(1, int(round(h0 * resize_w / max(1, w0))))
    diag = math.hypot(resize_w, resize_h)

    rows = []

    prev_gray = None
    prev_edge = None
    prev_hist = None
    prev_table = None

    limit = n0 if max_frames <= 0 else min(n0, max_frames)

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 160)
        hist = hist_hs(small)
        tbl = table_geometry(small, calib)

        if prev_gray is None:
            pix_diff = 0.0
            edge_diff = 0.0
            hist_diff = 0.0
            table_shift = 0.0
            table_area_log = 0.0
            table_iou = 1.0
        else:
            pix_diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            edge_diff = float(np.mean(cv2.absdiff(edge, prev_edge)) / 255.0)
            hist_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

            if int(tbl.get("ok", 0)) == 1 and int(prev_table.get("ok", 0)) == 1:
                dx = float(tbl["cx"]) - float(prev_table["cx"])
                dy = float(tbl["cy"]) - float(prev_table["cy"])
                table_shift = math.hypot(dx, dy) / max(1e-9, diag)

                a1 = max(1e-9, float(prev_table["area_ratio"]))
                a2 = max(1e-9, float(tbl["area_ratio"]))
                table_area_log = abs(math.log(a2 / a1))
                table_iou = bbox_iou(prev_table, tbl)
            else:
                table_shift = 0.0
                table_area_log = 0.0
                table_iou = 0.0

        rows.append({
            "review_id": review_id,
            "frame": int(fidx),
            "fps": round(fps, 3),
            "src_w": int(w0),
            "src_h": int(h0),
            "resize_w": int(resize_w),
            "resize_h": int(resize_h),
            "pix_diff": round(pix_diff, 6),
            "edge_diff": round(edge_diff, 6),
            "hist_diff": round(hist_diff, 6),
            "table_ok": int(tbl["ok"]),
            "table_x": tbl["x"],
            "table_y": tbl["y"],
            "table_w": tbl["w"],
            "table_h": tbl["h"],
            "table_cx": tbl["cx"],
            "table_cy": tbl["cy"],
            "table_area_ratio": tbl["area_ratio"],
            "table_fill": tbl["fill"],
            "table_white_edge_ratio": tbl["white_edge_ratio"],
            "table_shift_norm": round(table_shift, 6),
            "table_area_logdiff": round(table_area_log, 6),
            "table_iou_prev": round(table_iou, 6),
            "calib_ok": int(calib["ok"]),
            "calib_h": int(calib["h"]),
            "calib_s": int(calib["s"]),
            "calib_v": int(calib["v"]),
            "calib_h_tol": int(calib["h_tol"]),
        })

        prev_gray = gray
        prev_edge = edge
        prev_hist = hist
        prev_table = tbl

    cap.release()

    return pd.DataFrame(rows), calib


def classify(tl: pd.DataFrame, min_cut_gap: int):
    if tl.empty:
        return tl, {}

    tl = tl.copy()

    pix_thr = robust_thr(tl["pix_diff"], base_min=0.145, k=5.0)
    hist_thr = robust_thr(tl["hist_diff"], base_min=0.36, k=5.0)
    edge_thr = robust_thr(tl["edge_diff"], base_min=0.09, k=5.0)

    table_shift_thr = robust_thr(tl["table_shift_norm"], base_min=0.18, k=5.0)
    table_area_thr = robust_thr(tl["table_area_logdiff"], base_min=0.55, k=5.0)

    table_iou = to_num(tl["table_iou_prev"]).fillna(1.0)
    table_ok = to_num(tl["table_ok"]).fillna(0).astype(int)

    hard_image_cut = (
        (to_num(tl["hist_diff"]).fillna(0) >= hist_thr)
        & (to_num(tl["pix_diff"]).fillna(0) >= pix_thr)
    )

    table_jump = (
        (table_ok == 1)
        & (
            (to_num(tl["table_shift_norm"]).fillna(0) >= table_shift_thr)
            | (to_num(tl["table_area_logdiff"]).fillna(0) >= table_area_thr)
            | (table_iou <= 0.28)
        )
    )

    table_loss_jump = (
        (table_iou <= 0.05)
        & (to_num(tl["hist_diff"]).fillna(0) >= 0.22)
        & (to_num(tl["pix_diff"]).fillna(0) >= 0.08)
    )

    raw = (
        hard_image_cut
        | (
            table_jump
            & (to_num(tl["hist_diff"]).fillna(0) >= 0.18)
            & (to_num(tl["pix_diff"]).fillna(0) >= 0.06)
        )
        | table_loss_jump
    )

    tl["raw_camera_cut_005B2"] = raw.astype(int)

    segment = 1
    last_cut = -999999
    cuts = []
    segs = []

    for _, r in tl.iterrows():
        f = int(r["frame"])
        is_raw = int(r["raw_camera_cut_005B2"]) == 1

        if f == 0:
            is_cut = False
        elif is_raw and f - last_cut >= min_cut_gap:
            is_cut = True
            last_cut = f
            segment += 1
        else:
            is_cut = False

        cuts.append(int(is_cut))
        segs.append(int(segment))

    tl["camera_cut_005B2"] = cuts
    tl["camera_segment_id_005B2"] = segs

    tl["table_motion_score_005B2"] = (
        0.65 * to_num(tl["table_shift_norm"]).fillna(0)
        + 0.30 * to_num(tl["table_area_logdiff"]).fillna(0)
        + 0.05 * to_num(tl["pix_diff"]).fillna(0)
    )

    motion_thr = robust_thr(tl["table_motion_score_005B2"], base_min=0.065, k=4.0)

    tl["camera_motion_005B2"] = (
        (to_num(tl["table_motion_score_005B2"]).fillna(0) >= motion_thr)
        & (tl["camera_cut_005B2"].astype(int) == 0)
    ).astype(int)

    tl["cut_score_005B2"] = (
        0.35 * to_num(tl["hist_diff"]).fillna(0)
        + 0.25 * to_num(tl["pix_diff"]).fillna(0)
        + 0.20 * to_num(tl["table_shift_norm"]).fillna(0)
        + 0.20 * to_num(tl["table_area_logdiff"]).fillna(0)
        + 0.10 * (1.0 - to_num(tl["table_iou_prev"]).fillna(1.0))
    )

    thresholds = {
        "pix_thr": round(float(pix_thr), 6),
        "hist_thr": round(float(hist_thr), 6),
        "edge_thr": round(float(edge_thr), 6),
        "table_shift_thr": round(float(table_shift_thr), 6),
        "table_area_thr": round(float(table_area_thr), 6),
        "motion_thr": round(float(motion_thr), 6),
    }

    return tl, thresholds


def annotate_points(points: pd.DataFrame, timeline: pd.DataFrame):
    if points.empty or timeline.empty:
        return points

    pts = points.copy()

    if "frame_num" not in pts.columns:
        pts["frame_num"] = to_num(pts["frame"])

    pts["frame_num"] = to_num(pts["frame_num"]).fillna(-1).astype(int)

    small = timeline[[
        "review_id",
        "frame",
        "camera_segment_id_005B2",
        "camera_cut_005B2",
        "camera_motion_005B2",
        "table_ok",
        "table_x",
        "table_y",
        "table_w",
        "table_h",
        "table_area_ratio",
        "table_iou_prev",
        "table_shift_norm",
        "cut_score_005B2",
    ]].copy()

    small["frame_num"] = to_num(small["frame"]).fillna(-1).astype(int)
    small = small.drop(columns=["frame"])

    out = pts.merge(small, on=["review_id", "frame_num"], how="left")

    out["camera_segment_id_005B2"] = to_num(out["camera_segment_id_005B2"]).fillna(1).astype(int)
    out["camera_cut_005B2"] = to_num(out["camera_cut_005B2"]).fillna(0).astype(int)
    out["camera_motion_005B2"] = to_num(out["camera_motion_005B2"]).fillna(0).astype(int)

    return out


def make_overlay(review_id, clip_path, tl, out_path, max_video_frames):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = n if max_video_frames <= 0 else min(n, max_video_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = {int(r["frame"]): r for _, r in tl.iterrows()}

    for fidx in range(limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        r = by_frame.get(fidx)

        if r is not None:
            seg = int(r["camera_segment_id_005B2"])
            cut = int(r["camera_cut_005B2"]) == 1
            motion = int(r["camera_motion_005B2"]) == 1

            if int(r.get("table_ok", 0)) == 1:
                rw = float(r["resize_w"]); rh = float(r["resize_h"])
                sx = w / max(1.0, rw); sy = h / max(1.0, rh)

                x = int(float(r["table_x"]) * sx)
                y = int(float(r["table_y"]) * sy)
                bw = int(float(r["table_w"]) * sx)
                bh = int(float(r["table_h"]) * sy)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 180, 0), 2, cv2.LINE_AA)

            if cut:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                cv2.putText(frame, "CAMERA CUT RESET", (28, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            elif motion:
                cv2.putText(frame, "CAMERA MOTION / TABLE DRIFT", (28, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 185, 255), 2, cv2.LINE_AA)

            cv2.putText(
                frame,
                f"{review_id} f={fidx} seg={seg} H={int(r['calib_h'])} tol={int(r['calib_h_tol'])} iou={float(r['table_iou_prev']):.2f}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-summary", required=True)
    ap.add_argument("--display-points", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resize-w", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-cut-gap", type=int, default=24)
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=12)
    ap.add_argument("--max-video-frames", type=int, default=3000)
    args = ap.parse_args()

    root = Path.cwd()

    review_path = Path(args.review_summary)
    out_dir = Path(args.out_dir)

    if not review_path.is_absolute():
        review_path = root / review_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_csv(review_path).fillna("")

    timelines = []
    review_rows = []
    calib_by_review = {}
    thresholds_by_review = {}

    for i, (_, r) in enumerate(reviews.iterrows(), start=1):
        review_id = str(r["review_id"])
        clip_path = Path(str(r["clip_path"]))

        if not clip_path.is_absolute():
            clip_path = root / clip_path

        if not clip_path.is_file():
            print("WARN missing clip", review_id, clip_path)
            continue

        print(f"[{i}/{len(reviews)}] 005B2 scan {review_id}")

        tl, calib = scan_clip(review_id, clip_path, resize_w=args.resize_w, max_frames=args.max_frames)
        if tl.empty:
            continue

        tl, thresholds = classify(tl, min_cut_gap=args.min_cut_gap)

        timelines.append(tl)
        calib_by_review[review_id] = calib
        thresholds_by_review[review_id] = thresholds

        cuts = tl[tl["camera_cut_005B2"].astype(int).eq(1)]
        motion_n = int(tl["camera_motion_005B2"].astype(int).sum())

        review_rows.append({
            "review_id": review_id,
            "video_id": str(r.get("video_id", "")),
            "rally_id": str(r.get("rally_id", "")),
            "frames": int(len(tl)),
            "camera_segments_005B2": int(tl["camera_segment_id_005B2"].max()),
            "camera_cuts_005B2": int(len(cuts)),
            "cut_frames_005B2": "|".join(str(int(x)) for x in cuts["frame"].tolist()),
            "camera_motion_frames_005B2": motion_n,
            "table_ok_ratio_005B2": round(float(to_num(tl["table_ok"]).mean()), 4),
            "table_iou_med_005B2": round(float(to_num(tl["table_iou_prev"]).median()), 4),
            "table_shift_p95_005B2": round(float(to_num(tl["table_shift_norm"]).quantile(0.95)), 6),
            "calib_ok": int(calib["ok"]),
            "calib_h": int(calib["h"]),
            "calib_s": int(calib["s"]),
            "calib_v": int(calib["v"]),
            "calib_h_tol": int(calib["h_tol"]),
            "clip_path": str(clip_path),
            "overlay_005B2": "",
        })

    timeline = pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame()
    review_out = pd.DataFrame(review_rows)

    points_out = pd.DataFrame()

    if args.display_points:
        points_path = Path(args.display_points)
        if not points_path.is_absolute():
            points_path = root / points_path

        if points_path.is_file() and not timeline.empty:
            pts = pd.read_csv(points_path).fillna("")
            points_out = annotate_points(pts, timeline)

    overlays = []

    if args.make_video and not review_out.empty:
        viz = review_out.sort_values(
            ["camera_cuts_005B2", "camera_motion_frames_005B2", "table_ok_ratio_005B2"],
            ascending=[False, False, True],
        ).head(args.max_videos)

        for j, (_, r) in enumerate(viz.iterrows(), start=1):
            review_id = str(r["review_id"])
            clip_path = Path(str(r["clip_path"]))
            tl = timeline[timeline["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{j:03d}_{review_id}_005B2_table_camera_overlay.mp4"

            ok = make_overlay(
                review_id=review_id,
                clip_path=clip_path,
                tl=tl,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})
                review_out.loc[review_out["review_id"].astype(str).eq(review_id), "overlay_005B2"] = str(out_video)

    out_timeline = out_dir / "table_camera_timeline_005B2.csv"
    out_reviews = out_dir / "table_camera_review_summary_005B2.csv"
    out_points = out_dir / "display_points_table_camera_005B2.csv"
    out_json = out_dir / "table_camera_detection_summary_005B2.json"

    timeline.to_csv(out_timeline, index=False, encoding="utf-8")
    review_out.to_csv(out_reviews, index=False, encoding="utf-8")
    if not points_out.empty:
        points_out.to_csv(out_points, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "adaptive_table_color_geometry_for_camera_cut_and_motion_detection",
        "review_summary": str(review_path),
        "display_points": str(args.display_points),
        "params": {
            "resize_w": args.resize_w,
            "max_frames": args.max_frames,
            "min_cut_gap": args.min_cut_gap,
        },
        "reviews": int(review_out["review_id"].nunique()) if len(review_out) else 0,
        "frames": int(len(timeline)),
        "camera_cuts_total": int(review_out["camera_cuts_005B2"].sum()) if len(review_out) else 0,
        "reviews_with_cut": int((review_out["camera_cuts_005B2"] > 0).sum()) if len(review_out) else 0,
        "camera_motion_frames_total": int(review_out["camera_motion_frames_005B2"].sum()) if len(review_out) else 0,
        "table_ok_ratio_med": round(float(to_num(review_out["table_ok_ratio_005B2"]).median()), 4) if len(review_out) else 0,
        "calib_ok_count": int(to_num(review_out["calib_ok"]).sum()) if len(review_out) else 0,
        "calib_by_review": calib_by_review,
        "thresholds_by_review": thresholds_by_review,
        "overlays": overlays,
        "outputs": {
            "timeline": str(out_timeline),
            "review_summary": str(out_reviews),
            "display_points_table_camera": str(out_points) if not points_out.empty else "",
        },
        "next": "Compare with 005B. If cuts are cleaner, use 005B2 camera_segment_id to reset display trails and later normalize ball/table coordinates."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005B2 status=OK")
    print("reviews=", summary["reviews"])
    print("frames=", summary["frames"])
    print("camera_cuts_total=", summary["camera_cuts_total"])
    print("reviews_with_cut=", summary["reviews_with_cut"])
    print("camera_motion_frames_total=", summary["camera_motion_frames_total"])
    print("table_ok_ratio_med=", summary["table_ok_ratio_med"])
    print("calib_ok_count=", summary["calib_ok_count"])
    print("overlays=", len(overlays))
    print("wrote", out_timeline)
    print("wrote", out_reviews)
    if not points_out.empty:
        print("wrote", out_points)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
