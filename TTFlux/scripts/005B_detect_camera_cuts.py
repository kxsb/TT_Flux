from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005B"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def robust_thr(vals, base_min: float, k: float = 4.0) -> float:
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float32)
    if vals.size < 8:
        return base_min
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-9
    return max(base_min, med + k * mad)


def hist_hs(frame_small):
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def table_proxy(frame_small):
    """
    Proxy simple table bleue.
    Retourne bbox en coordonnées small-frame + aire ratio.
    Pas censé être parfait : juste détecter les gros sauts de géométrie.
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

    # Table souvent bleu/cyan. Assez large volontairement.
    lower = np.array([80, 35, 35], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return {
            "ok": 0,
            "x": "",
            "y": "",
            "w": "",
            "h": "",
            "cx": "",
            "cy": "",
            "area_ratio": 0.0,
        }

    h_img, w_img = mask.shape[:2]
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < (w_img * h_img * 0.01):
        return {
            "ok": 0,
            "x": "",
            "y": "",
            "w": "",
            "h": "",
            "cx": "",
            "cy": "",
            "area_ratio": 0.0,
        }

    x, y, w, h = cv2.boundingRect(cnt)

    return {
        "ok": 1,
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "cx": round(float(x + w / 2), 3),
        "cy": round(float(y + h / 2), 3),
        "area_ratio": round(float(area / max(1, w_img * h_img)), 6),
    }


def bbox_iou(a, b):
    if not a or not b or int(a.get("ok", 0)) != 1 or int(b.get("ok", 0)) != 1:
        return 0.0

    ax1 = float(a["x"])
    ay1 = float(a["y"])
    ax2 = ax1 + float(a["w"])
    ay2 = ay1 + float(a["h"])

    bx1 = float(b["x"])
    by1 = float(b["y"])
    bx2 = bx1 + float(b["w"])
    by2 = by1 + float(b["h"])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return float(inter / max(1e-9, area_a + area_b - inter))


def scan_clip(review_id: str, clip_path: Path, resize_w: int, max_frames: int) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print("WARN open failed", review_id, clip_path)
        return pd.DataFrame()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 50.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if w0 <= 0 or h0 <= 0:
        cap.release()
        return pd.DataFrame()

    resize_h = max(1, int(round(h0 * resize_w / max(1, w0))))
    diag = math.hypot(resize_w, resize_h)

    rows = []

    prev_gray = None
    prev_edge = None
    prev_hist = None
    prev_table = None

    fidx = 0
    limit = n0 if max_frames <= 0 else min(n0, max_frames)

    while fidx < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        small = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 160)
        hist = hist_hs(small)
        tbl = table_proxy(small)

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
                table_shift = float(math.hypot(dx, dy) / max(1e-9, diag))

                a1 = max(1e-9, float(prev_table.get("area_ratio", 0.0)))
                a2 = max(1e-9, float(tbl.get("area_ratio", 0.0)))
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
            "table_ok": int(tbl.get("ok", 0)),
            "table_x": tbl.get("x", ""),
            "table_y": tbl.get("y", ""),
            "table_w": tbl.get("w", ""),
            "table_h": tbl.get("h", ""),
            "table_cx": tbl.get("cx", ""),
            "table_cy": tbl.get("cy", ""),
            "table_area_ratio": tbl.get("area_ratio", 0.0),
            "table_shift_norm": round(table_shift, 6),
            "table_area_logdiff": round(table_area_log, 6),
            "table_iou_prev": round(table_iou, 6),
        })

        prev_gray = gray
        prev_edge = edge
        prev_hist = hist
        prev_table = tbl
        fidx += 1

    cap.release()
    return pd.DataFrame(rows)


def classify_cuts(timeline: pd.DataFrame, min_cut_gap: int) -> pd.DataFrame:
    if timeline.empty:
        return timeline

    timeline = timeline.copy()

    pix_thr = robust_thr(timeline["pix_diff"].tolist(), base_min=0.115, k=4.0)
    hist_thr = robust_thr(timeline["hist_diff"].tolist(), base_min=0.285, k=4.0)
    edge_thr = robust_thr(timeline["edge_diff"].tolist(), base_min=0.075, k=4.0)
    table_shift_thr = robust_thr(timeline["table_shift_norm"].tolist(), base_min=0.145, k=4.0)
    table_area_thr = robust_thr(timeline["table_area_logdiff"].tolist(), base_min=0.42, k=4.0)

    timeline["cut_score_005B"] = (
        0.42 * to_num(timeline["hist_diff"]).fillna(0)
        + 0.35 * to_num(timeline["pix_diff"]).fillna(0)
        + 0.16 * to_num(timeline["edge_diff"]).fillna(0)
        + 0.45 * to_num(timeline["table_shift_norm"]).fillna(0)
        + 0.12 * to_num(timeline["table_area_logdiff"]).fillna(0)
    )

    timeline["camera_motion_score_005B"] = (
        0.55 * to_num(timeline["table_shift_norm"]).fillna(0)
        + 0.25 * to_num(timeline["table_area_logdiff"]).fillna(0)
        + 0.20 * to_num(timeline["pix_diff"]).fillna(0)
    )

    raw_cut = (
        (
            (to_num(timeline["hist_diff"]) >= hist_thr)
            & (to_num(timeline["pix_diff"]) >= pix_thr)
        )
        | (
            (to_num(timeline["pix_diff"]) >= max(0.24, pix_thr * 1.35))
            & (to_num(timeline["edge_diff"]) >= edge_thr)
        )
        | (
            (to_num(timeline["table_shift_norm"]) >= table_shift_thr)
            & (to_num(timeline["hist_diff"]) >= max(0.16, hist_thr * 0.55))
        )
        | (
            (to_num(timeline["table_area_logdiff"]) >= table_area_thr)
            & (to_num(timeline["hist_diff"]) >= max(0.16, hist_thr * 0.50))
        )
    )

    timeline["raw_camera_cut_005B"] = raw_cut.astype(int)

    final_cut = []
    last_cut = -999999
    segment = 1
    segments = []

    for _, r in timeline.iterrows():
        f = int(r["frame"])
        is_cut = int(r["raw_camera_cut_005B"]) == 1

        if f == 0:
            is_final = False
        elif is_cut and (f - last_cut >= min_cut_gap):
            is_final = True
            last_cut = f
            segment += 1
        else:
            is_final = False

        final_cut.append(int(is_final))
        segments.append(int(segment))

    timeline["camera_cut_005B"] = final_cut
    timeline["camera_segment_id_005B"] = segments

    motion_thr = robust_thr(timeline["camera_motion_score_005B"].tolist(), base_min=0.055, k=3.0)
    timeline["camera_unstable_005B"] = (
        (to_num(timeline["camera_motion_score_005B"]).fillna(0) >= motion_thr)
        & (timeline["camera_cut_005B"].astype(int) == 0)
    ).astype(int)

    timeline.attrs["thresholds_005B"] = {
        "pix_thr": round(float(pix_thr), 6),
        "hist_thr": round(float(hist_thr), 6),
        "edge_thr": round(float(edge_thr), 6),
        "table_shift_thr": round(float(table_shift_thr), 6),
        "table_area_thr": round(float(table_area_thr), 6),
        "motion_thr": round(float(motion_thr), 6),
    }

    return timeline


def annotate_points(points: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if points.empty or timeline.empty:
        return points

    pts = points.copy()

    if "frame_num" not in pts.columns:
        if "frame" in pts.columns:
            pts["frame_num"] = to_num(pts["frame"])
        else:
            raise SystemExit("points: frame_num/frame absent")

    pts["frame_num"] = to_num(pts["frame_num"]).fillna(-1).astype(int)

    small = timeline[[
        "review_id",
        "frame",
        "camera_segment_id_005B",
        "camera_cut_005B",
        "camera_unstable_005B",
        "cut_score_005B",
        "camera_motion_score_005B",
    ]].copy()

    small["frame_num"] = to_num(small["frame"]).fillna(-1).astype(int)

    out = pts.merge(
        small.drop(columns=["frame"]),
        on=["review_id", "frame_num"],
        how="left",
    )

    out["camera_segment_id_005B"] = to_num(out["camera_segment_id_005B"]).fillna(1).astype(int)
    out["camera_cut_005B"] = to_num(out["camera_cut_005B"]).fillna(0).astype(int)
    out["camera_unstable_005B"] = to_num(out["camera_unstable_005B"]).fillna(0).astype(int)

    return out


def make_overlay(review_id: str, clip_path: Path, tl: pd.DataFrame, out_path: Path, max_video_frames: int) -> bool:
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
            seg = int(r["camera_segment_id_005B"])
            is_cut = int(r["camera_cut_005B"]) == 1
            unstable = int(r["camera_unstable_005B"]) == 1
            cut_score = float(r["cut_score_005B"])
            motion = float(r["camera_motion_score_005B"])

            # Table bbox scaled back to source coords.
            if int(r.get("table_ok", 0)) == 1:
                rw = float(r["resize_w"])
                rh = float(r["resize_h"])
                sx = w / max(1.0, rw)
                sy = h / max(1.0, rh)

                x = int(float(r["table_x"]) * sx)
                y = int(float(r["table_y"]) * sy)
                bw = int(float(r["table_w"]) * sx)
                bh = int(float(r["table_h"]) * sy)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 180, 0), 2, cv2.LINE_AA)

            if is_cut:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                cv2.putText(
                    frame,
                    "CAMERA CUT / RESET",
                    (28, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.05,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            elif unstable:
                cv2.putText(
                    frame,
                    "CAMERA MOTION",
                    (28, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"{review_id} f={fidx} cam_seg={seg} cut={cut_score:.3f} motion={motion:.3f}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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
    ap.add_argument("--display-points", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resize-w", type=int, default=192)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-cut-gap", type=int, default=14)

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

    if "clip_path" not in reviews.columns:
        raise SystemExit("review-summary doit contenir clip_path")

    all_tl = []
    review_rows = []
    thresholds_by_review = {}

    for i, (_, r) in enumerate(reviews.iterrows(), start=1):
        review_id = str(r["review_id"])
        clip_path = Path(str(r["clip_path"]))

        if not clip_path.is_absolute():
            clip_path = root / clip_path

        if not clip_path.is_file():
            print("WARN missing clip", review_id, clip_path)
            continue

        print(f"[{i}/{len(reviews)}] scan {review_id}")

        tl = scan_clip(
            review_id=review_id,
            clip_path=clip_path,
            resize_w=args.resize_w,
            max_frames=args.max_frames,
        )

        if tl.empty:
            continue

        tl = classify_cuts(tl, min_cut_gap=args.min_cut_gap)
        thresholds_by_review[review_id] = tl.attrs.get("thresholds_005B", {})

        cuts = tl[tl["camera_cut_005B"].astype(int).eq(1)].copy()
        unstable_n = int(tl["camera_unstable_005B"].astype(int).sum())

        all_tl.append(tl)

        review_rows.append({
            "review_id": review_id,
            "video_id": str(r.get("video_id", "")),
            "rally_id": str(r.get("rally_id", "")),
            "frames": int(len(tl)),
            "camera_segments_005B": int(tl["camera_segment_id_005B"].max()),
            "camera_cuts_005B": int(len(cuts)),
            "cut_frames_005B": "|".join(str(int(x)) for x in cuts["frame"].tolist()),
            "camera_unstable_frames_005B": unstable_n,
            "pix_diff_med": round(float(to_num(tl["pix_diff"]).median()), 6),
            "hist_diff_med": round(float(to_num(tl["hist_diff"]).median()), 6),
            "table_ok_ratio": round(float(to_num(tl["table_ok"]).mean()), 4),
            "clip_path": str(clip_path),
            "overlay_005B": "",
        })

    timeline = pd.concat(all_tl, ignore_index=True) if all_tl else pd.DataFrame()
    review_out = pd.DataFrame(review_rows)

    display_points_out = pd.DataFrame()

    if args.display_points:
        points_path = Path(args.display_points)
        if not points_path.is_absolute():
            points_path = root / points_path

        if points_path.is_file() and not timeline.empty:
            pts = pd.read_csv(points_path).fillna("")
            display_points_out = annotate_points(pts, timeline)

    overlays = []

    if args.make_video and not timeline.empty:
        viz = review_out.sort_values(
            ["camera_cuts_005B", "camera_unstable_frames_005B", "frames"],
            ascending=[False, False, False],
        ).head(args.max_videos).copy()

        for j, (_, r) in enumerate(viz.iterrows(), start=1):
            review_id = str(r["review_id"])
            clip_path = Path(str(r["clip_path"]))
            tl = timeline[timeline["review_id"].astype(str).eq(review_id)].copy()

            out_video = out_dir / "overlays" / f"{j:03d}_{review_id}_005B_camera_overlay.mp4"

            ok = make_overlay(
                review_id=review_id,
                clip_path=clip_path,
                tl=tl,
                out_path=out_video,
                max_video_frames=args.max_video_frames,
            )

            if ok:
                overlays.append({"review_id": review_id, "overlay": str(out_video)})
                review_out.loc[
                    review_out["review_id"].astype(str).eq(review_id),
                    "overlay_005B",
                ] = str(out_video)

    out_timeline = out_dir / "camera_timeline_005B.csv"
    out_reviews = out_dir / "camera_review_summary_005B.csv"
    out_points = out_dir / "display_points_camera_005B.csv"
    out_json = out_dir / "camera_detection_summary_005B.json"

    timeline.to_csv(out_timeline, index=False, encoding="utf-8")
    review_out.to_csv(out_reviews, index=False, encoding="utf-8")

    if not display_points_out.empty:
        display_points_out.to_csv(out_points, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "detect_camera_cuts_and_unstable_camera_motion_from_frame_and_table_proxy",
        "review_summary": str(review_path),
        "display_points": str(args.display_points),
        "params": {
            "resize_w": args.resize_w,
            "max_frames": args.max_frames,
            "min_cut_gap": args.min_cut_gap,
        },
        "reviews": int(review_out["review_id"].nunique()) if len(review_out) else 0,
        "frames": int(len(timeline)),
        "camera_cuts_total": int(review_out["camera_cuts_005B"].sum()) if len(review_out) else 0,
        "reviews_with_cut": int((review_out["camera_cuts_005B"] > 0).sum()) if len(review_out) else 0,
        "camera_unstable_frames_total": int(review_out["camera_unstable_frames_005B"].sum()) if len(review_out) else 0,
        "thresholds_by_review": thresholds_by_review,
        "overlays": overlays,
        "outputs": {
            "timeline": str(out_timeline),
            "review_summary": str(out_reviews),
            "display_points_camera": str(out_points) if not display_points_out.empty else "",
        },
        "next": "Use camera_segment_id_005B to reset trails/tracklets and prevent cross-camera continuity."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005B status=OK")
    print("reviews=", summary["reviews"])
    print("frames=", summary["frames"])
    print("camera_cuts_total=", summary["camera_cuts_total"])
    print("reviews_with_cut=", summary["reviews_with_cut"])
    print("camera_unstable_frames_total=", summary["camera_unstable_frames_total"])
    print("overlays=", len(overlays))
    print("wrote", out_timeline)
    print("wrote", out_reviews)
    if not display_points_out.empty:
        print("wrote", out_points)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
