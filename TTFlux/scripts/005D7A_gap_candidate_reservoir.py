from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PATCH_ID = "005D7A_gap_candidate_reservoir"


def log(msg: str) -> None:
    print(msg, flush=True)


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


def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    return None


def guess_latest_video(root: Path) -> Path | None:
    roots = [root / "runs", root / "data", root / "datasets", root]
    videos = []
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.mp4"):
            name = p.name.lower()
            if any(bad in name for bad in [
                "overlay",
                "render",
                "debug",
                "contact",
                "diagnostic",
                "gap_candidates",
            ]):
                continue
            videos.append(p)

    if not videos:
        for base in roots:
            if not base.exists():
                continue
            videos.extend(base.rglob("*.mp4"))

    if not videos:
        return None

    videos = sorted(videos, key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0]


def inspect_csv_has_track(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            cols = reader.fieldnames
            return (
                find_col(cols, ["frame", "frame_idx", "frame_id", "frame_number"]) is not None
                and find_col(cols, ["x", "cx", "ball_x", "center_x"]) is not None
                and find_col(cols, ["y", "cy", "ball_y", "center_y"]) is not None
            )
    except Exception:
        return False


def guess_latest_track_csv(root: Path) -> Path | None:
    roots = [root / "runs", root / "data", root]
    candidates = []
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.csv"):
            name = p.name.lower()
            if "005d7a" in name or "gap_candidate" in name:
                continue
            if inspect_csv_has_track(p):
                candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass
class TrackData:
    by_frame: dict[int, tuple[float, float, float]]
    chosen_sequence: str | None
    csv_path: str | None
    frame_count: int


def read_track_csv(path: Path | None, video_stem: str, sequence_key: str | None) -> TrackData:
    if path is None or not path.exists():
        return TrackData({}, None, None, 0)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    frame_col = find_col(cols, ["frame", "frame_idx", "frame_id", "frame_number"])
    x_col = find_col(cols, ["x", "cx", "ball_x", "center_x"])
    y_col = find_col(cols, ["y", "cy", "ball_y", "center_y"])
    score_col = find_col(cols, ["score", "conf", "confidence", "prob", "candidate_score"])
    seq_col = find_col(cols, ["sequence_key", "clip_id", "video_id", "segment_id", "source_id"])

    if frame_col is None or x_col is None or y_col is None:
        return TrackData({}, None, str(path), 0)

    if seq_col:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            k = str(r.get(seq_col, "")).strip()
            groups.setdefault(k, []).append(r)

        chosen = None
        if sequence_key and sequence_key in groups:
            chosen = sequence_key
        elif sequence_key:
            for k in groups:
                if sequence_key.lower() in k.lower() or k.lower() in sequence_key.lower():
                    chosen = k
                    break
        else:
            vs = video_stem.lower()
            for k in groups:
                kl = k.lower()
                if kl and (kl in vs or vs in kl):
                    chosen = k
                    break

        if chosen is None and groups:
            chosen = max(
                groups,
                key=lambda k: sum(
                    math.isfinite(safe_float(r.get(x_col))) and math.isfinite(safe_float(r.get(y_col)))
                    for r in groups[k]
                ),
            )

        rows = groups.get(chosen, rows)
    else:
        chosen = None

    by_frame = {}
    for r in rows:
        fr = safe_int(r.get(frame_col))
        x = safe_float(r.get(x_col))
        y = safe_float(r.get(y_col))
        sc = safe_float(r.get(score_col), 1.0) if score_col else 1.0
        if fr < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        by_frame[fr] = (x, y, sc)

    return TrackData(by_frame=by_frame, chosen_sequence=chosen, csv_path=str(path), frame_count=len(by_frame))


@dataclass
class Gap:
    gap_id: int
    start_frame: int
    end_frame: int
    left_frame: int | None
    right_frame: int | None
    left_xy: tuple[float, float] | None
    right_xy: tuple[float, float] | None

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


def build_gaps(total_frames: int, track: TrackData, min_gap_len: int, include_edges: bool) -> list[Gap]:
    obs = sorted(fr for fr in track.by_frame.keys() if 0 <= fr < total_frames)
    gaps: list[Gap] = []
    gid = 0

    if not obs:
        return [Gap(0, 0, max(0, total_frames - 1), None, None, None, None)]

    if include_edges and obs[0] > 0:
        length = obs[0]
        if length >= min_gap_len:
            gaps.append(Gap(gid, 0, obs[0] - 1, None, obs[0], None, track.by_frame[obs[0]][:2]))
            gid += 1

    for a, b in zip(obs[:-1], obs[1:]):
        missing = b - a - 1
        if missing >= min_gap_len:
            gaps.append(Gap(
                gid,
                a + 1,
                b - 1,
                a,
                b,
                track.by_frame[a][:2],
                track.by_frame[b][:2],
            ))
            gid += 1

    if include_edges and obs[-1] < total_frames - 1:
        length = total_frames - 1 - obs[-1]
        if length >= min_gap_len:
            gaps.append(Gap(
                gid,
                obs[-1] + 1,
                total_frames - 1,
                obs[-1],
                None,
                track.by_frame[obs[-1]][:2],
                None,
            ))

    return gaps


def predict_xy(gap: Gap, frame: int) -> tuple[float | None, float | None, str]:
    if gap.left_frame is not None and gap.right_frame is not None and gap.left_xy and gap.right_xy:
        denom = max(1, gap.right_frame - gap.left_frame)
        t = (frame - gap.left_frame) / denom
        x = gap.left_xy[0] * (1 - t) + gap.right_xy[0] * t
        y = gap.left_xy[1] * (1 - t) + gap.right_xy[1] * t
        return x, y, "interp"

    if gap.left_frame is not None and gap.left_xy:
        return gap.left_xy[0], gap.left_xy[1], "left_hold"

    if gap.right_frame is not None and gap.right_xy:
        return gap.right_xy[0], gap.right_xy[1], "right_hold"

    return None, None, "none"


def sample_video_frames(video_path: Path, max_samples: int) -> tuple[list[np.ndarray], int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total <= 0:
        total = 1

    idxs = np.linspace(0, max(0, total - 1), num=min(max_samples, total), dtype=np.int32)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    return frames, total, width, height, fps


def build_table_roi(video_path: Path, width: int, height: int, samples: int, expand_px: int) -> tuple[np.ndarray, str]:
    frames, _, _, _, _ = sample_video_frames(video_path, samples)
    if not frames:
        mask = np.zeros((height, width), np.uint8)
        mask[int(height * 0.20): int(height * 0.92), int(width * 0.03): int(width * 0.97)] = 255
        return mask, "fallback_no_samples"

    small_frames = []
    for fr in frames:
        if fr.shape[1] != width or fr.shape[0] != height:
            fr = cv2.resize(fr, (width, height))
        small_frames.append(fr)

    median = np.median(np.stack(small_frames, axis=0), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    # Table tennis videos often use blue/green tables. This mask is intentionally broad.
    blue = cv2.inRange(hsv, np.array([75, 25, 25]), np.array([145, 255, 255]))
    green = cv2.inRange(hsv, np.array([32, 25, 20]), np.array([90, 255, 255]))
    cyan = cv2.inRange(hsv, np.array([70, 20, 30]), np.array([105, 255, 255]))

    mask = cv2.bitwise_or(blue, green)
    mask = cv2.bitwise_or(mask, cyan)

    # Keep central court/table zone, suppress crowd/logo zones a bit.
    central = np.zeros_like(mask)
    central[int(height * 0.12): int(height * 0.95), int(width * 0.02): int(width * 0.98)] = 255
    mask = cv2.bitwise_and(mask, central)

    k1 = max(5, int(min(width, height) * 0.008))
    k2 = max(9, int(min(width, height) * 0.018))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k1, k1), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k2, k2), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        out = np.zeros((height, width), np.uint8)
        out[int(height * 0.18): int(height * 0.94), int(width * 0.03): int(width * 0.97)] = 255
        return out, "fallback_no_components"

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = int(np.argmax(areas)) + 1
    area_ratio = stats[best_i, cv2.CC_STAT_AREA] / float(width * height)

    if area_ratio < 0.035 or area_ratio > 0.82:
        out = np.zeros((height, width), np.uint8)
        out[int(height * 0.18): int(height * 0.94), int(width * 0.03): int(width * 0.97)] = 255
        roi_mode = f"fallback_area_ratio_{area_ratio:.3f}"
    else:
        out = np.zeros((height, width), np.uint8)
        out[labels == best_i] = 255
        roi_mode = f"table_color_component_area_ratio_{area_ratio:.3f}"

    if expand_px > 0:
        k = max(3, int(expand_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    return out, roi_mode


def frame_roi_mask(
    shape: tuple[int, int],
    table_mask: np.ndarray,
    pred_x: float | None,
    pred_y: float | None,
    radius: int,
) -> np.ndarray:
    h, w = shape
    if pred_x is None or pred_y is None:
        return table_mask.copy()

    circle = np.zeros((h, w), np.uint8)
    cv2.circle(circle, (int(round(pred_x)), int(round(pred_y))), int(radius), 255, -1)
    roi = cv2.bitwise_and(table_mask, circle)

    # If table mask misses too much, keep prediction circle as fallback.
    if int(np.count_nonzero(roi)) < 50:
        roi = circle

    return roi


def components_from_mask(
    mask: np.ndarray,
    gray: np.ndarray,
    motion: np.ndarray,
    source: str,
    frame_idx: int,
    pred_x: float | None,
    pred_y: float | None,
    min_area: int,
    max_area: int,
    max_box: int,
    max_aspect: float,
) -> list[dict]:
    out = []
    n, labels, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if bw <= 0 or bh <= 0:
            continue
        if bw > max_box or bh > max_box:
            continue

        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if aspect > max_aspect:
            continue

        cx = float(cent[i][0])
        cy = float(cent[i][1])
        comp = labels == i

        mean_gray = float(np.mean(gray[comp])) if np.any(comp) else 0.0
        max_gray = float(np.max(gray[comp])) if np.any(comp) else 0.0
        mean_motion = float(np.mean(motion[comp])) if np.any(comp) else 0.0

        if pred_x is not None and pred_y is not None:
            dist = float(math.hypot(cx - pred_x, cy - pred_y))
            pred_bonus = max(0.0, 1.0 - min(dist, 300.0) / 300.0)
        else:
            dist = float("nan")
            pred_bonus = 0.25

        # Reservoir score, not final tracking score.
        size_bonus = max(0.0, 1.0 - abs(area - 18.0) / 80.0)
        bright_score = min(1.0, max_gray / 255.0)
        motion_score = min(1.0, mean_motion / 80.0)
        score = 0.34 * bright_score + 0.33 * motion_score + 0.23 * pred_bonus + 0.10 * size_bonus

        out.append({
            "frame": frame_idx,
            "x": cx,
            "y": cy,
            "r": math.sqrt(area / math.pi),
            "area": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": bw,
            "bbox_h": bh,
            "aspect": aspect,
            "source": source,
            "mean_gray": mean_gray,
            "max_gray": max_gray,
            "mean_motion": mean_motion,
            "dist_pred": dist,
            "score": float(score),
        })

    return out


def dedupe_candidates(cands: list[dict], px: float) -> list[dict]:
    if not cands:
        return []

    ordered = sorted(cands, key=lambda c: c["score"], reverse=True)
    kept: list[dict] = []

    for c in ordered:
        merged = False
        for k in kept:
            d = math.hypot(c["x"] - k["x"], c["y"] - k["y"])
            if d <= px:
                if c["source"] not in k["source"]:
                    k["source"] = k["source"] + "+" + c["source"]
                    k["score"] = max(k["score"], c["score"] + 0.08)
                    k["mean_gray"] = max(k["mean_gray"], c["mean_gray"])
                    k["max_gray"] = max(k["max_gray"], c["max_gray"])
                    k["mean_motion"] = max(k["mean_motion"], c["mean_motion"])
                merged = True
                break
        if not merged:
            kept.append(c)

    return sorted(kept, key=lambda c: c["score"], reverse=True)


def detect_candidates_for_frame(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    roi: np.ndarray,
    frame_idx: int,
    pred_x: float | None,
    pred_y: float | None,
    args,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        motion = np.zeros_like(gray)
    else:
        motion = cv2.absdiff(gray, prev_gray)

    roi_bool = roi > 0
    roi_vals = gray[roi_bool]
    motion_vals = motion[roi_bool]

    if roi_vals.size == 0:
        return [], gray, motion, np.zeros_like(gray)

    bright_thr = max(args.bright_min, float(np.percentile(roi_vals, args.bright_percentile)))
    motion_thr = max(args.motion_min, float(np.percentile(motion_vals, args.motion_percentile))) if motion_vals.size else args.motion_min

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    white_ball = (gray >= bright_thr) & (sat <= args.white_sat_max) & roi_bool
    hot_ball = (val >= max(args.bright_min, bright_thr - 10)) & (
        ((hue <= 28) | (hue >= 165)) & (sat >= 45)
    ) & roi_bool
    bright_mask = (white_ball | hot_ball).astype(np.uint8) * 255

    motion_mask = ((motion >= motion_thr) & roi_bool).astype(np.uint8) * 255

    # Reduce player/racket noise: keep small compact blobs, but allow a slight dilation.
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    fused_seed = cv2.bitwise_and(
        cv2.dilate(bright_mask, np.ones((3, 3), np.uint8), iterations=1),
        cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1),
    )

    cands = []
    cands.extend(components_from_mask(
        fused_seed, gray, motion, "motion_bright",
        frame_idx, pred_x, pred_y,
        args.min_area, args.max_area, args.max_box, args.max_aspect,
    ))
    cands.extend(components_from_mask(
        bright_mask, gray, motion, "bright",
        frame_idx, pred_x, pred_y,
        args.min_area, args.max_area, args.max_box, args.max_aspect,
    ))
    cands.extend(components_from_mask(
        motion_mask, gray, motion, "motion",
        frame_idx, pred_x, pred_y,
        args.min_area, args.max_area, args.max_box, args.max_aspect,
    ))

    cands = dedupe_candidates(cands, args.dedupe_px)
    cands = cands[:args.max_candidates_per_frame]

    return cands, gray, motion, bright_mask


def draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    gap: Gap | None,
    pred_x: float | None,
    pred_y: float | None,
    pred_mode: str,
    roi: np.ndarray | None,
    cands: list[dict],
    track_xy: tuple[float, float, float] | None,
) -> np.ndarray:
    out = frame.copy()

    if roi is not None:
        contours, _ = cv2.findContours((roi > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (80, 80, 80), 1)

    if track_xy is not None:
        tx, ty, _ = track_xy
        cv2.circle(out, (int(round(tx)), int(round(ty))), 6, (180, 180, 180), 2)

    if pred_x is not None and pred_y is not None:
        px, py = int(round(pred_x)), int(round(pred_y))
        cv2.drawMarker(out, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

    for c in cands:
        cx, cy = int(round(c["x"])), int(round(c["y"]))
        src = c["source"]
        if "motion_bright" in src:
            color = (0, 255, 255)
        elif "bright" in src:
            color = (0, 220, 255)
        else:
            color = (0, 255, 0)
        cv2.circle(out, (cx, cy), max(3, int(round(c["r"])) + 2), color, 2)
        cv2.putText(out, f'{c["score"]:.2f}', (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    if gap is not None:
        label = f"{PATCH_ID} | frame={frame_idx} | gap={gap.gap_id} {gap.start_frame}-{gap.end_frame} | cands={len(cands)} | pred={pred_mode}"
    else:
        label = f"{PATCH_ID} | frame={frame_idx} | no-gap"

    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 980), 38), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "patch",
        "video",
        "track_csv",
        "chosen_sequence",
        "gap_id",
        "gap_start",
        "gap_end",
        "frame",
        "x",
        "y",
        "r",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "aspect",
        "source",
        "score",
        "mean_gray",
        "max_gray",
        "mean_motion",
        "pred_x",
        "pred_y",
        "pred_mode",
        "dist_pred",
        "roi_mode",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.environ.get("TTFLUX_VIDEO", ""))
    ap.add_argument("--track-csv", default=os.environ.get("TTFLUX_TRACK_CSV", ""))
    ap.add_argument("--sequence-key", default=os.environ.get("TTFLUX_SEQUENCE_KEY", ""))
    ap.add_argument("--out-dir", default="runs/005D7A_gap_candidates")
    ap.add_argument("--min-gap-len", type=int, default=2)
    ap.add_argument("--include-edge-gaps", action="store_true")
    ap.add_argument("--table-samples", type=int, default=80)
    ap.add_argument("--table-expand-px", type=int, default=95)
    ap.add_argument("--pred-radius", type=int, default=210)
    ap.add_argument("--pred-radius-long-gap-extra", type=int, default=2)
    ap.add_argument("--bright-percentile", type=float, default=99.55)
    ap.add_argument("--bright-min", type=float, default=158.0)
    ap.add_argument("--white-sat-max", type=float, default=95.0)
    ap.add_argument("--motion-percentile", type=float, default=99.30)
    ap.add_argument("--motion-min", type=float, default=14.0)
    ap.add_argument("--min-area", type=int, default=2)
    ap.add_argument("--max-area", type=int, default=180)
    ap.add_argument("--max-box", type=int, default=34)
    ap.add_argument("--max-aspect", type=float, default=3.2)
    ap.add_argument("--dedupe-px", type=float, default=7.5)
    ap.add_argument("--max-candidates-per-frame", type=int, default=12)
    ap.add_argument("--write-overlay", action="store_true")
    ap.add_argument("--overlay-full-video", action="store_true")
    ap.add_argument("--max-process-frames", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    root = Path.cwd()
    video_path = Path(args.video) if args.video else None
    if video_path is None or not video_path.exists():
        video_path = guess_latest_video(root)

    if video_path is None or not video_path.exists():
        raise RuntimeError("Aucune vidéo trouvée. Passe --video chemin\\vers\\rally.mp4 ou définis $env:TTFLUX_VIDEO.")

    track_csv = Path(args.track_csv) if args.track_csv else None
    if track_csv is None or not track_csv.exists():
        track_csv = guess_latest_track_csv(root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, total_frames, width, height, fps = sample_video_frames(video_path, max_samples=3)
    if width <= 0 or height <= 0:
        raise RuntimeError("Dimensions vidéo invalides.")

    track = read_track_csv(track_csv, video_path.stem, args.sequence_key or None)
    gaps = build_gaps(total_frames, track, args.min_gap_len, args.include_edge_gaps)

    table_mask, roi_mode = build_table_roi(
        video_path,
        width=width,
        height=height,
        samples=args.table_samples,
        expand_px=args.table_expand_px,
    )

    target_frames = set()
    gap_by_frame = {}
    for g in gaps:
        for fr in range(g.start_frame, g.end_frame + 1):
            if 0 <= fr < total_frames:
                target_frames.add(fr)
                gap_by_frame[fr] = g

    if args.max_process_frames and len(target_frames) > args.max_process_frames:
        target_frames = set(sorted(target_frames)[:args.max_process_frames])
        gap_by_frame = {fr: gap_by_frame[fr] for fr in target_frames}

    log("=" * 72)
    log(f"PATCH {PATCH_ID}")
    log(f"video        = {video_path}")
    log(f"track_csv    = {track.csv_path}")
    log(f"sequence     = {track.chosen_sequence}")
    log(f"video_frames = {total_frames} ({width}x{height}, fps={fps:.3f})")
    log(f"track_points = {track.frame_count}")
    log(f"gaps         = {len(gaps)}")
    log(f"gap_frames   = {len(target_frames)}")
    log(f"roi_mode     = {roi_mode}")
    log("=" * 72)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    overlay_path = out_dir / "005D7A_gap_candidates_overlay.mp4"
    writer = None
    if args.write_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps if fps > 0 else 25.0, (width, height))
        if not writer.isOpened():
            log("WARN: VideoWriter overlay non ouvert. Overlay désactivé.")
            writer = None

    rows: list[dict] = []
    frame_candidate_counts = {}
    prev_gray = None
    frame_idx = -1

    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        should_process = frame_idx in target_frames
        should_write = args.write_overlay and writer is not None and (args.overlay_full_video or should_process)

        cands = []
        gap = gap_by_frame.get(frame_idx)
        pred_x = pred_y = None
        pred_mode = "none"
        roi = None

        if should_process and gap is not None:
            pred_x, pred_y, pred_mode = predict_xy(gap, frame_idx)
            gap_len = max(1, gap.end_frame - gap.start_frame + 1)
            radius = int(args.pred_radius + min(180, gap_len * args.pred_radius_long_gap_extra))
            roi = frame_roi_mask((height, width), table_mask, pred_x, pred_y, radius)

            cands, _, _, _ = detect_candidates_for_frame(
                frame=frame,
                prev_gray=prev_gray,
                roi=roi,
                frame_idx=frame_idx,
                pred_x=pred_x,
                pred_y=pred_y,
                args=args,
            )

            frame_candidate_counts[frame_idx] = len(cands)

            for c in cands:
                row = dict(c)
                row.update({
                    "patch": PATCH_ID,
                    "video": str(video_path),
                    "track_csv": track.csv_path or "",
                    "chosen_sequence": track.chosen_sequence or "",
                    "gap_id": gap.gap_id,
                    "gap_start": gap.start_frame,
                    "gap_end": gap.end_frame,
                    "pred_x": "" if pred_x is None else float(pred_x),
                    "pred_y": "" if pred_y is None else float(pred_y),
                    "pred_mode": pred_mode,
                    "roi_mode": roi_mode,
                })
                rows.append(row)

        if should_write:
            overlay = draw_overlay(
                frame=frame,
                frame_idx=frame_idx,
                gap=gap,
                pred_x=pred_x,
                pred_y=pred_y,
                pred_mode=pred_mode,
                roi=roi,
                cands=cands,
                track_xy=track.by_frame.get(frame_idx),
            )
            writer.write(overlay)

        prev_gray = gray_now

    cap.release()
    if writer is not None:
        writer.release()

    csv_path = out_dir / "005D7A_gap_candidates.csv"
    json_path = out_dir / "005D7A_gap_summary.json"
    write_csv(csv_path, rows)

    gap_frames = len(target_frames)
    candidate_frames = sum(1 for v in frame_candidate_counts.values() if v > 0)
    empty_frames = gap_frames - candidate_frames
    counts = list(frame_candidate_counts.values())

    summary = {
        "patch": PATCH_ID,
        "video": str(video_path),
        "track_csv": track.csv_path,
        "chosen_sequence": track.chosen_sequence,
        "total_video_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "track_points": track.frame_count,
        "gap_count": len(gaps),
        "gap_frames": gap_frames,
        "candidate_frames": candidate_frames,
        "empty_gap_frames": empty_frames,
        "candidate_frame_ratio": 0.0 if gap_frames == 0 else candidate_frames / gap_frames,
        "candidates_total": len(rows),
        "candidate_count_mean_per_gap_frame": 0.0 if not counts else float(np.mean(counts)),
        "candidate_count_median_per_gap_frame": 0.0 if not counts else float(np.median(counts)),
        "candidate_count_max_per_gap_frame": 0 if not counts else int(np.max(counts)),
        "roi_mode": roi_mode,
        "csv": str(csv_path),
        "overlay": str(overlay_path) if args.write_overlay else None,
        "elapsed_sec": time.time() - t0,
        "params": vars(args),
        "gaps": [
            {
                "gap_id": g.gap_id,
                "start_frame": g.start_frame,
                "end_frame": g.end_frame,
                "length": g.end_frame - g.start_frame + 1,
                "left_frame": g.left_frame,
                "right_frame": g.right_frame,
            }
            for g in gaps
        ],
    }

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log("OK 005D7A")
    log(f"csv      = {csv_path}")
    log(f"summary  = {json_path}")
    if args.write_overlay:
        log(f"overlay  = {overlay_path}")
    log("")
    log(f"gap_frames={gap_frames} candidate_frames={candidate_frames} empty_gap_frames={empty_frames} candidates_total={len(rows)}")
    log(f"candidate_frame_ratio={summary['candidate_frame_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
