
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttflux.core.paths import project_root
from ttflux.datasets.openttgames_training_rows import load_openttgames_training_rows


def out_dir() -> Path:
    p = project_root() / "runs" / "007Q_gt_evolution_video"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _training_rows_csv() -> Path:
    payload = load_openttgames_training_rows(rebuild=False)
    path = Path(payload.get("outputs", {}).get("training_rows_csv", ""))

    if not path.exists():
        payload = load_openttgames_training_rows(rebuild=True)
        path = Path(payload.get("outputs", {}).get("training_rows_csv", ""))

    if not path.exists():
        raise FileNotFoundError("training rows CSV not found. Run 007P first.")

    return path


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None

    s = str(v).strip().replace(",", ".")
    if not s or s.lower() in {"none", "null", "nan"}:
        return None

    try:
        return float(s)
    except Exception:
        return None


def _int_or_none(v: Any) -> int | None:
    x = _float_or_none(v)
    if x is None:
        return None
    return int(round(x))


def _read_rows() -> list[dict[str, Any]]:
    path = _training_rows_csv()
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = _int_or_none(row.get("frame"))
            x = _float_or_none(row.get("x"))
            y = _float_or_none(row.get("y"))

            if frame is None:
                continue

            row["_frame"] = frame
            row["_x"] = x
            row["_y"] = y
            row["_valid_xy"] = str(row.get("valid_xy", "0")).strip() in {"1", "true", "True"}
            row["_usable_pos"] = str(row.get("usable_for_position", "0")).strip() in {"1", "true", "True"}
            row["_segment_idx"] = _int_or_none(row.get("segment_idx")) or 0
            row["_fps"] = _float_or_none(row.get("fps")) or 120.0
            row["_width"] = _int_or_none(row.get("width")) or 1920
            row["_height"] = _int_or_none(row.get("height")) or 1080
            rows.append(row)

    return rows


def _group_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        uid = str(r.get("segment_uid", "")).strip()
        if uid:
            by_uid[uid].append(r)

    segments = []

    for uid, pts in by_uid.items():
        pts = sorted(pts, key=lambda r: int(r["_frame"]))
        valid_pts = [p for p in pts if p["_valid_xy"]]

        if not pts:
            continue

        first = pts[0]
        frames = [int(p["_frame"]) for p in pts]
        valid_frames = [int(p["_frame"]) for p in valid_pts]

        seg = {
            "segment_uid": uid,
            "sample_id": first.get("sample_id", ""),
            "split": first.get("split", ""),
            "segment_idx": first.get("_segment_idx", 0),
            "video_path": first.get("video_path", ""),
            "fps": float(first.get("_fps", 120.0)),
            "width": int(first.get("_width", 1920)),
            "height": int(first.get("_height", 1080)),
            "frame_start": min(frames),
            "frame_end": max(frames),
            "valid_frame_start": min(valid_frames) if valid_frames else min(frames),
            "valid_frame_end": max(valid_frames) if valid_frames else max(frames),
            "point_count": len(pts),
            "valid_xy_count": len(valid_pts),
            "points": pts,
            "points_by_frame": {int(p["_frame"]): p for p in pts},
        }

        segments.append(seg)

    segments.sort(key=lambda s: (
        str(s["split"]),
        str(s["sample_id"]),
        int(s["segment_idx"]),
        int(s["frame_start"]),
    ))

    return segments


def _choose_segments(
    segments: list[dict[str, Any]],
    sample_id: str = "game_1",
    split: str = "training",
    min_valid_points: int = 24,
) -> list[dict[str, Any]]:
    selected = [
        s for s in segments
        if (not sample_id or s["sample_id"] == sample_id)
        and (not split or s["split"] == split)
        and int(s["valid_xy_count"]) >= min_valid_points
    ]

    if selected:
        return selected

    selected = [
        s for s in segments
        if (not split or s["split"] == split)
        and int(s["valid_xy_count"]) >= min_valid_points
    ]

    if selected:
        return selected

    return [s for s in segments if int(s["valid_xy_count"]) >= min_valid_points]


def _nearest_point(seg: dict[str, Any], frame: int, window: int = 8) -> dict[str, Any] | None:
    by_frame = seg["points_by_frame"]

    if frame in by_frame:
        return by_frame[frame]

    best = None
    best_delta = None

    for delta in range(1, window + 1):
        for f in (frame - delta, frame + delta):
            p = by_frame.get(f)
            if not p:
                continue

            if best is None or delta < best_delta:
                best = p
                best_delta = delta

        if best is not None:
            return best

    return None


def _trail_points(seg: dict[str, Any], frame: int, trail_frames: int = 72) -> list[dict[str, Any]]:
    out = []

    for f in range(max(int(seg["frame_start"]), frame - trail_frames), frame + 1):
        p = seg["points_by_frame"].get(f)
        if p and p.get("_valid_xy"):
            out.append(p)

    return out[-60:]


def _fit_frame(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]

    if w == width and h == height:
        return img

    scale = min(width / max(1, w), height / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    ox = (width - nw) // 2
    oy = (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized

    return canvas


def _draw_panel(
    img: np.ndarray,
    seg: dict[str, Any],
    frame: int,
    output_idx: int,
    output_total: int,
    segment_local_idx: int,
    selected_count: int,
    seconds: float,
) -> None:
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 118), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    title = "TTFluxV2 007Q | OpenTTGames GT ball evolution | 1080p"
    cv2.putText(img, title, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

    line2 = (
        f"sample={seg['sample_id']} | split={seg['split']} | segment={seg['segment_uid']} "
        f"({segment_local_idx}/{selected_count}) | source_frame={frame}"
    )
    cv2.putText(img, line2, (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 230, 255), 2, cv2.LINE_AA)

    t = output_idx / 30.0
    line3 = (
        f"video_time={t:05.2f}s / {seconds:.1f}s | GT points={seg['point_count']} | "
        f"valid={seg['valid_xy_count']} | source_fps={seg['fps']:.1f}"
    )
    cv2.putText(img, line3, (22, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 255, 190), 2, cv2.LINE_AA)

    # Progress bar.
    bar_x0 = 22
    bar_y0 = h - 34
    bar_w = w - 44
    bar_h = 13

    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (60, 60, 60), -1)

    prog = min(1.0, max(0.0, output_idx / max(1, output_total - 1)))
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + int(bar_w * prog), bar_y0 + bar_h), (0, 220, 255), -1)
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (240, 240, 240), 1)


def _draw_gt(
    img: np.ndarray,
    seg: dict[str, Any],
    frame: int,
    source_w: int,
    source_h: int,
) -> None:
    h, w = img.shape[:2]

    sx = w / max(1, source_w)
    sy = h / max(1, source_h)

    trail = _trail_points(seg, frame=frame, trail_frames=72)

    if len(trail) >= 2:
        pts = []

        for p in trail:
            x = p.get("_x")
            y = p.get("_y")

            if x is None or y is None or x < 0 or y < 0:
                continue

            pts.append((int(round(x * sx)), int(round(y * sy))))

        for i in range(1, len(pts)):
            age = i / max(1, len(pts) - 1)
            thickness = 1 + int(3 * age)
            cv2.line(img, pts[i - 1], pts[i], (0, 180 + int(75 * age), 255), thickness, cv2.LINE_AA)

        for i, pt in enumerate(pts):
            age = i / max(1, len(pts) - 1)
            radius = 2 + int(4 * age)
            cv2.circle(img, pt, radius, (0, 160 + int(90 * age), 255), -1, cv2.LINE_AA)

    point = _nearest_point(seg, frame=frame, window=8)

    if not point or not point.get("_valid_xy"):
        cv2.putText(img, "GT ball: not visible / no valid point", (24, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 180, 255), 2, cv2.LINE_AA)
        return

    x = point.get("_x")
    y = point.get("_y")

    if x is None or y is None:
        return

    cx = int(round(float(x) * sx))
    cy = int(round(float(y) * sy))

    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return

    cv2.circle(img, (cx, cy), 18, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(img, (cx - 34, cy), (cx + 34, cy), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 34), (cx, cy + 34), (0, 255, 255), 2, cv2.LINE_AA)

    label = f"GT ball ({int(round(float(x)))},{int(round(float(y)))})"
    tx = min(max(20, cx + 22), w - 320)
    ty = max(142, cy - 22)

    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2, cv2.LINE_AA)


class CaptureCache:
    def __init__(self) -> None:
        self.current_path: str | None = None
        self.cap: cv2.VideoCapture | None = None

    def read(self, path: str, frame: int) -> np.ndarray | None:
        if self.current_path != path:
            self.close()
            self.cap = cv2.VideoCapture(path)
            self.current_path = path

        if self.cap is None or not self.cap.isOpened():
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, img = self.cap.read()

        if not ok or img is None:
            return None

        return img

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.current_path = None


def _make_schedule(
    selected: list[dict[str, Any]],
    seconds: float,
    out_fps: int,
) -> list[tuple[dict[str, Any], int, int]]:
    target_frames = int(round(seconds * out_fps))
    schedule: list[tuple[dict[str, Any], int, int]] = []

    for seg_idx, seg in enumerate(selected, start=1):
        source_fps = float(seg.get("fps") or 120.0)
        source_step = max(1, int(round(source_fps / out_fps)))

        f0 = int(seg["frame_start"])
        f1 = int(seg["frame_end"])

        for f in range(f0, f1 + 1, source_step):
            schedule.append((seg, f, seg_idx))

            if len(schedule) >= target_frames:
                return schedule

    # Si le sample choisi ne suffit pas, on boucle sur les segments sélectionnés.
    i = 0
    while schedule and len(schedule) < target_frames:
        seg, f, seg_idx = schedule[i % len(schedule)]
        schedule.append((seg, f, seg_idx))
        i += 1

    return schedule[:target_frames]


def _open_writer(path: Path, out_fps: int, width: int, height: int) -> cv2.VideoWriter:
    codecs = ["mp4v", "avc1", "H264"]

    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(out_fps), (int(width), int(height)))

        if writer.isOpened():
            return writer

        writer.release()

    raise RuntimeError("cannot open cv2 VideoWriter with mp4v/avc1/H264")


def _try_h264_transcode(src: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return False

    tmp_out = Path(tempfile.gettempdir()) / f"ttfluxv2_007Q_h264_{int(time.time())}.mp4"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_out),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False

    if not tmp_out.exists() or tmp_out.stat().st_size < 10000:
        return False

    shutil.copy2(tmp_out, dst)

    try:
        tmp_out.unlink()
    except Exception:
        pass

    return True


def make_evolution_video(
    sample_id: str = "game_1",
    split: str = "training",
    seconds: float = 30.0,
    out_fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    min_valid_points: int = 24,
    h264: bool = True,
) -> dict[str, Any]:
    started = time.time()

    rows = _read_rows()
    segments = _group_segments(rows)
    selected = _choose_segments(
        segments,
        sample_id=sample_id,
        split=split,
        min_valid_points=min_valid_points,
    )

    if not selected:
        raise RuntimeError("no selected GT segment")

    schedule = _make_schedule(selected, seconds=seconds, out_fps=out_fps)

    if not schedule:
        raise RuntimeError("empty video schedule")

    od = out_dir()

    safe_sample = sample_id or "mixed"
    final_path = od / f"openttgames_gt_evolution_{safe_sample}_{int(seconds)}s_1080p_007Q.mp4"
    cv2_path = od / f"openttgames_gt_evolution_{safe_sample}_{int(seconds)}s_1080p_007Q_cv2.mp4"
    json_path = od / f"openttgames_gt_evolution_{safe_sample}_{int(seconds)}s_1080p_007Q.json"

    tmp_cv2 = Path(tempfile.gettempdir()) / f"ttfluxv2_007Q_cv2_{int(time.time())}.mp4"

    writer = _open_writer(tmp_cv2, out_fps=out_fps, width=width, height=height)
    cap_cache = CaptureCache()

    written = 0
    read_fail = 0

    selected_count = len(selected)
    source_meta = {}

    try:
        for out_idx, (seg, frame, seg_local_idx) in enumerate(schedule):
            img = cap_cache.read(seg["video_path"], frame)

            if img is None:
                read_fail += 1
                img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(img, "READ FAIL", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 255), 3, cv2.LINE_AA)
                cv2.putText(img, str(seg["video_path"]), (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
            else:
                img = _fit_frame(img, width=width, height=height)

            _draw_gt(
                img,
                seg=seg,
                frame=frame,
                source_w=int(seg.get("width") or 1920),
                source_h=int(seg.get("height") or 1080),
            )

            _draw_panel(
                img,
                seg=seg,
                frame=frame,
                output_idx=out_idx,
                output_total=len(schedule),
                segment_local_idx=seg_local_idx,
                selected_count=selected_count,
                seconds=seconds,
            )

            writer.write(img)
            written += 1

            if written % 60 == 0:
                print(f"  frames_written={written}/{len(schedule)}")

    finally:
        writer.release()
        cap_cache.close()

    if not tmp_cv2.exists() or tmp_cv2.stat().st_size < 10000:
        raise RuntimeError(f"output video missing or too small: {tmp_cv2}")

    shutil.copy2(tmp_cv2, cv2_path)

    transcode_ok = False

    if h264:
        transcode_ok = _try_h264_transcode(tmp_cv2, final_path)

    if not transcode_ok:
        shutil.copy2(tmp_cv2, final_path)

    try:
        tmp_cv2.unlink()
    except Exception:
        pass

    sample_counts = defaultdict(int)
    segment_uids = []

    for seg, _frame, _seg_idx in schedule:
        sample_counts[seg["sample_id"]] += 1
        if seg["segment_uid"] not in segment_uids:
            segment_uids.append(seg["segment_uid"])

    payload = {
        "patch": "007Q_TTFluxV2_openttgames_30s_gt_evolution_video",
        "created_at": time.time(),
        "elapsed_sec": round(time.time() - started, 3),
        "sample_id": sample_id,
        "split": split,
        "seconds": seconds,
        "out_fps": out_fps,
        "width": width,
        "height": height,
        "target_output_frames": int(round(seconds * out_fps)),
        "frames_written": written,
        "read_fail_count": read_fail,
        "selected_segment_count": selected_count,
        "used_segment_count": len(segment_uids),
        "used_segment_uids": segment_uids,
        "schedule_sample_frame_counts": dict(sample_counts),
        "h264_transcode_ok": transcode_ok,
        "outputs": {
            "mp4": str(final_path),
            "cv2_mp4": str(cv2_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return payload


def inspect_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {
            "opened": False,
            "path": str(path),
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "opened": True,
        "path": str(path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 else None,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="game_1")
    parser.add_argument("--split", default="training")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--min-valid-points", type=int, default=24)
    parser.add_argument("--no-h264", action="store_true")
    args = parser.parse_args()

    payload = make_evolution_video(
        sample_id=args.sample,
        split=args.split,
        seconds=args.seconds,
        out_fps=args.out_fps,
        width=args.width,
        height=args.height,
        min_valid_points=args.min_valid_points,
        h264=not args.no_h264,
    )

    info = inspect_video(Path(payload["outputs"]["mp4"]))

    print("OK 007Q")
    print("mp4=", payload["outputs"]["mp4"])
    print("json=", payload["outputs"]["json"])
    print("frames_written=", payload["frames_written"])
    print("duration_sec=", info.get("duration_sec"))
    print("fps=", info.get("fps"))
    print("size_bytes=", info.get("size_bytes"))
    print("h264_transcode_ok=", payload["h264_transcode_ok"])
    print("used_segment_count=", payload["used_segment_count"])
    print("read_fail_count=", payload["read_fail_count"])


if __name__ == "__main__":
    main()
