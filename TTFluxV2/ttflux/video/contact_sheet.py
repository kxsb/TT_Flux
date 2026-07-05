from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np

from ttflux.video.reader import read_video_meta


def _blank(width: int, height: int, text: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (18, 26, 47)
    cv2.putText(
        img,
        text[:40],
        (10, max(24, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


def _fit_thumb(frame: np.ndarray, thumb_w: int, thumb_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return _blank(thumb_w, thumb_h, "bad frame")

    scale = min(thumb_w / w, thumb_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    canvas[:, :] = (10, 15, 30)

    x0 = (thumb_w - nw) // 2
    y0 = (thumb_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_contact_sheet_jpeg(
    path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    count: int = 24,
    cols: int = 6,
    thumb_w: int = 220,
    quality: int = 88,
) -> bytes:
    path = Path(path)
    meta = read_video_meta(path)
    total = int(meta.get("frame_count") or 0)

    if total <= 0:
        raise RuntimeError("video_has_no_frames")

    start_frame = max(0, int(start_frame))
    if end_frame is None:
        end_frame = total - 1
    end_frame = min(total - 1, max(start_frame, int(end_frame)))

    count = max(1, min(80, int(count)))
    cols = max(1, min(10, int(cols)))

    if end_frame <= start_frame:
        frame_indices = [start_frame]
    else:
        frame_indices = np.linspace(start_frame, end_frame, count).round().astype(int).tolist()

    rows = int(math.ceil(len(frame_indices) / cols))

    video_w = int(meta.get("width") or 16)
    video_h = int(meta.get("height") or 9)
    thumb_h = max(90, int(round(thumb_w * video_h / max(1, video_w))))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv_open_failed: {path}")

    thumbs: list[np.ndarray] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            thumb = _blank(thumb_w, thumb_h, f"frame {idx} failed")
        else:
            thumb = _fit_thumb(frame, thumb_w, thumb_h)

        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (55, 65, 90), 1)
        cv2.rectangle(thumb, (0, 0), (92, 22), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"f {idx}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    cap.release()

    while len(thumbs) < rows * cols:
        thumbs.append(_blank(thumb_w, thumb_h, ""))

    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    sheet[:, :] = (7, 11, 22)

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")

    return buf.tobytes()
