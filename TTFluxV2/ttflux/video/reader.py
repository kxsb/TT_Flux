from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import cv2


@dataclass
class VideoMeta:
    path: str
    exists: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_sec: float = 0.0
    backend_ok: bool = False
    error: str = ""


def read_video_meta(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return asdict(VideoMeta(path=str(path), exists=False, error="file_not_found"))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return asdict(VideoMeta(path=str(path), exists=True, error="opencv_open_failed"))

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return asdict(
        VideoMeta(
            path=str(path),
            exists=True,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            backend_ok=True,
        )
    )


def extract_frame_jpeg(path: Path, frame_idx: int, quality: int = 90) -> bytes:
    path = Path(path)
    if frame_idx < 0:
        frame_idx = 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv_open_failed: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"frame_read_failed: frame={frame_idx}")

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")

    return buf.tobytes()



def read_frame_bgr(path: Path, frame_idx: int):
    path = Path(path)
    if frame_idx < 0:
        frame_idx = 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"opencv_open_failed: {path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"frame_read_failed: frame={frame_idx}")

    return frame


def encode_frame_jpeg(frame, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")
    return buf.tobytes()
