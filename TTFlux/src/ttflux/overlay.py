from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ttflux.segments import TrackSegment


YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (220, 220, 220)
BLACK = (0, 0, 0)


def _pt(x: float, y: float) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def _draw_header(
    image,
    title: str,
    subtitle: str,
    width: int,
) -> None:
    cv2.rectangle(
        image,
        (0, 0),
        (width, 60),
        BLACK,
        -1,
    )

    cv2.putText(
        image,
        title,
        (20, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        subtitle,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        GREY,
        1,
        cv2.LINE_AA,
    )


def draw_segment_on_frame(
    frame,
    segment: TrackSegment,
    title: str,
    subtitle: str,
):
    overlay = frame.copy()
    width = frame.shape[1]

    pts = np.array(
        [[p.x, p.y] for p in segment.points],
        dtype=np.float32,
    )

    for i in range(1, len(pts)):
        p0 = tuple(np.round(pts[i - 1]).astype(int))
        p1 = tuple(np.round(pts[i]).astype(int))

        cv2.line(
            overlay,
            p0,
            p1,
            YELLOW,
            2,
            cv2.LINE_AA,
        )

    for point in segment.points:
        x, y = _pt(point.x, point.y)

        cv2.circle(
            overlay,
            (x, y),
            4,
            YELLOW,
            -1,
        )

    marks = sorted(
        set([0, len(segment.points) // 2, len(segment.points) - 1])
    )

    for mark in marks:
        p = segment.points[mark]
        x, y = _pt(p.x, p.y)

        cv2.putText(
            overlay,
            f"{mark}/f{p.frame}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1,
            cv2.LINE_AA,
        )

    _draw_header(
        overlay,
        title,
        subtitle,
        width,
    )

    return overlay

def make_segment_png(
    video_path: str | Path,
    segment: TrackSegment,
    path: str | Path,
    title: str,
    subtitle: str,
) -> None:
    video_path = Path(video_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la vidéo: {video_path}"
        )

    mid = segment.points[len(segment.points) // 2].frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Impossible de lire la frame {mid}"
        )

    overlay = draw_segment_on_frame(
        frame,
        segment,
        title,
        subtitle,
    )

    cv2.imwrite(str(path), overlay)


def make_segment_video(
    video_path: str | Path,
    segment: TrackSegment,
    path: str | Path,
    fps: float,
    size: tuple[int, int],
    trail_keep_last: int,
    title_prefix: str,
) -> None:
    video_path = Path(video_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = size

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la vidéo: {video_path}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.first_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height),
    )

    frame_to_points = {}

    for idx, p in enumerate(segment.points):
        frame_to_points.setdefault(p.frame, [])
        frame_to_points[p.frame].append((idx, p.x, p.y))

    trail = []
    current = segment.first_frame

    while current <= segment.last_frame:
        ok, frame = cap.read()

        if not ok:
            break

        if current in frame_to_points:
            trail.extend(frame_to_points[current])

        out = frame.copy()

        for k in range(1, len(trail)):
            p0 = _pt(trail[k - 1][1], trail[k - 1][2])
            p1 = _pt(trail[k][1], trail[k][2])

            cv2.line(
                out,
                p0,
                p1,
                YELLOW,
                2,
                cv2.LINE_AA,
            )

        for _, x, y in trail[-trail_keep_last:]:
            cv2.circle(
                out,
                _pt(x, y),
                4,
                YELLOW,
                -1,
            )

        if trail:
            idx_last, x_last, y_last = trail[-1]
            x, y = _pt(x_last, y_last)

            cv2.circle(
                out,
                (x, y),
                6,
                CYAN,
                2,
            )

            cv2.putText(
                out,
                f"pt {idx_last}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                CYAN,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(
            out,
            (0, 0),
            (width, 48),
            BLACK,
            -1,
        )

        cv2.putText(
            out,
            f"{title_prefix} | f={current} | yellow=v71 trail",
            (20, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        writer.write(out)
        current += 1

    writer.release()
    cap.release()
