from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ttflux.track_io import TrackPoint


@dataclass
class TrackSegment:
    points: list[TrackPoint]
    first_frame: int
    last_frame: int
    span: int
    density: float
    travel: float
    x_range: float
    y_range: float
    score: float


def _segment_stats(
    points: list[TrackPoint],
) -> TrackSegment:
    frames = [p.frame for p in points]
    xs = np.array([p.x for p in points], dtype=np.float32)
    ys = np.array([p.y for p in points], dtype=np.float32)

    first_frame = frames[0]
    last_frame = frames[-1]
    span = last_frame - first_frame + 1

    if span > 0:
        density = len(points) / span
    else:
        density = 0.0

    travel = 0.0

    for i in range(1, len(points)):
        dx = points[i].x - points[i - 1].x
        dy = points[i].y - points[i - 1].y
        travel += float(math.hypot(dx, dy))

    if len(xs):
        x_range = float(xs.max() - xs.min())
        y_range = float(ys.max() - ys.min())
    else:
        x_range = 0.0
        y_range = 0.0

    score = (
        len(points) * density
        + 0.015 * travel
        + 0.01 * (x_range + y_range)
    )

    return TrackSegment(
        points=points,
        first_frame=first_frame,
        last_frame=last_frame,
        span=span,
        density=float(density),
        travel=float(travel),
        x_range=float(x_range),
        y_range=float(y_range),
        score=float(score),
    )


def find_candidate_segments(
    points: list[TrackPoint],
    window_span: int,
    min_points: int,
) -> list[TrackSegment]:
    candidates: list[TrackSegment] = []
    n = len(points)

    for i in range(n):
        first = points[i].frame
        segment: list[TrackPoint] = []

        for j in range(i, n):
            delta = points[j].frame - first

            if delta > window_span - 1:
                break

            segment.append(points[j])

        if len(segment) < min_points:
            continue

        candidates.append(_segment_stats(segment))

    return sorted(
        candidates,
        key=lambda seg: seg.score,
        reverse=True,
    )


def select_non_overlapping(
    candidates: list[TrackSegment],
    top_k: int,
    margin: int,
) -> list[TrackSegment]:
    selected: list[TrackSegment] = []

    for cand in candidates:
        keep = True

        for prev in selected:
            separated_left = (
                cand.last_frame < prev.first_frame - margin
            )
            separated_right = (
                cand.first_frame > prev.last_frame + margin
            )

            if not (separated_left or separated_right):
                keep = False
                break

        if keep:
            selected.append(cand)

        if len(selected) >= top_k:
            break

    return selected
