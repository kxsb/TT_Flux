from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BallCandidate:
    frame: int
    x: float
    y: float
    score: float
    source: str
    payload: dict | None = None


@dataclass
class BallTrackPoint:
    frame: int
    x: float
    y: float
    confidence: float
    state: str = "observed"
