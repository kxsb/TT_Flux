from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BallCandidateScoringInput:
    """Entr?e stable fournie ? un scorer de candidat balle."""

    previous_gray: np.ndarray
    current_gray: np.ndarray
    next_gray: np.ndarray

    x: float
    y: float

    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    area: int
    mean_brightness: float
    motion_strength: float
    fill_ratio: float
    circularity: float


class BallCandidateScorer(Protocol):
    """Contrat minimal d'un scorer interchangeable."""

    scorer_id: str

    def score(
        self,
        candidate: BallCandidateScoringInput,
    ) -> float:
        """Retourne un score o? une valeur sup?rieure est pr?f?rable."""
        ...


class HeuristicV1BallCandidateScorer:
    """Adaptateur exact du score heuristique canonique historique."""

    scorer_id = "heuristic_v1"

    def score(
        self,
        candidate: BallCandidateScoringInput,
    ) -> float:
        target_area = 28.0

        size_score = math.exp(
            -abs(float(candidate.area) - target_area) / target_area
        )
        motion_score = min(
            1.0,
            candidate.motion_strength / 100.0,
        )
        brightness_score = min(
            1.0,
            candidate.mean_brightness / 255.0,
        )
        compactness_score = min(
            1.0,
            candidate.fill_ratio / 0.70,
        )

        return float(
            0.38 * motion_score
            + 0.22 * brightness_score
            + 0.16 * compactness_score
            + 0.14 * candidate.circularity
            + 0.10 * size_score
        )
