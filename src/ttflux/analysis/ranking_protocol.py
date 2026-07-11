from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


HIT_RADIUS_PX = 20.0


def distance_hit(
    distance_to_gt_px: float | None,
    radius_px: float = HIT_RADIUS_PX,
) -> bool:
    if distance_to_gt_px is None:
        return False

    return float(distance_to_gt_px) <= float(radius_px)


def select_canonical_top1(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda row: (
            int(row["rank"]),
            str(row["candidate_id"]),
        ),
    )


def evaluate_ranking_frame(
    *,
    gt_visible: bool,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    top1 = select_canonical_top1(candidates)

    oracle_covered = (
        gt_visible
        and any(
            distance_hit(
                row.get("distance_to_gt_px")
            )
            for row in candidates
        )
    )

    if top1 is None:
        return {
            "candidate_count": 0,
            "oracle_covered": False,
            "visible_without_candidates":
                bool(gt_visible),
            "visible_candidates_without_ball":
                False,
            "baseline_candidate_id": "",
            "baseline_rank": None,
            "baseline_label": "",
            "baseline_distance_to_gt_px": None,
            "baseline_score": None,
            "baseline_hit_20px": False,
            "baseline_strict_label_hit": False,
            "baseline_ignore_hit_20px": False,
            "baseline_hard_negative": False,
        }

    top1_distance = top1.get(
        "distance_to_gt_px"
    )
    top1_label = str(top1["label"])

    tolerant_hit = (
        gt_visible
        and distance_hit(top1_distance)
    )

    strict_hit = (
        gt_visible
        and top1_label == "ball"
    )

    return {
        "candidate_count": len(candidates),
        "oracle_covered": oracle_covered,
        "visible_without_candidates": False,
        "visible_candidates_without_ball": (
            bool(gt_visible)
            and not oracle_covered
        ),
        "baseline_candidate_id":
            str(top1["candidate_id"]),
        "baseline_rank": int(top1["rank"]),
        "baseline_label": top1_label,
        "baseline_distance_to_gt_px":
            top1_distance,
        "baseline_score": top1.get("score"),
        "baseline_hit_20px": tolerant_hit,
        "baseline_strict_label_hit":
            strict_hit,
        "baseline_ignore_hit_20px": (
            tolerant_hit
            and top1_label == "ignore"
        ),
        "baseline_hard_negative": bool(
            top1.get("hard_negative", False)
        ),
    }
