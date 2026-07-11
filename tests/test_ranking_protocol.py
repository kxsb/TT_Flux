from __future__ import annotations

from ttflux.analysis.ranking_protocol import (
    distance_hit,
    evaluate_ranking_frame,
    select_canonical_top1,
)


def test_distance_hit_uses_inclusive_20px() -> None:
    assert distance_hit(19.999)
    assert distance_hit(20.0)
    assert not distance_hit(20.001)
    assert not distance_hit(None)


def test_canonical_top1_uses_rank() -> None:
    candidates = [
        {
            "candidate_id": "candidate_b",
            "rank": 2,
        },
        {
            "candidate_id": "candidate_a",
            "rank": 1,
        },
    ]

    selected = select_canonical_top1(
        candidates
    )

    assert selected is not None
    assert (
        selected["candidate_id"]
        == "candidate_a"
    )


def test_ignore_can_be_tolerant_hit() -> None:
    evaluation = evaluate_ranking_frame(
        gt_visible=True,
        candidates=[
            {
                "candidate_id": "rank_1",
                "rank": 1,
                "label": "ignore",
                "distance_to_gt_px": 15.69,
                "score": 0.6,
                "hard_negative": False,
            },
            {
                "candidate_id":
                    "designated_ball",
                "rank": 12,
                "label": "ball",
                "distance_to_gt_px": 5.78,
                "score": 0.4,
                "hard_negative": False,
            },
        ],
    )

    assert evaluation["oracle_covered"]
    assert evaluation["baseline_hit_20px"]
    assert not evaluation[
        "baseline_strict_label_hit"
    ]
    assert evaluation[
        "baseline_ignore_hit_20px"
    ]


def test_visible_frame_without_candidates() -> None:
    evaluation = evaluate_ranking_frame(
        gt_visible=True,
        candidates=[],
    )

    assert evaluation["candidate_count"] == 0
    assert evaluation[
        "visible_without_candidates"
    ]
    assert not evaluation["oracle_covered"]
    assert not evaluation[
        "baseline_hit_20px"
    ]
