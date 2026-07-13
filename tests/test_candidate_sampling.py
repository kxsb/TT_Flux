from __future__ import annotations

from ttflux.tracking.candidates.sampling import (
    HARD_NEGATIVE_CATEGORY,
    INVISIBLE_CATEGORY,
    POSITIVE_CATEGORY,
    SAME_FRAME_CATEGORY,
    VISIBLE_MISSING_CATEGORY,
    build_fold_plan,
    select_lowest_rank_per_frame,
    select_round_robin_by_frame,
)


def make_row(
    *,
    manifest_index: int,
    clip_id: str,
    local_frame: int,
    rank: int,
    category: str,
) -> dict[str, object]:
    return {
        "manifest_index": manifest_index,
        "candidate_id":
            f"candidate_{manifest_index}",
        "clip_id": clip_id,
        "local_frame": local_frame,
        "rank": rank,
        "category": category,
        "label": (
            "ball"
            if category == POSITIVE_CATEGORY
            else "not_ball"
        ),
    }


def test_lowest_rank_selection_per_frame() -> None:
    rows = [
        make_row(
            manifest_index=1,
            clip_id="clip_a",
            local_frame=0,
            rank=3,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=2,
            clip_id="clip_a",
            local_frame=0,
            rank=1,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=3,
            clip_id="clip_a",
            local_frame=0,
            rank=2,
            category=SAME_FRAME_CATEGORY,
        ),
    ]

    selected = select_lowest_rank_per_frame(
        rows,
        frame_keys={("clip_a", 0)},
        per_frame=2,
    )

    assert [
        row["rank"]
        for row in selected
    ] == [1, 2]


def test_round_robin_prefers_frame_coverage() -> None:
    rows = [
        make_row(
            manifest_index=1,
            clip_id="clip_a",
            local_frame=0,
            rank=1,
            category=VISIBLE_MISSING_CATEGORY,
        ),
        make_row(
            manifest_index=2,
            clip_id="clip_a",
            local_frame=0,
            rank=2,
            category=VISIBLE_MISSING_CATEGORY,
        ),
        make_row(
            manifest_index=3,
            clip_id="clip_a",
            local_frame=1,
            rank=1,
            category=VISIBLE_MISSING_CATEGORY,
        ),
        make_row(
            manifest_index=4,
            clip_id="clip_a",
            local_frame=1,
            rank=2,
            category=VISIBLE_MISSING_CATEGORY,
        ),
    ]

    selected = select_round_robin_by_frame(
        rows,
        budget=3,
    )

    assert [
        (
            row["local_frame"],
            row["rank"],
        )
        for row in selected
    ] == [
        (0, 1),
        (1, 1),
        (0, 2),
    ]


def test_fold_plan_excludes_holdout_and_ignore() -> None:
    rows = [
        make_row(
            manifest_index=0,
            clip_id="train",
            local_frame=0,
            rank=1,
            category=POSITIVE_CATEGORY,
        ),
        make_row(
            manifest_index=1,
            clip_id="train",
            local_frame=0,
            rank=2,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=2,
            clip_id="train",
            local_frame=0,
            rank=3,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=3,
            clip_id="train",
            local_frame=0,
            rank=4,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=4,
            clip_id="train",
            local_frame=0,
            rank=5,
            category=SAME_FRAME_CATEGORY,
        ),
        make_row(
            manifest_index=5,
            clip_id="train",
            local_frame=1,
            rank=1,
            category=HARD_NEGATIVE_CATEGORY,
        ),
        make_row(
            manifest_index=6,
            clip_id="train",
            local_frame=2,
            rank=1,
            category=VISIBLE_MISSING_CATEGORY,
        ),
        make_row(
            manifest_index=7,
            clip_id="train",
            local_frame=3,
            rank=1,
            category=INVISIBLE_CATEGORY,
        ),
        make_row(
            manifest_index=8,
            clip_id="holdout",
            local_frame=0,
            rank=1,
            category=POSITIVE_CATEGORY,
        ),
    ]

    plan = build_fold_plan(
        rows,
        holdout_clip="holdout",
        same_frame_per_positive=4,
        visible_missing_multiplier=2,
        invisible_multiplier=1,
    )

    assert all(
        row["clip_id"] == "train"
        for row in plan
    )

    categories = [
        row["category"]
        for row in plan
    ]

    assert categories.count(
        POSITIVE_CATEGORY
    ) == 1

    assert categories.count(
        SAME_FRAME_CATEGORY
    ) == 4

    assert categories.count(
        HARD_NEGATIVE_CATEGORY
    ) == 1

    assert categories.count(
        VISIBLE_MISSING_CATEGORY
    ) == 1

    assert categories.count(
        INVISIBLE_CATEGORY
    ) == 1

    assert [
        row["sample_order"]
        for row in plan
    ] == list(range(len(plan)))
