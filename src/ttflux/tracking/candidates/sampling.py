from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


POSITIVE_CATEGORY = "positive"
IGNORE_CATEGORY = "ignore"
HARD_NEGATIVE_CATEGORY = "hard_negative"
SAME_FRAME_CATEGORY = "same_frame_negative"
VISIBLE_MISSING_CATEGORY = "visible_missing_negative"
INVISIBLE_CATEGORY = "invisible_negative"

SAMPLER_CATEGORIES = (
    POSITIVE_CATEGORY,
    HARD_NEGATIVE_CATEGORY,
    SAME_FRAME_CATEGORY,
    VISIBLE_MISSING_CATEGORY,
    INVISIBLE_CATEGORY,
)


def candidate_sort_key(
    row: Mapping[str, Any],
) -> tuple[str, int, int, str, int]:
    return (
        str(row["clip_id"]),
        int(row["local_frame"]),
        int(row["rank"]),
        str(row["candidate_id"]),
        int(row["manifest_index"]),
    )


def frame_key(
    row: Mapping[str, Any],
) -> tuple[str, int]:
    return (
        str(row["clip_id"]),
        int(row["local_frame"]),
    )


def select_lowest_rank_per_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_keys: set[tuple[str, int]],
    per_frame: int,
) -> list[Mapping[str, Any]]:
    if per_frame <= 0:
        raise ValueError(
            "per_frame must be positive."
        )

    grouped: dict[
        tuple[str, int],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        key = frame_key(row)

        if key in frame_keys:
            grouped[key].append(row)

    selected: list[Mapping[str, Any]] = []

    for key in sorted(frame_keys):
        candidates = sorted(
            grouped.get(key, []),
            key=candidate_sort_key,
        )

        if len(candidates) < per_frame:
            raise ValueError(
                f"Frame {key} has only "
                f"{len(candidates)} candidates; "
                f"{per_frame} required."
            )

        selected.extend(
            candidates[:per_frame]
        )

    return selected


def select_round_robin_by_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> list[Mapping[str, Any]]:
    if budget < 0:
        raise ValueError(
            "budget cannot be negative."
        )

    if budget == 0:
        return []

    grouped: dict[
        tuple[str, int],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[frame_key(row)].append(row)

    ordered_groups = {
        key: sorted(
            grouped[key],
            key=candidate_sort_key,
        )
        for key in sorted(grouped)
    }

    selected: list[Mapping[str, Any]] = []
    depth = 0

    while len(selected) < budget:
        added = 0

        for key in ordered_groups:
            group = ordered_groups[key]

            if depth >= len(group):
                continue

            selected.append(group[depth])
            added += 1

            if len(selected) >= budget:
                break

        if added == 0:
            break

        depth += 1

    if len(selected) != min(
        budget,
        len(rows),
    ):
        raise RuntimeError(
            "Round-robin selection size mismatch."
        )

    return selected


def build_fold_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_clip: str,
    same_frame_per_positive: int = 4,
    visible_missing_multiplier: int = 2,
    invisible_multiplier: int = 1,
) -> list[dict[str, Any]]:
    if same_frame_per_positive <= 0:
        raise ValueError("same_frame_per_positive must be positive.")
    if visible_missing_multiplier < 0:
        raise ValueError(
            "visible_missing_multiplier cannot be negative."
        )
    if invisible_multiplier < 0:
        raise ValueError("invisible_multiplier cannot be negative.")

    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["clip_id"]) != holdout_clip:
            grouped[row["category"]].append(row)

    positives = sorted(
        grouped[POSITIVE_CATEGORY],
        key=candidate_sort_key,
    )
    hard_negatives = sorted(
        grouped[HARD_NEGATIVE_CATEGORY],
        key=candidate_sort_key,
    )
    positive_frames = {
        frame_key(row)
        for row in positives
    }

    if len(positive_frames) != len(positives):
        raise RuntimeError(
            "Expected one positive candidate per frame."
        )

    same_frame_selected = select_lowest_rank_per_frame(
        grouped[SAME_FRAME_CATEGORY],
        frame_keys=positive_frames,
        per_frame=same_frame_per_positive,
    )
    visible_missing_selected = select_round_robin_by_frame(
        grouped[VISIBLE_MISSING_CATEGORY],
        budget=min(
            visible_missing_multiplier * len(positives),
            len(grouped[VISIBLE_MISSING_CATEGORY]),
        ),
    )
    invisible_selected = select_round_robin_by_frame(
        grouped[INVISIBLE_CATEGORY],
        budget=min(
            invisible_multiplier * len(positives),
            len(grouped[INVISIBLE_CATEGORY]),
        ),
    )

    category_groups = (
        (positives, "all_positive"),
        (hard_negatives, "all_hard_negative"),
        (same_frame_selected, "top_rank_same_frame"),
        (
            visible_missing_selected,
            "round_robin_visible_missing",
        ),
        (invisible_selected, "round_robin_invisible"),
    )

    plan: list[dict[str, Any]] = []
    seen_manifest_indices: set[int] = set()

    for selected_rows, reason in category_groups:
        for row in selected_rows:
            manifest_index = int(row["manifest_index"])

            if manifest_index in seen_manifest_indices:
                raise RuntimeError(
                    "Duplicate manifest index in fold: "
                    f"{manifest_index}"
                )

            seen_manifest_indices.add(manifest_index)
            plan.append({
                **dict(row),
                "selection_reason": reason,
            })

    for sample_order, row in enumerate(plan):
        row["sample_order"] = sample_order
        row["holdout_clip"] = holdout_clip

    if any(str(row["clip_id"]) == holdout_clip for row in plan):
        raise RuntimeError("Holdout leakage detected.")

    if any(row["category"] == IGNORE_CATEGORY for row in plan):
        raise RuntimeError("Ignore candidate selected.")

    return plan
