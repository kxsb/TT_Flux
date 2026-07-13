from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ttflux.tracking.candidates.sampling import (  # noqa: E402
    HARD_NEGATIVE_CATEGORY,
    IGNORE_CATEGORY,
    INVISIBLE_CATEGORY,
    POSITIVE_CATEGORY,
    SAME_FRAME_CATEGORY,
    VISIBLE_MISSING_CATEGORY,
    build_fold_plan,
)


EXPECTED_INDEX_SHA256 = (
    "2f46d1633685da2ab88f936f86be1315aec2f1916b9a4301cec7cf2b09f892df"
)

EXPECTED_FRAME_SHA256 = (
    "6718c12bd0c69db641b93ecf5cae36160af6ee1e1dc444a04e4bf0cd6b468377"
)

EXPECTED_CLIPS = (
    "i12a_best_v61_2",
    "i12a_red_v61_7",
    "i12a_wide_v61_4",
)

EXPECTED_GLOBAL_CATEGORIES = {
    POSITIVE_CATEGORY: 322,
    IGNORE_CATEGORY: 37,
    HARD_NEGATIVE_CATEGORY: 81,
    SAME_FRAME_CATEGORY: 7303,
    VISIBLE_MISSING_CATEGORY: 1763,
    INVISIBLE_CATEGORY: 793,
}

EXPECTED_FOLD_COUNTS = {
    "i12a_best_v61_2": {
        POSITIVE_CATEGORY: 219,
        HARD_NEGATIVE_CATEGORY: 81,
        SAME_FRAME_CATEGORY: 876,
        VISIBLE_MISSING_CATEGORY: 438,
        INVISIBLE_CATEGORY: 219,
        "total": 1833,
    },
    "i12a_red_v61_7": {
        POSITIVE_CATEGORY: 217,
        HARD_NEGATIVE_CATEGORY: 48,
        SAME_FRAME_CATEGORY: 868,
        VISIBLE_MISSING_CATEGORY: 434,
        INVISIBLE_CATEGORY: 72,
        "total": 1639,
    },
    "i12a_wide_v61_4": {
        POSITIVE_CATEGORY: 208,
        HARD_NEGATIVE_CATEGORY: 33,
        SAME_FRAME_CATEGORY: 832,
        VISIBLE_MISSING_CATEGORY: 416,
        INVISIBLE_CATEGORY: 208,
        "total": 1697,
    },
}

PLAN_FIELDS = (
    "holdout_clip",
    "sample_order",
    "manifest_index",
    "shard_name",
    "shard_row",
    "candidate_id",
    "clip_id",
    "local_frame",
    "source_frame",
    "rank",
    "label",
    "label_id",
    "hard_negative",
    "category",
    "selection_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=Path(
            "runs/_ball_candidate_crops_003D_I15A5/"
            "i15a5_dataset_index.csv"
        ),
    )
    parser.add_argument(
        "--frame-universe",
        type=Path,
        default=Path(
            "runs/_ball_ranking_protocol_003D_I15A6A/"
            "i15a6a_frame_universe.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_candidate_sampler_003D_I15A6B1"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def parse_int(value: Any) -> int:
    text = str(value or "").strip()

    if not text:
        raise ValueError(
            f"Missing integer value: {value!r}"
        )

    return int(round(float(
        text.replace(",", ".")
    )))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    return result.stdout.strip()


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PLAN_FIELDS,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row[field]
                for field in PLAN_FIELDS
            })

    temporary.replace(path)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    temporary.replace(path)


def normalize_rows(
    index_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    frames: dict[
        tuple[str, int],
        dict[str, bool],
    ] = {}

    for raw in frame_rows:
        clip_id = str(raw["clip_id"]).strip()
        local_frame = parse_int(
            raw["local_frame"]
        )

        key = (
            clip_id,
            local_frame,
        )

        if key in frames:
            raise RuntimeError(
                f"Duplicate frame: {key}"
            )

        frames[key] = {
            "gt_visible": (
                parse_int(raw["gt_visible"])
                == 1
            ),
            "oracle_covered": (
                parse_int(raw["oracle_covered"])
                == 1
            ),
        }

    rows: list[dict[str, Any]] = []

    for raw in index_rows:
        clip_id = str(raw["clip_id"]).strip()
        local_frame = parse_int(
            raw["local_frame"]
        )

        key = (
            clip_id,
            local_frame,
        )

        if key not in frames:
            raise RuntimeError(
                f"Candidate without frame: {key}"
            )

        label = str(raw["label"]).strip()
        hard_negative = (
            parse_int(raw["hard_negative"])
            == 1
        )
        frame = frames[key]

        if label == "ball":
            category = POSITIVE_CATEGORY

        elif label == "ignore":
            category = IGNORE_CATEGORY

        elif label != "not_ball":
            raise RuntimeError(
                f"Unexpected label: {label}"
            )

        elif hard_negative:
            category = HARD_NEGATIVE_CATEGORY

        elif not frame["gt_visible"]:
            category = INVISIBLE_CATEGORY

        elif frame["oracle_covered"]:
            category = SAME_FRAME_CATEGORY

        else:
            category = VISIBLE_MISSING_CATEGORY

        rows.append({
            "manifest_index": parse_int(
                raw["manifest_index"]
            ),
            "shard_name": str(
                raw["shard_name"]
            ).strip(),
            "shard_row": parse_int(
                raw["shard_row"]
            ),
            "candidate_id": str(
                raw["candidate_id"]
            ).strip(),
            "clip_id": clip_id,
            "local_frame": local_frame,
            "source_frame": parse_int(
                raw["source_frame"]
            ),
            "rank": parse_int(raw["rank"]),
            "label": label,
            "label_id": parse_int(
                raw["label_id"]
            ),
            "hard_negative": int(
                hard_negative
            ),
            "category": category,
        })

    return rows


def category_counts(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    counter = Counter(
        row["category"]
        for row in rows
    )

    return {
        key: counter[key]
        for key in sorted(counter)
    }


def main() -> None:
    args = parse_args()

    index_path = args.dataset_index.resolve()
    frame_path = args.frame_universe.resolve()
    output_dir = args.output_dir.resolve()

    index_hash = sha256_file(index_path)
    frame_hash = sha256_file(frame_path)

    if index_hash != EXPECTED_INDEX_SHA256:
        raise RuntimeError(
            f"Dataset index hash mismatch: {index_hash}"
        )

    if frame_hash != EXPECTED_FRAME_SHA256:
        raise RuntimeError(
            f"Frame universe hash mismatch: {frame_hash}"
        )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise RuntimeError(
            f"Output directory is not empty: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_rows = read_csv(index_path)
    frame_rows = read_csv(frame_path)

    if len(index_rows) != 10299:
        raise RuntimeError(
            f"Expected 10299 index rows, "
            f"got {len(index_rows)}."
        )

    if len(frame_rows) != 450:
        raise RuntimeError(
            f"Expected 450 frame rows, "
            f"got {len(frame_rows)}."
        )

    rows = normalize_rows(
        index_rows,
        frame_rows,
    )

    global_counts = Counter(
        row["category"]
        for row in rows
    )

    if dict(global_counts) != EXPECTED_GLOBAL_CATEGORIES:
        raise RuntimeError(
            "Global category counts mismatch: "
            f"{dict(global_counts)}"
        )

    combined_plan: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}

    for holdout_clip in EXPECTED_CLIPS:
        fold_plan = build_fold_plan(
            rows,
            holdout_clip=holdout_clip,
            same_frame_per_positive=4,
            visible_missing_multiplier=2,
            invisible_multiplier=1,
        )

        counts = Counter(
            row["category"]
            for row in fold_plan
        )

        expected = EXPECTED_FOLD_COUNTS[
            holdout_clip
        ]

        for category in (
            POSITIVE_CATEGORY,
            HARD_NEGATIVE_CATEGORY,
            SAME_FRAME_CATEGORY,
            VISIBLE_MISSING_CATEGORY,
            INVISIBLE_CATEGORY,
        ):
            if counts[category] != expected[category]:
                raise RuntimeError(
                    f"{holdout_clip}/{category}: "
                    f"expected {expected[category]}, "
                    f"got {counts[category]}."
                )

        if len(fold_plan) != expected["total"]:
            raise RuntimeError(
                f"{holdout_clip}: expected "
                f"{expected['total']} rows, "
                f"got {len(fold_plan)}."
            )

        if [
            row["sample_order"]
            for row in fold_plan
        ] != list(range(len(fold_plan))):
            raise RuntimeError(
                f"{holdout_clip}: invalid sample order."
            )

        combined_plan.extend(fold_plan)

        positive_count = counts[
            POSITIVE_CATEGORY
        ]
        negative_count = (
            len(fold_plan)
            - positive_count
        )

        fold_reports[holdout_clip] = {
            "train_clips": [
                clip_id
                for clip_id in EXPECTED_CLIPS
                if clip_id != holdout_clip
            ],
            "selected_rows": len(fold_plan),
            "selected_categories": {
                key: counts[key]
                for key in (
                    POSITIVE_CATEGORY,
                    HARD_NEGATIVE_CATEGORY,
                    SAME_FRAME_CATEGORY,
                    VISIBLE_MISSING_CATEGORY,
                    INVISIBLE_CATEGORY,
                )
            },
            "positive_fraction": round(
                positive_count / len(fold_plan),
                9,
            ),
            "negative_per_positive": round(
                negative_count / positive_count,
                9,
            ),
        }

    expected_total = sum(
        item["total"]
        for item in EXPECTED_FOLD_COUNTS.values()
    )

    if len(combined_plan) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} combined rows, "
            f"got {len(combined_plan)}."
        )

    plan_path = (
        output_dir
        / "i15a6b1_sampler_plan.csv"
    )
    report_path = (
        output_dir
        / "i15a6b1_sampler_report.json"
    )

    atomic_write_csv(
        plan_path,
        combined_plan,
    )

    plan_hash = sha256_file(plan_path)

    report = {
        "experiment":
            "003D_I15A6B1_candidate_sampler",
        "schema_version": 1,
        "repository_head_at_freeze":
            git_head(),
        "inputs": {
            "dataset_index":
                args.dataset_index.as_posix(),
            "dataset_index_sha256":
                index_hash,
            "frame_universe":
                args.frame_universe.as_posix(),
            "frame_universe_sha256":
                frame_hash,
        },
        "contract": {
            "split_strategy":
                "leave_one_clip_out",
            "random_frame_split":
                "forbidden",
            "positive_selection":
                "all ball labels",
            "ignore_selection":
                "excluded",
            "hard_negative_selection":
                "all available",
            "same_frame_selection": {
                "count_per_positive_frame": 4,
                "ordering": [
                    "canonical rank",
                    "candidate_id",
                    "manifest_index",
                ],
            },
            "visible_missing_selection": {
                "budget_multiplier": 2,
                "strategy":
                    "round_robin_by_frame_then_rank",
            },
            "invisible_selection": {
                "budget_multiplier": 1,
                "strategy":
                    "round_robin_by_frame_then_rank",
            },
            "learned_features_forbidden": [
                "canonical rank",
                "heuristic score",
            ],
        },
        "global_available_categories": {
            key: global_counts[key]
            for key in (
                POSITIVE_CATEGORY,
                IGNORE_CATEGORY,
                HARD_NEGATIVE_CATEGORY,
                SAME_FRAME_CATEGORY,
                VISIBLE_MISSING_CATEGORY,
                INVISIBLE_CATEGORY,
            )
        },
        "folds": fold_reports,
        "combined_rows_across_folds":
            len(combined_plan),
        "artifacts": {
            "sampler_plan_csv":
                plan_path.name,
            "sampler_plan_sha256":
                plan_hash,
            "sampler_report_json":
                report_path.name,
        },
    }

    atomic_write_json(
        report_path,
        report,
    )

    if not args.quiet:
        print()
        print(
            "I15A6B1_SAMPLER_FREEZE_OK"
        )

        print()
        print("=== GLOBAL AVAILABLE ===")
        print(
            dict(
                report[
                    "global_available_categories"
                ]
            )
        )

        print()
        print("=== FOLDS ===")

        for holdout_clip in EXPECTED_CLIPS:
            print()
            print(
                "holdout =",
                holdout_clip,
            )

            for key, value in fold_reports[
                holdout_clip
            ].items():
                print(
                    " ",
                    key,
                    "=",
                    value,
                )

        print()
        print("=== ARTIFACTS ===")
        print(
            "sampler_plan_sha256 =",
            plan_hash,
        )
        print(
            "sampler_report_sha256 =",
            sha256_file(report_path),
        )
        print(
            "combined_rows_across_folds =",
            len(combined_plan),
        )
        print(
            "output_dir =",
            output_dir,
        )


if __name__ == "__main__":
    main()
