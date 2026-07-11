from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ttflux.analysis.ranking_protocol import (  # noqa: E402
    HIT_RADIUS_PX,
    evaluate_ranking_frame,
)


EXPECTED_MANIFEST_SHA256 = (
    "320a81f21c30e5ec363c7c560746cb46776e2355e26f8cc8879fd59e2b529b1d"
)

EXPECTED_CLIPS = (
    "i12a_best_v61_2",
    "i12a_red_v61_7",
    "i12a_wide_v61_4",
)

FRAME_FIELDS = (
    "clip_id",
    "split_group",
    "local_frame",
    "source_frame",
    "gt_visible",
    "candidate_count",
    "oracle_covered",
    "visible_without_candidates",
    "visible_candidates_without_ball",
    "baseline_candidate_id",
    "baseline_rank",
    "baseline_label",
    "baseline_distance_to_gt_px",
    "baseline_score",
    "baseline_hit_20px",
    "baseline_strict_label_hit",
    "baseline_ignore_hit_20px",
    "baseline_hard_negative",
)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "runs/_ball_gt_benchmark_003D_I12D/"
            "i12d_frozen_gt.csv"
        ),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path(
            "runs/_ball_candidate_labels_003D_I15A3/"
            "i15a3_candidate_label_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_ranking_protocol_003D_I15A6A"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args()


def parse_float(value: Any) -> float:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        raise ValueError(
            f"Missing numeric value: {value!r}"
        )

    number = float(text)

    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite numeric value: {value!r}"
        )

    return number


def parse_optional_float(
    value: Any,
) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    number = float(text)

    if not math.isfinite(number):
        return None

    return number


def parse_int(value: Any) -> int:
    return int(round(parse_float(value)))


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def format_optional_float(
    value: float | None,
) -> str:
    if value is None:
        return ""

    return format(float(value), ".12g")


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
            fieldnames=FRAME_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

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


def ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(
        numerator / denominator,
        9,
    )


def summarize_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()

    for row in rows:
        counts["frames"] += 1

        if row["gt_visible"]:
            counts["visible_frames"] += 1
        else:
            counts["invisible_frames"] += 1

        if row["candidate_count"] > 0:
            counts["frames_with_candidates"] += 1
        else:
            counts["frames_without_candidates"] += 1

        for key in (
            "oracle_covered",
            "visible_without_candidates",
            "visible_candidates_without_ball",
            "baseline_hit_20px",
            "baseline_strict_label_hit",
            "baseline_ignore_hit_20px",
            "baseline_hard_negative",
        ):
            if row[key]:
                counts[key] += 1

    return {
        "frames": counts["frames"],
        "visible_frames":
            counts["visible_frames"],
        "invisible_frames":
            counts["invisible_frames"],
        "frames_with_candidates":
            counts["frames_with_candidates"],
        "frames_without_candidates":
            counts["frames_without_candidates"],
        "oracle_covered_frames":
            counts["oracle_covered"],
        "visible_without_candidates":
            counts["visible_without_candidates"],
        "visible_candidates_without_ball":
            counts[
                "visible_candidates_without_ball"
            ],
        "visible_without_positive": (
            counts["visible_without_candidates"]
            + counts[
                "visible_candidates_without_ball"
            ]
        ),
        "baseline_hits_20px":
            counts["baseline_hit_20px"],
        "baseline_strict_label_hits":
            counts[
                "baseline_strict_label_hit"
            ],
        "baseline_ignore_hits_20px":
            counts[
                "baseline_ignore_hit_20px"
            ],
        "baseline_top1_hard_negative":
            counts["baseline_hard_negative"],
        "conditional_top1_20px": ratio(
            counts["baseline_hit_20px"],
            counts["oracle_covered"],
        ),
        "end_to_end_visible_recall_20px":
            ratio(
                counts["baseline_hit_20px"],
                counts["visible_frames"],
            ),
        "oracle_coverage": ratio(
            counts["oracle_covered"],
            counts["visible_frames"],
        ),
        "strict_label_top1_rate": ratio(
            counts[
                "baseline_strict_label_hit"
            ],
            counts["oracle_covered"],
        ),
    }


def main() -> None:
    args = parse_args()

    gt_path = args.gt.resolve()
    manifest_path = (
        args.candidate_manifest.resolve()
    )
    output_dir = args.output_dir.resolve()

    manifest_sha256 = sha256_file(
        manifest_path
    )

    if (
        manifest_sha256
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Candidate manifest hash mismatch: "
            f"{manifest_sha256}"
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

    gt_rows = read_csv(gt_path)
    candidate_rows = read_csv(
        manifest_path
    )

    if len(gt_rows) != 450:
        raise RuntimeError(
            f"Expected 450 GT rows, "
            f"got {len(gt_rows)}."
        )

    if len(candidate_rows) != 10299:
        raise RuntimeError(
            f"Expected 10299 candidate rows, "
            f"got {len(candidate_rows)}."
        )

    gt_by_frame: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    for raw in gt_rows:
        clip_id = str(
            raw["clip_id"]
        ).strip()
        local_frame = parse_int(
            raw["local_frame"]
        )

        if clip_id not in EXPECTED_CLIPS:
            raise RuntimeError(
                f"Unexpected GT clip: {clip_id}"
            )

        gt_visible = (
            str(raw["status"]).strip()
            != "not_visible"
            and bool(
                str(raw.get("x") or "").strip()
            )
            and bool(
                str(raw.get("y") or "").strip()
            )
        )

        key = (
            clip_id,
            local_frame,
        )

        if key in gt_by_frame:
            raise RuntimeError(
                f"Duplicate GT frame: {key}"
            )

        gt_by_frame[key] = {
            "clip_id": clip_id,
            "local_frame": local_frame,
            "source_frame": parse_int(
                raw["source_frame"]
            ),
            "gt_visible": gt_visible,
        }

    candidates_by_frame: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    candidate_ids = set()

    for raw in candidate_rows:
        candidate_id = str(
            raw["candidate_id"]
        ).strip()

        if candidate_id in candidate_ids:
            raise RuntimeError(
                f"Duplicate candidate_id: "
                f"{candidate_id}"
            )

        candidate_ids.add(candidate_id)

        clip_id = str(
            raw["clip_id"]
        ).strip()
        local_frame = parse_int(
            raw["local_frame"]
        )

        key = (
            clip_id,
            local_frame,
        )

        if key not in gt_by_frame:
            raise RuntimeError(
                f"Candidate without GT: {key}"
            )

        candidates_by_frame[key].append({
            "candidate_id": candidate_id,
            "rank": parse_int(raw["rank"]),
            "label": str(
                raw["label"]
            ).strip(),
            "distance_to_gt_px":
                parse_optional_float(
                    raw.get(
                        "distance_to_gt_px"
                    )
                ),
            "score": parse_float(
                raw["score"]
            ),
            "hard_negative": (
                parse_int(
                    raw["hard_negative"]
                )
                == 1
            ),
        })

    for key, rows in candidates_by_frame.items():
        ranks = sorted(
            int(row["rank"])
            for row in rows
        )

        expected = list(
            range(1, len(rows) + 1)
        )

        if ranks != expected:
            raise RuntimeError(
                f"Invalid ranks at {key}: "
                f"{ranks}"
            )

    frame_rows: list[dict[str, Any]] = []

    for clip_id in EXPECTED_CLIPS:
        for local_frame in range(150):
            key = (
                clip_id,
                local_frame,
            )

            if key not in gt_by_frame:
                raise RuntimeError(
                    f"Missing GT frame: {key}"
                )

            gt = gt_by_frame[key]
            candidates = candidates_by_frame.get(
                key,
                [],
            )

            evaluation = evaluate_ranking_frame(
                gt_visible=gt["gt_visible"],
                candidates=candidates,
            )

            frame_rows.append({
                "clip_id": clip_id,
                "split_group": clip_id,
                "local_frame": local_frame,
                "source_frame":
                    gt["source_frame"],
                "gt_visible": int(
                    gt["gt_visible"]
                ),
                "candidate_count":
                    evaluation[
                        "candidate_count"
                    ],
                "oracle_covered": int(
                    evaluation[
                        "oracle_covered"
                    ]
                ),
                "visible_without_candidates":
                    int(
                        evaluation[
                            "visible_without_candidates"
                        ]
                    ),
                "visible_candidates_without_ball":
                    int(
                        evaluation[
                            "visible_candidates_without_ball"
                        ]
                    ),
                "baseline_candidate_id":
                    evaluation[
                        "baseline_candidate_id"
                    ],
                "baseline_rank": (
                    evaluation["baseline_rank"]
                    if evaluation[
                        "baseline_rank"
                    ] is not None
                    else ""
                ),
                "baseline_label":
                    evaluation[
                        "baseline_label"
                    ],
                "baseline_distance_to_gt_px":
                    format_optional_float(
                        evaluation[
                            "baseline_distance_to_gt_px"
                        ]
                    ),
                "baseline_score":
                    format_optional_float(
                        evaluation[
                            "baseline_score"
                        ]
                    ),
                "baseline_hit_20px": int(
                    evaluation[
                        "baseline_hit_20px"
                    ]
                ),
                "baseline_strict_label_hit":
                    int(
                        evaluation[
                            "baseline_strict_label_hit"
                        ]
                    ),
                "baseline_ignore_hit_20px":
                    int(
                        evaluation[
                            "baseline_ignore_hit_20px"
                        ]
                    ),
                "baseline_hard_negative":
                    int(
                        evaluation[
                            "baseline_hard_negative"
                        ]
                    ),
            })

    global_summary = summarize_rows(
        frame_rows
    )

    expected_global = {
        "frames": 450,
        "visible_frames": 416,
        "invisible_frames": 34,
        "frames_with_candidates": 442,
        "frames_without_candidates": 8,
        "oracle_covered_frames": 322,
        "visible_without_candidates": 8,
        "visible_candidates_without_ball": 86,
        "visible_without_positive": 94,
        "baseline_hits_20px": 258,
        "baseline_strict_label_hits": 256,
        "baseline_ignore_hits_20px": 2,
        "baseline_top1_hard_negative": 22,
    }

    for key, expected in expected_global.items():
        actual = global_summary[key]

        if actual != expected:
            raise RuntimeError(
                f"{key}: expected {expected}, "
                f"got {actual}."
            )

    by_clip = {}

    for clip_id in EXPECTED_CLIPS:
        clip_rows = [
            row
            for row in frame_rows
            if row["clip_id"] == clip_id
        ]

        by_clip[clip_id] = summarize_rows(
            clip_rows
        )

    index_path = (
        output_dir
        / "i15a6a_frame_universe.csv"
    )
    report_path = (
        output_dir
        / "i15a6a_ranking_protocol.json"
    )

    atomic_write_csv(
        index_path,
        frame_rows,
    )

    index_sha256 = sha256_file(
        index_path
    )

    report = {
        "experiment":
            "003D_I15A6A_ranking_protocol",
        "schema_version": 1,
        "inputs": {
            "frozen_gt":
                args.gt.as_posix(),
            "frozen_gt_sha256":
                sha256_file(gt_path),
            "candidate_manifest":
                args.candidate_manifest.as_posix(),
            "candidate_manifest_sha256":
                manifest_sha256,
        },
        "contract": {
            "training": {
                "positive_label": "ball",
                "negative_label": "not_ball",
                "excluded_from_loss": [
                    "ignore"
                ],
                "random_candidate_split":
                    "forbidden",
                "split_strategy":
                    "leave_one_clip_out",
            },
            "baseline_selection": {
                "selector":
                    "minimum canonical rank",
                "tie_break":
                    "candidate_id",
                "recompute_from_csv_score":
                    False,
            },
            "learned_selection": {
                "selector":
                    "maximum model logit",
                "tie_break": [
                    "canonical rank",
                    "candidate_id",
                ],
            },
            "primary_metrics": {
                "conditional_top1_20px":
                    "selected candidate within "
                    "20 px divided by "
                    "oracle-covered visible frames",
                "end_to_end_visible_recall_20px":
                    "selected candidate within "
                    "20 px divided by all "
                    "visible GT frames",
                "oracle_coverage":
                    "visible frames containing "
                    "at least one candidate "
                    "within 20 px",
            },
            "secondary_metrics": {
                "strict_label_top1_rate":
                    "selected candidate has "
                    "the designated ball label",
                "top1_hard_negative_count":
                    "selected hard-negative "
                    "candidate count",
            },
            "hit_radius_px":
                HIT_RADIUS_PX,
        },
        "global": global_summary,
        "by_clip": by_clip,
        "leave_one_clip_out": [
            {
                "holdout_clip": holdout,
                "train_clips": [
                    clip_id
                    for clip_id in EXPECTED_CLIPS
                    if clip_id != holdout
                ],
            }
            for holdout in EXPECTED_CLIPS
        ],
        "artifacts": {
            "frame_universe_csv":
                index_path.name,
            "frame_universe_sha256":
                index_sha256,
            "report_json":
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
            "I15A6A_RANKING_PROTOCOL_FREEZE_OK"
        )

        print()
        print("=== GLOBAL ===")

        for key, value in global_summary.items():
            print(key, "=", value)

        print()
        print("=== BY CLIP ===")

        for clip_id in EXPECTED_CLIPS:
            print(
                clip_id,
                by_clip[clip_id],
            )

        print()
        print("=== ARTIFACTS ===")
        print(
            "frame_universe_sha256 =",
            index_sha256,
        )
        print(
            "report_sha256 =",
            sha256_file(report_path),
        )
        print(
            "output_dir =",
            output_dir,
        )


if __name__ == "__main__":
    main()
