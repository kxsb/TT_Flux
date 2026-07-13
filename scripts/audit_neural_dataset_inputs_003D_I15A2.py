from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2

from ttflux.tracking.candidates.scorers import (
    BallCandidateScoringInput,
)


EXPECTED_CLIPS = (
    "i12a_best_v61_2",
    "i12a_wide_v61_4",
    "i12a_red_v61_7",
)

WIDE_CLIP_ID = "i12a_wide_v61_4"

FALSE_POSITIVE_STATUSES = {
    "wrong_visible",
    "false_positive_invisible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/_ball_gt_seed_003D_I12A/"
            "i12a_frozen_manifest.csv"
        ),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "runs/_ball_gt_benchmark_003D_I12D/"
            "i12d_frozen_gt.csv"
        ),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "runs/_ball_candidate_oracle_003D_I13B/"
            "i13b_candidates_with_gt_distance.csv"
        ),
    )
    parser.add_argument(
        "--temporal-predictions",
        type=Path,
        default=Path(
            "runs/_ball_temporal_reranking_003D_I14C/"
            "i14c_temporal_predictions.csv"
        ),
    )
    parser.add_argument(
        "--i14d-dir",
        type=Path,
        default=Path(
            "runs/_ball_false_positive_gallery_003D_I14D"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_neural_dataset_inputs_003D_I15A2"
        ),
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    return columns, rows


def pick_column(
    columns: list[str],
    aliases: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    lowered = {
        column.strip().lower(): column
        for column in columns
    }

    for alias in aliases:
        match = lowered.get(alias.lower())

        if match is not None:
            return match

    if required:
        raise RuntimeError(
            "Colonne absente. Alternatives attendues : "
            + ", ".join(aliases)
        )

    return None


def parse_float(
    value: Any,
) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_int(
    value: Any,
) -> int:
    result = parse_float(value)

    if result is None:
        raise ValueError(f"Entier invalide : {value!r}")

    return int(round(result))


def resolve_video_path(
    raw: str,
    manifest_path: Path,
) -> Path:
    initial = Path(raw).expanduser()

    candidates = [
        initial,
        Path.cwd() / initial,
        manifest_path.parent / initial,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved.is_file():
            return resolved

    return initial.resolve()


def distribution(
    values: list[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        index = int(round((len(ordered) - 1) * ratio))
        return round(float(ordered[index]), 6)

    return {
        "count": len(ordered),
        "min": round(float(ordered[0]), 6),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": round(float(ordered[-1]), 6),
    }


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    gt_path = args.gt.resolve()
    candidate_path = args.candidates.resolve()
    temporal_path = args.temporal_predictions.resolve()
    i14d_dir = args.i14d_dir.resolve()
    output_dir = args.output_dir.resolve()

    blockers: list[str] = []
    warnings: list[str] = []

    manifest_columns, all_manifest_rows = read_csv(
        manifest_path
    )

    manifest_clip_column = pick_column(
        manifest_columns,
        ("clip_id",),
    )
    manifest_video_column = pick_column(
        manifest_columns,
        ("source_video", "video_path"),
    )
    manifest_start_column = pick_column(
        manifest_columns,
        ("start_frame",),
    )
    manifest_end_column = pick_column(
        manifest_columns,
        ("end_frame_exclusive",),
    )
    manifest_review_column = pick_column(
        manifest_columns,
        ("review_status",),
        required=False,
    )

    manifest_rows = []

    for row in all_manifest_rows:
        status = (
            str(row.get(manifest_review_column, "")).strip().lower()
            if manifest_review_column
            else "accepted"
        )

        if status == "accepted":
            manifest_rows.append(row)

    manifest_clip_ids = tuple(
        str(row[manifest_clip_column]).strip()
        for row in manifest_rows
    )

    if set(manifest_clip_ids) != set(EXPECTED_CLIPS):
        blockers.append(
            "Les trois clips acceptés ne correspondent pas au GT gelé."
        )

    if len(manifest_rows) != 3:
        blockers.append(
            f"Trois clips acceptés attendus, obtenu : "
            f"{len(manifest_rows)}."
        )

    video_inventory: list[dict[str, Any]] = []

    for row in manifest_rows:
        clip_id = str(row[manifest_clip_column]).strip()
        video_path = resolve_video_path(
            str(row[manifest_video_column]),
            manifest_path,
        )
        start_frame = parse_int(row[manifest_start_column])
        end_frame_exclusive = parse_int(
            row[manifest_end_column]
        )

        item: dict[str, Any] = {
            "clip_id": clip_id,
            "video_path": str(video_path),
            "exists": video_path.is_file(),
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame_exclusive,
            "requested_frames": (
                end_frame_exclusive - start_frame
            ),
        }

        if not video_path.is_file():
            blockers.append(
                f"{clip_id} : vidéo introuvable."
            )
            video_inventory.append(item)
            continue

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            blockers.append(
                f"{clip_id} : vidéo illisible par OpenCV."
            )
            item["opencv_opened"] = False
            video_inventory.append(item)
            continue

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        capture.release()

        context_before_available = start_frame >= 1
        context_after_available = (
            frame_count <= 0
            or end_frame_exclusive < frame_count
        )
        range_available = (
            start_frame >= 0
            and (
                frame_count <= 0
                or end_frame_exclusive <= frame_count
            )
        )

        item.update({
            "opencv_opened": True,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "range_available": range_available,
            "context_t_minus_1_available":
                context_before_available,
            "context_t_plus_1_available":
                context_after_available,
            "full_temporal_context_available": (
                context_before_available
                and context_after_available
            ),
        })

        if end_frame_exclusive - start_frame != 150:
            blockers.append(
                f"{clip_id} : le segment ne contient pas "
                "exactement 150 frames."
            )

        if not range_available:
            blockers.append(
                f"{clip_id} : plage GT hors de la vidéo."
            )

        if not (
            context_before_available
            and context_after_available
        ):
            warnings.append(
                f"{clip_id} : contexte t-1/t+1 incomplet ; "
                "une politique de bord sera nécessaire."
            )

        video_inventory.append(item)

    gt_columns, gt_rows = read_csv(gt_path)

    gt_clip_column = pick_column(
        gt_columns,
        ("clip_id",),
    )
    gt_frame_column = pick_column(
        gt_columns,
        ("local_frame", "frame"),
    )
    gt_x_column = pick_column(
        gt_columns,
        ("x", "gt_x", "ball_x"),
    )
    gt_y_column = pick_column(
        gt_columns,
        ("y", "gt_y", "ball_y"),
    )

    gt_index: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    gt_counts_by_clip: Counter[str] = Counter()
    visible_counts_by_clip: Counter[str] = Counter()

    duplicate_gt_keys = 0
    visible_frames = 0

    for row in gt_rows:
        clip_id = str(row[gt_clip_column]).strip()
        local_frame = parse_int(row[gt_frame_column])
        gt_x = parse_float(row.get(gt_x_column))
        gt_y = parse_float(row.get(gt_y_column))
        visible = gt_x is not None and gt_y is not None

        key = (clip_id, local_frame)

        if key in gt_index:
            duplicate_gt_keys += 1

        gt_index[key] = {
            "x": gt_x,
            "y": gt_y,
            "visible": visible,
        }

        gt_counts_by_clip[clip_id] += 1

        if visible:
            visible_frames += 1
            visible_counts_by_clip[clip_id] += 1

    invisible_frames = len(gt_rows) - visible_frames

    if len(gt_rows) != 450:
        blockers.append(
            f"450 lignes GT attendues, obtenu : {len(gt_rows)}."
        )

    if visible_frames != 416 or invisible_frames != 34:
        blockers.append(
            "Répartition GT inattendue : "
            f"{visible_frames} visibles / "
            f"{invisible_frames} invisibles."
        )

    if duplicate_gt_keys:
        blockers.append(
            f"{duplicate_gt_keys} clés GT dupliquées."
        )

    for clip_id in EXPECTED_CLIPS:
        frames = sorted(
            frame
            for current_clip, frame in gt_index
            if current_clip == clip_id
        )

        if frames != list(range(150)):
            blockers.append(
                f"{clip_id} : local_frame doit couvrir 0..149."
            )

    candidate_columns, candidate_rows = read_csv(
        candidate_path
    )

    candidate_clip_column = pick_column(
        candidate_columns,
        ("clip_id",),
    )
    candidate_frame_column = pick_column(
        candidate_columns,
        ("local_frame", "frame"),
    )
    candidate_id_column = pick_column(
        candidate_columns,
        ("candidate_id",),
    )
    candidate_rank_column = pick_column(
        candidate_columns,
        ("rank", "candidate_rank"),
    )
    candidate_x_column = pick_column(
        candidate_columns,
        ("x", "candidate_x"),
    )
    candidate_y_column = pick_column(
        candidate_columns,
        ("y", "candidate_y"),
    )

    numeric_feature_aliases = {
        "bbox_x": ("bbox_x",),
        "bbox_y": ("bbox_y",),
        "bbox_w": ("bbox_w",),
        "bbox_h": ("bbox_h",),
        "area": ("area",),
        "mean_brightness": ("mean_brightness",),
        "motion_strength": ("motion_strength",),
        "fill_ratio": ("fill_ratio",),
        "circularity": ("circularity",),
        "score": ("score",),
    }

    feature_columns: dict[str, str] = {}

    for feature, aliases in numeric_feature_aliases.items():
        feature_columns[feature] = str(
            pick_column(candidate_columns, aliases)
        )

    distance_column = pick_column(
        candidate_columns,
        (
            "gt_distance_px",
            "distance_to_gt_px",
            "distance_px",
            "gt_distance",
        ),
        required=False,
    )

    candidates_by_frame: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    candidate_keys: set[tuple[str, str]] = set()
    duplicate_candidate_keys = 0
    candidate_counts_by_clip: Counter[str] = Counter()
    max_rank = 0

    bbox_dimensions: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []

    for row in candidate_rows:
        clip_id = str(row[candidate_clip_column]).strip()
        local_frame = parse_int(
            row[candidate_frame_column]
        )
        candidate_id = str(
            row[candidate_id_column]
        ).strip()
        rank = parse_int(row[candidate_rank_column])
        x = parse_float(row.get(candidate_x_column))
        y = parse_float(row.get(candidate_y_column))

        if x is None or y is None:
            blockers.append(
                f"Candidat sans coordonnées : "
                f"{clip_id}/{candidate_id}."
            )
            continue

        key = (clip_id, candidate_id)

        if key in candidate_keys:
            duplicate_candidate_keys += 1

        candidate_keys.add(key)
        candidate_counts_by_clip[clip_id] += 1
        max_rank = max(max_rank, rank)

        bbox_w = parse_float(
            row.get(feature_columns["bbox_w"])
        )
        bbox_h = parse_float(
            row.get(feature_columns["bbox_h"])
        )

        if bbox_w is not None:
            bbox_widths.append(bbox_w)

        if bbox_h is not None:
            bbox_heights.append(bbox_h)

        if bbox_w is not None and bbox_h is not None:
            bbox_dimensions.append(max(bbox_w, bbox_h))

        gt = gt_index.get((clip_id, local_frame))
        distance = None

        if distance_column is not None:
            distance = parse_float(row.get(distance_column))

        if (
            distance is None
            and gt is not None
            and gt["visible"]
        ):
            distance = math.hypot(
                x - float(gt["x"]),
                y - float(gt["y"]),
            )

        candidates_by_frame[
            (clip_id, local_frame)
        ].append({
            "candidate_id": candidate_id,
            "rank": rank,
            "x": x,
            "y": y,
            "distance_px": distance,
            "row": row,
        })

    if duplicate_candidate_keys:
        blockers.append(
            f"{duplicate_candidate_keys} identifiants candidats "
            "dupliqués dans un même clip."
        )

    positive_candidate_keys: set[
        tuple[str, str]
    ] = set()

    candidate_visible_frames = 0
    positive_frames = 0
    visible_frames_without_positive = 0
    frames_with_multiple_near_candidates = 0
    additional_near_candidates = 0

    positive_dimensions: list[float] = []

    for key, gt in gt_index.items():
        frame_candidates = candidates_by_frame.get(key, [])

        if frame_candidates:
            candidate_visible_frames += 1

        if not gt["visible"]:
            continue

        near_candidates = [
            candidate
            for candidate in frame_candidates
            if candidate["distance_px"] is not None
            and float(candidate["distance_px"]) <= 20.0
        ]

        near_candidates.sort(
            key=lambda candidate: (
                float(candidate["distance_px"]),
                int(candidate["rank"]),
            )
        )

        if not near_candidates:
            visible_frames_without_positive += 1
            continue

        positive_frames += 1
        best = near_candidates[0]

        positive_candidate_keys.add(
            (key[0], str(best["candidate_id"]))
        )

        row = best["row"]
        bbox_w = parse_float(
            row.get(feature_columns["bbox_w"])
        )
        bbox_h = parse_float(
            row.get(feature_columns["bbox_h"])
        )

        if bbox_w is not None and bbox_h is not None:
            positive_dimensions.append(max(bbox_w, bbox_h))

        if len(near_candidates) > 1:
            frames_with_multiple_near_candidates += 1
            additional_near_candidates += (
                len(near_candidates) - 1
            )

    if positive_frames != 322:
        blockers.append(
            "Le réservoir cap=24 ne retrouve pas les "
            f"322 frames attendues : {positive_frames}."
        )

    temporal_inventory: dict[str, Any] = {
        "path": str(temporal_path),
        "exists": temporal_path.is_file(),
    }

    temporal_hard_negative_keys: set[
        tuple[str, str]
    ] = set()

    hard_negative_counts_by_clip: Counter[str] = Counter()
    hard_negative_rows = 0

    if temporal_path.is_file():
        temporal_columns, temporal_rows = read_csv(
            temporal_path
        )

        temporal_inventory.update({
            "columns": temporal_columns,
            "row_count": len(temporal_rows),
        })

        temporal_clip_column = pick_column(
            temporal_columns,
            ("clip_id",),
        )
        temporal_candidate_column = pick_column(
            temporal_columns,
            ("candidate_id",),
        )
        temporal_status_column = pick_column(
            temporal_columns,
            ("status", "evaluation_status"),
        )
        temporal_strategy_column = pick_column(
            temporal_columns,
            ("strategy",),
            required=False,
        )

        for row in temporal_rows:
            if (
                temporal_strategy_column is not None
                and str(
                    row.get(temporal_strategy_column, "")
                ).strip()
                != "temporal_abstention"
            ):
                continue

            status = str(
                row.get(temporal_status_column, "")
            ).strip()

            if status not in FALSE_POSITIVE_STATUSES:
                continue

            candidate_id = str(
                row.get(temporal_candidate_column, "")
            ).strip()

            if not candidate_id:
                continue

            clip_id = str(
                row.get(temporal_clip_column, "")
            ).strip()

            hard_negative_rows += 1
            hard_negative_counts_by_clip[clip_id] += 1
            temporal_hard_negative_keys.add(
                (clip_id, candidate_id)
            )
    else:
        blockers.append(
            "Les prédictions I14C nécessaires aux hard negatives "
            "sont absentes."
        )

    matched_hard_negatives = (
        temporal_hard_negative_keys
        & candidate_keys
    )
    unmatched_hard_negatives = (
        temporal_hard_negative_keys
        - candidate_keys
    )

    wide_hard_negative_count = hard_negative_counts_by_clip[
        WIDE_CLIP_ID
    ]

    if wide_hard_negative_count != 48:
        blockers.append(
            "48 faux positifs wide I14D attendus, obtenu : "
            f"{wide_hard_negative_count}."
        )

    if unmatched_hard_negatives:
        blockers.append(
            f"{len(unmatched_hard_negatives)} hard negatives "
            "I14C sont absents de la table candidate."
        )

    i14d_artifacts = []

    if i14d_dir.is_dir():
        for path in sorted(i14d_dir.rglob("*")):
            if not path.is_file():
                continue

            i14d_artifacts.append({
                "path": str(path.relative_to(i14d_dir)),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            })
    else:
        warnings.append(
            "Le dossier visuel I14D est absent ; "
            "les identifiants I14C restent disponibles."
        )

    scorer_contract_fields = [
        field.name
        for field in fields(BallCandidateScoringInput)
    ]

    split_folds = []

    for validation_clip in EXPECTED_CLIPS:
        split_folds.append({
            "validation_clip": validation_clip,
            "training_clips": [
                clip_id
                for clip_id in EXPECTED_CLIPS
                if clip_id != validation_clip
            ],
        })

    report = {
        "experiment":
            "003D_I15A2_neural_dataset_input_audit",
        "status":
            "ok" if not blockers else "blocked",
        "inputs": {
            "manifest": str(manifest_path),
            "gt": str(gt_path),
            "candidates": str(candidate_path),
            "temporal_predictions": str(temporal_path),
            "i14d_dir": str(i14d_dir),
        },
        "schemas": {
            "manifest_columns": manifest_columns,
            "gt_columns": gt_columns,
            "candidate_columns": candidate_columns,
            "temporal_columns":
                temporal_inventory.get("columns", []),
            "ball_candidate_scorer_input_fields":
                scorer_contract_fields,
            "numeric_candidate_features":
                feature_columns,
            "distance_column":
                distance_column or "recomputed_from_gt",
            "derived_features": [
                "max_bbox_dimension",
                "bbox_width_height_symmetry",
            ],
        },
        "ground_truth": {
            "rows": len(gt_rows),
            "visible_frames": visible_frames,
            "invisible_frames": invisible_frames,
            "duplicates": duplicate_gt_keys,
            "rows_by_clip": dict(gt_counts_by_clip),
            "visible_by_clip": dict(
                visible_counts_by_clip
            ),
        },
        "videos": video_inventory,
        "candidates": {
            "rows": len(candidate_rows),
            "unique_candidate_keys":
                len(candidate_keys),
            "duplicates":
                duplicate_candidate_keys,
            "max_rank": max_rank,
            "rows_by_clip":
                dict(candidate_counts_by_clip),
            "frames_with_candidates":
                len(candidates_by_frame),
            "bbox_width":
                distribution(bbox_widths),
            "bbox_height":
                distribution(bbox_heights),
            "bbox_max_dimension":
                distribution(bbox_dimensions),
        },
        "label_policy_probe": {
            "radius_px": 20,
            "proposed_positive_rule":
                "nearest_candidate_within_radius",
            "positive_candidates":
                len(positive_candidate_keys),
            "positive_frames":
                positive_frames,
            "visible_frames_without_positive":
                visible_frames_without_positive,
            "frames_with_multiple_candidates_within_radius":
                frames_with_multiple_near_candidates,
            "additional_candidates_within_radius":
                additional_near_candidates,
            "positive_bbox_max_dimension":
                distribution(positive_dimensions),
            "decision_pending":
                (
                    "other candidates within 20 px must be "
                    "labelled not_ball or ignored"
                ),
        },
        "hard_negatives": {
            "false_positive_rows":
                hard_negative_rows,
            "unique_candidate_keys":
                len(temporal_hard_negative_keys),
            "matched_to_candidate_table":
                len(matched_hard_negatives),
            "unmatched":
                len(unmatched_hard_negatives),
            "rows_by_clip":
                dict(hard_negative_counts_by_clip),
            "wide_i14d_player_attached_count":
                wide_hard_negative_count,
        },
        "i14d_artifacts": i14d_artifacts,
        "split_policy": {
            "name": "leave_one_clip_out",
            "random_frame_split_allowed": False,
            "folds": split_folds,
        },
        "blockers": blockers,
        "warnings": warnings,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "i15a2_neural_dataset_input_audit.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "I15A2_INPUT_AUDIT_OK"
        if not blockers
        else "I15A2_INPUT_AUDIT_BLOCKED"
    )

    print()
    print("=== GROUND TRUTH ===")
    print("clips =", len(manifest_rows))
    print("gt_rows =", len(gt_rows))
    print("visible_frames =", visible_frames)
    print("invisible_frames =", invisible_frames)

    print()
    print("=== RÉSERVOIR CAP 24 ===")
    print("candidate_rows =", len(candidate_rows))
    print("max_rank =", max_rank)
    print("positive_frames =", positive_frames)
    print(
        "visible_frames_without_positive =",
        visible_frames_without_positive,
    )
    print(
        "multi_near_gt_frames =",
        frames_with_multiple_near_candidates,
    )
    print(
        "additional_near_gt_candidates =",
        additional_near_candidates,
    )

    print()
    print("=== HARD NEGATIVES I14C / I14D ===")
    print("false_positive_rows =", hard_negative_rows)
    print(
        "unique_hard_negatives =",
        len(temporal_hard_negative_keys),
    )
    print(
        "matched_hard_negatives =",
        len(matched_hard_negatives),
    )
    print(
        "unmatched_hard_negatives =",
        len(unmatched_hard_negatives),
    )
    print(
        "wide_player_attached_hard_negatives =",
        wide_hard_negative_count,
    )

    print()
    print("=== CONTEXTE TEMPOREL ===")

    for item in video_inventory:
        print(
            item["clip_id"],
            "frames=",
            item.get("requested_frames"),
            "context_ok=",
            item.get(
                "full_temporal_context_available",
                False,
            ),
            "shape=",
            (
                item.get("width"),
                item.get("height"),
            ),
            "fps=",
            item.get("fps"),
        )

    print()
    print("=== BLOQUANTS ===")

    if blockers:
        for blocker in blockers:
            print("-", blocker)
    else:
        print("aucun")

    print()
    print("=== AVERTISSEMENTS ===")

    if warnings:
        for warning in warnings:
            print("-", warning)
    else:
        print("aucun")

    print()
    print("report =", report_path)

    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
