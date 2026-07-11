from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_CLIPS = (
    "i12a_best_v61_2",
    "i12a_wide_v61_4",
    "i12a_red_v61_7",
)

LABEL_RADIUS_PX = 20.0

FALSE_POSITIVE_STATUSES = {
    "wrong_visible",
    "false_positive_invisible",
}

OUTPUT_FIELDS = (
    "candidate_id",
    "clip_id",
    "source_key",
    "local_frame",
    "source_frame",
    "split_group",
    "rank",
    "x",
    "y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "mean_brightness",
    "motion_strength",
    "fill_ratio",
    "circularity",
    "score",
    "scorer_id",
    "gt_visible",
    "gt_x",
    "gt_y",
    "distance_to_gt_px",
    "label",
    "label_reason",
    "hard_negative",
    "hard_negative_source",
    "source_video",
    "clip_path",
    "source_width",
    "source_height",
    "source_fps",
)


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
        "--output-dir",
        type=Path,
        default=Path(
            "runs/_ball_candidate_labels_003D_I15A3"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def parse_int(value: Any) -> int:
    number = parse_float(value)

    if number is None:
        raise ValueError(f"Entier invalide : {value!r}")

    return int(round(number))


def format_float(
    value: float | None,
    digits: int = 6,
) -> str:
    if value is None:
        return ""

    text = f"{value:.{digits}f}"

    return text.rstrip("0").rstrip(".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def current_commit() -> str:
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
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    gt_path = args.gt.resolve()
    candidates_path = args.candidates.resolve()
    temporal_path = args.temporal_predictions.resolve()
    output_dir = args.output_dir.resolve()

    manifest_rows = [
        row
        for row in read_csv(manifest_path)
        if str(
            row.get("review_status") or ""
        ).strip().lower() == "accepted"
    ]

    manifest_by_clip = {
        str(row["clip_id"]).strip(): row
        for row in manifest_rows
    }

    if set(manifest_by_clip) != set(EXPECTED_CLIPS):
        raise RuntimeError(
            "Les clips accept?s ne correspondent pas "
            "au benchmark I12 gel?."
        )

    gt_rows = read_csv(gt_path)

    if len(gt_rows) != 450:
        raise RuntimeError(
            f"450 lignes GT attendues, obtenu : {len(gt_rows)}."
        )

    gt_by_frame: dict[
        tuple[str, int],
        dict[str, Any],
    ] = {}

    visible_frames = 0

    for row in gt_rows:
        clip_id = str(row["clip_id"]).strip()
        local_frame = parse_int(row["local_frame"])
        gt_x = parse_float(row.get("x"))
        gt_y = parse_float(row.get("y"))
        visible = gt_x is not None and gt_y is not None

        key = (clip_id, local_frame)

        if key in gt_by_frame:
            raise RuntimeError(f"GT dupliqu?e : {key}")

        gt_by_frame[key] = {
            "source_key": str(row["source_key"]).strip(),
            "source_frame": parse_int(row["source_frame"]),
            "visible": visible,
            "x": gt_x,
            "y": gt_y,
        }

        visible_frames += int(visible)

    if visible_frames != 416:
        raise RuntimeError(
            f"416 frames visibles attendues, obtenu : "
            f"{visible_frames}."
        )

    for clip_id in EXPECTED_CLIPS:
        clip_frames = sorted(
            local_frame
            for current_clip, local_frame in gt_by_frame
            if current_clip == clip_id
        )

        if clip_frames != list(range(150)):
            raise RuntimeError(
                f"{clip_id} ne couvre pas local_frame 0..149."
            )

    raw_candidates = read_csv(candidates_path)

    if len(raw_candidates) != 10299:
        raise RuntimeError(
            f"10299 candidats attendus, obtenu : "
            f"{len(raw_candidates)}."
        )

    candidates_by_frame: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    candidate_by_key: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for row in raw_candidates:
        clip_id = str(row["clip_id"]).strip()
        local_frame = parse_int(row["local_frame"])
        candidate_id = str(row["candidate_id"]).strip()
        rank = parse_int(row["rank"])

        if clip_id not in manifest_by_clip:
            raise RuntimeError(
                f"Clip candidat inattendu : {clip_id}"
            )

        if rank < 1 or rank > 24:
            raise RuntimeError(
                f"Rang candidat hors cap 24 : "
                f"{candidate_id}/{rank}"
            )

        x = parse_float(row.get("x"))
        y = parse_float(row.get("y"))

        if x is None or y is None:
            raise RuntimeError(
                f"Coordonn?es absentes : {candidate_id}"
            )

        frame_key = (clip_id, local_frame)
        gt = gt_by_frame.get(frame_key)

        if gt is None:
            raise RuntimeError(
                f"Frame GT absente : {frame_key}"
            )

        source_frame = parse_int(row["source_frame"])

        if source_frame != int(gt["source_frame"]):
            raise RuntimeError(
                f"source_frame incoh?rente : {candidate_id}"
            )

        candidate_key = (clip_id, candidate_id)

        if candidate_key in candidate_by_key:
            raise RuntimeError(
                f"Candidat dupliqu? : {candidate_key}"
            )

        distance = None

        if gt["visible"]:
            distance = math.hypot(
                x - float(gt["x"]),
                y - float(gt["y"]),
            )

            stored_distance = parse_float(
                row.get("distance_to_gt_px")
            )

            if (
                stored_distance is None
                or abs(stored_distance - distance) > 0.001
            ):
                raise RuntimeError(
                    f"Distance GT incoh?rente : {candidate_id}"
                )
        elif parse_float(row.get("distance_to_gt_px")) is not None:
            raise RuntimeError(
                f"Distance pr?sente sur GT invisible : "
                f"{candidate_id}"
            )

        candidate = {
            "raw": row,
            "clip_id": clip_id,
            "local_frame": local_frame,
            "candidate_id": candidate_id,
            "rank": rank,
            "x": x,
            "y": y,
            "distance": distance,
            "label": None,
            "label_reason": None,
        }

        candidates_by_frame[frame_key].append(candidate)
        candidate_by_key[candidate_key] = candidate

    positive_frame_count = 0
    visible_without_positive = 0
    ambiguous_frames = 0
    ignored_near_candidates = 0

    for frame_key, gt in gt_by_frame.items():
        frame_candidates = candidates_by_frame.get(
            frame_key,
            [],
        )

        frame_candidates.sort(
            key=lambda item: (
                int(item["rank"]),
                str(item["candidate_id"]),
            )
        )

        if not gt["visible"]:
            for candidate in frame_candidates:
                candidate["label"] = "not_ball"
                candidate["label_reason"] = "gt_not_visible"

            continue

        near_candidates = [
            candidate
            for candidate in frame_candidates
            if candidate["distance"] is not None
            and float(candidate["distance"])
            <= LABEL_RADIUS_PX
        ]

        near_candidates.sort(
            key=lambda item: (
                float(item["distance"]),
                int(item["rank"]),
                str(item["candidate_id"]),
            )
        )

        if not near_candidates:
            visible_without_positive += 1
        else:
            positive_frame_count += 1

            ball_candidate = near_candidates[0]
            ball_candidate["label"] = "ball"
            ball_candidate["label_reason"] = (
                "nearest_within_20px"
            )

            if len(near_candidates) > 1:
                ambiguous_frames += 1

            for candidate in near_candidates[1:]:
                candidate["label"] = "ignore"
                candidate["label_reason"] = (
                    "additional_within_20px"
                )
                ignored_near_candidates += 1

        for candidate in frame_candidates:
            if candidate["label"] is None:
                candidate["label"] = "not_ball"
                candidate["label_reason"] = "outside_20px"

    if positive_frame_count != 322:
        raise RuntimeError(
            f"322 frames positives attendues, obtenu : "
            f"{positive_frame_count}."
        )

    if visible_without_positive != 94:
        raise RuntimeError(
            f"94 frames visibles sans positif attendues, "
            f"obtenu : {visible_without_positive}."
        )

    if ambiguous_frames != 31:
        raise RuntimeError(
            f"31 frames ambigu?s attendues, obtenu : "
            f"{ambiguous_frames}."
        )

    if ignored_near_candidates != 37:
        raise RuntimeError(
            f"37 candidats ignor?s attendus, obtenu : "
            f"{ignored_near_candidates}."
        )

    temporal_rows = read_csv(temporal_path)

    hard_negative_keys: set[tuple[str, str]] = set()

    for row in temporal_rows:
        if str(row.get("strategy") or "").strip() != (
            "temporal_abstention"
        ):
            continue

        status = str(row.get("status") or "").strip()

        if status not in FALSE_POSITIVE_STATUSES:
            continue

        clip_id = str(row["clip_id"]).strip()
        candidate_id = str(
            row.get("candidate_id") or ""
        ).strip()

        if not candidate_id:
            raise RuntimeError(
                "Faux positif temporel sans candidate_id."
            )

        key = (clip_id, candidate_id)

        if key in hard_negative_keys:
            raise RuntimeError(
                f"Hard negative dupliqu? : {key}"
            )

        hard_negative_keys.add(key)

    if len(hard_negative_keys) != 81:
        raise RuntimeError(
            f"81 hard negatives attendus, obtenu : "
            f"{len(hard_negative_keys)}."
        )

    unmatched_hard_negatives = (
        hard_negative_keys - set(candidate_by_key)
    )

    if unmatched_hard_negatives:
        raise RuntimeError(
            f"{len(unmatched_hard_negatives)} hard negatives "
            "ne correspondent ? aucun candidat."
        )

    label_counts: Counter[str] = Counter()
    hard_negative_label_counts: Counter[str] = Counter()
    by_clip: dict[str, Counter[str]] = {
        clip_id: Counter()
        for clip_id in EXPECTED_CLIPS
    }

    clip_order = {
        clip_id: index
        for index, clip_id in enumerate(EXPECTED_CLIPS)
    }

    ordered_candidates = sorted(
        candidate_by_key.values(),
        key=lambda item: (
            clip_order[item["clip_id"]],
            int(item["local_frame"]),
            int(item["rank"]),
            str(item["candidate_id"]),
        ),
    )

    output_rows: list[dict[str, Any]] = []

    for candidate in ordered_candidates:
        raw = candidate["raw"]
        clip_id = str(candidate["clip_id"])
        candidate_id = str(candidate["candidate_id"])
        label = str(candidate["label"])
        candidate_key = (clip_id, candidate_id)
        hard_negative = candidate_key in hard_negative_keys

        if hard_negative and label != "not_ball":
            raise RuntimeError(
                "Un hard negative I14C n'est pas ?tiquet? "
                f"not_ball : {candidate_key}/{label}"
            )

        clip = manifest_by_clip[clip_id]
        gt = gt_by_frame[
            (clip_id, int(candidate["local_frame"]))
        ]

        label_counts[label] += 1
        by_clip[clip_id][label] += 1

        if hard_negative:
            hard_negative_label_counts[label] += 1

        output_rows.append({
            "candidate_id": candidate_id,
            "clip_id": clip_id,
            "source_key": raw["source_key"],
            "local_frame": int(candidate["local_frame"]),
            "source_frame": parse_int(raw["source_frame"]),
            "split_group": clip_id,
            "rank": int(candidate["rank"]),
            "x": format_float(float(candidate["x"]), 3),
            "y": format_float(float(candidate["y"]), 3),
            "bbox_x": parse_int(raw["bbox_x"]),
            "bbox_y": parse_int(raw["bbox_y"]),
            "bbox_w": parse_int(raw["bbox_w"]),
            "bbox_h": parse_int(raw["bbox_h"]),
            "area": parse_int(raw["area"]),
            "mean_brightness": raw["mean_brightness"],
            "motion_strength": raw["motion_strength"],
            "fill_ratio": raw["fill_ratio"],
            "circularity": raw["circularity"],
            "score": raw["score"],
            "scorer_id": "heuristic_v1",
            "gt_visible": int(bool(gt["visible"])),
            "gt_x": format_float(gt["x"], 3),
            "gt_y": format_float(gt["y"], 3),
            "distance_to_gt_px": format_float(
                candidate["distance"],
                6,
            ),
            "label": label,
            "label_reason": candidate["label_reason"],
            "hard_negative": int(hard_negative),
            "hard_negative_source": (
                "i14c_temporal_abstention"
                if hard_negative
                else ""
            ),
            "source_video": clip["source_video"],
            "clip_path": clip["clip_path"],
            "source_width": parse_int(clip["source_width"]),
            "source_height": parse_int(clip["source_height"]),
            "source_fps": clip["source_fps"],
        })

    expected_counts = {
        "ball": 322,
        "ignore": 37,
        "not_ball": 9940,
    }

    if dict(label_counts) != expected_counts:
        raise RuntimeError(
            "Comptes de labels inattendus : "
            f"{dict(label_counts)}"
        )

    if hard_negative_label_counts != Counter({
        "not_ball": 81,
    }):
        raise RuntimeError(
            "Les hard negatives doivent tous ?tre not_ball : "
            f"{dict(hard_negative_label_counts)}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "i15a3_candidate_label_manifest.csv"
    )
    report_path = (
        output_dir
        / "i15a3_candidate_label_report.json"
    )

    atomic_write_csv(
        csv_path,
        output_rows,
    )

    report = {
        "experiment":
            "003D_I15A3_candidate_label_manifest",
        "schema_version": 1,
        "source_commit": current_commit(),
        "input_sha256": {
            "manifest": sha256_file(manifest_path),
            "ground_truth": sha256_file(gt_path),
            "candidates": sha256_file(candidates_path),
            "temporal_predictions":
                sha256_file(temporal_path),
        },
        "policy": {
            "label_radius_px": LABEL_RADIUS_PX,
            "ball":
                "nearest candidate within radius",
            "ignore":
                "additional candidates within radius",
            "not_ball":
                "outside radius or GT invisible",
            "hard_negative":
                "temporal_abstention false positive",
            "random_frame_split_allowed": False,
            "split_group": "clip_id",
        },
        "ground_truth": {
            "frames": len(gt_rows),
            "visible_frames": visible_frames,
            "invisible_frames":
                len(gt_rows) - visible_frames,
        },
        "candidate_manifest": {
            "rows": len(output_rows),
            "labels": dict(label_counts),
            "positive_frames": positive_frame_count,
            "visible_frames_without_positive":
                visible_without_positive,
            "ambiguous_frames": ambiguous_frames,
            "ignored_near_candidates":
                ignored_near_candidates,
            "hard_negatives":
                len(hard_negative_keys),
            "hard_negative_labels":
                dict(hard_negative_label_counts),
            "counts_by_clip": {
                clip_id: dict(by_clip[clip_id])
                for clip_id in EXPECTED_CLIPS
            },
        },
        "artifacts": {
            "manifest_csv": csv_path.name,
            "manifest_sha256": sha256_file(csv_path),
            "report_json": report_path.name,
        },
    }

    atomic_write_json(
        report_path,
        report,
    )

    if not args.quiet:
        print()
        print("I15A3_LABEL_MANIFEST_OK")

        print()
        print("=== LABELS ===")
        print("total =", len(output_rows))
        print("ball =", label_counts["ball"])
        print("ignore =", label_counts["ignore"])
        print("not_ball =", label_counts["not_ball"])
        print(
            "hard_negative =",
            len(hard_negative_keys),
        )

        print()
        print("=== COUVERTURE ===")
        print(
            "positive_frames =",
            positive_frame_count,
        )
        print(
            "visible_frames_without_positive =",
            visible_without_positive,
        )
        print(
            "ambiguous_frames =",
            ambiguous_frames,
        )
        print(
            "ignored_near_candidates =",
            ignored_near_candidates,
        )

        print()
        print("=== PAR CLIP ===")

        for clip_id in EXPECTED_CLIPS:
            print(
                clip_id,
                dict(by_clip[clip_id]),
            )

        print()
        print("manifest =", csv_path)
        print("report =", report_path)
        print(
            "manifest_sha256 =",
            report["artifacts"]["manifest_sha256"],
        )


if __name__ == "__main__":
    main()
