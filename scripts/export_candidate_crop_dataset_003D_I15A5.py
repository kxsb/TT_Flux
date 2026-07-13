from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ttflux.tracking.candidates.crop_dataset import (  # noqa: E402
    CONTEXT_SIZE_PX,
    LABEL_TO_ID,
    LOCAL_SIZE_PX,
    SHARD_KEYS,
    TEMPORAL_OFFSETS,
    fixed_unicode_array,
    label_id,
    stack_temporal_rgb,
    validate_shard_arrays,
    write_npz_compressed,
)


EXPECTED_MANIFEST_SHA256 = (
    "320a81f21c30e5ec363c7c560746cb46776e2355e26f8cc8879fd59e2b529b1d"
)

EXPECTED_ROWS = 10299
EXPECTED_LABELS = {
    "ball": 322,
    "ignore": 37,
    "not_ball": 9940,
}
EXPECTED_HARD_NEGATIVES = 81

EXPECTED_CLIPS = (
    "i12a_best_v61_2",
    "i12a_wide_v61_4",
    "i12a_red_v61_7",
)

INDEX_FIELDS = (
    "manifest_index",
    "shard_name",
    "shard_row",
    "candidate_id",
    "clip_id",
    "split_group",
    "local_frame",
    "source_frame",
    "rank",
    "label",
    "label_id",
    "hard_negative",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
            "runs/_ball_candidate_crops_003D_I15A5"
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=512,
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


def parse_float(value: Any) -> float:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        raise ValueError(
            f"Missing numeric value: {value!r}"
        )

    number = float(text)

    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return number


def parse_int(value: Any) -> int:
    return int(round(parse_float(value)))


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
            fieldnames=INDEX_FIELDS,
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


def normalize_rows(
    raw_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    clip_order = {
        clip_id: index
        for index, clip_id in enumerate(
            EXPECTED_CLIPS
        )
    }

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        clip_id = str(raw["clip_id"]).strip()
        label = str(raw["label"]).strip()

        if clip_id not in clip_order:
            raise RuntimeError(
                f"Unexpected clip_id: {clip_id}"
            )

        if label not in LABEL_TO_ID:
            raise RuntimeError(
                f"Unexpected label: {label}"
            )

        hard_negative = parse_int(
            raw["hard_negative"]
        )

        if hard_negative not in (0, 1):
            raise RuntimeError(
                "hard_negative must be 0 or 1."
            )

        if (
            hard_negative == 1
            and label != "not_ball"
        ):
            raise RuntimeError(
                "Hard negatives must be not_ball."
            )

        rows.append({
            "candidate_id":
                str(raw["candidate_id"]).strip(),
            "clip_id": clip_id,
            "source_video":
                str(raw["source_video"]).strip(),
            "local_frame":
                parse_int(raw["local_frame"]),
            "source_frame":
                parse_int(raw["source_frame"]),
            "rank": parse_int(raw["rank"]),
            "x": parse_float(raw["x"]),
            "y": parse_float(raw["y"]),
            "label": label,
            "label_id": label_id(label),
            "hard_negative": hard_negative,
            "source_width":
                parse_int(raw["source_width"]),
            "source_height":
                parse_int(raw["source_height"]),
            "source_fps":
                parse_float(raw["source_fps"]),
        })

    rows.sort(
        key=lambda row: (
            clip_order[row["clip_id"]],
            row["local_frame"],
            row["rank"],
            row["candidate_id"],
        )
    )

    for manifest_index, row in enumerate(rows):
        row["manifest_index"] = manifest_index

    return rows


def validate_rows(
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"got {len(rows)}."
        )

    candidate_ids = [
        row["candidate_id"]
        for row in rows
    ]

    if len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise RuntimeError(
            "Duplicate candidate_id values."
        )

    label_counts = Counter(
        row["label"]
        for row in rows
    )

    if dict(label_counts) != EXPECTED_LABELS:
        raise RuntimeError(
            f"Unexpected labels: {dict(label_counts)}"
        )

    hard_negative_count = sum(
        row["hard_negative"]
        for row in rows
    )

    if (
        hard_negative_count
        != EXPECTED_HARD_NEGATIVES
    ):
        raise RuntimeError(
            "Unexpected hard negative count: "
            f"{hard_negative_count}"
        )

    clips = {
        row["clip_id"]
        for row in rows
    }

    if clips != set(EXPECTED_CLIPS):
        raise RuntimeError(
            f"Unexpected clip set: {clips}"
        )

    expected_indices = list(
        range(len(rows))
    )

    actual_indices = [
        row["manifest_index"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        raise RuntimeError(
            "Manifest indices are not contiguous."
        )


def inspect_video(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open source video: {path}"
        )

    information = {
        "frame_count": int(round(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )),
        "width": int(round(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )),
        "height": int(round(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )),
        "fps": float(
            capture.get(cv2.CAP_PROP_FPS)
        ),
    }

    capture.release()

    return information


def iter_clip_samples(
    clip_rows: list[dict[str, Any]],
) -> Iterator[
    tuple[
        dict[str, Any],
        np.ndarray,
        np.ndarray,
    ]
]:
    if not clip_rows:
        return

    source_videos = {
        row["source_video"]
        for row in clip_rows
    }

    if len(source_videos) != 1:
        raise RuntimeError(
            "A clip must map to one source video."
        )

    video_path = Path(
        next(iter(source_videos))
    )

    information = inspect_video(video_path)
    sample = clip_rows[0]

    if (
        information["width"]
        != sample["source_width"]
        or information["height"]
        != sample["source_height"]
    ):
        raise RuntimeError(
            f"Video geometry mismatch: {video_path}"
        )

    if abs(
        information["fps"]
        - sample["source_fps"]
    ) > 0.01:
        raise RuntimeError(
            f"Video fps mismatch: {video_path}"
        )

    rows_by_frame: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in clip_rows:
        rows_by_frame[
            row["source_frame"]
        ].append(row)

    for frame_rows in rows_by_frame.values():
        frame_rows.sort(
            key=lambda row: (
                row["rank"],
                row["candidate_id"],
            )
        )

    target_frames = sorted(rows_by_frame)
    minimum_target = target_frames[0]
    maximum_target = target_frames[-1]

    if minimum_target - 1 < 0:
        raise RuntimeError(
            "Missing source frame t-1."
        )

    if (
        maximum_target + 1
        >= information["frame_count"]
    ):
        raise RuntimeError(
            "Missing source frame t+1."
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot decode source video: {video_path}"
        )

    frame_buffer: dict[int, np.ndarray] = {}
    yielded = 0

    try:
        for frame_index in range(
            maximum_target + 2
        ):
            ok, frame = capture.read()

            if not ok:
                raise RuntimeError(
                    f"Decode stopped at frame "
                    f"{frame_index}: {video_path}"
                )

            frame_buffer[frame_index] = frame

            stale_frame = frame_index - 3
            frame_buffer.pop(
                stale_frame,
                None,
            )

            target_frame = frame_index - 1
            frame_rows = rows_by_frame.get(
                target_frame
            )

            if not frame_rows:
                continue

            temporal_frames = (
                frame_buffer[target_frame - 1],
                frame_buffer[target_frame],
                frame_buffer[target_frame + 1],
            )

            for row in frame_rows:
                local_rgb = stack_temporal_rgb(
                    temporal_frames,
                    row["x"],
                    row["y"],
                    LOCAL_SIZE_PX,
                )
                context_rgb = stack_temporal_rgb(
                    temporal_frames,
                    row["x"],
                    row["y"],
                    CONTEXT_SIZE_PX,
                )

                yielded += 1

                yield (
                    row,
                    local_rgb,
                    context_rgb,
                )
    finally:
        capture.release()

    if yielded != len(clip_rows):
        raise RuntimeError(
            f"Clip yielded {yielded} rows, "
            f"expected {len(clip_rows)}."
        )


def build_shard_arrays(
    pending: list[
        tuple[
            dict[str, Any],
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> dict[str, np.ndarray]:
    rows = [
        item[0]
        for item in pending
    ]

    arrays = {
        "local_rgb": np.stack(
            [
                item[1]
                for item in pending
            ],
            axis=0,
        ).astype(
            np.uint8,
            copy=False,
        ),
        "context_rgb": np.stack(
            [
                item[2]
                for item in pending
            ],
            axis=0,
        ).astype(
            np.uint8,
            copy=False,
        ),
        "label": np.asarray(
            [
                row["label_id"]
                for row in rows
            ],
            dtype=np.int8,
        ),
        "hard_negative": np.asarray(
            [
                row["hard_negative"]
                for row in rows
            ],
            dtype=np.uint8,
        ),
        "manifest_index": np.asarray(
            [
                row["manifest_index"]
                for row in rows
            ],
            dtype=np.int32,
        ),
        "candidate_id": fixed_unicode_array(
            [
                row["candidate_id"]
                for row in rows
            ],
            width=64,
        ),
        "clip_id": fixed_unicode_array(
            [
                row["clip_id"]
                for row in rows
            ],
            width=32,
        ),
        "local_frame": np.asarray(
            [
                row["local_frame"]
                for row in rows
            ],
            dtype=np.int32,
        ),
        "source_frame": np.asarray(
            [
                row["source_frame"]
                for row in rows
            ],
            dtype=np.int32,
        ),
        "rank": np.asarray(
            [
                row["rank"]
                for row in rows
            ],
            dtype=np.int16,
        ),
        "x": np.asarray(
            [
                row["x"]
                for row in rows
            ],
            dtype=np.float32,
        ),
        "y": np.asarray(
            [
                row["y"]
                for row in rows
            ],
            dtype=np.float32,
        ),
    }

    validate_shard_arrays(arrays)

    return arrays


def validate_written_shard(
    path: Path,
    expected_candidate_ids: list[str],
) -> None:
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        arrays = {
            key: archive[key]
            for key in SHARD_KEYS
        }

    validate_shard_arrays(arrays)

    actual_candidate_ids = (
        arrays["candidate_id"].tolist()
    )

    if (
        actual_candidate_ids
        != expected_candidate_ids
    ):
        raise RuntimeError(
            f"Candidate order mismatch in {path}."
        )


def dataset_fingerprint(
    index_sha256: str,
    shards: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        index_sha256.encode("ascii")
    )

    for shard in shards:
        digest.update(
            shard["filename"].encode("ascii")
        )
        digest.update(
            shard["sha256"].encode("ascii")
        )

    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    if args.shard_size <= 0:
        raise ValueError(
            "shard-size must be positive."
        )

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

    raw_rows = read_csv(manifest_path)
    rows = normalize_rows(raw_rows)
    validate_rows(rows)

    rows_by_clip: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        rows_by_clip[
            row["clip_id"]
        ].append(row)

    pending: list[
        tuple[
            dict[str, Any],
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    shard_reports: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    shard_index = 0
    exported_rows = 0

    def flush_pending() -> None:
        nonlocal shard_index
        nonlocal exported_rows

        if not pending:
            return

        arrays = build_shard_arrays(
            pending
        )

        shard_name = (
            f"i15a5_shard_{shard_index:03d}.npz"
        )
        shard_path = output_dir / shard_name

        write_npz_compressed(
            shard_path,
            arrays,
        )

        expected_candidate_ids = [
            item[0]["candidate_id"]
            for item in pending
        ]

        validate_written_shard(
            shard_path,
            expected_candidate_ids,
        )

        for shard_row, item in enumerate(
            pending
        ):
            row = item[0]

            index_rows.append({
                "manifest_index":
                    row["manifest_index"],
                "shard_name": shard_name,
                "shard_row": shard_row,
                "candidate_id":
                    row["candidate_id"],
                "clip_id": row["clip_id"],
                "split_group":
                    row["clip_id"],
                "local_frame":
                    row["local_frame"],
                "source_frame":
                    row["source_frame"],
                "rank": row["rank"],
                "label": row["label"],
                "label_id": row["label_id"],
                "hard_negative":
                    row["hard_negative"],
            })

        shard_reports.append({
            "filename": shard_name,
            "rows": len(pending),
            "first_manifest_index":
                pending[0][0][
                    "manifest_index"
                ],
            "last_manifest_index":
                pending[-1][0][
                    "manifest_index"
                ],
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(
                shard_path
            ),
        })

        exported_rows += len(pending)
        shard_index += 1
        pending.clear()

    for clip_id in EXPECTED_CLIPS:
        clip_rows = rows_by_clip[clip_id]

        for sample in iter_clip_samples(
            clip_rows
        ):
            pending.append(sample)

            if len(pending) >= args.shard_size:
                flush_pending()

    flush_pending()

    if exported_rows != len(rows):
        raise RuntimeError(
            f"Exported {exported_rows} rows, "
            f"expected {len(rows)}."
        )

    if len(index_rows) != len(rows):
        raise RuntimeError(
            "Dataset index row count mismatch."
        )

    if [
        row["manifest_index"]
        for row in index_rows
    ] != list(range(len(rows))):
        raise RuntimeError(
            "Dataset index order mismatch."
        )

    index_path = (
        output_dir
        / "i15a5_dataset_index.csv"
    )
    report_path = (
        output_dir
        / "i15a5_dataset_report.json"
    )

    atomic_write_csv(
        index_path,
        index_rows,
    )

    index_sha256 = sha256_file(
        index_path
    )

    total_shard_bytes = sum(
        shard["bytes"]
        for shard in shard_reports
    )

    clip_counts = {
        clip_id: len(
            rows_by_clip[clip_id]
        )
        for clip_id in EXPECTED_CLIPS
    }

    report = {
        "experiment":
            "003D_I15A5_candidate_crop_dataset",
        "schema_version": 1,
        "repository_head_at_export":
            git_head(),
        "generator": {
            "script_sha256":
                sha256_file(
                    Path(__file__).resolve()
                ),
            "module_sha256":
                sha256_file(
                    SOURCE_ROOT
                    / "ttflux"
                    / "analysis"
                    / "crop_dataset.py"
                ),
            "numpy_version":
                np.__version__,
            "opencv_version":
                cv2.__version__,
        },
        "input": {
            "candidate_manifest":
                args.candidate_manifest.as_posix(),
            "candidate_manifest_sha256":
                manifest_sha256,
        },
        "geometry": {
            "color_order": "RGB",
            "dtype": "uint8",
            "local_size_px":
                LOCAL_SIZE_PX,
            "context_size_px":
                CONTEXT_SIZE_PX,
            "temporal_offsets":
                list(TEMPORAL_OFFSETS),
            "centering_rule":
                "floor(candidate_coordinate + 0.5)",
            "padding_rule":
                "constant_zero_outside_source",
        },
        "tensor_schema": {
            "local_rgb":
                "[N, 3, 48, 48, 3] uint8",
            "context_rgb":
                "[N, 3, 96, 96, 3] uint8",
            "label":
                "[N] int8; ball=1, "
                "not_ball=0, ignore=-1",
            "hard_negative":
                "[N] uint8",
            "model_inputs": [
                "local_rgb",
                "context_rgb",
            ],
            "provenance_only": [
                "manifest_index",
                "candidate_id",
                "clip_id",
                "local_frame",
                "source_frame",
                "rank",
                "x",
                "y",
            ],
            "allow_pickle": False,
        },
        "dataset": {
            "rows": len(rows),
            "trainable_rows": sum(
                row["label"] != "ignore"
                for row in rows
            ),
            "labels": dict(Counter(
                row["label"]
                for row in rows
            )),
            "label_ids": LABEL_TO_ID,
            "hard_negatives": sum(
                row["hard_negative"]
                for row in rows
            ),
            "clips": clip_counts,
        },
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
        "sharding": {
            "requested_shard_size":
                args.shard_size,
            "shard_count":
                len(shard_reports),
            "total_shard_bytes":
                total_shard_bytes,
            "total_shard_mib":
                round(
                    total_shard_bytes
                    / (1024 * 1024),
                    3,
                ),
            "shards": shard_reports,
        },
        "artifacts": {
            "index_csv":
                index_path.name,
            "index_sha256":
                index_sha256,
            "report_json":
                report_path.name,
            "dataset_fingerprint":
                dataset_fingerprint(
                    index_sha256,
                    shard_reports,
                ),
        },
    }

    atomic_write_json(
        report_path,
        report,
    )

    if not args.quiet:
        print()
        print("I15A5_CROP_EXPORT_OK")

        print()
        print("=== DATASET ===")
        print("rows =", len(rows))
        print(
            "trainable_rows =",
            report["dataset"][
                "trainable_rows"
            ],
        )
        print(
            "labels =",
            report["dataset"]["labels"],
        )
        print(
            "hard_negatives =",
            report["dataset"][
                "hard_negatives"
            ],
        )

        print()
        print("=== TENSORS ===")
        print(
            "local_rgb =",
            "[N, 3, 48, 48, 3] uint8 RGB",
        )
        print(
            "context_rgb =",
            "[N, 3, 96, 96, 3] uint8 RGB",
        )
        print(
            "temporal_offsets =",
            list(TEMPORAL_OFFSETS),
        )

        print()
        print("=== SHARDS ===")
        print(
            "shard_size =",
            args.shard_size,
        )
        print(
            "shard_count =",
            len(shard_reports),
        )
        print(
            "total_shard_mib =",
            report["sharding"][
                "total_shard_mib"
            ],
        )
        print(
            "last_shard_rows =",
            shard_reports[-1]["rows"],
        )

        print()
        print("=== FINGERPRINT ===")
        print(
            "index_sha256 =",
            index_sha256,
        )
        print(
            "dataset_fingerprint =",
            report["artifacts"][
                "dataset_fingerprint"
            ],
        )
        print("output_dir =", output_dir)


if __name__ == "__main__":
    main()
