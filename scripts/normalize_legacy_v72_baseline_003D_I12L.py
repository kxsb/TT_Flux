from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


CLIP_PATTERN = re.compile(
    r"(?:^|\|)clip_id=([^|]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--legacy-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def parse_float(
    value: Any,
) -> float | None:
    text = str(value or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        parsed = float(text)
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def parse_int(
    value: Any,
) -> int | None:
    parsed = parse_float(value)

    if parsed is None:
        return None

    return int(round(parsed))


def extract_clip_id(
    sequence_key: Any,
) -> str:
    text = str(sequence_key or "").strip()

    match = CLIP_PATTERN.search(text)

    if match is not None:
        return match.group(1)

    return text


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            "Aucune ligne à écrire."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    legacy_path = args.legacy_csv.resolve()

    manifest = [
        row
        for row in read_csv(manifest_path)
        if str(
            row.get("review_status") or ""
        ).strip().lower() == "accepted"
    ]

    if len(manifest) != 3:
        raise RuntimeError(
            f"Trois clips acceptés attendus, "
            f"{len(manifest)} trouvés."
        )

    target_clip_ids = {
        Path(row["source_video"]).stem
        for row in manifest
    }

    legacy_index: dict[
        tuple[str, int],
        dict[str, str],
    ] = {}

    duplicate_rows = 0
    matched_legacy_rows = 0

    for row in read_csv(legacy_path):
        source_clip_id = extract_clip_id(
            row.get("sequence_key")
        )

        if source_clip_id not in target_clip_ids:
            continue

        frame = parse_int(
            row.get("frame")
        )

        if frame is None:
            continue

        matched_legacy_rows += 1

        key = (
            source_clip_id,
            frame,
        )

        if key in legacy_index:
            duplicate_rows += 1
            continue

        legacy_index[key] = row

    output_rows: list[dict[str, Any]] = []

    counts = {
        "frames":
            0,
        "legacy_row_present":
            0,
        "canonical_strong":
            0,
        "secondary_weak":
            0,
        "ghost_rejected":
            0,
        "invisible":
            0,
    }

    per_clip: dict[str, dict[str, int]] = {}

    for item in manifest:
        clip_id = str(item["clip_id"])
        source_key = str(item["source_key"])
        source_clip_id = Path(
            item["source_video"]
        ).stem

        start_frame = int(
            float(item["start_frame"])
        )

        end_frame = int(
            float(item["end_frame_exclusive"])
        )

        clip_counts = {
            "frames":
                0,
            "legacy_row_present":
                0,
            "canonical_strong":
                0,
            "secondary_weak":
                0,
            "ghost_rejected":
                0,
            "invisible":
                0,
        }

        for source_frame in range(
            start_frame,
            end_frame,
        ):
            local_frame = (
                source_frame - start_frame
            )

            legacy = legacy_index.get(
                (
                    source_clip_id,
                    source_frame,
                )
            )

            row_present = (
                legacy is not None
            )

            state = (
                str(
                    legacy.get(
                        "display_state"
                    )
                    or ""
                ).strip().lower()
                if legacy is not None
                else "missing"
            )

            reason = (
                str(
                    legacy.get(
                        "display_reason"
                    )
                    or ""
                ).strip()
                if legacy is not None
                else ""
            )

            display_x = (
                parse_float(
                    legacy.get("display_x")
                )
                if legacy is not None
                else None
            )

            display_y = (
                parse_float(
                    legacy.get("display_y")
                )
                if legacy is not None
                else None
            )

            display_score = (
                parse_float(
                    legacy.get(
                        "display_score"
                    )
                )
                if legacy is not None
                else None
            )

            has_xy = (
                display_x is not None
                and display_y is not None
            )

            canonical_valid = (
                state == "strong"
                and has_xy
            )

            secondary_valid = (
                state == "weak"
                and has_xy
            )

            ghost_rejected = (
                state == "ghost"
                and has_xy
            )

            output_rows.append({
                "clip_id":
                    clip_id,
                "source_key":
                    source_key,
                "source_clip_id":
                    source_clip_id,
                "local_frame":
                    local_frame,
                "source_frame":
                    source_frame,
                "legacy_row_present":
                    int(row_present),
                "legacy_display_state":
                    state,
                "legacy_display_reason":
                    reason,

                "canonical_valid":
                    int(canonical_valid),
                "canonical_x":
                    display_x
                    if canonical_valid
                    else "",
                "canonical_y":
                    display_y
                    if canonical_valid
                    else "",
                "canonical_score":
                    display_score
                    if canonical_valid
                    else "",
                "canonical_source":
                    (
                        "legacy_v71_motion_strong"
                        if canonical_valid
                        else ""
                    ),

                "secondary_valid":
                    int(secondary_valid),
                "secondary_x":
                    display_x
                    if secondary_valid
                    else "",
                "secondary_y":
                    display_y
                    if secondary_valid
                    else "",
                "secondary_score":
                    display_score
                    if secondary_valid
                    else "",
                "secondary_source":
                    (
                        "legacy_v70_clean_weak"
                        if secondary_valid
                        else ""
                    ),

                "interpolation_rejected":
                    int(ghost_rejected),
            })

            counts["frames"] += 1
            clip_counts["frames"] += 1

            if row_present:
                counts[
                    "legacy_row_present"
                ] += 1

                clip_counts[
                    "legacy_row_present"
                ] += 1

            if canonical_valid:
                counts[
                    "canonical_strong"
                ] += 1

                clip_counts[
                    "canonical_strong"
                ] += 1

            elif secondary_valid:
                counts[
                    "secondary_weak"
                ] += 1

                clip_counts[
                    "secondary_weak"
                ] += 1

            elif ghost_rejected:
                counts[
                    "ghost_rejected"
                ] += 1

                clip_counts[
                    "ghost_rejected"
                ] += 1

            else:
                counts["invisible"] += 1
                clip_counts["invisible"] += 1

        per_clip[clip_id] = clip_counts

    if len(output_rows) != 450:
        raise RuntimeError(
            f"450 lignes attendues, "
            f"{len(output_rows)} produites."
        )

    write_csv(
        args.output_csv.resolve(),
        output_rows,
    )

    payload = {
        "experiment":
            "003D_I12L_normalized_legacy_v72_baseline",
        "policy": {
            "canonical":
                "strong only",
            "secondary":
                "weak candidate only",
            "rejected":
                "ghost interpolation",
        },
        "manifest":
            str(manifest_path),
        "legacy_source":
            str(legacy_path),
        "manifest_sha256":
            sha256_file(manifest_path),
        "legacy_source_sha256":
            sha256_file(legacy_path),
        "output_csv":
            str(args.output_csv.resolve()),
        "output_csv_sha256":
            sha256_file(
                args.output_csv.resolve()
            ),
        "matched_legacy_rows":
            matched_legacy_rows,
        "indexed_legacy_rows":
            len(legacy_index),
        "duplicate_rows":
            duplicate_rows,
        "counts":
            counts,
        "per_clip":
            per_clip,
    }

    args.output_json.resolve().write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "I12L_LEGACY_BASELINE_NORMALIZED_OK"
    )

    print(
        "frames =",
        counts["frames"],
    )

    print(
        "legacy_rows_present =",
        counts["legacy_row_present"],
    )

    print(
        "canonical_strong =",
        counts["canonical_strong"],
    )

    print(
        "secondary_weak =",
        counts["secondary_weak"],
    )

    print(
        "ghost_rejected =",
        counts["ghost_rejected"],
    )

    print(
        "invisible_or_missing =",
        counts["invisible"],
    )

    print(
        "duplicates =",
        duplicate_rows,
    )

    print()
    print("=== I12L PAR CLIP ===")

    for clip_id, clip_counts in per_clip.items():
        print(
            clip_id,
            f"frames={clip_counts['frames']}",
            f"present={clip_counts['legacy_row_present']}",
            f"strong={clip_counts['canonical_strong']}",
            f"weak={clip_counts['secondary_weak']}",
            f"ghost_rejected={clip_counts['ghost_rejected']}",
            f"invisible={clip_counts['invisible']}",
        )

    print()
    print(
        "csv =",
        args.output_csv.resolve(),
    )

    print(
        "json =",
        args.output_json.resolve(),
    )


if __name__ == "__main__":
    main()
