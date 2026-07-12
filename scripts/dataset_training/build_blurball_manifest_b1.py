from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path.cwd()

DATASET_ROOT = (
    ROOT
    / "data"
    / "blurball_dataset"
)

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "dataset_manifest"
    / "b1_blurball"
)

FFPROBE = os.environ["TTFLUX_FFPROBE"]

EXPECTED_RALLIES = 463
EXPECTED_ROWS = 66595
EXPECTED_VISIBLE = 58547
EXPECTED_INVISIBLE = 8048


def root_only_files(
    pattern: str,
) -> list[Path]:
    nested_copy = (
        DATASET_ROOT
        / DATASET_ROOT.name
    )

    return sorted(
        path
        for path in DATASET_ROOT.rglob(pattern)
        if (
            path.is_file()
            and nested_copy not in path.parents
        )
    )


def parse_float(
    value: Any,
) -> float | None:
    text = str(value).strip()

    if text == "":
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def parse_int(
    value: Any,
) -> int | None:
    number = parse_float(value)

    if number is None:
        return None

    rounded = round(number)

    if abs(number - rounded) > 1e-6:
        return None

    return int(rounded)


def parse_rate(
    value: Any,
) -> float | None:
    text = str(value).strip()

    if text in {"", "N/A", "0/0"}:
        return None

    if "/" not in text:
        return parse_float(text)

    numerator_text, denominator_text = text.split(
        "/",
        1,
    )

    numerator = parse_float(numerator_text)
    denominator = parse_float(denominator_text)

    if (
        numerator is None
        or denominator is None
        or denominator == 0
    ):
        return None

    return numerator / denominator


def probe_video(
    path: Path,
) -> dict[str, Any]:
    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,codec_name,"
            "avg_frame_rate,r_frame_rate,"
            "nb_frames,duration:"
            "format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    payload = json.loads(
        completed.stdout
    )

    streams = payload.get(
        "streams",
        [],
    )

    if not streams:
        raise RuntimeError(
            f"Aucun flux vidéo : {path}"
        )

    stream = streams[0]
    format_info = payload.get(
        "format",
        {},
    )

    fps_average = parse_rate(
        stream.get("avg_frame_rate")
    )

    fps_declared = parse_rate(
        stream.get("r_frame_rate")
    )

    duration = parse_float(
        stream.get("duration")
    )

    if duration is None:
        duration = parse_float(
            format_info.get("duration")
        )

    frame_count = parse_int(
        stream.get("nb_frames")
    )

    frame_count_source = (
        "nb_frames"
        if frame_count is not None
        else None
    )

    if (
        frame_count is None
        and fps_average is not None
        and duration is not None
    ):
        frame_count = round(
            fps_average * duration
        )
        frame_count_source = "estimated"

    return {
        "width": parse_int(
            stream.get("width")
        ),
        "height": parse_int(
            stream.get("height")
        ),
        "codec": stream.get(
            "codec_name"
        ),
        "fps_average": fps_average,
        "fps_declared": fps_declared,
        "duration_s": duration,
        "frame_count": frame_count,
        "frame_count_source": (
            frame_count_source
        ),
    }


def nominal_fps(
    fps: float | None,
) -> str:
    if fps is None:
        return "unknown"

    candidates = (
        24,
        25,
        30,
        50,
        60,
        120,
    )

    nearest = min(
        candidates,
        key=lambda candidate: abs(
            fps - candidate
        ),
    )

    if abs(fps - nearest) <= 2.0:
        return str(nearest)

    return "other"


def video_key(
    path: Path,
) -> tuple[int, int] | None:
    relative = path.relative_to(
        DATASET_ROOT
    )

    parts = relative.parts

    if len(parts) != 3:
        return None

    match_directory = parts[0]
    rally_directory = parts[1]

    if rally_directory not in {
        "rallies_videos",
        "videos_rallies",
    }:
        return None

    if (
        not match_directory.isdigit()
        or not path.stem.isdigit()
    ):
        return None

    return (
        int(match_directory),
        int(path.stem),
    )


def annotation_key(
    member_name: str,
    convention: str,
) -> tuple[int, int] | None:
    filename = Path(
        member_name
    ).name

    if convention == "endpoint":
        pattern = (
            r"(\d+)_endpoint_csv_(\d+)\.csv"
        )
    else:
        pattern = (
            r"(\d+)_csv_(\d+)\.csv"
        )

    match = re.fullmatch(
        pattern,
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def read_annotation_archive(
    archive_path: Path,
    convention: str,
) -> dict[
    tuple[int, int],
    dict[str, Any],
]:
    records = {}

    with zipfile.ZipFile(
        archive_path
    ) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            key = annotation_key(
                member.filename,
                convention,
            )

            if key is None:
                continue

            text = archive.read(
                member
            ).decode(
                "utf-8-sig"
            )

            rows = {}

            reader = csv.DictReader(
                io.StringIO(text)
            )

            for source_row in reader:
                frame_index = parse_int(
                    source_row.get("Frame")
                )

                visibility = parse_int(
                    source_row.get("Visibility")
                )

                if (
                    frame_index is None
                    or visibility not in {0, 1}
                ):
                    raise RuntimeError(
                        "Annotation invalide : "
                        f"{member.filename}"
                    )

                if frame_index in rows:
                    raise RuntimeError(
                        "Frame dupliquée : "
                        f"{member.filename} "
                        f"frame={frame_index}"
                    )

                rows[frame_index] = {
                    "visibility": visibility,
                    "x": parse_float(
                        source_row.get("X")
                    ),
                    "y": parse_float(
                        source_row.get("Y")
                    ),
                    "theta": parse_float(
                        source_row.get("theta")
                    ),
                    "l": parse_float(
                        source_row.get("l")
                    ),
                }

            records[key] = {
                "member": member.filename,
                "rows": rows,
            }

    return records


def source_split(
    match_id: int,
) -> str:
    if match_id <= 21:
        return "train"

    return "test"


def ttflux_split(
    match_id: int,
) -> str:
    if match_id <= 19:
        return "train"

    if match_id <= 21:
        return "validation"

    return "test"


def inside_image(
    x: float | None,
    y: float | None,
    width: int,
    height: int,
) -> bool | None:
    if x is None or y is None:
        return None

    return (
        0 <= x < width
        and 0 <= y < height
    )


def values_equal(
    first: float | None,
    second: float | None,
) -> bool:
    if first is None or second is None:
        return first is second

    return abs(first - second) <= 1e-9


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_paths = root_only_files(
        "*.mp4"
    )

    video_map: dict[
        tuple[int, int],
        Path,
    ] = {}

    extra_videos = []
    folder_counts: Counter[str] = Counter()

    for path in video_paths:
        key = video_key(path)

        if key is None:
            extra_videos.append(
                path.relative_to(
                    DATASET_ROOT
                ).as_posix()
            )
            continue

        if key in video_map:
            raise RuntimeError(
                f"Vidéo dupliquée pour {key}"
            )

        video_map[key] = path
        folder_counts[
            path.parent.name
        ] += 1

    archive_paths = root_only_files(
        "*.zip"
    )

    endpoint_archive = next(
        path
        for path in archive_paths
        if "endpoint" in path.name.lower()
    )

    midpoint_archive = next(
        path
        for path in archive_paths
        if "midpoint" in path.name.lower()
    )

    endpoint_map = read_annotation_archive(
        endpoint_archive,
        "endpoint",
    )

    midpoint_map = read_annotation_archive(
        midpoint_archive,
        "midpoint",
    )

    endpoint_keys = set(endpoint_map)
    midpoint_keys = set(midpoint_map)
    video_keys = set(video_map)

    if endpoint_keys != midpoint_keys:
        raise RuntimeError(
            "Les clés endpoint et midpoint diffèrent."
        )

    if endpoint_keys != video_keys:
        missing_videos = sorted(
            endpoint_keys - video_keys
        )

        extra_keys = sorted(
            video_keys - endpoint_keys
        )

        raise RuntimeError(
            "Correspondance vidéo incomplète. "
            f"missing={missing_videos}, "
            f"extra={extra_keys}"
        )

    expected_matches = set(
        range(26)
    )

    actual_matches = {
        key[0]
        for key in endpoint_keys
    }

    if actual_matches != expected_matches:
        raise RuntimeError(
            "IDs de matchs inattendus : "
            f"{sorted(actual_matches)}"
        )

    frame_rows = []
    rally_rows = []

    visibility_counts: Counter[int] = Counter()
    source_split_frames: Counter[str] = Counter()
    ttflux_split_frames: Counter[str] = Counter()
    source_split_rallies: Counter[str] = Counter()
    ttflux_split_rallies: Counter[str] = Counter()
    nominal_fps_counts: Counter[str] = Counter()
    frame_min_counts: Counter[int] = Counter()

    endpoint_out_of_bounds = 0
    midpoint_out_of_bounds = 0

    for index, key in enumerate(
        sorted(endpoint_keys),
        start=1,
    ):
        if index == 1 or index % 50 == 0:
            print(
                "MANIFEST "
                f"{index}/{len(endpoint_keys)}"
            )

        match_id, rally_id = key
        video_path = video_map[key]

        endpoint = endpoint_map[key]
        midpoint = midpoint_map[key]

        endpoint_frames = set(
            endpoint["rows"]
        )

        midpoint_frames = set(
            midpoint["rows"]
        )

        if endpoint_frames != midpoint_frames:
            raise RuntimeError(
                f"Frames différentes pour {key}"
            )

        metadata = probe_video(
            video_path
        )

        width = metadata["width"]
        height = metadata["height"]

        if width is None or height is None:
            raise RuntimeError(
                f"Résolution absente : {video_path}"
            )

        exact_source_split = source_split(
            match_id
        )

        exact_ttflux_split = ttflux_split(
            match_id
        )

        source_split_rallies[
            exact_source_split
        ] += 1

        ttflux_split_rallies[
            exact_ttflux_split
        ] += 1

        exact_nominal_fps = nominal_fps(
            metadata["fps_average"]
        )

        nominal_fps_counts[
            exact_nominal_fps
        ] += 1

        rally_visible = 0
        rally_invisible = 0
        rally_endpoint_oob = 0
        rally_midpoint_oob = 0

        sorted_frames = sorted(
            endpoint_frames
        )

        frame_min_counts[
            min(sorted_frames)
        ] += 1

        for frame_index in sorted_frames:
            endpoint_row = (
                endpoint["rows"][frame_index]
            )

            midpoint_row = (
                midpoint["rows"][frame_index]
            )

            if (
                endpoint_row["visibility"]
                != midpoint_row["visibility"]
            ):
                raise RuntimeError(
                    "Visibility différente : "
                    f"{key} frame={frame_index}"
                )

            if not values_equal(
                endpoint_row["theta"],
                midpoint_row["theta"],
            ):
                raise RuntimeError(
                    "Theta différent : "
                    f"{key} frame={frame_index}"
                )

            if not values_equal(
                endpoint_row["l"],
                midpoint_row["l"],
            ):
                raise RuntimeError(
                    "Longueur de flou différente : "
                    f"{key} frame={frame_index}"
                )

            visibility = (
                midpoint_row["visibility"]
            )

            visibility_counts[
                visibility
            ] += 1

            source_split_frames[
                exact_source_split
            ] += 1

            ttflux_split_frames[
                exact_ttflux_split
            ] += 1

            if visibility == 1:
                rally_visible += 1
            else:
                rally_invisible += 1

            endpoint_inside = inside_image(
                endpoint_row["x"],
                endpoint_row["y"],
                width,
                height,
            )

            midpoint_inside = inside_image(
                midpoint_row["x"],
                midpoint_row["y"],
                width,
                height,
            )

            if (
                visibility == 1
                and endpoint_inside is False
            ):
                endpoint_out_of_bounds += 1
                rally_endpoint_oob += 1

            if (
                visibility == 1
                and midpoint_inside is False
            ):
                midpoint_out_of_bounds += 1
                rally_midpoint_oob += 1

            endpoint_delta = None

            if (
                visibility == 1
                and endpoint_row["x"] is not None
                and endpoint_row["y"] is not None
                and midpoint_row["x"] is not None
                and midpoint_row["y"] is not None
            ):
                endpoint_delta = math.hypot(
                    (
                        midpoint_row["x"]
                        - endpoint_row["x"]
                    ),
                    (
                        midpoint_row["y"]
                        - endpoint_row["y"]
                    ),
                )

            frame_rows.append(
                {
                    "source_dataset": "blurball",
                    "match_id": f"{match_id:02d}",
                    "rally_id": f"{rally_id:03d}",
                    "frame_index": frame_index,
                    "source_split": exact_source_split,
                    "ttflux_split": exact_ttflux_split,
                    "video_path": (
                        video_path.relative_to(
                            ROOT
                        ).as_posix()
                    ),
                    "video_width": width,
                    "video_height": height,
                    "fps_average": (
                        metadata["fps_average"]
                    ),
                    "fps_declared": (
                        metadata["fps_declared"]
                    ),
                    "fps_nominal": exact_nominal_fps,
                    "visibility": visibility,
                    "label_convention": "midpoint",
                    "x_px": (
                        midpoint_row["x"]
                        if visibility == 1
                        else None
                    ),
                    "y_px": (
                        midpoint_row["y"]
                        if visibility == 1
                        else None
                    ),
                    "midpoint_x_raw": (
                        midpoint_row["x"]
                    ),
                    "midpoint_y_raw": (
                        midpoint_row["y"]
                    ),
                    "endpoint_x_raw": (
                        endpoint_row["x"]
                    ),
                    "endpoint_y_raw": (
                        endpoint_row["y"]
                    ),
                    "endpoint_delta_px": (
                        endpoint_delta
                    ),
                    "blur_theta_raw": (
                        midpoint_row["theta"]
                    ),
                    "blur_l_raw": (
                        midpoint_row["l"]
                    ),
                    "midpoint_inside_image": (
                        midpoint_inside
                    ),
                    "endpoint_inside_image": (
                        endpoint_inside
                    ),
                    "midpoint_annotation_member": (
                        midpoint["member"]
                    ),
                    "endpoint_annotation_member": (
                        endpoint["member"]
                    ),
                }
            )

        rally_rows.append(
            {
                "source_dataset": "blurball",
                "match_id": f"{match_id:02d}",
                "rally_id": f"{rally_id:03d}",
                "source_split": exact_source_split,
                "ttflux_split": exact_ttflux_split,
                "video_path": (
                    video_path.relative_to(
                        ROOT
                    ).as_posix()
                ),
                "video_directory_variant": (
                    video_path.parent.name
                ),
                "video_width": width,
                "video_height": height,
                "codec": metadata["codec"],
                "fps_average": (
                    metadata["fps_average"]
                ),
                "fps_declared": (
                    metadata["fps_declared"]
                ),
                "fps_nominal": exact_nominal_fps,
                "duration_s": (
                    metadata["duration_s"]
                ),
                "video_frame_count": (
                    metadata["frame_count"]
                ),
                "video_frame_count_source": (
                    metadata[
                        "frame_count_source"
                    ]
                ),
                "annotation_frame_min": (
                    min(sorted_frames)
                ),
                "annotation_frame_max": (
                    max(sorted_frames)
                ),
                "annotation_row_count": (
                    len(sorted_frames)
                ),
                "visible_rows": rally_visible,
                "invisible_rows": rally_invisible,
                "midpoint_out_of_bounds": (
                    rally_midpoint_oob
                ),
                "endpoint_out_of_bounds": (
                    rally_endpoint_oob
                ),
                "primary_label_convention": (
                    "midpoint"
                ),
                "midpoint_annotation_member": (
                    midpoint["member"]
                ),
                "endpoint_annotation_member": (
                    endpoint["member"]
                ),
            }
        )

    if len(rally_rows) != EXPECTED_RALLIES:
        raise RuntimeError(
            "Nombre d'échanges inattendu : "
            f"{len(rally_rows)}"
        )

    if len(frame_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            "Nombre de frames inattendu : "
            f"{len(frame_rows)}"
        )

    if visibility_counts[1] != EXPECTED_VISIBLE:
        raise RuntimeError(
            "Nombre visible inattendu : "
            f"{visibility_counts[1]}"
        )

    if visibility_counts[0] != EXPECTED_INVISIBLE:
        raise RuntimeError(
            "Nombre invisible inattendu : "
            f"{visibility_counts[0]}"
        )

    if midpoint_out_of_bounds != 0:
        raise RuntimeError(
            "Des midpoints visibles sont hors image : "
            f"{midpoint_out_of_bounds}"
        )

    if endpoint_out_of_bounds != 1:
        raise RuntimeError(
            "Nombre d'endpoints hors image inattendu : "
            f"{endpoint_out_of_bounds}"
        )

    frames_path = (
        OUTPUT_DIR
        / "blurball_frames.csv"
    )

    rallies_path = (
        OUTPUT_DIR
        / "blurball_rallies.csv"
    )

    write_csv(
        frames_path,
        frame_rows,
    )

    write_csv(
        rallies_path,
        rally_rows,
    )

    summary = {
        "schema_version": 1,
        "dataset": "blurball",
        "primary_label_convention": "midpoint",
        "rally_count": len(rally_rows),
        "frame_row_count": len(frame_rows),
        "visible_row_count": (
            visibility_counts[1]
        ),
        "invisible_row_count": (
            visibility_counts[0]
        ),
        "endpoint_out_of_bounds": (
            endpoint_out_of_bounds
        ),
        "midpoint_out_of_bounds": (
            midpoint_out_of_bounds
        ),
        "video_directory_variants": dict(
            folder_counts
        ),
        "extra_videos_excluded": (
            extra_videos
        ),
        "frame_index_min_distribution": {
            str(key): value
            for key, value in sorted(
                frame_min_counts.items()
            )
        },
        "source_split_rallies": dict(
            source_split_rallies
        ),
        "source_split_frames": dict(
            source_split_frames
        ),
        "ttflux_split_rallies": dict(
            ttflux_split_rallies
        ),
        "ttflux_split_frames": dict(
            ttflux_split_frames
        ),
        "nominal_fps_rallies": dict(
            nominal_fps_counts
        ),
        "rallies_manifest": str(
            rallies_path
        ),
        "frames_manifest": str(
            frames_path
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        "DSET_B1_BLURBALL_MANIFEST_GENERATED"
    )
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()