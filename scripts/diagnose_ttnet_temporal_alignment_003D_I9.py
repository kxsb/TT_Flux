from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "model_id",
    "source_name",
    "gt_x",
    "gt_y",
    "global_x",
    "global_y",
    "refined_x",
    "refined_y",
}

OUTPUTS = {
    "global": ("global_x", "global_y", "global_valid"),
    "refined": ("refined_x", "refined_y", "refined_valid"),
}


def find_prediction_csv(root: Path) -> tuple[Path, list[Path]]:
    matches: list[Path] = []

    for path in root.glob("runs/**/*.csv"):
        try:
            columns = set(
                pd.read_csv(
                    path,
                    nrows=0,
                    encoding="utf-8-sig",
                ).columns
            )
        except Exception:
            continue

        if REQUIRED_COLUMNS.issubset(columns) and (
            "source_frame" in columns or "local_frame" in columns
        ):
            matches.append(path.resolve())

    if not matches:
        raise FileNotFoundError(
            "Aucun CSV de prédictions TTNet compatible trouvé dans runs/."
        )

    matches.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0], matches


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def boolean_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "oui",
    }


def error_metrics(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {
            "n": 0,
            "median_error_px": None,
            "p90_error_px": None,
            "recall_5px": None,
            "recall_10px": None,
            "recall_20px": None,
            "recall_50px": None,
        }

    series = pd.Series(errors, dtype="float64")

    return {
        "n": len(errors),
        "median_error_px": float(series.median()),
        "p90_error_px": float(series.quantile(0.90)),
        "recall_5px": float((series <= 5).mean()),
        "recall_10px": float((series <= 10).mean()),
        "recall_20px": float((series <= 20).mean()),
        "recall_50px": float((series <= 50).mean()),
    }


def group_columns(frame: pd.DataFrame) -> list[str]:
    result = ["source_name"]

    if "window_id" in frame.columns:
        result.append("window_id")

    return result


def aligned_errors(
    frame: pd.DataFrame,
    frame_column: str,
    prediction_x: str,
    prediction_y: str,
    valid_column: str,
    offset: int,
    valid_only: bool,
) -> tuple[list[float], dict[str, list[float]]]:
    errors: list[float] = []
    errors_by_source: dict[str, list[float]] = {}

    for _, group in frame.groupby(
        group_columns(frame),
        dropna=False,
        sort=False,
    ):
        gt_by_frame: dict[int, tuple[float, float]] = {}

        for row in group.to_dict("records"):
            source_frame = numeric(row.get(frame_column))
            gt_x = numeric(row.get("gt_x"))
            gt_y = numeric(row.get("gt_y"))

            if (
                source_frame is None
                or gt_x is None
                or gt_y is None
                or gt_x < 0
                or gt_y < 0
            ):
                continue

            gt_by_frame.setdefault(
                int(round(source_frame)),
                (gt_x, gt_y),
            )

        source_name = str(group.iloc[0]["source_name"])
        source_errors = errors_by_source.setdefault(source_name, [])

        for row in group.to_dict("records"):
            source_frame = numeric(row.get(frame_column))
            pred_x = numeric(row.get(prediction_x))
            pred_y = numeric(row.get(prediction_y))

            if source_frame is None or pred_x is None or pred_y is None:
                continue

            if (
                valid_only
                and valid_column in group.columns
                and not boolean_value(row.get(valid_column))
            ):
                continue

            target_frame = int(round(source_frame)) + offset
            target = gt_by_frame.get(target_frame)

            if target is None:
                continue

            error = math.hypot(
                pred_x - target[0],
                pred_y - target[1],
            )

            errors.append(error)
            source_errors.append(error)

    return errors, errors_by_source


def select_best(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    baseline = next(
        (
            row for row in rows
            if row["offset_frames"] == 0
        ),
        None,
    )

    if baseline is None or baseline["n"] == 0:
        return None

    minimum_n = max(
        20,
        int(round(0.80 * int(baseline["n"]))),
    )

    eligible = [
        row
        for row in rows
        if (
            row["n"] >= minimum_n
            and row["median_error_px"] is not None
        )
    ]

    if not eligible:
        return None

    return min(
        eligible,
        key=lambda row: (
            float(row["median_error_px"]),
            -float(row["recall_20px"] or 0.0),
            abs(int(row["offset_frames"])),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV de prédictions I7. Détection automatique sinon.",
    )
    parser.add_argument(
        "--min-offset",
        type=int,
        default=-24,
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=24,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.input is None:
        input_path, matches = find_prediction_csv(root)

        print("=== CSV COMPATIBLES ===")
        for path in matches:
            print(" ", path)

        print()
        print("CSV sélectionné :", input_path)
    else:
        input_path = args.input.resolve()

    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    frame_column = (
        "source_frame"
        if "source_frame" in frame.columns
        else "local_frame"
    )

    print("Colonne temporelle :", frame_column)
    print("Lignes :", len(frame))

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for model_id, model_frame in frame.groupby(
        "model_id",
        dropna=False,
        sort=True,
    ):
        stride_values = []

        if "temporal_stride" in model_frame.columns:
            stride_values = sorted(
                {
                    int(value)
                    for value in pd.to_numeric(
                        model_frame["temporal_stride"],
                        errors="coerce",
                    ).dropna()
                }
            )

        for output_name, (
            prediction_x,
            prediction_y,
            valid_column,
        ) in OUTPUTS.items():
            scopes = ["all"]

            if valid_column in model_frame.columns:
                scopes.append("valid_only")

            for scope in scopes:
                local_rows: list[dict[str, Any]] = []

                for offset in range(
                    args.min_offset,
                    args.max_offset + 1,
                ):
                    errors, by_source = aligned_errors(
                        model_frame,
                        frame_column,
                        prediction_x,
                        prediction_y,
                        valid_column,
                        offset,
                        valid_only=(scope == "valid_only"),
                    )

                    metrics = error_metrics(errors)

                    row = {
                        "model_id": str(model_id),
                        "temporal_stride": ",".join(
                            str(value) for value in stride_values
                        ),
                        "output": output_name,
                        "scope": scope,
                        "offset_frames": offset,
                        **metrics,
                    }

                    summary_rows.append(row)
                    local_rows.append(row)

                    for source_name, source_errors in by_source.items():
                        source_rows.append(
                            {
                                "model_id": str(model_id),
                                "output": output_name,
                                "scope": scope,
                                "source_name": source_name,
                                "offset_frames": offset,
                                **error_metrics(source_errors),
                            }
                        )

                best = select_best(local_rows)

                if best is not None:
                    best_rows.append(best)

    output_dir = (
        root
        / "runs"
        / "_ttnet_temporal_alignment_003D_I9"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "temporal_offset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(source_rows).to_csv(
        output_dir / "temporal_offset_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(best_rows).to_csv(
        output_dir / "best_offsets.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "experiment": "003D-I9",
        "input_csv": str(input_path),
        "frame_column": frame_column,
        "offset_range": [
            args.min_offset,
            args.max_offset,
        ],
        "best_offsets": best_rows,
    }

    (
        output_dir / "temporal_alignment.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=== MEILLEURS OFFSETS ===")

    for row in best_rows:
        baseline = next(
            candidate
            for candidate in summary_rows
            if (
                candidate["model_id"] == row["model_id"]
                and candidate["output"] == row["output"]
                and candidate["scope"] == row["scope"]
                and candidate["offset_frames"] == 0
            )
        )

        print()
        print(
            f"{row['model_id']} | "
            f"{row['output']} | "
            f"{row['scope']}"
        )
        print(
            "  offset 0 : "
            f"N={baseline['n']} "
            f"médiane={baseline['median_error_px']:.2f}px "
            f"R20={100 * baseline['recall_20px']:.2f}%"
        )
        print(
            f"  meilleur : offset={row['offset_frames']:+d} "
            f"N={row['n']} "
            f"médiane={row['median_error_px']:.2f}px "
            f"R20={100 * row['recall_20px']:.2f}%"
        )

    print()
    print("Rapport :", output_dir / "temporal_alignment.json")


if __name__ == "__main__":
    main()
