from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(
    os.environ["TTFLUX_ROOT"]
)

BASELINE_CHECKPOINT = Path(
    os.environ[
        "TTFLUX_TTNET_BASELINE_CHECKPOINT"
    ]
)

FINAL_CHECKPOINT = Path(
    os.environ[
        "TTFLUX_TTNET_FINAL_CHECKPOINT"
    ]
)

POLICY_PATH = Path(
    os.environ["TTFLUX_POLICY_PATH"]
)

D1D0_MODULE_PATH = Path(
    os.environ["TTFLUX_D1D0_MODULE"]
)

I10_SCRIPT = Path(
    os.environ["TTFLUX_I10_SCRIPT"]
)

RALLIES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_RALLIES"]
)

FRAMES_MANIFEST = Path(
    os.environ["TTFLUX_BLURBALL_FRAMES"]
)

OUTPUT_DIR = Path(
    os.environ["TTFLUX_D1E_OUTPUT"]
)

CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080

TEST_MATCH_IDS = {
    22,
    23,
    24,
    25,
}


def load_module(
    name: str,
    path: Path,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Import impossible : {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

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
            f"Aucune ligne à écrire : {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def in_bounds(
    x: float,
    y: float,
    width: int,
    height: int,
) -> bool:
    return (
        0 <= x < width
        and 0 <= y < height
    )


def metric_delta(
    final: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
) -> float | None:
    final_value = final.get(key)
    baseline_value = baseline.get(key)

    if (
        final_value is None
        or baseline_value is None
    ):
        return None

    return (
        float(final_value)
        - float(baseline_value)
    )


def metrics_by_group(
    rows: list[dict[str, Any]],
    group_field: str,
    d1d0: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            str(row[group_field])
        ].append(row)

    return {
        key: {
            "global":
                d1d0.stage_metrics(
                    group_rows,
                    "global",
                ),
            "refined":
                d1d0.stage_metrics(
                    group_rows,
                    "refined",
                ),
        }
        for key, group_rows
        in sorted(
            grouped.items()
        )
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible."
        )

    policy = json.loads(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        policy["prediction_policy"]
        != "refined_only"
    ):
        raise RuntimeError(
            "Politique finale non conforme."
        )

    global_threshold = float(
        policy["global_threshold"]
    )

    local_threshold = float(
        policy["local_threshold"]
    )

    if (
        global_threshold != 0.05
        or local_threshold != 0.05
    ):
        raise RuntimeError(
            "Les seuils finaux ne sont "
            "pas 0.05 / 0.05."
        )

    if set(
        policy[
            "frozen_test_match_ids"
        ]
    ) != TEST_MATCH_IDS:
        raise RuntimeError(
            "Liste des matchs test incorrecte."
        )

    d1d0 = load_module(
        "ttflux_d1d0_reference_d1e",
        D1D0_MODULE_PATH,
    )

    i10 = load_module(
        "ttflux_i10_reference_d1e",
        I10_SCRIPT,
    )

    i10.ORIGINAL_WIDTH = (
        CANONICAL_WIDTH
    )

    i10.ORIGINAL_HEIGHT = (
        CANONICAL_HEIGHT
    )

    device = torch.device(
        "cuda:0"
    )

    print(
        "Chargement du checkpoint TTNet "
        "d'origine..."
    )

    baseline_model, baseline_metadata = (
        i10.load_model(
            BASELINE_CHECKPOINT,
            device,
        )
    )

    print(
        "Chargement du checkpoint final D1C..."
    )

    final_model, final_metadata = (
        i10.load_model(
            FINAL_CHECKPOINT,
            device,
        )
    )

    for model_id, metadata in (
        (
            "baseline",
            baseline_metadata,
        ),
        (
            "final",
            final_metadata,
        ),
    ):
        if (
            metadata[
                "matched_parameters"
            ]
            != metadata[
                "model_parameters"
            ]
        ):
            raise RuntimeError(
                "Chargement incomplet : "
                f"{model_id}"
            )

    models = {
        "ttnet_original":
            baseline_model,
        "ttnet_d1_final":
            final_model,
    }

    rally_rows = d1d0.read_csv(
        RALLIES_MANIFEST
    )

    frame_rows = d1d0.read_csv(
        FRAMES_MANIFEST
    )

    test_rallies = [
        row
        for row in rally_rows
        if row["ttflux_split"]
        == "test"
    ]

    test_rallies.sort(
        key=d1d0.rally_key
    )

    if len(test_rallies) != 80:
        raise RuntimeError(
            "80 échanges test attendus, "
            f"obtenu={len(test_rallies)}"
        )

    observed_matches = {
        d1d0.parse_int(
            row["match_id"]
        )
        for row in test_rallies
    }

    if observed_matches != TEST_MATCH_IDS:
        raise RuntimeError(
            "Matchs test inattendus : "
            f"{sorted(observed_matches)}"
        )

    annotations_by_key: dict[
        tuple[int, int],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in frame_rows:
        annotations_by_key[
            d1d0.rally_key(row)
        ].append(row)

    all_predictions: dict[
        str,
        list[dict[str, Any]],
    ] = {
        model_id: []
        for model_id in models
    }

    per_rally_rows = []

    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()

    started = time.perf_counter()

    for rally_index, rally in enumerate(
        test_rallies,
        start=1,
    ):
        match_id, rally_id = (
            d1d0.rally_key(rally)
        )

        if match_id not in TEST_MATCH_IDS:
            raise RuntimeError(
                "Échange hors test détecté."
            )

        video_path = (
            ROOT
            / Path(
                rally["video_path"]
            )
        ).resolve()

        annotations = sorted(
            d1d0.eligible_annotations(
                annotations_by_key[
                    (
                        match_id,
                        rally_id,
                    )
                ]
            ),
            key=lambda row:
                d1d0.parse_int(
                    row["frame_index"]
                ),
        )

        (
            resized_frames,
            source_width,
            source_height,
            actual_fps,
        ) = d1d0.load_resized_video(
            video_path
        )

        source_scale_x = (
            source_width
            / CANONICAL_WIDTH
        )

        source_scale_y = (
            source_height
            / CANONICAL_HEIGHT
        )

        rally_predictions: dict[
            str,
            list[dict[str, Any]],
        ] = {
            model_id: []
            for model_id in models
        }

        for annotation in annotations:
            frame_index = (
                d1d0.parse_int(
                    annotation[
                        "frame_index"
                    ]
                )
            )

            tensor = i10.make_sequence_tensor(
                resized_frames,
                frame_index,
                1,
                device,
            )

            visibility = (
                d1d0.parse_int(
                    annotation[
                        "visibility"
                    ]
                )
            )

            gt_x = d1d0.parse_float(
                annotation["x_px"]
            )

            gt_y = d1d0.parse_float(
                annotation["y_px"]
            )

            if (
                visibility == 1
                and (
                    gt_x is None
                    or gt_y is None
                )
            ):
                raise RuntimeError(
                    "GT visible absente : "
                    f"{match_id:02d}/"
                    f"{rally_id:03d}/"
                    f"{frame_index}"
                )

            for model_id, model in (
                models.items()
            ):
                with torch.inference_mode():
                    (
                        pred_global,
                        pred_local,
                        _pred_events,
                        _pred_seg,
                    ) = model.run_demo(
                        tensor
                    )

                decoded = (
                    i10.decode_prediction(
                        pred_global,
                        pred_local,
                        0.0,
                    )
                )

                global_x = (
                    float(
                        decoded[
                            "global_x"
                        ]
                    )
                    * source_scale_x
                )

                global_y = (
                    float(
                        decoded[
                            "global_y"
                        ]
                    )
                    * source_scale_y
                )

                refined_x = (
                    float(
                        decoded[
                            "refined_x"
                        ]
                    )
                    * source_scale_x
                )

                refined_y = (
                    float(
                        decoded[
                            "refined_y"
                        ]
                    )
                    * source_scale_y
                )

                global_conf = float(
                    decoded["global_conf"]
                )

                local_conf = float(
                    decoded["local_conf"]
                )

                global_valid = bool(
                    global_conf
                    >= global_threshold
                    and in_bounds(
                        global_x,
                        global_y,
                        source_width,
                        source_height,
                    )
                )

                refined_valid = bool(
                    global_valid
                    and local_conf
                    >= local_threshold
                    and in_bounds(
                        refined_x,
                        refined_y,
                        source_width,
                        source_height,
                    )
                )

                global_error = None
                refined_error = None

                if visibility == 1:
                    global_error = math.hypot(
                        global_x - gt_x,
                        global_y - gt_y,
                    )

                    refined_error = math.hypot(
                        refined_x - gt_x,
                        refined_y - gt_y,
                    )

                prediction_row = {
                    "model_id":
                        model_id,
                    "split":
                        "test",
                    "match_id":
                        f"{match_id:02d}",
                    "rally_id":
                        f"{rally_id:03d}",
                    "frame_index":
                        frame_index,
                    "fps_nominal":
                        rally[
                            "fps_nominal"
                        ],
                    "fps_actual":
                        actual_fps,
                    "source_width":
                        source_width,
                    "source_height":
                        source_height,
                    "visibility":
                        visibility,
                    "gt_x":
                        gt_x,
                    "gt_y":
                        gt_y,
                    "global_x":
                        round(
                            global_x,
                            4,
                        ),
                    "global_y":
                        round(
                            global_y,
                            4,
                        ),
                    "global_conf":
                        round(
                            global_conf,
                            8,
                        ),
                    "global_valid":
                        global_valid,
                    "global_error_px":
                        (
                            round(
                                global_error,
                                4,
                            )
                            if global_error
                            is not None
                            else None
                        ),
                    "refined_x":
                        round(
                            refined_x,
                            4,
                        ),
                    "refined_y":
                        round(
                            refined_y,
                            4,
                        ),
                    "local_conf":
                        round(
                            local_conf,
                            8,
                        ),
                    "refined_valid":
                        refined_valid,
                    "refined_error_px":
                        (
                            round(
                                refined_error,
                                4,
                            )
                            if refined_error
                            is not None
                            else None
                        ),
                }

                all_predictions[
                    model_id
                ].append(
                    prediction_row
                )

                rally_predictions[
                    model_id
                ].append(
                    prediction_row
                )

        baseline_global = (
            d1d0.stage_metrics(
                rally_predictions[
                    "ttnet_original"
                ],
                "global",
            )
        )

        baseline_refined = (
            d1d0.stage_metrics(
                rally_predictions[
                    "ttnet_original"
                ],
                "refined",
            )
        )

        final_global = (
            d1d0.stage_metrics(
                rally_predictions[
                    "ttnet_d1_final"
                ],
                "global",
            )
        )

        final_refined = (
            d1d0.stage_metrics(
                rally_predictions[
                    "ttnet_d1_final"
                ],
                "refined",
            )
        )

        per_rally_rows.append(
            {
                "match_id":
                    f"{match_id:02d}",
                "rally_id":
                    f"{rally_id:03d}",
                "fps_nominal":
                    rally[
                        "fps_nominal"
                    ],
                "frames":
                    len(annotations),
                "baseline_global_f1_at_20px":
                    baseline_global[
                        "f1_at_20px"
                    ],
                "baseline_refined_precision_at_20px":
                    baseline_refined[
                        "precision_at_20px"
                    ],
                "baseline_refined_recall_at_20px":
                    baseline_refined[
                        "recall_at_20px"
                    ],
                "baseline_refined_f1_at_20px":
                    baseline_refined[
                        "f1_at_20px"
                    ],
                "final_global_f1_at_20px":
                    final_global[
                        "f1_at_20px"
                    ],
                "final_refined_precision_at_20px":
                    final_refined[
                        "precision_at_20px"
                    ],
                "final_refined_recall_at_20px":
                    final_refined[
                        "recall_at_20px"
                    ],
                "final_refined_f1_at_20px":
                    final_refined[
                        "f1_at_20px"
                    ],
                "refined_f1_delta":
                    (
                        final_refined[
                            "f1_at_20px"
                        ]
                        - baseline_refined[
                            "f1_at_20px"
                        ]
                    ),
            }
        )

        print(
            f"[{rally_index:02d}/80] "
            f"{match_id:02d}/"
            f"{rally_id:03d} "
            f"frames={len(annotations):4d} "
            "base_F1="
            f"{baseline_refined['f1_at_20px']:.3f} "
            "final_F1="
            f"{final_refined['f1_at_20px']:.3f}"
        )

        del resized_frames

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - started
    )

    baseline_rows = all_predictions[
        "ttnet_original"
    ]

    final_rows = all_predictions[
        "ttnet_d1_final"
    ]

    if len(baseline_rows) != len(
        final_rows
    ):
        raise RuntimeError(
            "Nombre de prédictions incohérent "
            "entre les deux modèles."
        )

    baseline_global = (
        d1d0.stage_metrics(
            baseline_rows,
            "global",
        )
    )

    baseline_refined = (
        d1d0.stage_metrics(
            baseline_rows,
            "refined",
        )
    )

    final_global = (
        d1d0.stage_metrics(
            final_rows,
            "global",
        )
    )

    final_refined = (
        d1d0.stage_metrics(
            final_rows,
            "refined",
        )
    )

    baseline_transitions = (
        d1d0.transition_metrics(
            baseline_rows
        )
    )

    final_transitions = (
        d1d0.transition_metrics(
            final_rows
        )
    )

    baseline_by_fps = metrics_by_group(
        baseline_rows,
        "fps_nominal",
        d1d0,
    )

    final_by_fps = metrics_by_group(
        final_rows,
        "fps_nominal",
        d1d0,
    )

    baseline_by_match = metrics_by_group(
        baseline_rows,
        "match_id",
        d1d0,
    )

    final_by_match = metrics_by_group(
        final_rows,
        "match_id",
        d1d0,
    )

    combined_predictions = (
        baseline_rows
        + final_rows
    )

    predictions_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    per_rally_path = (
        OUTPUT_DIR
        / "test_by_rally.csv"
    )

    write_csv(
        predictions_path,
        combined_predictions,
    )

    write_csv(
        per_rally_path,
        per_rally_rows,
    )

    comparison = {
        "global": {
            "precision_at_20px_delta":
                metric_delta(
                    final_global,
                    baseline_global,
                    "precision_at_20px",
                ),
            "recall_at_20px_delta":
                metric_delta(
                    final_global,
                    baseline_global,
                    "recall_at_20px",
                ),
            "f1_at_20px_delta":
                metric_delta(
                    final_global,
                    baseline_global,
                    "f1_at_20px",
                ),
            "median_valid_error_px_delta":
                metric_delta(
                    final_global,
                    baseline_global,
                    "median_valid_error_px",
                ),
            "false_positive_rate_invisible_delta":
                metric_delta(
                    final_global,
                    baseline_global,
                    "false_positive_rate_invisible",
                ),
        },
        "refined": {
            "precision_at_20px_delta":
                metric_delta(
                    final_refined,
                    baseline_refined,
                    "precision_at_20px",
                ),
            "recall_at_20px_delta":
                metric_delta(
                    final_refined,
                    baseline_refined,
                    "recall_at_20px",
                ),
            "f1_at_20px_delta":
                metric_delta(
                    final_refined,
                    baseline_refined,
                    "f1_at_20px",
                ),
            "median_valid_error_px_delta":
                metric_delta(
                    final_refined,
                    baseline_refined,
                    "median_valid_error_px",
                ),
            "false_positive_rate_invisible_delta":
                metric_delta(
                    final_refined,
                    baseline_refined,
                    "false_positive_rate_invisible",
                ),
        },
    }

    summary = {
        "schema_version": 1,
        "experiment":
            "DSET_D1E_BlurBall_final_test",
        "status":
            "final_test_opened_once",
        "policy": {
            "prediction_policy":
                policy[
                    "prediction_policy"
                ],
            "global_threshold":
                global_threshold,
            "local_threshold":
                local_threshold,
            "distance_threshold_px":
                float(
                    policy[
                        "distance_threshold_px"
                    ]
                ),
            "calibration_split":
                policy[
                    "selection_split"
                ],
        },
        "dataset": {
            "split":
                "test",
            "match_ids":
                sorted(
                    TEST_MATCH_IDS
                ),
            "rallies":
                len(test_rallies),
            "evaluated_frames_per_model":
                len(final_rows),
            "total_prediction_rows":
                len(
                    combined_predictions
                ),
        },
        "models": {
            "ttnet_original": {
                "checkpoint_path":
                    str(
                        BASELINE_CHECKPOINT
                    ),
                "checkpoint_sha256":
                    sha256_file(
                        BASELINE_CHECKPOINT
                    ),
                "epoch":
                    baseline_metadata[
                        "epoch"
                    ],
                "global":
                    baseline_global,
                "refined":
                    baseline_refined,
                "transitions":
                    baseline_transitions,
                "by_fps":
                    baseline_by_fps,
                "by_match":
                    baseline_by_match,
            },
            "ttnet_d1_final": {
                "checkpoint_path":
                    str(
                        FINAL_CHECKPOINT
                    ),
                "checkpoint_sha256":
                    sha256_file(
                        FINAL_CHECKPOINT
                    ),
                "epoch":
                    final_metadata[
                        "epoch"
                    ],
                "global":
                    final_global,
                "refined":
                    final_refined,
                "transitions":
                    final_transitions,
                "by_fps":
                    final_by_fps,
                "by_match":
                    final_by_match,
            },
        },
        "comparison_final_minus_original":
            comparison,
        "runtime": {
            "seconds":
                elapsed,
            "frames_per_second_per_model_equivalent":
                (
                    (
                        len(final_rows)
                        * 2
                    )
                    / elapsed
                    if elapsed > 0
                    else None
                ),
            "device":
                torch.cuda.get_device_name(
                    0
                ),
        },
        "artifacts": {
            "predictions_csv":
                str(
                    predictions_path
                ),
            "per_rally_csv":
                str(
                    per_rally_path
                ),
        },
        "protocol_note": (
            "Le test 22-25 est désormais ouvert. "
            "Ces résultats ne doivent pas servir à "
            "modifier les seuils ou réentraîner le "
            "même modèle D1."
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
        "DSET_D1E_FINAL_TEST_GENERATED"
    )

    print("")
    print(
        "=== TTNET ORIGINAL — REFINED ==="
    )

    print(
        json.dumps(
            baseline_refined,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        "=== TTNET D1 FINAL — REFINED ==="
    )

    print(
        json.dumps(
            final_refined,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        "=== DELTA FINAL - ORIGINAL ==="
    )

    print(
        json.dumps(
            comparison["refined"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print("")
    print(
        "TEST_OPENED_DO_NOT_RECALIBRATE_D1"
    )

    print(f"summary={summary_path}")
    print(
        f"predictions={predictions_path}"
    )
    print(
        f"per_rally={per_rally_path}"
    )


if __name__ == "__main__":
    main()