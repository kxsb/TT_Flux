from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()
INPUT_CSV = (
    ROOT
    / "runs"
    / "_ttnet_gt_evaluation_003D_I7"
    / "ttnet_gt_predictions.csv"
)
OUT_DIR = ROOT / "runs" / "_ttnet_coordinate_audit_003D_I8"
OUT_JSON = OUT_DIR / "coordinate_audit.json"


def find_column(
    df: pd.DataFrame,
    exact: list[str],
    required_tokens: list[str] | None = None,
) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}

    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]

    if required_tokens:
        for col in df.columns:
            normalized = re.sub(r"[^a-z0-9]+", "_", str(col).lower())
            if all(token in normalized for token in required_tokens):
                return str(col)

    return None


def find_stage_coord(df: pd.DataFrame, stage: str, axis: str) -> str | None:
    return find_column(
        df,
        [
            f"{stage}_{axis}",
            f"{stage}_pred_{axis}",
            f"pred_{stage}_{axis}",
            f"{axis}_{stage}",
            f"{stage}_ball_{axis}",
            f"ball_{stage}_{axis}",
        ],
        [stage, axis],
    )


def finite_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def quantiles(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}

    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def distance_stats(gt_x, gt_y, pred_x, pred_y) -> dict:
    err = np.hypot(gt_x - pred_x, gt_y - pred_y)

    return {
        "count": int(len(err)),
        "median_error_px": float(np.median(err)),
        "p90_error_px": float(np.quantile(err, 0.90)),
        "recall_5": float(np.mean(err <= 5)),
        "recall_10": float(np.mean(err <= 10)),
        "recall_20": float(np.mean(err <= 20)),
        "recall_40": float(np.mean(err <= 40)),
    }


def analyse_coordinates(
    part: pd.DataFrame,
    gt_x_col: str,
    gt_y_col: str,
    pred_x_col: str,
    pred_y_col: str,
) -> dict:
    gx = finite_values(part[gt_x_col])
    gy = finite_values(part[gt_y_col])
    px = finite_values(part[pred_x_col])
    py = finite_values(part[pred_y_col])

    valid = (
        np.isfinite(gx)
        & np.isfinite(gy)
        & np.isfinite(px)
        & np.isfinite(py)
    )

    gx = gx[valid]
    gy = gy[valid]
    px = px[valid]
    py = py[valid]

    result = {
        "valid_rows": int(valid.sum()),
        "gt_x": quantiles(gx),
        "gt_y": quantiles(gy),
        "pred_x": quantiles(px),
        "pred_y": quantiles(py),
    }

    if len(gx) < 3:
        return result

    result["raw"] = distance_stats(gx, gy, px, py)

    # Décalage constant médian.
    dx = float(np.median(gx - px))
    dy = float(np.median(gy - py))

    result["median_offset"] = {
        "dx": dx,
        "dy": dy,
        "corrected": distance_stats(gx, gy, px + dx, py + dy),
    }

    # Test d'inversion des axes X/Y.
    result["xy_swap"] = distance_stats(gx, gy, py, px)

    # Régression indépendante par axe :
    # gt_x = ax * pred_x + bx
    # gt_y = ay * pred_y + by
    x_design = np.column_stack([px, np.ones(len(px))])
    y_design = np.column_stack([py, np.ones(len(py))])

    coef_x, *_ = np.linalg.lstsq(x_design, gx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(y_design, gy, rcond=None)

    axis_fit_x = x_design @ coef_x
    axis_fit_y = y_design @ coef_y

    result["axis_scale_offset_fit"] = {
        "gt_x_equals": {
            "pred_x_scale": float(coef_x[0]),
            "offset": float(coef_x[1]),
        },
        "gt_y_equals": {
            "pred_y_scale": float(coef_y[0]),
            "offset": float(coef_y[1]),
        },
        "corrected": distance_stats(gx, gy, axis_fit_x, axis_fit_y),
    }

    # Transformation affine complète :
    # permet de révéler crop, redimensionnement anisotrope,
    # axes inversés ou mauvais retour vers l'image source.
    design = np.column_stack([px, py, np.ones(len(px))])

    affine_x, *_ = np.linalg.lstsq(design, gx, rcond=None)
    affine_y, *_ = np.linalg.lstsq(design, gy, rcond=None)

    affine_pred_x = design @ affine_x
    affine_pred_y = design @ affine_y

    result["affine_fit"] = {
        "gt_x_coefficients": {
            "pred_x": float(affine_x[0]),
            "pred_y": float(affine_x[1]),
            "constant": float(affine_x[2]),
        },
        "gt_y_coefficients": {
            "pred_x": float(affine_y[0]),
            "pred_y": float(affine_y[1]),
            "constant": float(affine_y[2]),
        },
        "corrected": distance_stats(
            gx,
            gy,
            affine_pred_x,
            affine_pred_y,
        ),
    }

    return result


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\n=== COLONNES DISPONIBLES ===")
    for col in df.columns:
        print(f"  {col}")

    model_col = find_column(
        df,
        ["model_id", "model", "checkpoint_id"],
        ["model"],
    )
    source_col = find_column(
        df,
        ["source_name", "source", "video_id"],
        ["source"],
    )
    frame_col = find_column(
        df,
        ["frame", "frame_idx", "frame_index", "gt_frame"],
        ["frame"],
    )

    gt_x_col = find_column(
        df,
        ["gt_x", "x_gt", "target_x", "true_x", "label_x"],
        ["gt", "x"],
    )
    gt_y_col = find_column(
        df,
        ["gt_y", "y_gt", "target_y", "true_y", "label_y"],
        ["gt", "y"],
    )

    detected = {
        "model": model_col,
        "source": source_col,
        "frame": frame_col,
        "gt_x": gt_x_col,
        "gt_y": gt_y_col,
        "global_x": find_stage_coord(df, "global", "x"),
        "global_y": find_stage_coord(df, "global", "y"),
        "refined_x": find_stage_coord(df, "refined", "x"),
        "refined_y": find_stage_coord(df, "refined", "y"),
    }

    print("\n=== COLONNES DÉTECTÉES ===")
    for key, value in detected.items():
        print(f"{key:12s}: {value}")

    report = {
        "experiment": "003D_I8_TTNet_coordinate_audit",
        "input_csv": str(INPUT_CSV),
        "row_count": int(len(df)),
        "columns": list(map(str, df.columns)),
        "detected_columns": detected,
        "models": {},
    }

    if not gt_x_col or not gt_y_col:
        report["error"] = "Colonnes GT X/Y non détectées."
        OUT_JSON.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\nERREUR : colonnes GT non détectées.")
        print(df.head(5).to_string())
        return

    if model_col:
        model_values = df[model_col].fillna("unknown").astype(str).unique()
    else:
        model_values = ["all"]

    for model_id in model_values:
        if model_col:
            part = df[df[model_col].fillna("unknown").astype(str) == model_id]
        else:
            part = df

        model_report = {
            "rows": int(len(part)),
            "sources": (
                sorted(part[source_col].dropna().astype(str).unique().tolist())
                if source_col
                else []
            ),
            "stages": {},
        }

        for stage in ("global", "refined"):
            px_col = detected[f"{stage}_x"]
            py_col = detected[f"{stage}_y"]

            if not px_col or not py_col:
                model_report["stages"][stage] = {
                    "error": "Colonnes de prédiction non détectées."
                }
                continue

            model_report["stages"][stage] = analyse_coordinates(
                part,
                gt_x_col,
                gt_y_col,
                px_col,
                py_col,
            )

        report["models"][str(model_id)] = model_report

    # Recherche de colonnes de score/peak utiles au checkpoint 30 fps.
    score_columns = [
        str(col)
        for col in df.columns
        if any(
            token in str(col).lower()
            for token in ("score", "peak", "confidence", "prob")
        )
    ]

    report["score_columns"] = {}

    for col in score_columns:
        report["score_columns"][col] = quantiles(finite_values(df[col]))

    OUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RÉSUMÉ DE L'AUDIT ===")

    for model_id, model_report in report["models"].items():
        print(f"\nMODEL {model_id}")

        for stage, stage_report in model_report["stages"].items():
            print(f"  {stage}")

            if "raw" not in stage_report:
                print(
                    f"    aucune coordonnée exploitable "
                    f"({stage_report.get('valid_rows', 0)} lignes)"
                )
                continue

            raw = stage_report["raw"]
            offset = stage_report["median_offset"]
            scaled = stage_report["axis_scale_offset_fit"]
            affine = stage_report["affine_fit"]

            print(
                f"    brut       : médiane={raw['median_error_px']:.2f}px "
                f"p90={raw['p90_error_px']:.2f}px "
                f"R20={100 * raw['recall_20']:.2f}%"
            )
            print(
                f"    offset     : dx={offset['dx']:.2f} "
                f"dy={offset['dy']:.2f} "
                f"médiane={offset['corrected']['median_error_px']:.2f}px"
            )
            print(
                "    axe X      : "
                f"GT = {scaled['gt_x_equals']['pred_x_scale']:.5f} * PRED "
                f"+ {scaled['gt_x_equals']['offset']:.2f}"
            )
            print(
                "    axe Y      : "
                f"GT = {scaled['gt_y_equals']['pred_y_scale']:.5f} * PRED "
                f"+ {scaled['gt_y_equals']['offset']:.2f}"
            )
            print(
                f"    axe corrigé: "
                f"médiane={scaled['corrected']['median_error_px']:.2f}px "
                f"R20={100 * scaled['corrected']['recall_20']:.2f}%"
            )
            print(
                f"    affine     : "
                f"médiane={affine['corrected']['median_error_px']:.2f}px "
                f"R20={100 * affine['corrected']['recall_20']:.2f}%"
            )

    print(f"\nRapport : {OUT_JSON}")


if __name__ == "__main__":
    main()
