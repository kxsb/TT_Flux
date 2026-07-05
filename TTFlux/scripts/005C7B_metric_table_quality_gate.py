from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VERSION = "005C7B"

CANON_W = 520
CANON_H = 936


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def imwrite_unicode(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))


def read_frame(clip_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def snap_quad_src(row: dict) -> np.ndarray:
    return np.asarray([
        [float(row["snap_tl_x_005C7A"]), float(row["snap_tl_y_005C7A"])],
        [float(row["snap_tr_x_005C7A"]), float(row["snap_tr_y_005C7A"])],
        [float(row["snap_br_x_005C7A"]), float(row["snap_br_y_005C7A"])],
        [float(row["snap_bl_x_005C7A"]), float(row["snap_bl_y_005C7A"])],
    ], dtype=np.float32)


def old_quad_src(row: dict) -> np.ndarray:
    return np.asarray([
        [float(row["quad_tl_x_005C3"]), float(row["quad_tl_y_005C3"])],
        [float(row["quad_tr_x_005C3"]), float(row["quad_tr_y_005C3"])],
        [float(row["quad_br_x_005C3"]), float(row["quad_br_y_005C3"])],
        [float(row["quad_bl_x_005C3"]), float(row["quad_bl_y_005C3"])],
    ], dtype=np.float32)


def quad_area(q: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2))))


def warp_quad(frame: np.ndarray, quad: np.ndarray, w: int = CANON_W, h: int = CANON_H):
    dst = np.asarray([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
    warped = cv2.warpPerspective(frame, H, (w, h))

    return warped


def line_metrics(warped: np.ndarray):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Blanc / lignes claires : faible saturation + valeur assez haute.
    white = ((hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 135)).astype(np.uint8)

    # Edges : utiles quand les lignes ne sont pas vraiment blanches à cause compression/lumière.
    edges = cv2.Canny(gray, 55, 145)
    edge = (edges > 0).astype(np.uint8)

    h, w = warped.shape[:2]

    outer_band = max(12, int(min(w, h) * 0.045))
    inner_band = max(10, int(min(w, h) * 0.035))

    bands = {
        "top":        (slice(0, outer_band), slice(0, w)),
        "bottom":     (slice(h - outer_band, h), slice(0, w)),
        "left":       (slice(0, h), slice(0, outer_band)),
        "right":      (slice(0, h), slice(w - outer_band, w)),
        "net":        (slice(max(0, h // 2 - inner_band), min(h, h // 2 + inner_band)), slice(0, w)),
        "centerline": (slice(0, h), slice(max(0, w // 2 - inner_band), min(w, w // 2 + inner_band))),
    }

    out = {}

    for name, sl in bands.items():
        ww = float(white[sl].mean())
        ee = float(edge[sl].mean())

        # Seuils empiriques, volontairement tolérants.
        if name in {"top", "bottom", "left", "right"}:
            support = int((ww >= 0.018) or (ee >= 0.045))
        else:
            support = int((ww >= 0.012) or (ee >= 0.035))

        out[f"{name}_white_ratio"] = round(ww, 6)
        out[f"{name}_edge_ratio"] = round(ee, 6)
        out[f"{name}_support"] = support

    outer_count = (
        out["top_support"]
        + out["bottom_support"]
        + out["left_support"]
        + out["right_support"]
    )

    internal_count = out["net_support"] + out["centerline_support"]

    # Texture centrale : si le warp est un panneau/mur/joueur, ça varie souvent beaucoup.
    central = warped[
        int(h * 0.18):int(h * 0.82),
        int(w * 0.18):int(w * 0.82),
    ]

    if central.size:
        chsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_std = float(np.std(chsv[:, :, 0]) / 180.0)
        sat_std = float(np.std(chsv[:, :, 1]) / 255.0)
        val_std = float(np.std(chsv[:, :, 2]) / 255.0)
    else:
        hue_std = sat_std = val_std = 1.0

    out.update({
        "outer_support_count": int(outer_count),
        "internal_support_count": int(internal_count),
        "central_hue_std": round(hue_std, 6),
        "central_sat_std": round(sat_std, 6),
        "central_val_std": round(val_std, 6),
    })

    return out, white, edge


def classify_quality(row: dict, metrics: dict):
    snap_status = str(row.get("snap_status_005C7A", ""))
    snap_conf = float(to_num(pd.Series([row.get("snap_confidence_005C7A", 0)])).fillna(0).iloc[0])
    poly_support = float(to_num(pd.Series([row.get("snap_poly_support_005C7A", 0)])).fillna(0).iloc[0])
    mask_in_poly = float(to_num(pd.Series([row.get("snap_mask_in_poly_005C7A", 0)])).fillna(0).iloc[0])
    inside = float(to_num(pd.Series([row.get("audit_inside_ratio", 0)])).fillna(0).iloc[0])
    points = int(to_num(pd.Series([row.get("audit_points_projected", 0)])).fillna(0).iloc[0])

    outer = int(metrics["outer_support_count"])
    internal = int(metrics["internal_support_count"])

    # Homographie métrique stricte : pas juste un patch bleu.
    if (
        snap_status == "SNAP_METRIC_TABLE_OK"
        and snap_conf >= 0.68
        and poly_support >= 0.50
        and mask_in_poly >= 0.48
        and outer >= 2
        and (internal >= 1 or snap_conf >= 0.86)
        and (points < 20 or inside >= 0.70)
    ):
        return "METRIC_TABLE_STRICT", "snap_supported_by_table_lines_and_mask"

    # Candidat crédible mais pas encore figé.
    if (
        snap_status in {"SNAP_METRIC_TABLE_OK", "SNAP_METRIC_CANDIDATE"}
        and snap_conf >= 0.55
        and poly_support >= 0.38
        and mask_in_poly >= 0.35
        and (outer >= 1 or internal >= 1)
    ):
        return "METRIC_TABLE_REVIEW", "metric_candidate_needs_visual_or_later_temporal_validation"

    # Masque table utile, mais pas de métrique globale.
    if snap_status in {"SNAP_METRIC_TABLE_OK", "SNAP_METRIC_CANDIDATE", "SNAP_PARTIAL_TABLE"}:
        return "PARTIAL_TABLE_OBJECT", "table_visible_but_no_full_metric_structure"

    return "BAD_TABLE_OBJECT", "bad_or_missing_snap"


def draw_grid(warped):
    h, w = warped.shape[:2]
    out = warped.copy()

    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(out, (0, h // 2), (w - 1, h // 2), (0, 180, 255), 2, cv2.LINE_AA)
    cv2.line(out, (w // 2, 0), (w // 2, h - 1), (255, 255, 0), 2, cv2.LINE_AA)

    for x in np.linspace(0, w - 1, 5).astype(int):
        cv2.line(out, (x, 0), (x, h - 1), (80, 180, 180), 1, cv2.LINE_AA)

    for y in np.linspace(0, h - 1, 7).astype(int):
        cv2.line(out, (0, y), (w - 1, y), (80, 180, 180), 1, cv2.LINE_AA)

    return out


def draw_quad(frame, q, color, label):
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    qi = np.round(q).astype(int)

    for a, b in zip(qi, np.vstack([qi[1:], qi[:1]])):
        cv2.line(frame, tuple(a), tuple(b), color, 3, cv2.LINE_AA)

    for p in qi:
        cv2.circle(frame, tuple(p), 6, color, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        label,
        (int(qi[0, 0]) + 8, int(qi[0, 1]) + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def make_report_image(frame, old_q, snap_q, warped_grid, quality, reason, metrics):
    left = frame.copy()

    draw_quad(left, old_q, (0, 255, 255), "old")
    draw_quad(left, snap_q, (255, 0, 255), "snap")

    cv2.putText(
        left,
        f"{quality}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        left,
        f"{reason}",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        left,
        f"outer={metrics['outer_support_count']} internal={metrics['internal_support_count']}",
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Redimension droite.
    h1, w1 = left.shape[:2]
    h2, w2 = warped_grid.shape[:2]
    target_right_w = 360
    scale = target_right_w / max(1, w2)
    right = cv2.resize(warped_grid, (target_right_w, int(round(h2 * scale))), interpolation=cv2.INTER_AREA)

    h = max(left.shape[0], right.shape[0])

    def pad(img):
        dh = h - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    return cv2.hconcat([pad(left), pad(right)])


def build_html(cards):
    parts = []

    for c in cards:
        parts.append(f"""
<section class="card {html.escape(str(c["quality"]))}">
  <h2>{html.escape(str(c["review_id"]))} · seg {c["seg"]} · {html.escape(str(c["quality"]))}</h2>
  <p>
    reason={html.escape(str(c["reason"]))}
    · outer={c["outer"]}
    · internal={c["internal"]}
    · snap={html.escape(str(c["snap_status"]))}
    · snap_conf={c["snap_conf"]}
  </p>
  <img src="images/{html.escape(Path(c["image"]).name)}">
</section>
""")

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>005C7B table structural QA</title>
<style>
body{{margin:0;background:#101219;color:#eef1f8;font-family:system-ui,Segoe UI,sans-serif}}
header{{padding:16px 22px;background:#171b25;border-bottom:1px solid #303746}}
.card{{margin:18px;padding:16px;background:#181d27;border:1px solid #303746;border-radius:12px}}
.card.METRIC_TABLE_STRICT{{border-color:#2d8a4d}}
.card.METRIC_TABLE_REVIEW{{border-color:#4a8a8a}}
.card.PARTIAL_TABLE_OBJECT{{border-color:#9b842e}}
.card.BAD_TABLE_OBJECT{{border-color:#9b3939}}
img{{max-width:100%;border-radius:8px;border:1px solid #303746}}
h1{{margin:0;font-size:20px}}
h2{{font-size:17px;margin:0 0 8px}}
p{{color:#c3cada}}
</style>
</head>
<body>
<header>
<h1>TTFlux 005C7B · QA structurelle table métrique</h1>
<p>Gauche = frame + old/snap. Droite = warp canonique du snap. On vérifie la présence de structure table, pas seulement un patch bleu.</p>
</header>
{''.join(parts)}
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-csv", default="runs/rally_table_snap_geometry_005C7A/table_object_snap_geometry_005C7A.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_metric_qa_005C7B")
    ap.add_argument("--make-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=101)
    args = ap.parse_args()

    root = Path.cwd()

    snap_path = Path(args.snap_csv)
    out_dir = Path(args.out_dir)

    if not snap_path.is_absolute():
        snap_path = root / snap_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    snap = pd.read_csv(snap_path).fillna("")

    rows = []
    cards = []

    for i, (_, r) in enumerate(snap.iterrows(), start=1):
        row = r.to_dict()
        review_id = str(row["review_id"])
        seg = int(to_num(pd.Series([row["camera_segment_id_005B2"]])).fillna(1).iloc[0])

        clip_path = Path(str(row["clip_path"]))
        if not clip_path.is_absolute():
            clip_path = root / clip_path

        frame_idx = int(to_num(pd.Series([
            row.get("mask_frame_005C6B", row.get("best_frame_005C1", row.get("first_frame", 0)))
        ])).fillna(0).iloc[0])

        frame = read_frame(clip_path, frame_idx)

        if frame is None:
            out = dict(row)
            out.update({
                "metric_quality_005C7B": "BAD_TABLE_OBJECT",
                "metric_reason_005C7B": "missing_frame",
            })
            rows.append(out)
            continue

        try:
            q_snap = snap_quad_src(row)
            q_old = old_quad_src(row)
            warped = warp_quad(frame, q_snap)
            metrics, white, edge = line_metrics(warped)
            quality, reason = classify_quality(row, metrics)
        except Exception as e:
            q_snap = np.zeros((4, 2), dtype=np.float32)
            q_old = np.zeros((4, 2), dtype=np.float32)
            warped = np.zeros((CANON_H, CANON_W, 3), dtype=np.uint8)
            metrics = {
                "outer_support_count": 0,
                "internal_support_count": 0,
            }
            quality = "BAD_TABLE_OBJECT"
            reason = f"exception:{type(e).__name__}"

        out = dict(row)
        out.update(metrics)
        out.update({
            "metric_quality_005C7B": quality,
            "metric_reason_005C7B": reason,
            "metric_frame_005C7B": frame_idx,
        })

        rows.append(out)

        if args.make_images:
            warped_grid = draw_grid(warped)
            report = make_report_image(
                frame=frame,
                old_q=q_old,
                snap_q=q_snap,
                warped_grid=warped_grid,
                quality=quality,
                reason=reason,
                metrics=metrics,
            )

            fname = f"{i:03d}_{review_id}_seg{seg}_005C7B_metric_qa.jpg"
            out_img = out_dir / "images" / fname
            imwrite_unicode(out_img, report)

            cards.append({
                "review_id": review_id,
                "seg": seg,
                "quality": quality,
                "reason": reason,
                "outer": metrics.get("outer_support_count", 0),
                "internal": metrics.get("internal_support_count", 0),
                "snap_status": str(row.get("snap_status_005C7A", "")),
                "snap_conf": float(to_num(pd.Series([row.get("snap_confidence_005C7A", 0)])).fillna(0).iloc[0]),
                "image": str(out_img),
            })

    out_df = pd.DataFrame(rows)

    order = {
        "BAD_TABLE_OBJECT": 0,
        "PARTIAL_TABLE_OBJECT": 1,
        "METRIC_TABLE_REVIEW": 2,
        "METRIC_TABLE_STRICT": 3,
    }

    out_df["_order"] = out_df["metric_quality_005C7B"].astype(str).map(order).fillna(9)
    out_df = out_df.sort_values(
        ["_order", "review_id", "camera_segment_id_005B2"],
        ascending=[True, True, True],
    ).drop(columns=["_order"]).copy()

    out_csv = out_dir / "table_metric_quality_005C7B.csv"
    out_json = out_dir / "table_metric_quality_summary_005C7B.json"
    out_html = out_dir / "table_metric_quality_report_005C7B.html"

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    if args.make_images:
        cards = sorted(
            cards,
            key=lambda c: (
                order.get(str(c["quality"]), 9),
                str(c["review_id"]),
                int(c["seg"]),
            ),
        )[:args.max_images]

        out_html.write_text(build_html(cards), encoding="utf-8")

    counts = out_df["metric_quality_005C7B"].astype(str).value_counts().to_dict()
    snap_counts = out_df["snap_status_005C7A"].astype(str).value_counts().to_dict()

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "structural_quality_gate_for_snapped_metric_table_using_canonical_warp_line_support",
        "snap_csv": str(snap_path),
        "objects_total": int(len(out_df)),
        "snap_status_counts_005C7A": snap_counts,
        "metric_quality_counts_005C7B": counts,
        "outputs": {
            "csv": str(out_csv),
            "html": str(out_html) if args.make_images else "",
        },
        "next": "Use METRIC_TABLE_STRICT for reliable metric table. Use METRIC_TABLE_REVIEW only after visual vote or temporal consistency. Keep PARTIAL_TABLE_OBJECT as local mask only."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C7B status=OK")
    print("objects_total=", summary["objects_total"])
    print("snap_status_counts_005C7A=", json.dumps(snap_counts, ensure_ascii=False))
    print("metric_quality_counts_005C7B=", json.dumps(counts, ensure_ascii=False))
    print("wrote", out_csv)
    if args.make_images:
        print("wrote", out_html)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
