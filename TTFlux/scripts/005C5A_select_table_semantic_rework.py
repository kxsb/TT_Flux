from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "005C5A"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", default="runs/rally_table_object_005C4_canonical_audit/table_object_canonical_audit_005C4.csv")
    ap.add_argument("--table-objects", default="runs/rally_table_object_005C3_corrected/table_objects_corrected_005C3.csv")
    ap.add_argument("--out-dir", default="runs/rally_table_object_005C5_semantic_rework")
    ap.add_argument("--inside-max", type=float, default=0.35)
    ap.add_argument("--min-points", type=int, default=20)
    args = ap.parse_args()

    root = Path.cwd()

    audit_path = Path(args.audit_csv)
    table_path = Path(args.table_objects)
    out_dir = Path(args.out_dir)

    if not audit_path.is_absolute():
        audit_path = root / audit_path
    if not table_path.is_absolute():
        table_path = root / table_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(audit_path).fillna("")
    table = pd.read_csv(table_path).fillna("")

    audit["points_projected_num"] = to_num(audit.get("points_projected", 0)).fillna(0).astype(int)
    audit["inside_ratio_num"] = to_num(audit.get("inside_ratio", 0)).fillna(0.0)
    audit["camera_segment_id_005B2"] = to_num(audit["camera_segment_id"]).fillna(1).astype(int)

    # Cas suspects : beaucoup de points projetés, mais presque rien dans la table.
    audit["suspect_reason_005C5A"] = ""

    mask_zero = (
        (audit["points_projected_num"] >= args.min_points)
        & (audit["inside_ratio_num"] <= args.inside_max)
    )

    audit.loc[mask_zero, "suspect_reason_005C5A"] = (
        "many_projected_points_but_low_inside_ratio"
    )

    # Les corrections humaines avec inside faible sont prioritaires :
    # elles indiquent probablement un ordre sémantique faux ou un mauvais coin.
    mask_human_bad = (
        audit["table_source_005C3"].astype(str).eq("005C2_human")
        & (audit["points_projected_num"] >= 1)
        & (audit["inside_ratio_num"] <= 0.50)
    )

    audit.loc[mask_human_bad, "suspect_reason_005C5A"] = (
        audit.loc[mask_human_bad, "suspect_reason_005C5A"].astype(str)
        + "|human_correction_needs_semantic_rework"
    )

    suspects = audit[audit["suspect_reason_005C5A"].astype(str).str.len() > 0].copy()

    # Jointure avec table objects pour que le serveur puisse charger les clips.
    keep_cols = [
        "review_id",
        "camera_segment_id_005B2",
        "video_id",
        "rally_id",
        "first_frame",
        "last_frame",
        "best_frame_005C1",
        "table_source_005C3",
        "table_status_005C3",
        "table_confidence_005C1",
        "quad_tl_x_005C3",
        "quad_tl_y_005C3",
        "quad_tr_x_005C3",
        "quad_tr_y_005C3",
        "quad_br_x_005C3",
        "quad_br_y_005C3",
        "quad_bl_x_005C3",
        "quad_bl_y_005C3",
        "src_w",
        "src_h",
        "clip_path",
    ]

    keep_cols = [c for c in keep_cols if c in table.columns]

    merged = suspects.merge(
        table[keep_cols],
        on=["review_id", "camera_segment_id_005B2"],
        how="left",
        suffixes=("", "_table"),
    )

    merged = merged.sort_values(
        ["suspect_reason_005C5A", "points_projected_num", "inside_ratio_num"],
        ascending=[True, False, True],
    )

    out_suspects = out_dir / "table_semantic_rework_candidates_005C5A.csv"
    out_json = out_dir / "table_semantic_rework_summary_005C5A.json"

    merged.to_csv(out_suspects, index=False, encoding="utf-8")

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "select_table_objects_requiring_semantic_corner_rework",
        "audit_csv": str(audit_path),
        "table_objects": str(table_path),
        "objects_audited": int(len(audit)),
        "candidates": int(len(merged)),
        "inside_max": args.inside_max,
        "min_points": args.min_points,
        "review_ids": merged["review_id"].astype(str).tolist(),
        "outputs": {
            "candidates": str(out_suspects),
        },
        "next": "Run 005C5B semantic corner correction server on these candidates."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005C5A status=OK")
    print("objects_audited=", summary["objects_audited"])
    print("candidates=", summary["candidates"])
    print("review_ids=", json.dumps(summary["review_ids"], ensure_ascii=False))
    print("wrote", out_suspects)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
