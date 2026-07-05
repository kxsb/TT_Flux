from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "005D6A"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def find_optional_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_by_fragment(df: pd.DataFrame, fragments: list[str]):
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}

    for frag in fragments:
        frag = frag.lower()
        for c in cols:
            if frag in low[c]:
                return c

    return None


def scene_status_map(scene: pd.DataFrame):
    out = {}

    if scene.empty:
        return out

    review_col = find_optional_col(scene, ["review_id", "rally_id"])
    seg_col = find_optional_col(scene, ["camera_segment_id", "camera_segment_id_005B2"])
    status_col = find_optional_col(scene, ["scene_table_status_005C9B"])

    if not review_col or not seg_col or not status_col:
        return out

    for _, r in scene.iterrows():
        review_id = str(r[review_col])
        seg = int(to_num(pd.Series([r[seg_col]])).fillna(-1).iloc[0])
        out[(review_id, seg)] = str(r[status_col])

    return out


def camera_timeline_probe(timeline: pd.DataFrame, review_id: str, gap_start: int, gap_end: int):
    if timeline.empty:
        return {
            "timeline_available": 0,
            "timeline_frames": 0,
            "timeline_table_ok_ratio": "",
            "timeline_camera_segments": "",
        }

    frame_col = find_optional_col(timeline, ["frame_num", "frame", "frame_idx", "local_frame"])
    if not frame_col:
        return {
            "timeline_available": 0,
            "timeline_frames": 0,
            "timeline_table_ok_ratio": "",
            "timeline_camera_segments": "",
        }

    review_col = find_optional_col(timeline, ["review_id", "rally_id", "clip_id", "sequence_key"])

    t = timeline.copy()
    t[frame_col] = to_num(t[frame_col]).fillna(-999999).astype(int)
    t = t[t[frame_col].between(int(gap_start), int(gap_end))].copy()

    if review_col:
        if review_col == "review_id":
            t = t[t[review_col].astype(str).eq(str(review_id))].copy()
        else:
            t = t[t[review_col].astype(str).str.contains(str(review_id), regex=False, na=False)].copy()

    if t.empty:
        return {
            "timeline_available": 1,
            "timeline_frames": 0,
            "timeline_table_ok_ratio": "",
            "timeline_camera_segments": "",
        }

    table_ok_col = find_optional_col(
        t,
        [
            "table_ok",
            "table_ok_005B2",
            "calib_ok",
            "calib_ok_005B2",
            "table_visible_ok",
            "mask_ok",
        ],
    )

    if not table_ok_col:
        table_ok_col = find_by_fragment(t, ["table_ok", "calib_ok", "table_visible", "mask_ok"])

    if table_ok_col:
        ok = to_num(t[table_ok_col]).fillna(0)
        table_ok_ratio = round(float(ok.mean()), 4)
    else:
        table_ok_ratio = ""

    seg_col = find_optional_col(t, ["camera_segment_id", "camera_segment_id_005B2"])

    if seg_col:
        segs = sorted(set(to_num(t[seg_col]).dropna().astype(int).tolist()))
        segs_str = ",".join(map(str, segs[:12]))
    else:
        segs_str = ""

    return {
        "timeline_available": 1,
        "timeline_frames": int(len(t)),
        "timeline_table_ok_ratio": table_ok_ratio,
        "timeline_camera_segments": segs_str,
    }


def triage_gap(gap_kind: str, scene_status: str, timeline_table_ok_ratio):
    metric_statuses = {"METRIC_TABLE_TRUSTED", "METRIC_TABLE_REVIEW"}

    timeline_known = str(timeline_table_ok_ratio) != ""

    if timeline_known:
        try:
            table_ok = float(timeline_table_ok_ratio)
        except Exception:
            table_ok = None
    else:
        table_ok = None

    if gap_kind == "FILTERED_LOW_SCORE":
        return (
            "RECOVERABLE_FILTERED_SOURCE",
            "points_source_exist_but_score_or_gate_rejected_them",
        )

    if gap_kind == "BRIDGE_TOO_STRICT_OR_SEGMENT_SPLIT":
        return (
            "RELINK_POLICY_TOO_STRICT",
            "safe_points_exist_but_relink_did_not_connect_them",
        )

    if gap_kind == "NO_SOURCE_POINT":
        if scene_status in metric_statuses:
            if table_ok is None or table_ok >= 0.50:
                return (
                    "LIVE_TABLE_NO_SOURCE_POINT",
                    "table_context_exists_but_selected_source_has_no_ball_point",
                )
            return (
                "LIKELY_CAMERA_OR_NON_TABLE_VIEW",
                "camera_timeline_says_table_not_reliably_visible",
            )

        if scene_status in {"PARTIAL_TABLE_MASK", "NO_TABLE_METRIC", ""}:
            if table_ok is not None and table_ok >= 0.50:
                return (
                    "TABLE_VISIBLE_BUT_NO_METRIC_SOURCE",
                    "table_may_be_visible_but_not_promoted_to_metric_status",
                )
            return (
                "NON_METRIC_OR_BROADCAST_VIEW",
                "no_trusted_metric_table_for_this_gap",
            )

    return (
        "UNKNOWN_GAP_STATE",
        "insufficient_context_to_classify_gap",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", default="runs/rally_ball_gap_audit_005D5/ball_tracking_gap_audit_005D5.csv")
    ap.add_argument("--scene-tables", default="runs/rally_scene_table_objects_005C9B/scene_table_objects_005C9B.csv")
    ap.add_argument("--camera-timeline", default="runs/rally_table_camera_005B2_pass33/table_camera_timeline_005B2.csv")
    ap.add_argument("--out-dir", default="runs/rally_ball_gap_triage_005D6A")
    args = ap.parse_args()

    root = Path.cwd()

    gaps_path = Path(args.gaps)
    scene_path = Path(args.scene_tables)
    timeline_path = Path(args.camera_timeline)
    out_dir = Path(args.out_dir)

    if not gaps_path.is_absolute():
        gaps_path = root / gaps_path
    if not scene_path.is_absolute():
        scene_path = root / scene_path
    if not timeline_path.is_absolute():
        timeline_path = root / timeline_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    gaps = pd.read_csv(gaps_path).fillna("")
    scene = pd.read_csv(scene_path).fillna("") if scene_path.is_file() else pd.DataFrame()
    timeline = pd.read_csv(timeline_path).fillna("") if timeline_path.is_file() else pd.DataFrame()

    status_map = scene_status_map(scene)

    rows = []

    for _, r in gaps.iterrows():
        review_id = str(r["review_id"])
        seg = int(to_num(pd.Series([r["camera_segment_id_005B2"]])).fillna(-1).iloc[0])

        gap_start = int(to_num(pd.Series([r["gap_start"]])).fillna(-1).iloc[0])
        gap_end = int(to_num(pd.Series([r["gap_end"]])).fillna(-1).iloc[0])
        gap_kind = str(r["gap_kind"])

        scene_status = status_map.get((review_id, seg), "")

        tinfo = camera_timeline_probe(
            timeline=timeline,
            review_id=review_id,
            gap_start=gap_start,
            gap_end=gap_end,
        )

        triage_kind, triage_action = triage_gap(
            gap_kind=gap_kind,
            scene_status=scene_status,
            timeline_table_ok_ratio=tinfo["timeline_table_ok_ratio"],
        )

        out = r.to_dict()
        out.update({
            "scene_table_status_005C9B": scene_status,
            **tinfo,
            "gap_triage_kind_005D6A": triage_kind,
            "gap_triage_action_005D6A": triage_action,
        })

        rows.append(out)

    triage = pd.DataFrame(rows)

    out_csv = out_dir / "ball_tracking_gap_triage_005D6A.csv"
    out_top = out_dir / "ball_tracking_gap_triage_top_005D6A.csv"
    out_json = out_dir / "ball_tracking_gap_triage_summary_005D6A.json"

    triage.to_csv(out_csv, index=False, encoding="utf-8")

    if not triage.empty:
        top = triage.sort_values(["gap_len", "source_points_in_gap"], ascending=[False, False]).head(80).copy()
    else:
        top = pd.DataFrame()

    top.to_csv(out_top, index=False, encoding="utf-8")

    triage_counts = triage["gap_triage_kind_005D6A"].astype(str).value_counts().to_dict() if not triage.empty else {}
    scene_counts = triage["scene_table_status_005C9B"].astype(str).value_counts().to_dict() if not triage.empty else {}

    summary = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "classify_tracking_gaps_as_live_table_source_missing_vs_non_metric_camera_view_vs_filtered_source",
        "gaps": str(gaps_path),
        "scene_tables": str(scene_path),
        "camera_timeline": str(timeline_path),
        "gaps_total": int(len(triage)),
        "triage_counts": {str(k): int(v) for k, v in triage_counts.items()},
        "scene_status_counts": {str(k): int(v) for k, v in scene_counts.items()},
        "gap_len_median_by_triage": {
            str(k): round(float(g["gap_len"].median()), 4)
            for k, g in triage.groupby("gap_triage_kind_005D6A", dropna=False)
        } if not triage.empty else {},
        "gap_len_max_by_triage": {
            str(k): int(g["gap_len"].max())
            for k, g in triage.groupby("gap_triage_kind_005D6A", dropna=False)
        } if not triage.empty else {},
        "outputs": {
            "triage_csv": str(out_csv),
            "top_csv": str(out_top),
        },
        "next": "If LIVE_TABLE_NO_SOURCE_POINT dominates, audit candidate reservoir and improve detector/ranker. If NON_METRIC_OR_BROADCAST_VIEW dominates, improve live-camera/table segmentation before tracking."
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("005D6A status=OK")
    print("gaps_total=", summary["gaps_total"])
    print("triage_counts=", json.dumps(summary["triage_counts"], ensure_ascii=False))
    print("scene_status_counts=", json.dumps(summary["scene_status_counts"], ensure_ascii=False))
    print("gap_len_median_by_triage=", json.dumps(summary["gap_len_median_by_triage"], ensure_ascii=False))
    print("gap_len_max_by_triage=", json.dumps(summary["gap_len_max_by_triage"], ensure_ascii=False))
    print("wrote", out_csv)
    print("wrote", out_top)
    print("wrote", out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
