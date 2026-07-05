from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PATCH_ID = "006A2_supervised_labels_build"


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    keys = []
    seen = set()

    preferred = [
        "label_uid",
        "sample_id",
        "video_path",
        "video_frame",
        "label_task",
        "label_source",
        "visible",
        "x",
        "y",
        "label",
        "usable_for_visibility",
        "usable_for_position",
        "training_weight",
        "notes",
    ]

    for k in preferred:
        seen.add(k)
        keys.append(k)

    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def norm_str(v):
    return "" if v is None else str(v).strip()


def norm_visible(v):
    s = norm_str(v).lower()
    if s in {"1", "true", "yes", "y", "visible", "vis"}:
        return "1"
    if s in {"0", "false", "no", "n", "invisible", "not_visible", "missing"}:
        return "0"
    return ""


def norm_float(v):
    s = norm_str(v).replace(",", ".")
    if not s:
        return ""
    try:
        return str(round(float(s), 3))
    except Exception:
        return ""


def norm_frame(v):
    s = norm_str(v)
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return ""


def first_nonempty(row, keys, default=""):
    for k in keys:
        v = norm_str(row.get(k, ""))
        if v:
            return v
    return default


def convert_rows(rows, task, source_name):
    out = []

    for idx, r in enumerate(rows, 1):
        frame = norm_frame(first_nonempty(r, ["video_frame", "frame", "frame_id"]))
        visible = norm_visible(first_nonempty(r, ["visible", "is_visible"]))
        x = norm_float(first_nonempty(r, ["x", "ball_x", "cx", "center_x"]))
        y = norm_float(first_nonempty(r, ["y", "ball_y", "cy", "center_y"]))
        label = first_nonempty(r, ["label", "class", "status"])

        sample_id = first_nonempty(r, ["sample_id", "clip_id", "video_id", "sequence_key"], "human_006A")
        video_path = first_nonempty(r, ["video_path", "video", "source_video"], "")

        clicked = bool(x and y)
        usable_for_visibility = bool(frame and visible in {"0", "1"})
        usable_for_position = bool(frame and visible == "1" and clicked)

        notes = []
        if not frame:
            notes.append("missing_frame")
        if visible not in {"0", "1"}:
            notes.append("missing_visible")
        if visible == "1" and not clicked and task == "click":
            notes.append("visible_without_xy")
        if visible == "0" and clicked:
            notes.append("invisible_with_xy_check")

        out.append({
            "label_uid": f"{PATCH_ID}_{task}_{idx:04d}",
            "sample_id": sample_id,
            "video_path": video_path,
            "video_frame": frame,
            "label_task": task,
            "label_source": source_name,
            "visible": visible,
            "x": x,
            "y": y,
            "label": label,
            "usable_for_visibility": "1" if usable_for_visibility else "0",
            "usable_for_position": "1" if usable_for_position else "0",
            "training_weight": "1.0",
            "notes": "|".join(notes),
            **{f"raw_{k}": v for k, v in r.items()},
        })

    return out


def summarize(rows):
    return {
        "row_count": len(rows),
        "usable_for_visibility": sum(1 for r in rows if r.get("usable_for_visibility") == "1"),
        "usable_for_position": sum(1 for r in rows if r.get("usable_for_position") == "1"),
        "visible_count": sum(1 for r in rows if r.get("visible") == "1"),
        "invisible_count": sum(1 for r in rows if r.get("visible") == "0"),
        "with_notes": sum(1 for r in rows if r.get("notes")),
        "frame_count_unique": len({r.get("video_frame") for r in rows if r.get("video_frame")}),
    }


def main():
    out_dir = Path("runs/006A_human_frame_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    visibility_csv = out_dir / "006A_visibility_labels.csv"
    click_csv = out_dir / "006A_click_labels.csv"

    visibility_rows = read_csv(visibility_csv)
    click_rows = read_csv(click_csv)

    supervised = []
    supervised.extend(convert_rows(visibility_rows, "visibility", visibility_csv.name))
    supervised.extend(convert_rows(click_rows, "click_position", click_csv.name))

    visibility_training = [
        r for r in supervised
        if r.get("usable_for_visibility") == "1"
    ]

    position_training = [
        r for r in supervised
        if r.get("usable_for_position") == "1"
    ]

    supervised_path = out_dir / "006A2_supervised_labels.csv"
    visibility_path = out_dir / "006A2_visibility_training_rows.csv"
    position_path = out_dir / "006A2_position_training_rows.csv"
    summary_path = out_dir / "006A2_supervised_labels_summary.json"

    write_csv(supervised_path, supervised)
    write_csv(visibility_path, visibility_training)
    write_csv(position_path, position_training)

    summary = {
        "patch": PATCH_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "visibility_csv": str(visibility_csv),
            "click_csv": str(click_csv),
            "visibility_rows": len(visibility_rows),
            "click_rows": len(click_rows),
        },
        "outputs": {
            "supervised_labels": str(supervised_path),
            "visibility_training_rows": str(visibility_path),
            "position_training_rows": str(position_path),
        },
        "supervised": summarize(supervised),
        "visibility_training": summarize(visibility_training),
        "position_training": summarize(position_training),
        "ok_for_next": len(visibility_training) >= 30 and len(position_training) >= 10,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006A2")
    print(f"supervised = {supervised_path}")
    print(f"visibility = {visibility_path}")
    print(f"position   = {position_path}")
    print(f"summary    = {summary_path}")
    print("")
    print("SUPERVISED")
    print(summary["supervised"])
    print("")
    print("VISIBILITY TRAINING")
    print(summary["visibility_training"])
    print("")
    print("POSITION TRAINING")
    print(summary["position_training"])
    print("")
    print(f"ok_for_next={summary['ok_for_next']}")


if __name__ == "__main__":
    main()
