from __future__ import annotations

import csv
import json
from pathlib import Path


PATCH_ID = "006A1_human_labels_audit"


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    visible = [r for r in rows if r.get("visible") == "1"]
    invisible = [r for r in rows if r.get("visible") == "0"]
    unsure = [r for r in rows if r.get("label") == "unsure"]
    clicked = [r for r in rows if r.get("x") and r.get("y")]

    frames = []
    for r in rows:
        v = str(r.get("video_frame", "")).strip()
        if not v:
            continue
        try:
            frames.append(int(float(v)))
        except Exception:
            pass

    return {
        "row_count": len(rows),
        "visible_count": len(visible),
        "invisible_count": len(invisible),
        "unsure_count": len(unsure),
        "clicked_count": len(clicked),
        "frame_count_unique": len(set(frames)),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "frames": sorted(set(frames)),
    }


def main():
    out_dir = Path("runs/006A_human_frame_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    visibility_csv = out_dir / "006A_visibility_labels.csv"
    click_csv = out_dir / "006A_click_labels.csv"

    visibility_rows = read_csv(visibility_csv)
    click_rows = read_csv(click_csv)

    summary = {
        "patch": PATCH_ID,
        "visibility_csv": str(visibility_csv),
        "click_csv": str(click_csv),
        "visibility": summarize(visibility_rows),
        "click": summarize(click_rows),
        "ok_for_next": len(visibility_rows) > 0 and len(click_rows) > 0,
    }

    summary_path = out_dir / "006A1_human_labels_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"PATCH {PATCH_ID}")
    print("=" * 72)
    print("")
    print("OK 006A1")
    print(f"summary = {summary_path}")
    print("")
    print("VISIBILITY")
    print(summary["visibility"])
    print("")
    print("CLICK")
    print(summary["click"])
    print("")
    print(f"ok_for_next={summary['ok_for_next']}")


if __name__ == "__main__":
    main()
