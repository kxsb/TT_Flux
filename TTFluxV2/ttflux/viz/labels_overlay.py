from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COLOR_BY_TYPE = {
    "ball": (255, 190, 40),
    "ball_missing": (90, 90, 255),
    "table_corner": (0, 215, 255),
    "table_edge": (0, 180, 255),
    "player": (120, 255, 120),
    "scoreboard": (255, 150, 230),
    "note": (210, 210, 210),
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def draw_label_annotations(frame: np.ndarray, annotations: list[dict[str, Any]], frame_idx: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    frame_idx = int(frame_idx)

    rows = [a for a in annotations if int(a.get("frame", -999999)) == frame_idx]

    for ann in rows:
        if not ann.get("visible", True):
            continue

        x = _num(ann.get("x"))
        y = _num(ann.get("y"))
        if x is None or y is None:
            continue
        if not (-w * 0.2 <= x <= w * 1.2 and -h * 0.2 <= y <= h * 1.2):
            continue

        label_type = str(ann.get("label_type") or "note")
        color = COLOR_BY_TYPE.get(label_type, (220, 220, 220))
        pt = (int(round(x)), int(round(y)))

        cv2.circle(out, pt, 8, color, -1, cv2.LINE_AA)
        cv2.circle(out, pt, 10, (0, 0, 0), 2, cv2.LINE_AA)

        label = label_type
        obj = str(ann.get("object_id") or "").strip()
        if obj:
            label += f":{obj}"

        cv2.putText(out, label[:28], (pt[0] + 12, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label[:28], (pt[0] + 12, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    if rows:
        cv2.rectangle(out, (8, 38), (255, 64), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"manual labels: {len(rows)}",
            (16, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return out
