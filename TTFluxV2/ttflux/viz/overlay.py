from __future__ import annotations

import cv2
import numpy as np


def draw_crosshair(frame: np.ndarray, x: float, y: float, radius: int = 12, thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    x_i = int(round(x))
    y_i = int(round(y))
    cv2.line(out, (x_i - radius, y_i), (x_i + radius, y_i), (255, 255, 255), thickness)
    cv2.line(out, (x_i, y_i - radius), (x_i, y_i + radius), (255, 255, 255), thickness)
    cv2.circle(out, (x_i, y_i), radius, (0, 0, 0), max(1, thickness - 1))
    return out
