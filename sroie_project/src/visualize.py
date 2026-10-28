"""Vẽ khung OCR lên ảnh."""
import cv2
import numpy as np


def draw_boxes(img: np.ndarray, lines) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    thick = max(1, out.shape[1] // 600)
    for i, l in enumerate(lines):
        x1, y1, x2, y2 = [int(v) for v in l.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 160, 0), thick)
        cv2.putText(out, str(i), (max(0, x1 - 4), max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, (0, 0, 220), thick)
    return out
