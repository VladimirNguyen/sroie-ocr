"""Tiền xử lý ảnh hóa đơn trước khi OCR.

variant:
  raw      : giữ nguyên ảnh gốc
  basic    : xám + phóng to (nếu ảnh nhỏ)
  improved : xám + phóng to + khử nhiễu nhẹ + CLAHE + chỉnh nghiêng + ngưỡng thích nghi + viền trắng
"""
from __future__ import annotations

import cv2
import numpy as np

VARIANTS = ("raw", "basic", "improved")


def read_image(path) -> np.ndarray:
    """Đọc ảnh BGR (an toàn với đường dẫn có dấu / Unicode trên Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _upscale(gray: np.ndarray, target_w: int = 1000, max_side: int = 4000) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = target_w / w if w < target_w else 1.0
    scale = min(scale, max_side / max(h, w))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def estimate_skew(gray: np.ndarray, max_angle: float = 8.0, step: float = 0.5) -> float:
    """Ước lượng góc nghiêng bằng projection profile trên ảnh thu nhỏ (độ)."""
    h, w = gray.shape[:2]
    s = 400 / max(w, 1)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hh, ww = bw.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
        rot = cv2.warpAffine(bw, M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0)
        prof = rot.sum(axis=1).astype(np.float64)
        score = float(np.sum(np.diff(prof) ** 2))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def deskew(gray: np.ndarray) -> np.ndarray:
    angle = estimate_skew(gray)
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def preprocess(img: np.ndarray, variant: str = "improved") -> np.ndarray:
    if variant == "raw":
        return img
    gray = _upscale(_to_gray(img))
    if variant == "basic":
        return gray
    if variant != "improved":
        raise ValueError(f"variant không hợp lệ: {variant} (chọn {VARIANTS})")
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = deskew(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 15)
    return cv2.copyMakeBorder(bw, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
