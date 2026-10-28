"""Bọc OCR có sẵn: Tesseract (mặc định) hoặc EasyOCR (tùy chọn). Kết quả là danh sách dòng chữ."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import TESSERACT_CMD


class OCRNotAvailable(RuntimeError):
    pass


@dataclass
class OCRLine:
    text: str
    bbox: tuple  # x1, y1, x2, y2
    conf: float

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": [int(v) for v in self.bbox], "conf": round(float(self.conf), 2)}

    @staticmethod
    def from_dict(d: dict) -> "OCRLine":
        return OCRLine(d["text"], tuple(d["bbox"]), d.get("conf", -1))


def group_words_into_lines(words: list[tuple]) -> list[OCRLine]:
    """words: (x1,y1,x2,y2,text,conf). Gom các từ cùng hàng thành dòng, sắp từ trên xuống, trái sang phải."""
    if not words:
        return []
    med_h = float(np.median([w[3] - w[1] for w in words])) or 10.0
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    groups: list[list[tuple]] = []
    cur_y = None
    for w in words:
        yc = (w[1] + w[3]) / 2
        if cur_y is not None and abs(yc - cur_y) < 0.6 * med_h:
            groups[-1].append(w)
            cur_y = float(np.mean([(g[1] + g[3]) / 2 for g in groups[-1]]))
        else:
            groups.append([w])
            cur_y = yc
    lines = []
    for g in groups:
        g.sort(key=lambda w: w[0])
        lines.append(OCRLine(
            text=" ".join(w[4] for w in g),
            bbox=(min(w[0] for w in g), min(w[1] for w in g), max(w[2] for w in g), max(w[3] for w in g)),
            conf=float(np.mean([w[5] for w in g])),
        ))
    return lines


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6):
        try:
            import pytesseract
        except ImportError as e:
            raise OCRNotAvailable("Chưa cài pytesseract: pip install pytesseract") from e
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise OCRNotAvailable(
                "Không tìm thấy chương trình Tesseract. Windows: cài từ "
                "https://github.com/UB-Mannheim/tesseract/wiki rồi đặt biến môi trường TESSERACT_CMD "
                "(ví dụ C:\\Program Files\\Tesseract-OCR\\tesseract.exe) hoặc thêm vào PATH."
            ) from e
        self.pt, self.lang, self.psm = pytesseract, lang, psm

    def run(self, img: np.ndarray) -> list[OCRLine]:
        import cv2
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        d = self.pt.image_to_data(arr, lang=self.lang, config=f"--oem 3 --psm {self.psm}",
                                  output_type=self.pt.Output.DICT)
        groups: dict[tuple, list[tuple]] = {}
        for i, t in enumerate(d["text"]):
            t = t.strip()
            try:
                conf = float(d["conf"][i])
            except (ValueError, TypeError):
                conf = -1
            if not t or conf < 0:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            groups.setdefault(key, []).append((x, y, x + w, y + h, t, conf))
        lines = []
        for words in groups.values():
            words.sort(key=lambda w: w[0])
            lines.append(OCRLine(
                " ".join(w[4] for w in words),
                (min(w[0] for w in words), min(w[1] for w in words),
                 max(w[2] for w in words), max(w[3] for w in words)),
                float(np.mean([w[5] for w in words])),
            ))
        lines.sort(key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
        return lines


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, langs=("en",)):
        try:
            import easyocr
        except ImportError as e:
            raise OCRNotAvailable("Chưa cài easyocr: pip install -r requirements-optional.txt") from e
        self.reader = easyocr.Reader(list(langs), gpu=False)

    def run(self, img: np.ndarray) -> list[OCRLine]:
        words = []
        for poly, text, conf in self.reader.readtext(img):
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            words.append((min(xs), min(ys), max(xs), max(ys), text.strip(), float(conf) * 100))
        return group_words_into_lines([w for w in words if w[4]])


def get_engine(name: str = "tesseract", **kw):
    if name == "tesseract":
        return TesseractEngine(**kw)
    if name == "easyocr":
        return EasyOCREngine()
    raise ValueError(f"OCR engine không hợp lệ: {name}")
