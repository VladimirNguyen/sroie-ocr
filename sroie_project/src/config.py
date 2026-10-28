"""Cấu hình chung của project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Thư mục chứa bộ SROIE đã giải nén. Có thể đổi bằng biến môi trường SROIE_DIR.
# Cấu trúc mong đợi:  <DATA_ROOT>/train/{img,box,entities}  và  <DATA_ROOT>/test/{img,box,entities}
DATA_ROOT = Path(os.environ.get("SROIE_DIR", ROOT / "data" / "raw" / "SROIE2019"))

OUTPUT_DIR = ROOT / "outputs"
OCR_CACHE_DIR = OUTPUT_DIR / "ocr_cache"
RESULTS_DIR = OUTPUT_DIR / "results"
EDA_DIR = OUTPUT_DIR / "eda"
ERR_DIR = OUTPUT_DIR / "error_analysis"

FIELDS = ("company", "date", "address", "total")

SEED = 42
VAL_RATIO = 0.2  # tỷ lệ validation tách từ train chính thức


def _find_tesseract():
    env = os.environ.get("TESSERACT_CMD")
    if env:
        return env
    for p in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(p).exists():
            return p
    return None  # trên Linux/macOS: tesseract nằm trong PATH


TESSERACT_CMD = _find_tesseract()
