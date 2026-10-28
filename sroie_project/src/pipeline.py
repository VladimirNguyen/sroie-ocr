"""Ghép tiền xử lý -> OCR -> trích trường; cache kết quả OCR ra JSON để khỏi chạy lại."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import OCR_CACHE_DIR
from src.extractors import EXTRACTORS
from src.ocr_engine import OCRLine
from src.preprocess import preprocess


def run_ocr(img_bgr, engine, variant: str):
    """Trả về (danh sách OCRLine, ảnh đã xử lý, số giây OCR+tiền xử lý)."""
    t0 = time.perf_counter()
    proc = preprocess(img_bgr, variant)
    lines = engine.run(proc)
    return lines, proc, time.perf_counter() - t0


def extract(lines: list[OCRLine] | list[str], extractor: str = "improved") -> dict:
    texts = [l.text if isinstance(l, OCRLine) else l for l in lines]
    return EXTRACTORS[extractor](texts)


def cache_path(engine_name: str, variant: str, sample_id: str) -> Path:
    return OCR_CACHE_DIR / f"{engine_name}_{variant}" / f"{sample_id}.json"


def save_ocr(engine_name, variant, sample_id, lines, seconds) -> None:
    p = cache_path(engine_name, variant, sample_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"id": sample_id, "engine": engine_name, "variant": variant,
                             "seconds": round(seconds, 3), "lines": [l.to_dict() for l in lines]},
                            ensure_ascii=False), encoding="utf-8")


def load_ocr(engine_name, variant, sample_id) -> dict | None:
    p = cache_path(engine_name, variant, sample_id)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d["lines"] = [OCRLine.from_dict(x) for x in d["lines"]]
    return d
