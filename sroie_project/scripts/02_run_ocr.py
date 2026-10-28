"""Bước 2: chạy OCR và lưu cache JSON (mỗi ảnh một file).

Ví dụ:
  python scripts/02_run_ocr.py --split val  --limit 200 --variant basic
  python scripts/02_run_ocr.py --split val  --limit 200 --variant improved
  python scripts/02_run_ocr.py --split test --limit 200 --variant improved   (chạy cuối cùng)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data_loader import get_splits
from src.ocr_engine import OCRNotAvailable, get_engine
from src.pipeline import cache_path, run_ocr, save_ocr
from src.preprocess import VARIANTS, read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--engine", choices=["tesseract", "easyocr"], default="tesseract")
    ap.add_argument("--variant", choices=VARIANTS, default="basic")
    ap.add_argument("--psm", type=int, default=6, help="chế độ phân trang Tesseract (6 hoặc 4)")
    ap.add_argument("--force", action="store_true", help="chạy lại dù đã có cache")
    args = ap.parse_args()

    samples = get_splits()[args.split][:args.limit]
    try:
        engine = get_engine(args.engine, psm=args.psm) if args.engine == "tesseract" else get_engine(args.engine)
    except OCRNotAvailable as e:
        sys.exit(f"[LỖI] {e}")

    done = 0
    for s in tqdm(samples, desc=f"OCR {args.engine}/{args.variant}/{args.split}"):
        if cache_path(args.engine, args.variant, s.id).exists() and not args.force:
            continue
        try:
            lines, _, sec = run_ocr(read_image(s.image_path), engine, args.variant)
        except Exception as e:  # ảnh hỏng không làm dừng cả lượt chạy
            print(f"\n[BỎ QUA] {s.id}: {e}")
            continue
        save_ocr(args.engine, args.variant, s.id, lines, sec)
        done += 1
    print(f"Xong: {done} ảnh mới, còn lại lấy từ cache.")


if __name__ == "__main__":
    main()
