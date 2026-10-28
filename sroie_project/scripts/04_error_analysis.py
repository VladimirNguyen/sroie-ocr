"""Bước 4: phân tích lỗi — thống kê loại lỗi và xuất các ca sai điển hình (kèm ảnh có khung OCR)."""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import ERR_DIR, FIELDS
from src.data_loader import get_splits
from src.evaluate import make_record
from src.extractors import EXTRACTORS
from src.pipeline import load_ocr
from src.preprocess import preprocess, read_image
from src.visualize import draw_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--variant", default="improved")
    ap.add_argument("--extractor", default="improved")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--per-group", type=int, default=2, help="số ca minh họa cho mỗi (trường, loại lỗi)")
    args = ap.parse_args()
    ERR_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_splits()[args.split][:args.limit]
    counts = {f: Counter() for f in FIELDS}
    groups = defaultdict(list)
    for s in samples:
        o = load_ocr(args.engine, args.variant, s.id)
        if not o:
            continue
        lines = [l.text for l in o["lines"]]
        rec = make_record(s.id, s.labels(), EXTRACTORS[args.extractor](lines), lines)
        for f in FIELDS:
            et = rec["error_type"][f]
            counts[f][et] += 1
            if et in ("ocr_error", "extraction_error"):
                groups[(f, et)].append((s, rec, o))
    if not any(counts[f] for f in FIELDS):
        sys.exit("Chưa có cache OCR phù hợp. Chạy scripts/02_run_ocr.py trước.")

    rng = random.Random(0)
    out = ["# Phân tích lỗi", f"\nCấu hình: {args.engine}/{args.variant}/{args.extractor}, tập {args.split}\n",
           "## Thống kê loại lỗi theo trường\n", "| Trường | ok | sai do OCR | sai do luật | no_gt |", "|---|---|---|---|---|"]
    for f in FIELDS:
        c = counts[f]
        out.append(f"| {f} | {c['ok']} | {c['ocr_error']} | {c['extraction_error']} | {c['no_gt']} |")
    out.append("\n## Các ca sai điển hình\n")
    shown = set()
    for (f, et), items in sorted(groups.items()):
        out.append(f"### {f} — {'sai do đọc chữ (OCR)' if et == 'ocr_error' else 'sai do tìm trường (luật)'}\n")
        for s, rec, o in rng.sample(items, min(args.per_group, len(items))):
            out += [f"- **{s.id}**", f"  - Nhãn đúng : `{rec['gt'][f]}`", f"  - Dự đoán   : `{rec['pred'][f]}`",
                    f"  - Ảnh: `{s.id}.jpg`  *(Nguyên nhân: điền nhận xét của bạn — mờ/nghiêng/nhiều giá tiền/tách dòng...)*"]
            if s.id not in shown:
                shown.add(s.id)
                proc = preprocess(read_image(s.image_path), args.variant)
                cv2.imwrite(str(ERR_DIR / f"{s.id}.jpg"), draw_boxes(proc, o["lines"]))
        out.append("")
    (ERR_DIR / "report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:12]))
    print(f"\nĐã lưu {ERR_DIR / 'report.md'} và {len(shown)} ảnh minh họa.")


if __name__ == "__main__":
    main()
