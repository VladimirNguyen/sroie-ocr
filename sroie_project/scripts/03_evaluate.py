"""Bước 3: so sánh baseline và cải tiến trên CÙNG tập ảnh.

Các cấu hình được so sánh:
  <variant>/baseline, <variant>/improved   cho từng variant tiền xử lý đã chạy OCR
  oracle/<extractor>                       chạy luật trên văn bản NHÃN GỐC (trần của luật, không có lỗi OCR)

Ví dụ: python scripts/03_evaluate.py --split val --variants basic,improved
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import FIELDS, RESULTS_DIR
from src.data_loader import get_splits
from src.evaluate import compute_metrics, make_record
from src.extractors import EXTRACTORS
from src.pipeline import load_ocr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--variants", default="basic,improved")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    variants = args.variants.split(",")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = get_splits()[args.split][:args.limit]
    cached = {v: {s.id: load_ocr(args.engine, v, s.id) for s in samples} for v in variants}
    # chỉ giữ ảnh có OCR ở MỌI variant để so sánh công bằng
    samples = [s for s in samples if all(cached[v][s.id] for v in variants)]
    if not samples:
        sys.exit("Chưa có cache OCR. Chạy scripts/02_run_ocr.py trước (cho từng variant).")
    print(f"Đánh giá trên {len(samples)} ảnh ({args.split}).")

    table, per_sample = [], []
    configs = [(v, e) for v in variants for e in EXTRACTORS] + [("oracle", e) for e in EXTRACTORS]
    for v, e in configs:
        recs, secs = [], []
        for s in samples:
            gt = s.labels()
            if v == "oracle":
                lines = s.gt_lines()
                if not lines:
                    continue
            else:
                o = cached[v][s.id]
                lines, secs = [l.text for l in o["lines"]], secs + [o["seconds"]]
            pred = EXTRACTORS[e](lines)
            rec = make_record(s.id, gt, pred, lines)
            recs.append(rec)
            per_sample.append({"config": f"{v}/{e}", "id": s.id,
                               **{f"{f}_gt": gt[f] for f in FIELDS},
                               **{f"{f}_pred": pred[f] for f in FIELDS},
                               **{f"{f}_ok": rec["ok"][f] for f in FIELDS},
                               **{f"{f}_err": rec["error_type"][f] for f in FIELDS}})
        if not recs:
            continue
        m = compute_metrics(recs)
        row = {"config": f"{v}/{e}", "n": len(recs)}
        for f in FIELDS:
            row[f"EM_{f}"] = round(m[f]["exact_match"], 3)
        row["EM_mean"] = round(m["mean"]["exact_match"], 3)
        row["F1_mean"] = round(m["mean"]["f1"], 3)
        row["sim_company"] = round(m["company"]["avg_similarity"], 3)
        row["sim_address"] = round(m["address"]["avg_similarity"], 3)
        row["sec/img"] = round(sum(secs) / len(secs), 2) if secs else None
        row["ocr_err"] = sum(m[f]["ocr_error"] for f in FIELDS)
        row["extract_err"] = sum(m[f]["extraction_error"] for f in FIELDS)
        table.append(row)

    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    tag = f"{args.engine}_{args.split}"
    df.to_csv(RESULTS_DIR / f"metrics_{tag}.csv", index=False)
    pd.DataFrame(per_sample).to_csv(RESULTS_DIR / f"per_sample_{tag}.csv", index=False)
    (RESULTS_DIR / f"metrics_{tag}.md").write_text(df.to_markdown(index=False) if _has_tabulate() else df.to_csv(index=False),
                                                   encoding="utf-8")
    print(f"\nĐã lưu vào {RESULTS_DIR}")
    print("Gợi ý đọc bảng: EM = exact match sau chuẩn hóa; ocr_err = sai do đọc chữ; extract_err = sai do luật tìm trường.")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
