"""Đánh giá: exact match sau chuẩn hóa, precision/recall/F1 theo trường, phân loại nguồn lỗi."""
from __future__ import annotations

from difflib import SequenceMatcher

from src.config import FIELDS
from src.normalize import NORMALIZERS, find_amounts_in_line, find_dates, normalize_text


def is_correct(field: str, pred: str | None, gt: str | None) -> bool:
    if not pred or not gt:
        return False
    norm = NORMALIZERS[field]
    return norm(pred) == norm(gt)


def similarity(field: str, pred: str | None, gt: str | None) -> float:
    if not pred or not gt:
        return 0.0
    norm = NORMALIZERS[field]
    return SequenceMatcher(None, norm(pred), norm(gt)).ratio()


def gt_in_text(field: str, gt: str, lines: list[str]) -> bool:
    """Giá trị đúng có xuất hiện trong văn bản OCR không? (để phân biệt sai do đọc chữ / sai do tìm trường)."""
    if field in ("company", "address"):
        blob = "".join(normalize_text(l).replace(" ", "") for l in lines)
        return normalize_text(gt).replace(" ", "") in blob
    text = "\n".join(lines)
    if field == "date":
        target = NORMALIZERS["date"](gt)
        return any(iso == target for _, iso in find_dates(text))
    target = NORMALIZERS["total"](gt)
    return any(f"{v:.2f}" == target for l in lines for _, v in find_amounts_in_line(l))


def classify(field: str, pred, gt, lines) -> str:
    if not gt:
        return "no_gt"
    if is_correct(field, pred, gt):
        return "ok"
    return "extraction_error" if gt_in_text(field, gt, lines) else "ocr_error"


def make_record(sample_id: str, gt: dict, pred: dict, lines: list[str]) -> dict:
    rec = {"id": sample_id, "gt": gt, "pred": pred, "ok": {}, "error_type": {}, "sim": {}}
    for f in FIELDS:
        rec["ok"][f] = is_correct(f, pred.get(f), gt.get(f))
        rec["error_type"][f] = classify(f, pred.get(f), gt.get(f), lines)
        rec["sim"][f] = similarity(f, pred.get(f), gt.get(f))
    return rec


def compute_metrics(records: list[dict]) -> dict:
    out = {}
    for f in FIELDS:
        n = sum(1 for r in records if r["gt"].get(f))
        n_pred = sum(1 for r in records if r["gt"].get(f) and r["pred"].get(f))
        correct = sum(1 for r in records if r["gt"].get(f) and r["ok"][f])
        prec = correct / n_pred if n_pred else 0.0
        rec = correct / n if n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        sims = [r["sim"][f] for r in records if r["gt"].get(f)]
        out[f] = {"n": n, "exact_match": rec, "precision": prec, "recall": rec, "f1": f1,
                  "avg_similarity": sum(sims) / len(sims) if sims else 0.0,
                  "ocr_error": sum(1 for r in records if r["error_type"][f] == "ocr_error"),
                  "extraction_error": sum(1 for r in records if r["error_type"][f] == "extraction_error")}
    out["mean"] = {
        "exact_match": sum(out[f]["exact_match"] for f in FIELDS) / len(FIELDS),
        "f1": sum(out[f]["f1"] for f in FIELDS) / len(FIELDS),
    }
    return out
