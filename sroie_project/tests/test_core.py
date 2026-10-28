import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import classify, compute_metrics, is_correct, make_record
from src.extractors import baseline_extract, improved_extract
from src.normalize import find_amounts_in_line, find_dates, normalize_date, normalize_total

RECEIPT = """BOOK TA .K (TAMAN DAYA) SDN BHD
(789417-W)
NO.5, JALAN SAGU 18,
TAMAN DAYA,
81100 JOHOR BAHRU,
JOHOR.
TEL: 07-3558788
GST ID: 000123456
TAX INVOICE
DATE: 25/12/2018 12:34
BOOK A 10.00
BOOK B 7.00
SUBTOTAL 17.00
GST 6% 1.02
TOTAL 18.02
ROUNDING 0.03
ROUNDED TOTAL 18.05
CASH 50.00
CHANGE 31.95""".splitlines()


def test_dates():
    assert normalize_date("25/12/2018") == "2018-12-25"
    assert normalize_date("25-DEC-18") == "2018-12-25"
    assert normalize_date("15 JAN 2019") == "2019-01-15"
    assert normalize_date("2018-03-26 10:00") == "2018-03-26"
    assert normalize_date("Dec 5, 2018") == "2018-12-05"
    assert find_dates("Tel 03-1234-5678") == []


def test_amounts():
    assert find_amounts_in_line("TOTAL RM 1,234.50")[0][1] == 1234.50
    assert find_amounts_in_line("TOTAL 1O.5O")[0][1] == 10.50      # O -> 0
    assert find_amounts_in_line("25.12.2018") == []                 # ngày không phải tiền
    assert normalize_total("RM 9.00") == "9.00"


def test_improved_extract():
    r = improved_extract(RECEIPT)
    assert r["company"] == "BOOK TA .K (TAMAN DAYA) SDN BHD"
    assert r["date"] == "25/12/2018"
    assert r["total"] == "18.05"
    assert r["address"] == "NO.5, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU,"


def test_baseline_is_weaker():
    r = baseline_extract(RECEIPT)
    assert r["total"] == "50.00"  # lấy nhầm số tiền khách đưa -> lý do cần luật từ khóa


def test_missing_fields_return_none():
    r = improved_extract(["hello world", "no numbers here"])
    assert r["date"] is None and r["total"] is None


def test_empty_input():
    r = improved_extract([])
    assert all(v is None for v in r.values())


def test_error_classification():
    gt = {"company": "A SDN BHD", "date": "25/12/2018", "address": "ZZ QQ 999", "total": "18.05"}
    pred = {"company": "A SDN BHD", "date": "25/12/2018", "address": None, "total": "50.00"}
    rec = make_record("t", gt, pred, RECEIPT)
    assert rec["ok"]["total"] is False
    assert rec["error_type"]["total"] == "extraction_error"   # 18.05 có trong văn bản OCR
    assert rec["error_type"]["address"] == "ocr_error"        # địa chỉ nhãn không có trong văn bản
    assert is_correct("date", "25-12-2018", "25/12/2018")
    m = compute_metrics([rec])
    assert m["total"]["exact_match"] == 0.0 and m["date"]["exact_match"] == 1.0
