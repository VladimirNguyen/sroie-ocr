"""Chuẩn hóa chuỗi / ngày / số tiền. Chỉ dùng thư viện chuẩn (dễ test)."""
from __future__ import annotations

import re
from datetime import date as _date


# ---------------------------------------------------------------- văn bản
def normalize_text(s: str | None) -> str:
    """Viết hoa, bỏ dấu câu, gộp khoảng trắng -> so khớp công ty / địa chỉ."""
    if s is None:
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


# ---------------------------------------------------------------- ngày
_MON = (
    r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|"
    r"AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?"
)
_MON_NUM = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

P_YMD = re.compile(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)")
P_DMY = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)")
P_DMONY = re.compile(rf"(?<!\d)(\d{{1,2}})[\s\-/.]*({_MON})[\s\-/.,']*(\d{{4}}|\d{{2}})(?!\d)", re.I)
P_MONDY = re.compile(rf"(?<![A-Za-z])({_MON})[\s\-/.]*(\d{{1,2}})[\s,\-/.]+(\d{{4}})(?!\d)", re.I)


def _make_date(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000 if y < 70 else 1900
    try:
        return _date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[str, str]]:
    """Trả về [(chuỗi gốc, ISO yyyy-mm-dd)] theo thứ tự xuất hiện, bỏ ngày không hợp lệ."""
    found: list[tuple[int, int, str, str]] = []
    taken: list[tuple[int, int]] = []

    def overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in taken)

    def add(m, iso):
        if iso and not overlaps(m.start(), m.end()):
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), m.group(0), iso))

    for m in P_YMD.finditer(text):
        add(m, _make_date(int(m[1]), int(m[2]), int(m[3])))
    for m in P_DMY.finditer(text):
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if mo > 12 and d <= 12:  # có thể là mm/dd/yyyy
            d, mo = mo, d
        add(m, _make_date(y, mo, d))
    for m in P_DMONY.finditer(text):
        add(m, _make_date(int(m[3]), _MON_NUM[m[2][:3].upper()], int(m[1])))
    for m in P_MONDY.finditer(text):
        add(m, _make_date(int(m[3]), _MON_NUM[m[1][:3].upper()], int(m[2])))
    found.sort()
    return [(raw, iso) for _, _, raw, iso in found]


def normalize_date(s: str | None) -> str:
    if not s:
        return ""
    ds = find_dates(s)
    return ds[0][1] if ds else normalize_text(s)


# ---------------------------------------------------------------- tiền
AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)\s?\.\s?(\d{2})(?!\d|\.\d)")
_NUMTOKEN = re.compile(r"[0-9OoIl.,]{3,}")


def _fix_ocr_digits(line: str) -> str:
    """Sửa lỗi OCR hay gặp trong token số: O->0, l/I->1 (chỉ khi token có chữ số và dấu '.')."""
    def fix(m):
        tok = m.group(0)
        if "." in tok and any(c.isdigit() for c in tok):
            return tok.translate(str.maketrans("OoIl", "0011"))
        return tok
    return _NUMTOKEN.sub(fix, line)


def find_amounts_in_line(line: str) -> list[tuple[str, float]]:
    """[(chuỗi 'xxx.yy', giá trị float)] của mọi số tiền dạng có 2 chữ số thập phân."""
    out = []
    for m in AMOUNT_RE.finditer(_fix_ocr_digits(line)):
        raw = f"{m.group(1).replace(',', '')}.{m.group(2)}"
        out.append((raw, float(raw)))
    return out


def parse_amount(s: str | None) -> float | None:
    if s is None:
        return None
    t = re.sub(r"(?i)\bRM\b|MYR|\$|\s", "", s)
    if re.fullmatch(r"\d{1,3}(,\d{3})*(\.\d+)?|\d+(\.\d+)?", t):
        return float(t.replace(",", ""))
    if re.fullmatch(r"\d+,\d{1,2}", t):
        return float(t.replace(",", "."))
    return None


def normalize_total(s: str | None) -> str:
    if not s:
        return ""
    am = find_amounts_in_line(s)
    if am:
        return f"{am[0][1]:.2f}"
    v = parse_amount(s)
    return f"{v:.2f}" if v is not None else normalize_text(s)


NORMALIZERS = {
    "company": normalize_text,
    "address": normalize_text,
    "date": normalize_date,
    "total": normalize_total,
}
