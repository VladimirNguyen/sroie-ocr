"""Trích 4 trường (company, date, address, total) từ danh sách dòng văn bản OCR.

- baseline_extract : luật đơn giản (dòng đầu, regex ngày, số tiền lớn nhất)
- improved_extract : luật theo vị trí + từ khóa + chấm điểm ứng viên

Cả hai chỉ nhận list[str] nên chạy được cả trên văn bản OCR lẫn văn bản nhãn gốc (oracle).
Trường không tìm thấy trả về None.
"""
from __future__ import annotations

import re

from src.normalize import find_amounts_in_line, find_dates

# =====================================================================
# BASELINE
# =====================================================================
def baseline_extract(lines: list[str]) -> dict:
    lines = [l.strip() for l in lines if l and l.strip()]
    text = "\n".join(lines)

    company = lines[0] if lines else None

    m = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
    date = m.group(0) if m else None

    address = " ".join(lines[1:3]) if len(lines) > 1 else None

    amounts = [v for l in lines for _, v in find_amounts_in_line(l)]
    total = f"{max(amounts):.2f}" if amounts else None

    return {"company": company, "date": date, "address": address, "total": total}


# =====================================================================
# IMPROVED
# =====================================================================
def clean_lines(lines: list[str]) -> list[str]:
    """Chuẩn hóa chuỗi: bỏ ký tự rác do OCR, gộp khoảng trắng, bỏ dòng rỗng."""
    out = []
    for l in lines:
        l = re.sub(r"[|_~`]+", " ", l or "")
        l = re.sub(r"\s+", " ", l).strip()
        if l:
            out.append(l)
    return out


# ---------------- company
_HINT_RE = re.compile(
    r"\b(SDN|BHD|S/B|PLT|ENTERPRISE|ENTERPRISES|TRADING|RESTAURANT|RESTORAN|CAFE|STORE|MART|"
    r"LTD|INC|CORP|CORPORATION|HOLDINGS|BAKERY|MARKET|SHOP|PHARMACY|KEDAI|BOOK|BOOKS|HARDWARE|"
    r"SUPERMARKET|GIFT|TEXTILE|FOOD|MEDICAL|OPTICAL|STATIONERY|INDUSTRIES|CENTRE|CENTER|TRADERS)\b"
)
_NOT_COMPANY_RE = re.compile(
    r"\b(TAX INVOICE|INVOICE|RECEIPT|TEL|PHONE|FAX|GST|DATE|WELCOME|CASH SALE|CASHIER|BILL|COPY|"
    r"OFFICIAL|DUPLICATE|TERMINAL|ORDER)\b"
)
_REG_NO_RE = re.compile(
    r"^\(?\s*\d{4,}\s*-?\s*[A-Z]?\s*\)?$|\bCO\.?\s*(REG|NO)|\bREG(ISTRATION)?\s*NO|\bCOMPANY\s*NO|"
    r"\(\s*\d{5,}[\s-]*[A-Z]?\s*\)"
)
_REG_PAREN_RE = re.compile(r"\(\s*\d{4,}[\s-]*[A-Z]?\s*\)")
_COMPANY_SUFFIX_RE = re.compile(r"^\(?\s*(SDN\.?|S/B|BHD\.?|PLT)\b")


def _company_score(line: str, rank: int) -> float:
    up = line.upper()
    letters = sum(c.isalpha() for c in up)
    if letters < 3:
        return -99.0
    digits = sum(c.isdigit() for c in up)
    score = -0.5 * rank
    if _HINT_RE.search(up):
        score += 3
    if sum(c.isupper() for c in line) / letters > 0.8:
        score += 1
    if digits / max(1, len(up)) > 0.25:
        score -= 3
    if _NOT_COMPANY_RE.search(up):
        score -= 5
    if _REG_NO_RE.search(up):
        score -= 5
    return score


def _company(lines: list[str]) -> tuple[str | None, int]:
    if not lines:
        return None, -1
    best, best_s = 0, -1e9
    for i, l in enumerate(lines[:8]):
        s = _company_score(l, i)
        if s > best_s:
            best, best_s = i, s
    idx = best
    name = lines[best]
    if best_s < -3:  # không dòng nào đáng tin -> lấy dòng đầu
        idx, name = 0, lines[0]
    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
    if _COMPANY_SUFFIX_RE.match(nxt.upper()) and not re.search(r"\b(SDN|BHD|S/B|PLT)\b", name.upper()):
        name = f"{name} {nxt}"
        idx += 1
    name = _REG_PAREN_RE.sub("", name).strip(" ,-")
    return name or None, idx


# ---------------- date
def _date(lines: list[str]) -> str | None:
    cands = []  # (ưu tiên, thứ tự, raw)
    for i, l in enumerate(lines):
        for raw, _iso in find_dates(l):
            prio = 0 if re.search(r"\bDATE\b|\bDT\b", l.upper()) else 1
            cands.append((prio, i, raw))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


# ---------------- address
_STOP_RE = re.compile(
    r"\b(TEL|PHONE|FAX|GST|TAX INVOICE|INVOICE|RECEIPT|DATE|CASHIER|WWW|EMAIL|E-MAIL|BILL|DOC|SLIP|TABLE|CASH SALE)\b|@|\.COM"
)
_PHONE_ONLY_RE = re.compile(r"^\+?[\d\s\-()]{8,}$")
_POSTCODE_RE = re.compile(r"\b\d{5}\b")
_STATE_END_RE = re.compile(
    r"(JOHOR|SELANGOR|KUALA LUMPUR|PENANG|PULAU PINANG|PERAK|KEDAH|KELANTAN|TERENGGANU|PAHANG|MELAKA|MALACCA|"
    r"NEGERI SEMBILAN|SABAH|SARAWAK|PUTRAJAYA|LABUAN|PERLIS|MALAYSIA)\W*$"
)


def _address(lines: list[str], c_idx: int) -> str | None:
    if c_idx < 0:
        return None
    parts: list[str] = []
    for l in lines[c_idx + 1:c_idx + 8]:
        up = l.upper()
        if _REG_NO_RE.search(up):
            continue
        if _PHONE_ONLY_RE.match(l):
            break
        m = _STOP_RE.search(up)
        if m:
            prefix = l[:m.start()].strip(" ,:;-")
            if prefix and re.search(r"[A-Za-z]", prefix):
                parts.append(prefix)
            break
        parts.append(l)
        if _POSTCODE_RE.search(up) or _STATE_END_RE.search(up):
            break
        if len(parts) >= 4:
            break
    return " ".join(parts) if parts else None


# ---------------- total
_TOTAL_TIERS = [
    re.compile(r"ROUNDED|\bROUND\s*(TOTAL|AMT|AMOUNT)"),
    re.compile(r"GRAND\s*TOTAL|AMOUNT\s*DUE|TOTAL\s*DUE|NETT?\s*(TOTAL|AMOUNT|AMT)|AMOUNT\s*PAYABLE|"
               r"TOTAL\s*(AMOUNT|AMT|PAYABLE)|BALANCE\s*DUE"),
    re.compile(r"\bTOTAL\b|\bTTL\b|\bTOTA1\b"),
]
_TOTAL_EXCLUDE = re.compile(
    r"SUB\s*-?\s*TOTAL|TOTAL\s*(QTY|ITEM|GST|TAX|SAVING|DISC|POINT|EXCL|BEFORE)|\bQTY\b|\bGST\b|\bTAX\b|"
    r"DISCOUNT|SAVING|EXCL|POINT|CHANGE|ROUNDING|\bADJ"
)
_PAYMENT_RE = re.compile(r"CASH|CHANGE|TENDER|PAID|VISA|CARD|MASTER|DEBIT|CREDIT|PAYMENT")


def _total(lines: list[str]) -> str | None:
    by_tier: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for i, l in enumerate(lines):
        up = l.upper()
        if _TOTAL_EXCLUDE.search(up) and "INCL" not in up:
            continue
        tier = next((t for t, rx in enumerate(_TOTAL_TIERS) if rx.search(up)), None)
        if tier is None:
            continue
        am = find_amounts_in_line(l)
        if not am and i + 1 < len(lines):  # số tiền nằm ở dòng kế tiếp
            am = find_amounts_in_line(lines[i + 1])
        if am and am[-1][1] > 0:
            by_tier[tier].append(am[-1][1])
    for tier in (0, 1, 2):
        if by_tier[tier]:
            return f"{by_tier[tier][-1]:.2f}"  # lần xuất hiện cuối của mức ưu tiên cao nhất
    # dự phòng: số lớn nhất ở các dòng không phải thanh toán
    vals = [v for l in lines if not _PAYMENT_RE.search(l.upper()) for _, v in find_amounts_in_line(l)]
    return f"{max(vals):.2f}" if vals else None


def improved_extract(lines: list[str]) -> dict:
    lines = clean_lines(lines)
    company, c_idx = _company(lines)
    return {
        "company": company,
        "date": _date(lines),
        "address": _address(lines, c_idx),
        "total": _total(lines),
    }


EXTRACTORS = {"baseline": baseline_extract, "improved": improved_extract}
