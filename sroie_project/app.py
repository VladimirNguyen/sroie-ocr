"""Demo Streamlit: tải ảnh hóa đơn -> OCR -> JSON 4 trường (company, date, address, total).

Chạy:  streamlit run app.py
"""
import json
import time

import cv2
import numpy as np
import streamlit as st

from src.normalize import normalize_date, normalize_total
from src.ocr_engine import OCRNotAvailable, get_engine
from src.pipeline import extract
from src.preprocess import VARIANTS, preprocess
from src.visualize import draw_boxes

st.set_page_config(page_title="Trích xuất hóa đơn", layout="wide")
st.title("Trích xuất thông tin từ ảnh hóa đơn (OCR + luật)")
st.caption("Bài tập lớn Nhập môn AI – Chủ đề 05. Dữ liệu: ICDAR 2019 SROIE. Không phải hệ thống kế toán thực tế.")

with st.sidebar:
    st.header("Cấu hình")
    engine_name = st.selectbox("OCR", ["tesseract", "easyocr"])
    variant = st.selectbox("Tiền xử lý ảnh", VARIANTS, index=2)
    extractor = st.selectbox("Bộ luật trích trường", ["improved", "baseline"])
    psm = st.selectbox("Tesseract psm", [6, 4, 11], index=0)
    show_boxes = st.checkbox("Vẽ khung OCR", value=True)


@st.cache_resource(show_spinner="Đang khởi tạo OCR...")
def load_engine(name: str, psm: int):
    return get_engine(name, psm=psm) if name == "tesseract" else get_engine(name)


up = st.file_uploader("Tải ảnh hóa đơn (jpg/png)", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Hãy tải một ảnh hóa đơn. Chỉ dùng ảnh bạn được phép chia sẻ; hóa đơn thật có thể chứa thông tin nhạy cảm.")
    st.stop()

img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Không đọc được ảnh. Hãy thử file jpg/png khác.")
    st.stop()
if min(img.shape[:2]) < 100:
    st.warning("Ảnh rất nhỏ, kết quả OCR có thể kém.")

try:
    engine = load_engine(engine_name, psm)
except OCRNotAvailable as e:
    st.error(str(e))
    st.stop()

t0 = time.perf_counter()
proc = preprocess(img, variant)
lines = engine.run(proc)
sec = time.perf_counter() - t0

if not lines:
    st.error("OCR không đọc được chữ nào trong ảnh.")
    st.stop()

result = extract(lines, extractor)
missing = [k for k, v in result.items() if not v]

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.subheader("Ảnh")
    shown = draw_boxes(proc, lines) if show_boxes else proc
    st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB) if shown.ndim == 3 else shown, use_container_width=True)
with c2:
    st.subheader("Văn bản OCR")
    st.text_area("", "\n".join(f"{i:02d}: {l.text}" for i, l in enumerate(lines)), height=500)
with c3:
    st.subheader("JSON 4 trường")
    st.json(result)
    if missing:
        st.warning("Không tìm thấy: " + ", ".join(missing))
    with st.expander("Giá trị đã chuẩn hóa (bản gốc vẫn giữ ở trên)"):
        st.json({"date_iso": normalize_date(result["date"]) if result["date"] else None,
                 "total_value": normalize_total(result["total"]) if result["total"] else None})
    st.download_button("Tải JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="receipt.json", mime="application/json")
    st.caption(f"OCR + tiền xử lý: {sec:.2f}s · {len(lines)} dòng")
