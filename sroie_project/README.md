# Chủ đề 05 – Trích xuất thông tin từ ảnh hóa đơn bán lẻ (OCR + luật)

Bài tập lớn *Nhập môn Trí tuệ nhân tạo*. Từ ảnh hóa đơn, hệ thống đọc chữ bằng OCR có sẵn (Tesseract / EasyOCR)
rồi dùng luật/regex để tìm 4 trường: **company, date, address, total**. Đầu ra là JSON.

- Dữ liệu: **ICDAR 2019 SROIE** (https://rrc.cvc.uab.es/?ch=13, repo tham khảo https://github.com/metatatt/sorie)
- Không cần GPU, không huấn luyện OCR, chạy được trên CPU.
- Thiết kế thực nghiệm: **baseline** (luật đơn giản, ảnh ít xử lý) → **cải tiến** (làm sạch ảnh + luật theo vị trí/từ khóa).

## 1. Cài đặt (VS Code)

1. Cài Python 3.10+ và VS Code (extension *Python*).
2. Cài **Tesseract OCR** (chương trình riêng, không phải thư viện pip):
   - Windows: tải bản cài từ https://github.com/UB-Mannheim/tesseract/wiki (mặc định cài vào `C:\Program Files\Tesseract-OCR`). Project tự dò đường dẫn này; nếu cài chỗ khác, đặt biến môi trường `TESSERACT_CMD`.
   - Ubuntu: `sudo apt install tesseract-ocr` · macOS: `brew install tesseract`
3. Mở thư mục project trong VS Code, mở Terminal và chạy:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   # tùy chọn (nặng): pip install -r requirements-optional.txt   # EasyOCR
   ```
4. VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → chọn `.venv`.
5. Kiểm tra: `python -m pytest tests -q` (7 test cho luật trích xuất, chuẩn hóa, đánh giá).

## 2. Lấy dữ liệu

Tải SROIE từ trang cuộc thi hoặc bản Kaggle/GitHub được cung cấp, **đọc giấy phép của bản bạn tải**, rồi giải nén thành:

```
data/raw/SROIE2019/
├── train/  img/  box/  entities/
└── test/   img/  box/  entities/
```
- `img/` ảnh hóa đơn · `box/` vùng chữ + nội dung chữ (`x1,y1,...,x4,y4,text`) · `entities/` nhãn 4 trường (JSON).
- Tên thư mục khác (`key` thay `entities`, v.v.) thì sửa nhẹ trong `src/data_loader.py` hoặc đổi tên. Có thể đặt thư mục khác bằng biến môi trường `SROIE_DIR`.
- Nếu bản tải không có nhãn test, chương trình tự giữ 15% train làm test (có in cảnh báo).
- Ghi lại: **nguồn, phiên bản, ngày tải, giấy phép** vào `docs/DATA_CARD.md`.

**Chia dữ liệu:** `train` chính thức tách thành train/val (80/20, seed 42); `test` chính thức chỉ dùng **một lần ở cuối** để báo cáo. Mỗi hóa đơn là một mẫu độc lập nên chia ngẫu nhiên có seed là chấp nhận được. Mọi chỉnh sửa luật/tham số dựa trên **val**, không nhìn test.

## 3. Quy trình chạy (theo thứ tự)

```bash
# 1) Khảo sát dữ liệu: kích thước, độ mờ, định dạng ngày -> outputs/eda/
python scripts/01_explore_data.py

# 2) Chạy OCR trên 200 ảnh val, hai kiểu tiền xử lý (kết quả được cache vào outputs/ocr_cache/)
python scripts/02_run_ocr.py --split val --limit 200 --variant basic
python scripts/02_run_ocr.py --split val --limit 200 --variant improved

# 3) So sánh baseline / cải tiến -> outputs/results/metrics_*.csv|md, per_sample_*.csv
python scripts/03_evaluate.py --split val --variants basic,improved

# 4) Phân tích lỗi -> outputs/error_analysis/report.md + ảnh có khung OCR
python scripts/04_error_analysis.py --split val --variant improved

# 5) Khi đã chốt phương pháp: chạy một lần trên TEST
python scripts/02_run_ocr.py --split test --limit 200 --variant basic
python scripts/02_run_ocr.py --split test --limit 200 --variant improved
python scripts/03_evaluate.py --split test

# 6) Demo
streamlit run app.py
```
Trong VS Code có sẵn các cấu hình chạy ở *Run and Debug* (`.vscode/launch.json`).
Thử OCR nhanh với `--limit 20` trước khi chạy 200 ảnh (khoảng 1–3 giây/ảnh trên CPU).

## 4. Cấu trúc project

```
src/config.py        đường dẫn, seed, tên trường
src/data_loader.py   đọc ảnh/box/entities, chia train/val/test
src/preprocess.py    raw | basic | improved (xám, phóng to, khử nhiễu, CLAHE, chỉnh nghiêng, ngưỡng thích nghi)
src/ocr_engine.py    Tesseract / EasyOCR -> danh sách dòng (text, bbox, conf)
src/extractors.py    baseline_extract, improved_extract (thuần Python, dễ test)
src/normalize.py     chuẩn hóa ngày -> ISO, số tiền -> 2 chữ số thập phân, chuỗi -> IN HOA bỏ dấu câu
src/evaluate.py      exact match, precision/recall/F1, phân loại lỗi OCR vs lỗi luật
src/pipeline.py      ghép tiền xử lý + OCR + trích trường, cache JSON
src/visualize.py     vẽ khung OCR
scripts/             01 EDA · 02 OCR · 03 đánh giá · 04 phân tích lỗi
app.py               demo Streamlit
tests/               kiểm thử
docs/DATA_CARD.md    mẫu ghi nguồn dữ liệu
```

## 5. Phương pháp

| | Baseline | Cải tiến |
|---|---|---|
| Ảnh | xám + phóng to nếu nhỏ (`basic`) | + khử nhiễu, CLAHE, chỉnh nghiêng, ngưỡng thích nghi (`improved`) |
| company | dòng đầu tiên | chấm điểm 8 dòng đầu (hậu tố SDN/BHD…, chữ hoa, ít chữ số, loại dòng TAX INVOICE/TEL/số đăng ký), ghép dòng "SDN BHD" bị tách |
| date | regex `dd/mm/yy(yy)` đầu tiên | nhiều định dạng (dd/mm/yyyy, yyyy-mm-dd, 15 JAN 2019, Dec 5 2018…), ưu tiên dòng có "DATE", loại ngày không hợp lệ |
| address | 2 dòng sau dòng đầu | các dòng sau company đến khi gặp TEL/GST/INVOICE/… hoặc dòng có mã bưu chính/tên bang |
| total | số tiền lớn nhất | từ khóa theo mức ưu tiên (ROUNDED TOTAL > GRAND TOTAL/AMOUNT DUE > TOTAL), loại SUBTOTAL/GST/ROUNDING/CASH/CHANGE, sửa lỗi O→0 trong số |

**Chuẩn hóa:** ngày → `yyyy-mm-dd`, tiền → `xx.yy`, công ty/địa chỉ → IN HOA bỏ dấu câu. **Giữ nguyên bản gốc** trong JSON đầu ra.
**Không sửa nhãn kiểm thử** sau khi xem dự đoán.

### Cách phân biệt sai do đọc chữ và sai do tìm trường
- Với mỗi trường sai, kiểm tra giá trị đúng có xuất hiện trong văn bản OCR không: có → `extraction_error` (luật chọn sai); không → `ocr_error`.
- Dòng `oracle/*` trong bảng kết quả chạy luật trên **văn bản nhãn gốc** (không có lỗi OCR), cho biết **trần** của bộ luật.

## 6. Việc bạn cần tự làm (để có điểm cao)

1. Chạy các bước ở mục 3 với số ảnh ≥ 200 và **tự đọc kết quả**; viết nhận xét thật, đừng chép số liệu mẫu.
2. Mở `outputs/results/metrics_*.md`, so baseline với cải tiến và nêu **khi nào cải tiến giúp, khi nào không**. Tiền xử lý `improved` không đảm bảo luôn tốt hơn `basic` — hãy báo cáo đúng kết quả đo được.
3. Điền nguyên nhân từng ca lỗi trong `outputs/error_analysis/report.md` (≥ 3 lỗi: ảnh mờ/nghiêng, nhiều giá tiền, địa chỉ tách dòng…) và đưa ảnh minh họa vào báo cáo.
4. Thử thêm: `--psm 4`, EasyOCR, chỉnh luật dựa trên **val**; ghi lại mỗi lần thay đổi.
5. Điền `docs/DATA_CARD.md`; chụp ảnh demo từ ảnh được phép chia sẻ.
6. Báo cáo PDF 10–15 trang, slide (problem / data / baseline / method / results / error / demo), nộp kèm `requirements.txt` và README này.

## 7. Hạn chế cần nêu trong báo cáo

- Bộ luật tinh chỉnh theo hóa đơn Malaysia tiếng Anh (SDN BHD, GST, bang…); không đảm bảo cho định dạng khác.
- Kết quả phụ thuộc mạnh vào chất lượng OCR; ảnh mờ/nghiêng/nhăn giảm điểm.
- Địa chỉ dùng exact match nên rất khắt khe; có báo thêm độ tương tự chuỗi (`sim_address`) để tham khảo.
- Luật dừng địa chỉ khi gặp tên bang có thể cắt sớm với tên khu dân cư trùng tên bang.
- Tập SROIE đã chú thích theo quy ước riêng, một số bản phân phối bị sửa nhãn hoặc khác số ảnh.
- Hóa đơn thật có thể chứa thông tin nhạy cảm; chỉ dùng ảnh trong bộ dữ liệu công khai và tuân thủ giấy phép, không tự thu thập ảnh riêng tư.

## 8. Ghi công nguồn

ICDAR 2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE), Huang et al., 2019.
OCR: Tesseract (Apache 2.0), EasyOCR (Apache 2.0). Xử lý ảnh: OpenCV.
