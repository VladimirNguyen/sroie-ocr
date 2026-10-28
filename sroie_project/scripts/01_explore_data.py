"""Bước 1: khảo sát dữ liệu (số ảnh, kích thước, độ mờ, định dạng ngày/tiền) và lưu biểu đồ."""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import EDA_DIR
from src.data_loader import get_splits
from src.normalize import find_amounts_in_line


def date_format(s: str) -> str:
    if not s:
        return "none"
    for name, rx in [("yyyy-mm-dd", r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"), ("dd/mm/yyyy", r"\d{1,2}/\d{1,2}/\d{4}"),
                     ("dd-mm-yyyy", r"\d{1,2}-\d{1,2}-\d{4}"), ("dd.mm.yyyy", r"\d{1,2}\.\d{1,2}\.\d{4}"),
                     ("dd/mm/yy", r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2}\b"),
                     ("dd MON yyyy", r"\d{1,2}[\s\-]*[A-Za-z]{3,9}[\s\-,]*\d{2,4}"),
                     ("MON dd yyyy", r"[A-Za-z]{3,9}\s*\d{1,2},?\s*\d{4}")]:
        if re.search(rx, s):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur-sample", type=int, default=150, help="số ảnh dùng đo độ mờ")
    args = ap.parse_args()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    splits = get_splits()

    rows = []
    for name, samples in splits.items():
        print(f"{name}: {len(samples)} ảnh")
        for s in samples:
            with Image.open(s.image_path) as im:
                w, h = im.size
            lab = s.labels() or {}
            rows.append({"id": s.id, "split": name, "w": w, "h": h, "aspect": h / w,
                         "date_format": date_format(lab.get("date")),
                         "total_str": lab.get("total"), "company": lab.get("company")})
    df = pd.DataFrame(rows)
    df.to_csv(EDA_DIR / "dataset_summary.csv", index=False)
    print(df.groupby("split")[["w", "h"]].describe().round(0).T)

    # độ mờ (variance of Laplacian) trên một tập con
    import cv2
    from src.preprocess import read_image
    blur = []
    for s in splits["train"][:args.blur_sample]:
        g = cv2.cvtColor(read_image(s.image_path), cv2.COLOR_BGR2GRAY)
        blur.append(cv2.Laplacian(g, cv2.CV_64F).var())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(df["w"], bins=30); ax[0].set_title("Chiều rộng ảnh (px)")
    ax[1].hist(df["h"], bins=30); ax[1].set_title("Chiều cao ảnh (px)")
    ax[2].hist(blur, bins=30); ax[2].set_title("Độ nét (var Laplacian) - thấp = mờ")
    plt.tight_layout(); plt.savefig(EDA_DIR / "image_stats.png", dpi=120); plt.close()

    fc = Counter(df["date_format"])
    plt.figure(figsize=(8, 4))
    plt.bar(fc.keys(), fc.values()); plt.xticks(rotation=30); plt.title("Định dạng ngày trong nhãn")
    plt.tight_layout(); plt.savefig(EDA_DIR / "date_formats.png", dpi=120); plt.close()
    print("Định dạng ngày:", dict(fc))

    # lưới ảnh mẫu
    sm = splits["train"][:6]
    fig, ax = plt.subplots(1, 6, figsize=(18, 6))
    for a, s in zip(ax, sm):
        a.imshow(Image.open(s.image_path)); a.set_title(s.id, fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(EDA_DIR / "sample_images.png", dpi=110); plt.close()
    print(f"Đã lưu biểu đồ vào {EDA_DIR}")


if __name__ == "__main__":
    main()
