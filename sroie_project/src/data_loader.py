"""Đọc bộ SROIE 2019 (ảnh + box + entities)."""
from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import DATA_ROOT, FIELDS, SEED, VAL_RATIO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    id: str
    split: str
    image_path: Path
    box_path: Path | None = None
    entity_path: Path | None = None

    def labels(self) -> dict | None:
        return load_entities(self.entity_path) if self.entity_path else None

    def gt_lines(self) -> list[str]:
        return load_box_lines(self.box_path) if self.box_path else []


def load_entities(path: Path) -> dict:
    """Đọc file nhãn key-value (JSON) -> {company,date,address,total}."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = ast.literal_eval(raw)
        except Exception:
            data = {}
            for f in FIELDS:
                m = re.search(rf'"{f}"\s*:\s*"(.*?)"\s*[,}}]', raw, re.I | re.S)
                if m:
                    data[f] = m.group(1)
    out = {}
    for k, v in (data or {}).items():
        k = str(k).lower().strip()
        if k in FIELDS:
            out[k] = " ".join(str(v).split())
    return {f: out.get(f) for f in FIELDS}


def load_box_lines(path: Path) -> list[str]:
    """Mỗi dòng box: x1,y1,x2,y2,x3,y3,x4,y4,text (text có thể chứa dấu phẩy)."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(",", 8)
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return lines


def _subdir(base: Path, names: list[str]) -> Path | None:
    for n in names:
        if (base / n).is_dir():
            return base / n
    return None


def list_samples(split_dir: Path, split_name: str) -> list[Sample]:
    img_dir = _subdir(split_dir, ["img", "images", "image"])
    if img_dir is None:
        raise FileNotFoundError(f"Không thấy thư mục ảnh (img/) trong {split_dir}")
    box_dir = _subdir(split_dir, ["box", "boxes"])
    ent_dir = _subdir(split_dir, ["entities", "key", "entity"])
    samples = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        box = box_dir / f"{p.stem}.txt" if box_dir else None
        ent = ent_dir / f"{p.stem}.txt" if ent_dir else None
        samples.append(Sample(
            id=p.stem, split=split_name, image_path=p,
            box_path=box if box and box.exists() else None,
            entity_path=ent if ent and ent.exists() else None,
        ))
    return samples


def _split_dir(name: str) -> Path:
    d = DATA_ROOT / name
    if not d.is_dir():
        raise FileNotFoundError(
            f"Không thấy {d}. Hãy tải SROIE và giải nén theo hướng dẫn trong README "
            f"(hoặc đặt biến môi trường SROIE_DIR)."
        )
    return d


def get_splits(seed: int = SEED, val_ratio: float = VAL_RATIO) -> dict[str, list[Sample]]:
    """train / val (tách ngẫu nhiên có seed từ train chính thức) / test (test chính thức).

    Nếu test chính thức không có nhãn -> giữ lại 15% train làm test (có cảnh báo).
    """
    train_all = [s for s in list_samples(_split_dir("train"), "train") if s.entity_path]
    rng = random.Random(seed)
    rng.shuffle(train_all)

    try:
        test = [s for s in list_samples(_split_dir("test"), "test") if s.entity_path]
    except FileNotFoundError:
        test = []

    if not test:
        print("[CẢNH BÁO] Không có test có nhãn -> tách 15% train làm test.")
        n_test = int(len(train_all) * 0.15)
        test = [Sample(**{**s.__dict__, "split": "test"}) for s in train_all[:n_test]]
        train_all = train_all[n_test:]

    n_val = int(len(train_all) * val_ratio)
    val = [Sample(**{**s.__dict__, "split": "val"}) for s in train_all[:n_val]]
    train = train_all[n_val:]
    return {"train": sorted(train, key=lambda s: s.id),
            "val": sorted(val, key=lambda s: s.id),
            "test": sorted(test, key=lambda s: s.id)}
