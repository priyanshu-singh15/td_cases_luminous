"""Weak pseudo-labels from edge/saliency for bootstrap training until manual labels exist."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import load_config, resolve_path

DAMAGE_IDX_OFFSET = 1  # 0 = product


def _saliency_boxes(gray: np.ndarray, max_boxes: int = 3) -> list[tuple[float, float, float, float, int]]:
    """Return YOLO-normalized boxes (xc, yc, w, h, class_id)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[float, float, float, float, float]] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.002 * w * h or area > 0.65 * w * h:
            continue
        boxes.append((area, x, y, bw, bh))

    boxes.sort(key=lambda b: b[0], reverse=True)
    product_box = (0.5, 0.5, 0.85, 0.85, 0)  # full-frame product prior
    out = [product_box]

    for _, x, y, bw, bh in boxes[:max_boxes]:
        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        nw, nh = bw / w, bh / h
        if nw < 0.03 or nh < 0.03:
            continue
        cls = random.randint(1, 8)  # damage class placeholder
        out.append((xc, yc, nw, nh, cls))
    return out


def generate_weak_labels(
    clean_dir: Path | None = None,
    labels_dir: Path | None = None,
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    limit: int | None = 800,
) -> dict:
    cfg = load_config()
    clean = clean_dir or resolve_path(f"{cfg['paths']['data_root']}/clean/images")
    base = labels_dir or resolve_path(f"{cfg['paths']['data_root']}/labels/yolo")
    images = sorted(clean.glob("*.jpg"))
    if limit:
        images = images[:limit]
    random.seed(cfg["project"]["seed"])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    counts = {"train": 0, "val": 0, "test": 0}
    for split, paths in splits.items():
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(paths, desc=f"Bootstrap {split}"):
            img = cv2.imread(str(src))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = _saliency_boxes(gray)
            dst_img = img_dir / src.name
            cv2.imwrite(str(dst_img), img)
            lbl_path = lbl_dir / f"{src.stem}.txt"
            lines = [f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h, cls in boxes]
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] += 1
    return counts
