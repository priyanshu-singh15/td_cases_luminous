"""Phase 2: EDA — class balance, resolution/blur distributions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.data.quality import laplacian_blur_score


def run_eda(
    clean_dir: Path | None = None,
    labels_dir: Path | None = None,
    report_dir: Path | None = None,
) -> pd.DataFrame:
    cfg = load_config()
    clean = clean_dir or resolve_path(f"{cfg['paths']['data_root']}/clean/images")
    labels = labels_dir or resolve_path(f"{cfg['paths']['data_root']}/labels/yolo/train/labels")
    report = report_dir or resolve_path(f"{cfg['paths']['data_root']}/reports")
    report.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(sorted(clean.glob("*.jpg"))[:2000], desc="EDA scan"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbl_file = labels / f"{img_path.stem}.txt"
        n_boxes = 0
        classes: list[int] = []
        if lbl_file.exists():
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    classes.append(int(parts[0]))
                    n_boxes += 1
        rows.append(
            {
                "case_id": img_path.stem,
                "width": w,
                "height": h,
                "blur": laplacian_blur_score(gray),
                "n_boxes": n_boxes,
                "classes": classes,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(report / "eda_samples.csv", index=False)

    if not df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].hist(df["blur"], bins=40, color="#2563eb", edgecolor="white")
        axes[0, 0].set_title("Blur (Laplacian variance)")
        axes[0, 1].hist(df["width"], bins=30, alpha=0.7, label="width")
        axes[0, 1].hist(df["height"], bins=30, alpha=0.7, label="height")
        axes[0, 1].legend()
        axes[0, 1].set_title("Resolution distribution")
        all_cls = [c for cs in df["classes"] for c in cs]
        if all_cls:
            names = [cfg["product_class"]] + cfg["damage_classes"]
            cnt = Counter(all_cls)
            labels_x = [names[i] if i < len(names) else str(i) for i in cnt.keys()]
            axes[1, 0].bar(labels_x, list(cnt.values()), color="#dc2626")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].set_title("Class distribution (weak labels)")
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(report / "eda_plots.png", dpi=120)
        plt.close()

    imbalance = {}
    all_cls = [c for cs in df["classes"] for c in cs]
    if all_cls:
        total = sum(Counter(all_cls).values())
        imbalance = {str(k): v / total for k, v in Counter(all_cls).items()}
    summary = {
        "n_images": len(df),
        "mean_blur": float(df["blur"].mean()) if len(df) else 0,
        "class_imbalance": imbalance,
        "recommendation": "Use class weights + oversample minority damage types during YOLO training",
    }
    with open(report / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return df
