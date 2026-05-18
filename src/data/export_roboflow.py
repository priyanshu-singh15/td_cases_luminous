"""Export clean images + empty label stubs for Roboflow/CVAT annotation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.config import load_config, resolve_path

DAMAGE_CLASSES = None  # filled from config at runtime


def export_for_labeling(
    clean_dir: Path | None = None,
    export_dir: Path | None = None,
    max_images: int | None = 500,
) -> Path:
    cfg = load_config()
    classes = cfg["damage_classes"] + [cfg["product_class"]]
    clean = clean_dir or resolve_path(f"{cfg['paths']['data_root']}/clean/images")
    export = export_dir or resolve_path(f"{cfg['paths']['data_root']}/labeling/roboflow_export")
    images_out = export / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    images = sorted(clean.glob("*.jpg"))
    if max_images:
        images = images[:max_images]

    for img in images:
        shutil.copy2(img, images_out / img.name)

    data_yaml = {
        "path": str(export.resolve()),
        "train": "images",
        "val": "images",
        "names": {i: c for i, c in enumerate(classes)},
        "note": "Annotate in Roboflow or CVAT, then import labels to data/labels/yolo/",
    }
    with open(export / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(data_yaml, f, indent=2)

    readme = export / "LABELING_GUIDE.txt"
    readme.write_text(
        "Roboflow/CVAT labeling guide\n"
        "----------------------------\n"
        f"Classes: {', '.join(classes)}\n"
        "1. Upload images/ to Roboflow project\n"
        "2. Draw bounding boxes per damage region\n"
        "3. Export YOLOv8 format -> data/labels/yolo/\n"
        "4. Run: python scripts/run_phase3_train.py\n",
        encoding="utf-8",
    )
    return export
