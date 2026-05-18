"""YOLOv8m detector wrapper for Luminous TD damage."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.config import load_config, resolve_path


def build_data_yaml(labels_root: Path, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    classes = [cfg["product_class"]] + cfg["damage_classes"]
    yaml_path = labels_root / "data.yaml"
    content = f"""path: {labels_root.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def train_detector(
    labels_root: Path | None = None,
    epochs: int | None = None,
    resume: bool = False,
) -> Path:
    cfg = load_config()
    labels_root = labels_root or resolve_path(f"{cfg['paths']['data_root']}/labels/yolo")
    data_yaml = build_data_yaml(labels_root, cfg)
    det_cfg = cfg["detector"]
    model = YOLO(f"{det_cfg['model_size']}.pt")
    aug = cfg["augmentation"]
    results = model.train(
        data=str(data_yaml),
        epochs=epochs or det_cfg["epochs"],
        imgsz=det_cfg["imgsz"],
        batch=det_cfg["batch"],
        patience=det_cfg["patience"],
        project=str(resolve_path(cfg["paths"]["models_dir"])),
        name="yolov8_td_detector",
        exist_ok=True,
        resume=resume,
        hsv_h=aug["hsv_h"],
        hsv_s=aug["hsv_s"],
        hsv_v=aug["hsv_v"],
        degrees=aug["degrees"],
        translate=aug["translate"],
        scale=aug["scale"],
        shear=aug["shear"],
        perspective=aug["perspective"],
        flipud=aug["flipud"],
        fliplr=aug["fliplr"],
        mosaic=aug["mosaic"],
        mixup=aug["mixup"],
        copy_paste=aug["copy_paste"],
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = resolve_path(f"{cfg['paths']['models_dir']}/detector_best.pt")
    if best.exists():
        dest.write_bytes(best.read_bytes())
    return dest


def load_detector(weights: Path | None = None) -> YOLO:
    cfg = load_config()
    w = weights or resolve_path(f"{cfg['paths']['models_dir']}/detector_best.pt")
    if w.exists():
        return YOLO(str(w))
    return YOLO(f"{cfg['detector']['model_size']}.pt")
