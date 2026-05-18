"""Phase 5: mAP, precision, recall + validation on held-out TD cases."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config, resolve_path
from src.models.detector import load_detector


def evaluate_detector(labels_root: Path | None = None) -> dict:
    cfg = load_config()
    labels_root = labels_root or resolve_path(f"{cfg['paths']['data_root']}/labels/yolo")
    data_yaml = labels_root / "data.yaml"
    if not data_yaml.exists():
        from src.models.detector import build_data_yaml

        build_data_yaml(labels_root, cfg)

    det = load_detector()
    metrics = det.val(data=str(data_yaml), split="test", verbose=False)
    out = {
        "mAP50": float(metrics.box.map50) if metrics.box else 0,
        "mAP50-95": float(metrics.box.map) if metrics.box else 0,
        "precision": float(metrics.box.mp) if metrics.box else 0,
        "recall": float(metrics.box.mr) if metrics.box else 0,
    }
    report_path = resolve_path(f"{cfg['paths']['data_root']}/reports/eval_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
