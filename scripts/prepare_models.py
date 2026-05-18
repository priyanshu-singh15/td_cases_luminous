"""Ensure model weights exist for inference (bootstrap if training incomplete)."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_path


def main() -> None:
    models = resolve_path("models")
    models.mkdir(parents=True, exist_ok=True)
    det_dest = models / "detector_best.pt"
    yolo_root = ROOT / "yolov8m.pt"
    yolo_train = models / "yolov8_td_detector" / "weights" / "best.pt"

    if not det_dest.exists():
        if yolo_train.exists():
            shutil.copy2(yolo_train, det_dest)
            print(f"Copied trained weights -> {det_dest}")
        elif yolo_root.exists():
            shutil.copy2(yolo_root, det_dest)
            print(f"Using pretrained YOLOv8m -> {det_dest} (fine-tune for TD classes)")
        else:
            print("No YOLO weights found. Run: python scripts/run_phase3_train.py --quick")

    sev_dest = models / "severity_best.pt"
    if not sev_dest.exists():
        print("Severity weights missing — training 3 epochs...")
        from src.training.train_severity import train_severity

        train_severity(epochs=3)
        print(f"Saved {sev_dest}")


if __name__ == "__main__":
    main()
