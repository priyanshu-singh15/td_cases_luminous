"""Week 3: Train YOLOv8m detector + EfficientNet severity head."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.detector import train_detector
from src.training.train_severity import train_severity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--severity-epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true", help="Short run for smoke test")
    args = parser.parse_args()
    if args.quick:
        args.detector_epochs = 3
        args.severity_epochs = 3

    print("=== Phase 3: Train detector ===")
    det_weights = train_detector(epochs=args.detector_epochs)
    print(f"Detector saved: {det_weights}")

    print("=== Phase 3: Train severity head ===")
    sev_weights = train_severity(epochs=args.severity_epochs)
    print(f"Severity saved: {sev_weights}")


if __name__ == "__main__":
    main()
