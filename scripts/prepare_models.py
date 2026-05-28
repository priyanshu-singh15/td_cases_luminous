"""Download YOLO-World weights for open-vocabulary damage detection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("Loading YOLO-World (downloads weights on first run)...")
    from src.models.damage_engine import _load_yolo_world

    model = _load_yolo_world()
    model.set_classes(["dent on product", "scratch on battery"])
    print("Ready:", ROOT / "yolov8s-worldv2.pt")


if __name__ == "__main__":
    main()
