"""Generate demo annotated outputs for manager presentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import TDPipeline


def main() -> None:
    pipe = TDPipeline()
    out = ROOT / "outputs" / "demo_manager"
    out.mkdir(parents=True, exist_ok=True)
    samples = sorted((ROOT / "data" / "clean" / "images").glob("*.jpg"))[10:18]
    for img in samples:
        r = pipe.analyze(img, case_id=f"demo_{img.stem[:12]}")
        print(f"OK {img.name} -> {r.overall_severity} | {r.message}")
    print(f"\nDemo images saved under outputs/cases/demo_*")
    print("Open dashboard -> Results Viewer with those case IDs")


if __name__ == "__main__":
    main()
