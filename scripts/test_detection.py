"""Quick test damage detection on sample images."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.pipeline import TDPipeline


def main() -> None:
    pipe = TDPipeline()
    samples = sorted((ROOT / "data" / "clean" / "images").glob("*.jpg"))[:5]
    print(f"Testing {len(samples)} images...\n")
    for img in samples:
        r = pipe.analyze(img)
        types = [f.class_name for f in r.findings]
        print(f"{img.name}: {len(r.findings)} findings {types} | {r.overall_severity} | {r.message}")


if __name__ == "__main__":
    main()
