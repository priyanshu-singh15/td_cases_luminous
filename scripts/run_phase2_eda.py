"""Week 2: EDA & preprocessing report."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bootstrap_labels import generate_weak_labels
from src.preprocessing.eda import run_eda


def main() -> None:
    print("=== Phase 2: Bootstrap weak labels + EDA ===")
    counts = generate_weak_labels(limit=800)
    print("Bootstrap splits:", counts)
    df = run_eda()
    print(f"EDA on {len(df)} samples — see data/reports/")


if __name__ == "__main__":
    main()
