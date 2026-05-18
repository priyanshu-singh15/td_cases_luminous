"""Week 5: Evaluation metrics on test split."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_detector


def main() -> None:
    print("=== Phase 5: Evaluate ===")
    metrics = evaluate_detector()
    print(metrics)


if __name__ == "__main__":
    main()
