"""Week 1: Data collection — ingest UploadTdcaseDocuments."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.export_roboflow import export_for_labeling
from src.data.ingest import ingest_raw


def main() -> None:
    print("=== Phase 1: Ingest & clean ===")
    stats = ingest_raw()
    print(stats)
    export_path = export_for_labeling(max_images=500)
    print(f"Roboflow export ready: {export_path}")


if __name__ == "__main__":
    main()
