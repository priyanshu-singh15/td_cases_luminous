"""Full pipeline setup: ingest → bootstrap → train (quick) → eval."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "scripts/run_phase1_ingest.py"])
    run([py, "scripts/run_phase2_eda.py"])
    run([py, "scripts/run_phase3_train.py", "--quick"])
    run([py, "scripts/run_phase5_eval.py"])
    print("\nSetup complete. Start services:")
    print(f"  {py} -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
    print(f"  {py} -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
