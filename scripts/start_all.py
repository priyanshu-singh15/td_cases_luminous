"""Start FastAPI + Streamlit from project root (correct PYTHONPATH)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_models.py")], cwd=ROOT, check=False)
    print("\nStarting API on http://127.0.0.1:8000")
    print("Starting Streamlit on http://127.0.0.1:8501\n")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
    )
    st = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "dashboard" / "app.py"),
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
    )
    print("Press Ctrl+C to stop both.")
    try:
        st.wait()
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
        st.terminate()


if __name__ == "__main__":
    main()
