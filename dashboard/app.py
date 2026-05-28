"""Streamlit dashboard — 4 pages for Luminous TD damage analysis."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ensure project root is on path (fixes blank page when streamlit cwd != root)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.config import load_config, resolve_path

cfg = load_config()
API_URL = cfg["dashboard"]["api_url"]
OUT = resolve_path(cfg["paths"]["outputs_dir"])
HISTORY = resolve_path(cfg["paths"]["cases_db"])
SAMPLE_DIR = resolve_path(f"{cfg['paths']['data_root']}/clean/images")

st.set_page_config(
    page_title="Luminous TD Damage AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Luminous TD AI")
st.sidebar.caption("Transit damage detection for dealer uploads")
page = st.sidebar.radio(
    "Navigate",
    ["Upload & Analyse", "Results Viewer", "Case History", "Analytics Dashboard"],
)
st.sidebar.divider()
st.sidebar.markdown(f"**API:** `{API_URL}`")


@st.cache_resource
def _get_local_pipeline():
    from src.inference.pipeline import TDPipeline

    return TDPipeline()


def _api_online() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _analyze_image(file_bytes: bytes, filename: str) -> dict:
    """Analyze via API if up, else run pipeline in-process."""
    if _api_online():
        r = httpx.post(
            f"{API_URL}/analyze",
            files={"file": (filename, file_bytes, "image/jpeg")},
            timeout=180.0,
        )
        r.raise_for_status()
        return r.json()

    suffix = Path(filename).suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(file_bytes)
        report = _get_local_pipeline().analyze(path)
    return {
        "case_id": report.case_id,
        "timestamp": report.timestamp,
        "quality": report.quality if isinstance(report.quality, dict) else {},
        "product_detected": report.product_detected,
        "overall_severity": report.overall_severity,
        "flagged_for_review": report.flagged_for_review,
        "findings": [asdict(f) for f in report.findings],
        "annotated_url": report.annotated_path,
        "gradcam_url": report.gradcam_path,
        "message": report.message + " (local pipeline)",
    }


def _load_history() -> pd.DataFrame:
    if not HISTORY.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line)
        for line in HISTORY.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "case_id": r.get("case_id"),
                "timestamp": r.get("timestamp"),
                "overall_severity": r.get("overall_severity"),
                "flagged_for_review": r.get("flagged_for_review"),
                "product_detected": r.get("product_detected"),
                "n_findings": r.get("n_findings", 0),
                "source": r.get("source", ""),
            }
            for r in rows
        ]
    )


# --- Page: Upload & Analyse ---
if page == "Upload & Analyse":
    st.title("Upload & Analyse")
    st.caption("Uses YOLO-World open-vocabulary AI + classical dent/scratch CV — no manual labels required.")
    online = _api_online()
    if online:
        st.success("Backend API is online.")
    else:
        st.warning(
            "API offline — using **local pipeline** (start API: "
            "`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`)"
        )

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.subheader("Quick sample")
        samples = sorted(SAMPLE_DIR.glob("*.jpg"))[:12] if SAMPLE_DIR.exists() else []
        if samples:
            pick = st.selectbox("Dealer image", [s.name for s in samples])
            if st.button("Analyze sample"):
                data = _analyze_image(samples[[s.name for s in samples].index(pick)].read_bytes(), pick)
                st.session_state["last_report"] = data
                st.rerun()
        else:
            st.info("Run ingest first to populate samples.")

    with col_a:
        uploaded = st.file_uploader("Damage image", type=["jpg", "jpeg", "png", "webp"])
        if st.button("Run analysis", type="primary", disabled=uploaded is None):
            with st.spinner("Quality → detect → edge severity → Grad-CAM..."):
                try:
                    data = _analyze_image(
                        uploaded.getvalue(),
                        uploaded.name,
                    )
                    st.session_state["last_report"] = data
                    st.success(f"Case **{data['case_id']}** — **{data['overall_severity']}**")
                    st.info(data.get("message", ""))
                    if data.get("flagged_for_review"):
                        st.warning("Some regions have lower confidence — recommend human review.")
                    cid = data["case_id"]
                    ann = OUT / "cases" / cid / "annotated.jpg"
                    cam = OUT / "cases" / cid / "gradcam.jpg"
                    if ann.exists() and cam.exists():
                        v1, v2 = st.columns(2)
                        v1.image(Image.open(ann), caption="Damage boxes", use_container_width=True)
                        v2.image(Image.open(cam), caption="Damage heatmap", use_container_width=True)
                    with st.expander("Full JSON report"):
                        st.json(data)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# --- Page: Results Viewer ---
elif page == "Results Viewer":
    st.title("Results Viewer")
    report = st.session_state.get("last_report")
    case_id = st.text_input("Case ID", value=(report or {}).get("case_id", ""))
    if not case_id:
        st.info("Run an analysis on **Upload & Analyse** first.")
    else:
        ann = OUT / "cases" / case_id / "annotated.jpg"
        cam = OUT / "cases" / case_id / "gradcam.jpg"
        c1, c2, c3 = st.columns(3)
        if ann.exists():
            c1.image(Image.open(ann), caption="Annotated detections", use_container_width=True)
        else:
            c1.warning("Annotated image not found.")
        if cam.exists():
            c2.image(Image.open(cam), caption="Grad-CAM severity", use_container_width=True)
        else:
            c2.warning("Grad-CAM not found.")
        if report and report.get("case_id") == case_id:
            c3.markdown("### Findings")
            for f in report.get("findings", []):
                src = f.get("detection_source", "")
                c3.markdown(
                    f"- **{f['class_name']}** — {f['severity']} "
                    f"(conf {f['confidence']:.0%})"
                    + (f" _{src}_" if src else "")
                )
            if not report.get("findings"):
                c3.write("No damage regions detected above threshold.")

# --- Page: Case History ---
elif page == "Case History":
    st.title("Case History")
    df = _load_history()
    if df.empty:
        st.info("No cases yet. Analyze an image on **Upload & Analyse**.")
    else:
        sev_opts = df["overall_severity"].dropna().unique().tolist()
        sev = st.multiselect("Filter severity", sev_opts, default=sev_opts)
        flagged = st.checkbox("Review flagged only", value=False)
        filtered = df[df["overall_severity"].isin(sev)] if sev else df
        if flagged:
            filtered = filtered[filtered["flagged_for_review"] == True]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Export CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"td_cases_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# --- Page: Analytics ---
else:
    st.title("Analytics Dashboard")
    df = _load_history()
    if df.empty:
        st.info("Charts appear after you analyze cases.")
        if SAMPLE_DIR.exists():
            st.metric("Clean images ready", len(list(SAMPLE_DIR.glob("*.jpg"))))
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cases", len(df))
        m2.metric("Flagged for review", int(df["flagged_for_review"].sum()))
        m3.metric("Severe cases", int((df["overall_severity"] == "Severe").sum()))
        c1, c2 = st.columns(2)
        fig_pie = px.pie(df, names="overall_severity", title="Severity distribution")
        c1.plotly_chart(fig_pie, use_container_width=True)
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        trend = df.groupby("date").size().reset_index(name="cases")
        fig_line = px.line(trend, x="date", y="cases", title="Cases over time")
        c2.plotly_chart(fig_line, use_container_width=True)
