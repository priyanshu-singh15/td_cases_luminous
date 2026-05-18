"""FastAPI backend for Luminous TD damage analysis."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import load_config, resolve_path
from src.inference.pipeline import TDPipeline, TDReport

app = FastAPI(title="Luminous TD Damage API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: TDPipeline | None = None


class ReportResponse(BaseModel):
    case_id: str
    timestamp: str
    quality: dict
    product_detected: bool
    overall_severity: str
    flagged_for_review: bool
    findings: list[dict]
    annotated_url: str | None = None
    gradcam_url: str | None = None
    message: str


@app.on_event("startup")
def startup() -> None:
    global pipeline
    pipeline = TDPipeline()


@app.get("/")
def root() -> dict:
    return {
        "service": "luminous-td-api",
        "dashboard": "http://127.0.0.1:8501",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "luminous-td-api"}


@app.post("/analyze", response_model=ReportResponse)
async def analyze(file: UploadFile = File(...)) -> ReportResponse:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(400, "Unsupported image format")
    case_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"{case_id}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        report: TDReport = pipeline.analyze(dest, case_id=case_id)
    return ReportResponse(
        case_id=report.case_id,
        timestamp=report.timestamp,
        quality=report.quality if isinstance(report.quality, dict) else {},
        product_detected=report.product_detected,
        overall_severity=report.overall_severity,
        flagged_for_review=report.flagged_for_review,
        findings=[asdict(f) for f in report.findings],
        annotated_url=f"/artifacts/{report.case_id}/annotated.jpg" if report.annotated_path else None,
        gradcam_url=f"/artifacts/{report.case_id}/gradcam.jpg" if report.gradcam_path else None,
        message=report.message,
    )


@app.get("/artifacts/{case_id}/{name}")
def get_artifact(case_id: str, name: str):
    cfg = load_config()
    path = resolve_path(cfg["paths"]["outputs_dir"]) / "cases" / case_id / name
    if not path.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path)


@app.get("/history")
def history(limit: int = 200) -> list[dict]:
    cfg = load_config()
    db = resolve_path(cfg["paths"]["cases_db"])
    if not db.exists():
        return []
    rows = []
    for line in db.read_text(encoding="utf-8").strip().splitlines()[-limit:]:
        rows.append(json.loads(line))
    return rows[::-1]


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run("api.main:app", host=cfg["api"]["host"], port=cfg["api"]["port"], reload=False)
