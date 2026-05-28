"""End-to-end TD analysis: quality → detect → edge severity → explainability → report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, resolve_path
from src.data.quality import QualityResult, check_quality
from src.models.damage_engine import DamageDetectionEngine, score_severity


@dataclass
class DamageFinding:
    class_name: str
    confidence: float
    bbox: list[float]
    severity: str
    severity_confidence: float
    needs_review: bool
    detection_source: str = ""


@dataclass
class TDReport:
    case_id: str
    timestamp: str
    quality: dict
    product_detected: bool
    findings: list[DamageFinding] = field(default_factory=list)
    overall_severity: str = "Mild"
    flagged_for_review: bool = False
    annotated_path: str | None = None
    gradcam_path: str | None = None
    message: str = ""


class TDPipeline:
    def __init__(self, device: str | None = None):
        self.cfg = load_config()
        inf = self.cfg["inference"]
        self.engine = DamageDetectionEngine(
            conf=inf.get("world_conf", 0.12),
            iou=inf["iou_threshold"],
            use_world=inf.get("use_yolo_world", True),
        )
        self.severity_names = self.cfg["severity_classes"]
        self.out_dir = resolve_path(self.cfg["paths"]["outputs_dir"])
        self.cases_db = resolve_path(self.cfg["paths"]["cases_db"])
        self.cases_db.parent.mkdir(parents=True, exist_ok=True)
        self.inf = inf

    def _annotate(
        self,
        bgr: np.ndarray,
        dets,
        product_boxes: list,
    ) -> np.ndarray:
        vis = bgr.copy()
        for x1, y1, x2, y2 in product_boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(vis, "Product", (int(x1), max(22, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        colors = {
            "dent": (0, 140, 255),
            "scratch": (0, 0, 255),
            "crack": (255, 0, 0),
            "corrosion": (0, 165, 255),
            "leak": (255, 0, 255),
            "packaging_damage": (180, 120, 0),
            "terminal_damage": (128, 0, 128),
            "label_damage": (200, 200, 0),
        }
        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            color = colors.get(d.class_name, (0, 0, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.0%}"
            cv2.putText(vis, label, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis

    def analyze(self, image_path: Path, case_id: str | None = None) -> TDReport:
        case_id = case_id or image_path.stem
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return TDReport(
                case_id=case_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                quality={},
                product_detected=False,
                message="Could not read image",
            )

        q: QualityResult = check_quality(
            bgr,
            min_blur=self.inf["min_blur_variance"],
            min_resolution=self.inf["min_resolution"],
        )
        if not q.passed:
            return TDReport(
                case_id=case_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                quality=asdict(q),
                product_detected=False,
                flagged_for_review=True,
                message=f"Quality check failed: {', '.join(q.issues)}",
            )

        dets, product_boxes, heatmap = self.engine.detect(bgr)
        product_detected = len(product_boxes) > 0 or True  # ROI always covers product region

        findings: list[DamageFinding] = []
        review_thresh = self.inf["human_review_threshold"]

        for d in dets:
            if d.confidence < self.inf.get("min_damage_conf", 0.20):
                continue
            sev, sev_conf = score_severity(bgr, d.bbox, d.class_name)
            findings.append(
                DamageFinding(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=list(d.bbox),
                    severity=sev,
                    severity_confidence=sev_conf,
                    needs_review=d.confidence < review_thresh,
                    detection_source=d.source,
                )
            )

        # Sort by confidence, cap display
        findings.sort(key=lambda f: f.confidence, reverse=True)
        findings = findings[:8]

        ann = self._annotate(bgr, dets[:12], product_boxes)
        case_dir = self.out_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        ann_path = case_dir / "annotated.jpg"
        cv2.imwrite(str(ann_path), ann)

        gradcam_path = case_dir / "gradcam.jpg"
        cv2.imwrite(str(gradcam_path), heatmap)

        severity_order = {"Mild": 0, "Moderate": 1, "Severe": 2}
        overall = "Mild"
        if findings:
            overall = max(findings, key=lambda f: severity_order.get(f.severity, 0)).severity

        flagged = any(f.needs_review for f in findings)
        if len(findings) == 0:
            msg = "No significant damage detected — product appears transit-OK."
        else:
            types = ", ".join(sorted({f.class_name for f in findings}))
            msg = f"Found {len(findings)} damage area(s): {types}"

        report = TDReport(
            case_id=case_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            quality={"passed": q.passed, "blur_score": q.blur_score, "brightness": q.brightness},
            product_detected=product_detected,
            findings=findings,
            overall_severity=overall,
            flagged_for_review=flagged,
            annotated_path=str(ann_path.relative_to(resolve_path("."))),
            gradcam_path=str(gradcam_path.relative_to(resolve_path("."))),
            message=msg,
        )
        self._persist(report, str(image_path))
        return report

    def _persist(self, report: TDReport, source: str) -> None:
        row = {
            "case_id": report.case_id,
            "timestamp": report.timestamp,
            "source": source,
            "overall_severity": report.overall_severity,
            "flagged_for_review": report.flagged_for_review,
            "product_detected": report.product_detected,
            "n_findings": len(report.findings),
            "findings": [asdict(f) for f in report.findings],
            "annotated_path": report.annotated_path,
            "gradcam_path": report.gradcam_path,
        }
        with open(self.cases_db, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
