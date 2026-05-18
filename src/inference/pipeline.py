"""End-to-end TD analysis: quality → detect → segment → edge → severity → report."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_config, resolve_path
from src.data.quality import QualityResult, check_quality
from src.explainability.gradcam import generate_gradcam, save_gradcam
from src.models.detector import load_detector
from src.models.severity_head import load_severity_model
from src.preprocessing.edge_channel import compute_edge_channel, rgb_edge_4channel


@dataclass
class DamageFinding:
    class_name: str
    confidence: float
    bbox: list[float]
    severity: str
    severity_confidence: float
    needs_review: bool


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = [self.cfg["product_class"]] + self.cfg["damage_classes"]
        self.severity_names = self.cfg["severity_classes"]
        det_path = resolve_path(f"{self.cfg['paths']['models_dir']}/detector_best.pt")
        sev_path = resolve_path(f"{self.cfg['paths']['models_dir']}/severity_best.pt")
        self.detector = load_detector(det_path if det_path.exists() else None)
        self.severity_model = load_severity_model(
            sev_path if sev_path.exists() else None,
            num_classes=len(self.severity_names),
            device=self.device,
        )
        self.out_dir = resolve_path(self.cfg["paths"]["outputs_dir"])
        self.cases_db = resolve_path(self.cfg["paths"]["cases_db"])
        self.cases_db.parent.mkdir(parents=True, exist_ok=True)

    def _annotate(self, bgr: np.ndarray, boxes, labels, severities) -> np.ndarray:
        vis = bgr.copy()
        for box, lbl, sev in zip(boxes, labels, severities):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 180, 255) if lbl == self.cfg["product_class"] else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{lbl} | {sev}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis

    def analyze(self, image_path: Path, case_id: str | None = None) -> TDReport:
        inf = self.cfg["inference"]
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
            min_blur=inf["min_blur_variance"],
            min_resolution=inf["min_resolution"],
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

        # Edge enhancement map (used in severity path)
        edge_map = compute_edge_channel(bgr)
        edge_overlay = (edge_map * 255).astype(np.uint8)
        edge_overlay = cv2.cvtColor(edge_overlay, cv2.COLOR_GRAY2BGR)

        results = self.detector.predict(
            source=bgr,
            conf=inf["conf_threshold"],
            iou=inf["iou_threshold"],
            verbose=False,
        )[0]

        h, w = bgr.shape[:2]
        findings: list[DamageFinding] = []
        draw_boxes, draw_labels, draw_sevs = [], [], []
        product_detected = False
        primary_crop = bgr
        max_damage_conf = 0.0

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                if name == self.cfg["product_class"]:
                    product_detected = True
                    continue
                crop = bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                tensor = torch.from_numpy(rgb_edge_4channel(crop, 224)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.severity_model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    sev_idx = int(probs.argmax())
                    sev_conf = float(probs[sev_idx])
                sev_name = self.severity_names[sev_idx]
                needs_review = conf < inf["human_review_threshold"] or sev_conf < inf["human_review_threshold"]
                findings.append(
                    DamageFinding(
                        class_name=name,
                        confidence=conf,
                        bbox=xyxy,
                        severity=sev_name,
                        severity_confidence=sev_conf,
                        needs_review=needs_review,
                    )
                )
                draw_boxes.append(xyxy)
                draw_labels.append(name)
                draw_sevs.append(sev_name)
                if conf > max_damage_conf:
                    max_damage_conf = conf
                    primary_crop = crop
        else:
            product_detected = True
            primary_crop = bgr

        annotated = self._annotate(bgr, draw_boxes, draw_labels, draw_sevs)
        ann_path = self.out_dir / "cases" / case_id / "annotated.jpg"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ann_path), annotated)

        gradcam_path = self.out_dir / "cases" / case_id / "gradcam.jpg"
        overlay, _, _ = generate_gradcam(self.severity_model, primary_crop, device=self.device)
        save_gradcam(overlay, gradcam_path)

        severity_order = {"Mild": 0, "Moderate": 1, "Severe": 2}
        overall = "Mild"
        if findings:
            overall = max(findings, key=lambda f: severity_order.get(f.severity, 0)).severity
        flagged = any(f.needs_review for f in findings) or not product_detected

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
            message="Analysis complete",
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
