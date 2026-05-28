"""
Production damage detection: YOLO-World (open vocabulary) + classical CV fusion.
No custom labels required — works on dealer phone photos out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Open-vocabulary prompts tuned for Luminous batteries / inverters / packaging
DAMAGE_PROMPTS = [
    "dent on product",
    "scratch on product surface",
    "crack on battery or plastic case",
    "rust or corrosion on metal",
    "damaged torn cardboard box",
    "broken battery terminal",
    "torn or damaged product label",
    "liquid leak stain",
    "physical damage on electronic device",
    "deformed swollen battery",
]

PRODUCT_PROMPTS = [
    "battery",
    "inverter",
    "cardboard box",
    "electronic device",
    "solar panel",
]

PROMPT_TO_CLASS = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "rust": "corrosion",
    "corrosion": "corrosion",
    "cardboard": "packaging_damage",
    "packaging": "packaging_damage",
    "box": "packaging_damage",
    "terminal": "terminal_damage",
    "label": "label_damage",
    "leak": "leak",
    "liquid": "leak",
    "swollen": "dent",
    "deform": "dent",
    "damage": "dent",
    "physical": "dent",
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy
    source: str  # yolo_world | cv_dent | cv_scratch | cv_anomaly


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(dets: list[Detection], iou_thresh: float = 0.45) -> list[Detection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    for d in dets:
        if all(_iou(d.bbox, k.bbox) < iou_thresh or d.class_name != k.class_name for k in keep):
            keep.append(d)
    return keep


def _map_prompt_label(label: str) -> str:
    low = label.lower()
    for key, cls in PROMPT_TO_CLASS.items():
        if key in low:
            return cls
    return "dent"


def _product_roi(bgr: np.ndarray, product_boxes: list[tuple[float, float, float, float]]) -> tuple[int, int, int, int]:
    h, w = bgr.shape[:2]
    if product_boxes:
        x1 = max(0, int(min(b[0] for b in product_boxes)))
        y1 = max(0, int(min(b[1] for b in product_boxes)))
        x2 = min(w, int(max(b[2] for b in product_boxes)))
        y2 = min(h, int(max(b[3] for b in product_boxes)))
        pad_x, pad_y = int(0.03 * w), int(0.03 * h)
        return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    # Center crop — most dealer photos frame the product
    mx, my = int(w * 0.08), int(h * 0.08)
    return mx, my, w - mx, h - my


class CVDamageDetector:
    """Classical computer vision: dents (black-hat), scratches (edges), stains (color)."""

    def detect(self, bgr: np.ndarray, roi: tuple[int, int, int, int]) -> list[Detection]:
        x1, y1, x2, y2 = roi
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        dets: list[Detection] = []

        # Dents / depressions: black-hat highlights dark valleys
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, dent_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
        dets.extend(self._mask_to_dets(dent_mask, "dent", x1, y1, w, h, min_area=0.0025, source="cv_dent", scale=0.62))

        # Scratches: strong oriented edges
        edges = cv2.Canny(gray, 80, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        dets.extend(self._mask_to_dets(edges, "scratch", x1, y1, w, h, min_area=0.0012, source="cv_scratch", scale=0.58))

        # Corrosion / leaks: saturation + dark outliers in HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        stain = cv2.bitwise_or(
            cv2.threshold(sat, 140, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(255 - val, 50, 255, cv2.THRESH_BINARY)[1],
        )
        stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        dets.extend(self._mask_to_dets(stain, "corrosion", x1, y1, w, h, min_area=0.002, source="cv_anomaly", scale=0.48))

        # Cracks: thin long contours on edge map
        crack_edges = cv2.Canny(gray, 30, 90)
        crack_edges = cv2.morphologyEx(crack_edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dets.extend(self._mask_to_dets(crack_edges, "crack", x1, y1, w, h, min_area=0.0003, source="cv_anomaly", scale=0.52, max_area=0.15))

        return dets

    def _mask_to_dets(
        self,
        mask: np.ndarray,
        cls: str,
        ox: int,
        oy: int,
        w: int,
        h: int,
        min_area: float,
        source: str,
        scale: float,
        max_area: float = 0.4,
    ) -> list[Detection]:
        out: list[Detection] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = w * h
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area * img_area or area > max_area * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8:
                continue
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if cls == "scratch" and aspect < 2.0 and area < 0.01 * img_area:
                continue
            conf = min(0.92, scale + 0.35 * (area / img_area) ** 0.3)
            out.append(
                Detection(
                    class_name=cls,
                    confidence=float(conf),
                    bbox=(float(ox + x), float(oy + y), float(ox + x + bw), float(oy + y + bh)),
                    source=source,
                )
            )
        return out


@lru_cache(maxsize=1)
def _load_yolo_world():
    from ultralytics import YOLOWorld

    root = Path(__file__).resolve().parents[2]
    weights = root / "yolov8s-worldv2.pt"
    model = YOLOWorld(str(weights) if weights.exists() else "yolov8s-worldv2.pt")
    return model


class DamageDetectionEngine:
    def __init__(self, conf: float = 0.12, iou: float = 0.45, use_world: bool = True):
        self.conf = conf
        self.iou = iou
        self.use_world = use_world
        self.cv = CVDamageDetector()
        self._world = None
        self._world_ok = False

    def _ensure_world(self) -> bool:
        if not self.use_world:
            return False
        if self._world_ok:
            return True
        try:
            self._world = _load_yolo_world()
            self._world.set_classes(DAMAGE_PROMPTS + PRODUCT_PROMPTS)
            self._world_ok = True
            return True
        except Exception:
            self._world_ok = False
            return False

    def detect(self, bgr: np.ndarray) -> tuple[list[Detection], list[tuple[float, float, float, float]], np.ndarray]:
        """
        Returns (damage_detections, product_boxes, saliency_heatmap BGR for explainability).
        """
        h, w = bgr.shape[:2]
        all_dets: list[Detection] = []
        product_boxes: list[tuple[float, float, float, float]] = []

        if self._ensure_world():
            results = self._world.predict(bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
            if results.boxes is not None:
                names = results.names or {}
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    xyxy = tuple(box.xyxy[0].cpu().numpy().tolist())
                    low = label.lower()
                    if any(p in low for p in ("battery", "inverter", "box", "device", "solar", "electronic")):
                        if conf >= 0.15:
                            product_boxes.append(xyxy)
                    else:
                        cls = _map_prompt_label(label)
                        all_dets.append(
                            Detection(class_name=cls, confidence=conf, bbox=xyxy, source="yolo_world")
                        )

        roi = _product_roi(bgr, product_boxes)
        cv_dets = self.cv.detect(bgr, roi)
        all_dets.extend(cv_dets)

        merged = _nms(all_dets, self.iou)
        img_area = h * w
        filtered = []
        for d in merged:
            x1, y1, x2, y2 = d.bbox
            ba = (x2 - x1) * (y2 - y1)
            if ba > 0.70 * img_area or ba < 0.0008 * img_area:
                continue
            # CV-only needs higher confidence to reduce texture false alarms
            if d.source.startswith("cv_") and d.confidence < 0.45:
                continue
            filtered.append(d)

        filtered.sort(key=lambda d: (d.source == "yolo_world", d.confidence), reverse=True)
        if len(filtered) > 6:
            filtered = [d for d in filtered if d.confidence >= 0.38][:8]

        heatmap = self._build_heatmap(bgr, roi, filtered)
        return filtered, product_boxes, heatmap

    def _build_heatmap(self, bgr: np.ndarray, roi: tuple[int, int, int, int], dets: list[Detection]) -> np.ndarray:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = roi
        crop = bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        edges = cv2.Canny(gray, 50, 130)
        sal = cv2.normalize(bh.astype(np.float32) + edges.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        sal = sal.astype(np.uint8)
        sal_full = np.zeros((h, w), dtype=np.uint8)
        sal_full[y1:y2, x1:x2] = sal
        for d in dets:
            bx1, by1, bx2, by2 = map(int, d.bbox)
            sal_full[by1:by2, bx1:bx2] = np.clip(sal_full[by1:by2, bx1:bx2].astype(np.int32) + 80, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(sal_full, cv2.COLORMAP_JET)
        return cv2.addWeighted(bgr, 0.55, colored, 0.45, 0)


def score_severity(bgr: np.ndarray, bbox: tuple[float, float, float, float], class_name: str) -> tuple[str, float]:
    """Visual severity from defect size, contrast, and edge strength (no unreliable classifier)."""
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "Mild", 0.5
    h, w = bgr.shape[:2]
    rel_area = ((x2 - x1) * (y2 - y1)) / (h * w + 1e-6)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge_strength = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    score = 0.0
    score += min(1.0, rel_area * 25) * 0.45
    score += min(1.0, edge_strength / 800) * 0.35
    score += min(1.0, contrast / 60) * 0.20
    if class_name in ("crack", "leak", "terminal_damage"):
        score += 0.15
    if class_name == "packaging_damage" and rel_area > 0.08:
        score += 0.1

    score = min(1.0, score)
    if score < 0.28:
        return "Mild", 0.55 + score
    if score < 0.55:
        return "Moderate", 0.60 + score * 0.35
    return "Severe", 0.70 + score * 0.25
