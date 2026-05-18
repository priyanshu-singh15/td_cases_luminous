"""Image quality checks for dealer phone-camera uploads."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityResult:
    passed: bool
    blur_score: float
    brightness: float
    width: int
    height: int
    issues: list[str]


def laplacian_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_quality(
    image_bgr: np.ndarray,
    min_blur: float = 80.0,
    min_resolution: int = 320,
    min_brightness: float = 25.0,
    max_brightness: float = 240.0,
) -> QualityResult:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = laplacian_blur_score(gray)
    brightness = float(np.mean(gray))
    issues: list[str] = []

    if min(h, w) < min_resolution:
        issues.append(f"low_resolution ({w}x{h})")
    if blur < min_blur:
        issues.append(f"too_blurry (score={blur:.1f})")
    if brightness < min_brightness:
        issues.append(f"underexposed ({brightness:.1f})")
    if brightness > max_brightness:
        issues.append(f"overexposed ({brightness:.1f})")

    return QualityResult(
        passed=len(issues) == 0,
        blur_score=blur,
        brightness=brightness,
        width=w,
        height=h,
        issues=issues,
    )
