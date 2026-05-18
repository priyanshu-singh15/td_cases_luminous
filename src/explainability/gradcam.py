"""Grad-CAM heatmaps for severity decisions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.severity_head import SeverityNet4Ch
from src.preprocessing.edge_channel import rgb_edge_4channel


class _SeverityWrapper(torch.nn.Module):
    def __init__(self, model: SeverityNet4Ch):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def generate_gradcam(
    model: SeverityNet4Ch,
    bgr_crop: np.ndarray,
    target_class: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, int, float]:
    """Returns overlay BGR, predicted class, confidence."""
    size = 224
    bgr = cv2.resize(bgr_crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_edge_4channel(bgr, size)).unsqueeze(0).to(device)
    wrapped = _SeverityWrapper(model).to(device)
    wrapped.eval()
    target_layers = [model.encoder._blocks[-1]]
    cam = GradCAM(model=wrapped, target_layers=target_layers)
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_f, grayscale, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    return overlay_bgr, pred, conf


def save_gradcam(overlay: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
    return path
