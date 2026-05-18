"""4th input channel: Canny edge map fused with RGB for severity head."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def compute_edge_channel(bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    return edges.astype(np.float32) / 255.0


def rgb_edge_4channel(bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Returns (4, H, W) float32 in [0,1]."""
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = resized.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # CHW
    edge = compute_edge_channel(resized)
    edge = edge[np.newaxis, ...]
    return np.concatenate([rgb, edge], axis=0)


def to_tensor_4ch(bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    arr = rgb_edge_4channel(bgr, size=size)
    return torch.from_numpy(arr).float()
