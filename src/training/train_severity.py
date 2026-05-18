"""Train EfficientNet severity head on damage crops (4-channel RGB+edge)."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config, resolve_path
from src.models.severity_head import SeverityNet4Ch
from src.preprocessing.edge_channel import rgb_edge_4channel


class SeverityCropDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, size: int = 224):
        self.size = size
        self.samples: list[tuple[Path, int]] = []
        for lbl in label_dir.glob("*.txt"):
            img = image_dir / f"{lbl.stem}.jpg"
            if not img.exists():
                continue
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5 or int(parts[0]) == 0:
                    continue
                _, xc, yc, w, h = map(float, parts[:5])
                area = w * h
                if area < 0.08:
                    sev = 0
                elif area < 0.22:
                    sev = 1
                else:
                    sev = 2
                self.samples.append((img, sev))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, sev = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        ch, cw = int(h * 0.5), int(w * 0.5)
        y1, x1 = max(0, ch - 112), max(0, cw - 112)
        crop = bgr[y1 : y1 + 224, x1 : x1 + 224]
        if crop.shape[0] < 224 or crop.shape[1] < 224:
            crop = cv2.resize(bgr, (224, 224))
        tensor = torch.from_numpy(rgb_edge_4channel(crop, self.size))
        return tensor, sev


def train_severity(
    labels_root: Path | None = None,
    epochs: int | None = None,
    device: str | None = None,
) -> Path:
    cfg = load_config()
    labels_root = labels_root or resolve_path(f"{cfg['paths']['data_root']}/labels/yolo")
    sev_cfg = cfg["severity"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SeverityCropDataset(
        labels_root / "train" / "images",
        labels_root / "train" / "labels",
        size=sev_cfg["input_size"],
    )
    if len(train_ds) < 10:
        raise RuntimeError("Not enough damage crops for severity training. Run bootstrap_labels first.")
    loader = DataLoader(train_ds, batch_size=sev_cfg["batch"], shuffle=True, num_workers=0)
    model = SeverityNet4Ch(num_classes=len(cfg["severity_classes"]), backbone=sev_cfg["backbone"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=sev_cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    n_epochs = epochs or sev_cfg["epochs"]

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"Severity epoch {epoch+1}/{n_epochs}"):
            x, y = x.to(device), torch.tensor(y, device=device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1} loss={total_loss/len(loader):.4f}")

    out = resolve_path(f"{cfg['paths']['models_dir']}/severity_best.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    return out
