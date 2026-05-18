"""Phase 1: ingest unclean dealer uploads — images, PDFs, dedupe, inventory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import fitz  # pymupdf
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import ROOT, load_config, resolve_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}
SKIP_EXTS = {".docx", ".doc", ".xlsx", ".txt"}


def file_hash(path: Path, block: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        stem = f"{pdf_path.stem}_page{i+1:02d}"
        out_path = out_dir / f"{stem}.jpg"
        pix.save(str(out_path))
        saved.append(out_path)
    doc.close()
    return saved


def normalize_image(src: Path, dst: Path, max_side: int = 1920) -> bool:
    try:
        img = Image.open(src).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=92, optimize=True)
    return True


def ingest_raw(
    upload_dir: Path | None = None,
    clean_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    cfg = load_config()
    upload = upload_dir or resolve_path(cfg["paths"]["raw_upload"])
    clean = clean_dir or resolve_path(f"{cfg['paths']['data_root']}/clean/images")
    pdf_extract = clean.parent / "pdf_pages"
    clean.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_files": 0,
        "images_copied": 0,
        "pdfs_extracted": 0,
        "skipped_doc": 0,
        "duplicates": 0,
        "failed": 0,
    }
    seen_hashes: set[str] = set()
    inventory: list[dict] = []

    files = sorted([p for p in upload.iterdir() if p.is_file()])
    stats["total_files"] = len(files)

    for src in tqdm(files, desc="Ingesting uploads"):
        ext = src.suffix.lower()
        if ext in SKIP_EXTS:
            stats["skipped_doc"] += 1
            continue

        paths_to_process: list[Path] = []
        if ext in PDF_EXTS:
            try:
                paths_to_process = pdf_to_images(src, pdf_extract)
                stats["pdfs_extracted"] += len(paths_to_process)
            except Exception:
                stats["failed"] += 1
                continue
        elif ext in IMAGE_EXTS:
            paths_to_process = [src]
        else:
            stats["failed"] += 1
            continue

        for img_path in paths_to_process:
            try:
                h = file_hash(img_path)
            except Exception:
                stats["failed"] += 1
                continue
            if h in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(h)

            case_id = img_path.stem
            dst = clean / f"{case_id}.jpg"
            if normalize_image(img_path, dst):
                stats["images_copied"] += 1
                inventory.append(
                    {
                        "case_id": case_id,
                        "source": str(src.name),
                        "clean_path": str(dst.relative_to(ROOT)),
                        "hash": h,
                    }
                )
            else:
                stats["failed"] += 1

    inv_path = clean.parent / "inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    report = report_path or resolve_path(f"{cfg['paths']['data_root']}/reports/ingest_report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    result = ingest_raw()
    print(result)
