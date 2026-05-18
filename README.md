# Luminous Transit Damage (TD) AI

Deep learning pipeline for dealer-uploaded transit damage cases: **YOLOv8m detection** + **EfficientNet severity** (RGB + edge 4th channel) + **Grad-CAM** + **FastAPI** + **Streamlit**.

## 6-week plan

| Week | Phase | Deliverable |
|------|--------|-------------|
| 1 | Data collection & labeling | `scripts/run_phase1_ingest.py`, Roboflow export in `data/labeling/` |
| 2 | EDA & preprocessing | Class imbalance report, edge channel in `src/preprocessing/edge_channel.py` |
| 3 | Model training | `models/detector_best.pt`, `models/severity_best.pt` |
| 4 | Explainability | Grad-CAM in `src/explainability/gradcam.py`, review flags |
| 5 | Evaluation | mAP / precision / recall via `scripts/run_phase5_eval.py` |
| 6 | Deployment | `api/main.py`, `dashboard/app.py` (4 pages) |

## Pipeline flow

```
Dealer image → Quality check → Product/damage detection (YOLO)
→ Edge-enhanced severity (EfficientNet 4ch) → Grad-CAM + report
```

## Quick start

```bash
cd project_td
pip install -r requirements.txt
python scripts/run_setup.py          # ingest, bootstrap labels, quick train
# Dashboard UI (open this in browser):
python -m streamlit run dashboard/app.py --server.port 8501
# -> http://localhost:8501

# API backend (optional; dashboard falls back to local pipeline):
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Or start both:
python scripts/start_all.py
```

**Important:** `UploadTdcaseDocuments` is unlabeled. Bootstrap weak labels enable training immediately; replace with **Roboflow/CVAT** exports in `data/labels/yolo/` for production accuracy.

## Labeling (recommended)

1. Run Phase 1 → `data/labeling/roboflow_export/images`
2. Annotate classes: `luminous_product`, `dent`, `scratch`, `crack`, …
3. Export YOLOv8 → `data/labels/yolo/{train,val,test}/`
4. `python scripts/run_phase3_train.py`

## Config

Edit `configs/config.yaml` for classes, thresholds, and augmentation.
