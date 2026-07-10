# CBCT Dental Segmentation

Automated tooth segmentation from CBCT (Cone-Beam CT) scans using a 2D U-Net, wrapped in a FastAPI + vanilla JS web app for interactive 3D inference.

**Test Dice: 0.7479 · Test IoU: 0.6948 · 48 patients · 14,790 training slices**

---

## What it does

Upload a raw `.nii` CBCT scan → the model slices the volume, runs each slice through a trained U-Net on GPU, stitches predictions back into 3D, cleans up noise, and returns:

- An interactive 3D visualization (rotate/zoom, toggle CBCT scan on/off)
- A slice-by-slice comparison report (original vs. predicted)
- A downloadable predicted mask (`.nii`) and report (`.png`)
- Live, real progress updates while it processes (not a fake loading bar)

## Features

- 2D U-Net (7.7M params) trained on 48 patients / 14,790 axial slices
- FastAPI backend with background job processing and live status polling
- Interactive Plotly 3D viewer with a scan/teeth visibility toggle
- Slice comparison report generation (matplotlib)
- File validation, friendly error handling, backend-connectivity check
- Zero cloud dependency — runs entirely locally on your GPU

## Project structure

```
dental-segmentation/
├── app/
│   ├── main.py                  # FastAPI backend
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   ├── app.js
│   │   └── results/             # per-job outputs (viz, report, prediction)
│   └── uploads/                 # temp storage for uploaded scans
├── data/
│   ├── raw/CBCT_upload/images/  # raw .nii scans
│   └── processed/               # preprocessed .npy slices
├── src/
│   ├── dataset.py                # PyTorch Dataset + train/val/test split
│   ├── model.py                  # 2D U-Net architecture
│   ├── train.py                  # training loop
│   ├── evaluate.py               # test-set Dice/IoU evaluation
│   ├── postprocess.py            # noise removal, hole filling
│   ├── visualize.py               # inference + 3D plotly rendering
│   └── inference.py              # standalone .nii → .nii inference
├── outputs/
│   ├── checkpoints/best_model.pth
│   └── figures/
├── rank_test_patients.py         # ranks test patients by their own Dice score
├── requirements.txt
└── README.md
```

## Setup

**1. Create and activate a virtual environment** (Python 3.11):

```
python -m venv venv
```

Windows Command Prompt:
```
venv\Scripts\activate
```

Windows PowerShell:
```
.\venv\Scripts\Activate.ps1
```
> If PowerShell blocks the script with an execution-policy error, run this once first:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**2. Install PyTorch with CUDA first** (see note in `requirements.txt` — plain PyPI won't give you the CUDA build):

```
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**3. Install the rest:**

```
pip install -r requirements.txt
```

> **Windows drive-switching gotcha:** if your project is on a different drive than your terminal's current one, plain `cd D:\path` won't actually switch drives in Command Prompt — use `cd /d D:\path` instead. In PowerShell, plain `cd D:\path` works fine.

## Running the app

```
cd app
python main.py
```

Once you see `Application startup complete`, open **http://127.0.0.1:8000** in your browser, upload a `.nii` scan, and click **Run Segmentation**.

A scan typically takes 30–50 seconds end to end on an RTX 4050 (6GB VRAM).

## Finding a good demo scan

Run this to rank your test-set patients by their own Dice score (useful for picking a clean scan for a live demo):

```
python rank_test_patients.py
```

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Upload a `.nii` scan, starts a background inference job, returns a `job_id` |
| `/status/{job_id}` | GET | Live processing status — current step + progress % |
| `/download/nii/{job_id}` | GET | Download the predicted segmentation mask (`.nii`) |
| `/download/report/{job_id}` | GET | Download the slice-comparison report (`.png`) |
| `/health` | GET | Backend/model status, used for connectivity checks |

## Model

| | |
|---|---|
| Architecture | 2D U-Net, 7.7M parameters |
| Input | 400×400 single-channel grayscale slice |
| Output | 400×400 binary mask (0 = background, 1 = tooth) |
| Loss / Optimizer | BCEWithLogitsLoss / Adam |
| Epochs | 10 |
| Training data | 48 patients, 14,790 slices |
| Test Dice | 0.7479 (mean), range 0.6172–0.8864 across test patients |
| Test IoU | 0.6948 |

## Known limitations

Predictions are made independently per 2D slice, so there's no explicit constraint enforcing consistency between adjacent slices — this can cause fragmented (disconnected) predictions in 3D, more noticeably on lower-scoring scans. A 3D U-Net or a slice-consistency loss term would likely address this; not implemented here due to GPU memory constraints (6GB VRAM).

## Tech stack

Python, PyTorch, FastAPI, Uvicorn, Plotly, Matplotlib, NiBabel, vanilla HTML/CSS/JS.

## Author

Manvansh Singh — B.Tech Robotics & AI, Chandigarh Engineering College, CGC Landran
