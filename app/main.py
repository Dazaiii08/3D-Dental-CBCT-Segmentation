"""
FastAPI wrapper around the existing dental CBCT segmentation pipeline.
Does NOT modify anything in src/ — imports and reuses load_model,
postprocess_volume directly. Adds a job-based background pipeline so the
frontend can poll REAL progress (not simulated) while GPU inference runs.
"""

import re
import sys
import time
import shutil
import uuid
import threading
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # D:\dental-segmentation
SRC_DIR = BASE_DIR / "src"
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"

APP_DIR = Path(__file__).resolve().parent                  # D:\dental-segmentation\app
UPLOAD_DIR = APP_DIR / "uploads"
STATIC_DIR = APP_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from visualize import load_model  # noqa: E402
from postprocess import postprocess_volume  # noqa: E402

# ---------------------------------------------------------------------
# Model metadata shown on the "Model Info" card — from your actual
# training run (train.py / evaluate.py results)
# ---------------------------------------------------------------------
MODEL_INFO = {
    "architecture": "2D U-Net (7.7M parameters)",
    "input_size": "400 x 400 (single-channel grayscale slice)",
    "training_patients": 48,
    "training_slices": 14790,
    "epochs": 10,
    "loss_function": "BCEWithLogitsLoss",
    "optimizer": "Adam",
    "test_dice": 0.7479,
    "test_iou": 0.6948,
}

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(title="DentaSeg Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
INFERENCE_LOCK = threading.Lock()   # serialize GPU access across jobs

JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


@app.on_event("startup")
def startup_event():
    global MODEL
    print(f"[startup] Using device: {DEVICE}")
    print(f"[startup] Loading checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    MODEL = load_model(str(CHECKPOINT_PATH), DEVICE)
    print("[startup] Model loaded and ready.")


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "model_info": MODEL_INFO,
    }


# ---------------------------------------------------------------------
# Pipeline pieces (deliberately re-implemented here, not imported from
# src/visualize.py, so we can hook in real progress reporting and add
# the scan/teeth visibility toggle without touching src/ at all)
# ---------------------------------------------------------------------

def run_inference_with_progress(model, img_volume, device, job_id, batch_size=16):
    H, W, D = img_volume.shape
    pred_volume = np.zeros((H, W, D), dtype=np.float32)
    slices = [img_volume[:, :, s].astype(np.float32) for s in range(D)]

    with torch.no_grad():
        for start in range(0, D, batch_size):
            batch = slices[start:start + batch_size]
            batch_tensor = torch.tensor(np.array(batch)[:, np.newaxis, :, :]).to(device)
            outputs = torch.sigmoid(model(batch_tensor))
            preds = (outputs > 0.5).float().cpu().numpy()
            for i, s in enumerate(range(start, min(start + batch_size, D))):
                pred_volume[:, :, s] = preds[i, 0]

            pct = min(100, round(((start + batch_size) / D) * 100))
            update_job(job_id, step="inference", progress=pct)

    return pred_volume


def find_tooth_slices(pred_volume, n=4, min_voxels=50):
    D = pred_volume.shape[2]
    candidates = [i for i in range(D) if pred_volume[:, :, i].sum() > min_voxels]
    if len(candidates) < n:
        candidates = list(range(0, D, max(1, D // n)))
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


def generate_report_image(img_volume, pred_volume, stats, out_path):
    """Combined slice-comparison + stats report — used for both the
    inline 'slice comparison viewer' and the downloadable report PNG."""
    slices = find_tooth_slices(pred_volume, n=4)

    fig = plt.figure(figsize=(11, 3 * len(slices) + 1.6), facecolor="white")
    gs = fig.add_gridspec(len(slices) + 1, 2, height_ratios=[0.5] + [1] * len(slices))

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    header_text = (
        f"DentaSeg — Segmentation Report\n"
        f"File: {stats['filename']}    |    Volume: {stats['volume_shape']}    |    "
        f"Tooth Voxels: {stats['tooth_voxels']:,}    |    Time: {stats['processing_seconds']}s\n"
        f"Model: {MODEL_INFO['architecture']}    |    Test Dice: {MODEL_INFO['test_dice']}    |    "
        f"Test IoU: {MODEL_INFO['test_iou']}"
    )
    ax_header.text(0.0, 0.5, header_text, fontsize=11, va="center", family="monospace")

    for i, s in enumerate(slices):
        ax1 = fig.add_subplot(gs[i + 1, 0])
        ax1.imshow(img_volume[:, :, s], cmap="gray")
        ax1.set_title(f"Slice {s} — Original", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[i + 1, 1])
        ax2.imshow(img_volume[:, :, s], cmap="gray")
        ax2.imshow(pred_volume[:, :, s], alpha=0.45, cmap="cool")
        ax2.set_title(f"Slice {s} — Predicted Teeth", fontsize=10)
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_3d_html_with_toggle(img_volume, pred_volume, out_path):
    """Same visualization as src/visualize.py's save_3d_visualization,
    plus a toggle for CBCT scan visibility. Uses custom-styled HTML/JS
    buttons instead of Plotly's built-in updatemenus, since the native
    button menu doesn't give enough control over active-state styling
    (was washing out to unreadable text). aspectmode='data' keeps the
    scene sized to the real data proportions instead of a forced cube,
    which was clipping the point cloud."""
    step = 4
    img_ds = img_volume[::step, ::step, ::step]
    pred_ds = pred_volume[::step, ::step, ::step]

    H, W, D = img_ds.shape
    xs, ys, zs = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="xy")

    thresh = 0.35
    mask_scan = img_ds > thresh

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs[mask_scan].flatten(), y=ys[mask_scan].flatten(), z=zs[mask_scan].flatten(),
        mode="markers",
        marker=dict(size=1.4, color=img_ds[mask_scan].flatten(), colorscale="Gray", opacity=0.25),
        name="CBCT Scan",
        visible=False,
    ))

    mask_pred = pred_ds > 0.5
    fig.add_trace(go.Scatter3d(
        x=xs[mask_pred].flatten(), y=ys[mask_pred].flatten(), z=zs[mask_pred].flatten(),
        mode="markers",
        marker=dict(size=2.5, color="cyan", opacity=0.7),
        name="Predicted Teeth",
        visible=True,
    ))

    fig.update_layout(
        title=dict(text="3D CBCT Scan with Predicted Tooth Segmentation", font=dict(size=16)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z (Slice)",
            bgcolor="black",
            xaxis=dict(backgroundcolor="black", color="white"),
            yaxis=dict(backgroundcolor="black", color="white"),
            zaxis=dict(backgroundcolor="black", color="white"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(font=dict(size=12)),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True, div_id="dentaviz")

    # Custom toggle bar — full styling control, no Plotly native-button
    # contrast issues. Talks to the plot via Plotly.restyle().
    control_style = """
    <style>
      body { margin: 0; position: relative; background: black; }
      .scan-toggle-bar {
        position: absolute; bottom: 16px; left: 16px; z-index: 10;
        display: flex; gap: 8px; padding: 8px;
        background: rgba(10, 14, 19, 0.6); border-radius: 10px;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
      }
      .scan-toggle-btn {
        font-size: 12px; padding: 9px 16px; border-radius: 6px;
        border: 1px solid #2dd4e8; background: #0d141a; color: #2dd4e8;
        cursor: pointer; transition: background 0.15s, color 0.15s;
      }
      .scan-toggle-btn:hover { background: #16232a; }
      .scan-toggle-btn.active { background: #2dd4e8; color: #05141a; font-weight: 600; }
    </style>
    """
    control_html = """
    <div class="scan-toggle-bar">
      <button id="btnTeethOnly" class="scan-toggle-btn active" type="button">Teeth Only</button>
      <button id="btnShowScan" class="scan-toggle-btn" type="button">Show Scan + Teeth</button>
    </div>
    """
    control_script = """
    <script>
      window.addEventListener('load', function() {
        var btnTeeth = document.getElementById('btnTeethOnly');
        var btnScan = document.getElementById('btnShowScan');
        function setActive(showScan) {
          Plotly.restyle('dentaviz', {visible: [showScan, true]}, [0, 1]);
          btnTeeth.classList.toggle('active', !showScan);
          btnScan.classList.toggle('active', showScan);
        }
        btnTeeth.addEventListener('click', function () { setActive(false); });
        btnScan.addEventListener('click', function () { setActive(true); });
      });
    </script>
    """

    html_str = html_str.replace("</head>", control_style + "</head>")
    html_str = re.sub(r"<body([^>]*)>", r"<body\1>" + control_html, html_str, count=1)
    html_str = html_str.replace("</body>", control_script + "</body>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return out_path


# ---------------------------------------------------------------------
# Background job pipeline
# ---------------------------------------------------------------------

def process_job(job_id: str, saved_path: Path, original_filename: str):
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        update_job(job_id, status="running", step="loading", progress=0, error=None)

        try:
            nii = nib.load(str(saved_path))
            img = nii.get_fdata(dtype=np.float32)
        except Exception:
            raise ValueError(
                "This doesn't look like a valid CBCT (.nii) file. "
                "Please check the file and try again."
            )

        if img.max() > 0:
            img = img / img.max()
        update_job(job_id, step="loading", progress=100)

        update_job(job_id, step="inference", progress=0)
        with INFERENCE_LOCK:
            pred_raw = run_inference_with_progress(MODEL, img, DEVICE, job_id)

        update_job(job_id, step="postprocess", progress=0)
        pred_clean = postprocess_volume(pred_raw)
        update_job(job_id, step="postprocess", progress=100)

        tooth_voxels = int(pred_clean.sum())
        elapsed_so_far = round(time.time() - t0, 1)

        stats = {
            "filename": original_filename,
            "volume_shape": list(img.shape),
            "tooth_voxels": tooth_voxels,
            "processing_seconds": elapsed_so_far,
        }

        update_job(job_id, step="report", progress=0)
        report_path = job_dir / "report.png"
        generate_report_image(img, pred_clean, stats, report_path)
        update_job(job_id, step="report", progress=100)

        update_job(job_id, step="visualization", progress=0)
        viz_path = job_dir / "viz.html"
        generate_3d_html_with_toggle(img, pred_clean, viz_path)
        update_job(job_id, step="visualization", progress=100)

        update_job(job_id, step="saving", progress=0)
        stem = Path(original_filename).stem.replace(".nii", "")
        nii_out_path = job_dir / f"{stem}_prediction.nii"
        pred_nii = nib.Nifti1Image(pred_clean, affine=nii.affine, header=nii.header)
        nib.save(pred_nii, str(nii_out_path))
        update_job(job_id, step="saving", progress=100)

        total_elapsed = round(time.time() - t0, 1)

        result = {
            "filename": original_filename,
            "volume_shape": list(img.shape),
            "tooth_voxels": tooth_voxels,
            "processing_seconds": total_elapsed,
            "html_url": f"/static/results/{job_id}/viz.html",
            "report_url": f"/static/results/{job_id}/report.png",
            "download_nii_url": f"/download/nii/{job_id}",
            "download_report_url": f"/download/report/{job_id}",
        }

        update_job(job_id, status="done", step="done", progress=100, result=result)

    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="error", error=str(e))

    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    name_lower = file.filename.lower()
    if not (name_lower.endswith(".nii") or name_lower.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Please upload a .nii or .nii.gz file.")

    job_id = uuid.uuid4().hex[:12]
    saved_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    try:
        size = 0
        with open(saved_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                f.write(chunk)
    finally:
        await file.close()

    if size == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "step": "queued", "progress": 0, "error": None, "result": None}

    thread = threading.Thread(target=process_job, args=(job_id, saved_path, file.filename), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job


@app.get("/download/nii/{job_id}")
def download_nii(job_id: str):
    job_dir = RESULTS_DIR / job_id
    matches = list(job_dir.glob("*_prediction.nii")) if job_dir.exists() else []
    if not matches:
        raise HTTPException(status_code=404, detail="Prediction file not found.")
    return FileResponse(str(matches[0]), filename=matches[0].name, media_type="application/octet-stream")


@app.get("/download/report/{job_id}")
def download_report(job_id: str):
    report_path = RESULTS_DIR / job_id / "report.png"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(str(report_path), filename=f"dentaseg_report_{job_id}.png", media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
