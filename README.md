# Dental CBCT Tooth Segmentation Pipeline

End-to-end deep learning pipeline for automated tooth segmentation on 3D Cone Beam CT (CBCT) scans, built for the Dobbe AI ML assignment.

## Results

| Metric | Score |
|--------|-------|
| Mean Dice Score | 0.7479 |
| Mean IoU Score | 0.6948 |
| Median Dice | 0.9104 |
| Best Val Dice (training) | 0.6825 |

## Project Structure
dental-segmentation/

├── data/
│   ├── raw/              ← original NIfTI scans
│   └── processed/        ← preprocessed .npy volumes
├── src/
│   ├── dataset.py        ← data loading and train/val/test splits
│   ├── model.py          ← U-Net architecture (7.76M parameters)
│   ├── train.py          ← training loop with Dice loss
│   ├── evaluate.py       ← test set evaluation (Dice, IoU)
│   ├── postprocess.py    ← noise removal and hole filling
│   ├── visualize.py      ← 2D slice grid + interactive 3D HTML
│   └── inference.py      ← run model on any new NIfTI scan
├── outputs/
│   ├── checkpoints/      ← best_model.pth
│   ├── predictions/      ← predicted segmentation NIfTI files
│   └── figures/          ← slice_comparison.png, 3d_visualization.html
├── report/               ← PDF report
└── requirements.txt


## Setup

**Requirements:** Python 3.11, NVIDIA GPU with CUDA

```bash
# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel scipy scikit-image matplotlib plotly tqdm scikit-learn
```

## Dataset

CBCT Teeth Segmentation dataset from Kaggle (`detectioncla/cbct-teeth-segmentation`).
50 CBCT volumes in NIfTI format with per-tooth segmentation labels.
Split: 34 train / 7 validation / 8 test patients.

```bash
# Download via Kaggle API
kaggle datasets download detectioncla/cbct-teeth-segmentation -p data/raw --unzip
```

## Preprocessing

Converts raw NIfTI volumes to numpy arrays, normalizes intensities to [0,1], and binarizes labels (tooth vs background):

```bash
# Run from explore.ipynb or adapt the preprocessing cell
```

## Training

```bash
python src/train.py
```

Trains a 2D U-Net slice-by-slice on axial CBCT slices. Hyperparameters:
- Batch size: 8
- Epochs: 10
- Learning rate: 1e-4 (with ReduceLROnPlateau)
- Loss: BCEWithLogitsLoss

## Evaluation

```bash
python src/evaluate.py
```

## Visualization

```bash
python src/visualize.py
```

Generates:
- `outputs/figures/slice_comparison.png` — 6-slice grid showing original, ground truth, and prediction
- `outputs/figures/3d_visualization.html` — interactive 3D render of scan + predicted teeth

## Inference on New Scans

```bash
python src/inference.py
```

Or use programmatically:

```python
from src.inference import run_inference
run_inference(
    nii_path="path/to/scan.nii",
    checkpoint_path="outputs/checkpoints/best_model.pth",
    output_dir="outputs/predictions"
)
```

## Model Architecture

2D U-Net with encoder-decoder structure and skip connections:
- Input: single-channel 400×400 axial CBCT slice
- Encoder: 4 downsampling blocks [32, 64, 128, 256 channels]
- Bottleneck: 512 channels
- Decoder: 4 upsampling blocks with skip connections
- Output: single-channel binary segmentation mask
- Parameters: 7,762,465