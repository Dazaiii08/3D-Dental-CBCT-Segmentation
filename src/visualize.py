import sys
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet
from dataset import get_file_splits


def load_model(checkpoint_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_volume(model, img_volume, device, batch_size=16):
    """Run model on all slices of a 3D volume."""
    H, W, D = img_volume.shape
    pred_volume = np.zeros((H, W, D), dtype=np.float32)

    slices = []
    for s in range(D):
        slc = img_volume[:, :, s].astype(np.float32)
        slices.append(slc)

    with torch.no_grad():
        for start in range(0, D, batch_size):
            batch = slices[start:start + batch_size]
            batch_tensor = torch.tensor(
                np.array(batch)[:, np.newaxis, :, :]
            ).to(device)
            outputs = torch.sigmoid(model(batch_tensor))
            preds = (outputs > 0.5).float().cpu().numpy()
            for i, s in enumerate(range(start, min(start + batch_size, D))):
                pred_volume[:, :, s] = preds[i, 0]

    return pred_volume


def save_slice_comparison(img_volume, lbl_volume, pred_volume, out_path, n_slices=6):
    """Save a grid of slice comparisons: original | ground truth | prediction."""
    D = img_volume.shape[2]
    # Pick evenly spaced slices that have tooth content
    tooth_slices = [i for i in range(D) if lbl_volume[:, :, i].sum() > 50]
    if len(tooth_slices) < n_slices:
        tooth_slices = list(range(0, D, D // n_slices))
    step = max(1, len(tooth_slices) // n_slices)
    selected = tooth_slices[::step][:n_slices]

    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices))
    fig.suptitle("Dental CBCT Segmentation Results", fontsize=16, fontweight='bold')

    for row, s in enumerate(selected):
        img = img_volume[:, :, s]
        lbl = lbl_volume[:, :, s]
        pred = pred_volume[:, :, s]

        # Original
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title(f'Slice {s} — Original CBCT')
        axes[row, 0].axis('off')

        # Ground truth overlay
        axes[row, 1].imshow(img, cmap='gray')
        axes[row, 1].imshow(lbl, alpha=0.4, cmap='Reds')
        axes[row, 1].set_title(f'Slice {s} — Ground Truth')
        axes[row, 1].axis('off')

        # Prediction overlay
        axes[row, 2].imshow(img, cmap='gray')
        axes[row, 2].imshow(pred, alpha=0.4, cmap='Blues')
        axes[row, 2].set_title(f'Slice {s} — Model Prediction')
        axes[row, 2].axis('off')

    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Ground Truth')
    blue_patch = mpatches.Patch(color='blue', alpha=0.4, label='Prediction')
    fig.legend(handles=[red_patch, blue_patch], loc='lower center',
               ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved slice comparison → {out_path}")


def save_3d_visualization(img_volume, pred_volume, out_path):
    """Save interactive 3D HTML: CBCT scan surface + predicted tooth mask."""
    print("Building 3D visualization (this takes ~1 min)...")

    # Downsample for performance
    step = 4
    img_ds = img_volume[::step, ::step, ::step]
    pred_ds = pred_volume[::step, ::step, ::step]

    H, W, D = img_ds.shape
    xs, ys, zs = np.meshgrid(
        np.arange(W), np.arange(H), np.arange(D), indexing='xy'
    )

    # CBCT scan — show voxels above intensity threshold
    thresh = 0.35
    mask_scan = img_ds > thresh
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs[mask_scan].flatten(),
        y=ys[mask_scan].flatten(),
        z=zs[mask_scan].flatten(),
        mode='markers',
        marker=dict(
            size=1.2,
            color=img_ds[mask_scan].flatten(),
            colorscale='Gray',
            opacity=0.08,
        ),
        name='CBCT Scan'
    ))

    # Predicted teeth — show in cyan
    mask_pred = pred_ds > 0.5
    if mask_pred.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=xs[mask_pred].flatten(),
            y=ys[mask_pred].flatten(),
            z=zs[mask_pred].flatten(),
            mode='markers',
            marker=dict(
                size=2.5,
                color='cyan',
                opacity=0.6,
            ),
            name='Predicted Teeth'
        ))

    fig.update_layout(
        title=dict(text='3D CBCT Scan with Predicted Tooth Segmentation',
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (Slice)',
            bgcolor='black',
            xaxis=dict(backgroundcolor='black', color='white'),
            yaxis=dict(backgroundcolor='black', color='white'),
            zaxis=dict(backgroundcolor='black', color='white'),
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(font=dict(size=12))
    )

    fig.write_html(str(out_path))
    print(f"Saved 3D visualization → {out_path}")


def main():
    PROCESSED_DIR = r"D:\dental-segmentation\data\processed"
    CHECKPOINT = r"D:\dental-segmentation\outputs\checkpoints\best_model.pth"
    FIGURES_DIR = Path(r"D:\dental-segmentation\outputs\figures")
    FIGURES_DIR.mkdir(exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    model = load_model(CHECKPOINT, DEVICE)

    # Pick one test patient to visualize
    _, _, test_stems = get_file_splits(PROCESSED_DIR)
    stem = test_stems[0]
    print(f"\nVisualizing patient: {stem}")

    processed = Path(PROCESSED_DIR)
    img_volume = np.load(str(processed / f"{stem}_img.npy"))
    lbl_volume = np.load(str(processed / f"{stem}_lbl.npy"))

    print(f"Volume shape: {img_volume.shape}")

    # Run inference on full volume
    pred_volume = predict_volume(model, img_volume, DEVICE)

    # 1. Slice comparison grid
    save_slice_comparison(
        img_volume, lbl_volume, pred_volume,
        FIGURES_DIR / "slice_comparison.png"
    )

    # 2. Interactive 3D HTML
    save_3d_visualization(
        img_volume, pred_volume,
        FIGURES_DIR / "3d_visualization.html"
    )

    print("\nAll visualizations saved!")
    print(f"  → {FIGURES_DIR / 'slice_comparison.png'}")
    print(f"  → {FIGURES_DIR / '3d_visualization.html'}")


if __name__ == "__main__":
    main()