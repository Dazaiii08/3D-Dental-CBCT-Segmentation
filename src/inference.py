import sys
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet
from visualize import load_model, predict_volume
from postprocess import postprocess_volume


def run_inference(nii_path, checkpoint_path, output_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading scan: {nii_path}")
    nii = nib.load(str(nii_path))
    img = nii.get_fdata(dtype=np.float32)
    if img.max() > 0:
        img = img / img.max()

    print(f"Volume shape: {img.shape}")

    model = load_model(checkpoint_path, device)
    print("Running inference...")
    pred_raw = predict_volume(model, img, device)

    print("Postprocessing...")
    pred_clean = postprocess_volume(pred_raw)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save prediction as NIfTI (same space as input)
    pred_nii = nib.Nifti1Image(pred_clean, affine=nii.affine, header=nii.header)
    out_path = out_dir / f"{Path(nii_path).stem}_prediction.nii"
    nib.save(pred_nii, str(out_path))

    print(f"Saved prediction → {out_path}")
    print(f"Tooth voxels found: {int(pred_clean.sum()):,}")
    return pred_clean


if __name__ == "__main__":
    # Example: run on first test patient's raw NIfTI
    from dataset import get_file_splits

    PROCESSED_DIR = r"D:\dental-segmentation\data\processed"
    RAW_DIR = r"D:\dental-segmentation\data\raw\CBCT_upload\images"
    CHECKPOINT = r"D:\dental-segmentation\outputs\checkpoints\best_model.pth"
    OUTPUT_DIR = r"D:\dental-segmentation\outputs\predictions"

    _, _, test_stems = get_file_splits(PROCESSED_DIR)
    stem = test_stems[0]

    run_inference(
        nii_path=Path(RAW_DIR) / f"{stem}.nii",
        checkpoint_path=CHECKPOINT,
        output_dir=OUTPUT_DIR
    )