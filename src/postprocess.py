import numpy as np
from scipy import ndimage


def remove_small_components(pred_mask, min_size=500):
    """Remove isolated small predicted regions (noise)."""
    labeled, num_features = ndimage.label(pred_mask)
    cleaned = np.zeros_like(pred_mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() >= min_size:
            cleaned[component] = 1
    return cleaned


def fill_holes(pred_mask):
    """Fill small holes inside predicted tooth regions."""
    return ndimage.binary_fill_holes(pred_mask).astype(np.float32)


def postprocess_volume(pred_volume, min_size=500):
    """Apply postprocessing to a full 3D predicted volume."""
    cleaned = remove_small_components(pred_volume > 0.5, min_size=min_size)
    filled = fill_holes(cleaned)
    return filled.astype(np.float32)


if __name__ == "__main__":
    import sys
    import torch
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from model import UNet
    from dataset import get_file_splits
    from visualize import load_model, predict_volume

    PROCESSED_DIR = r"D:\dental-segmentation\data\processed"
    CHECKPOINT = r"D:\dental-segmentation\outputs\checkpoints\best_model.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_stems = get_file_splits(PROCESSED_DIR)
    stem = test_stems[0]

    processed = Path(PROCESSED_DIR)
    img_volume = np.load(str(processed / f"{stem}_img.npy"))
    lbl_volume = np.load(str(processed / f"{stem}_lbl.npy"))

    model = load_model(CHECKPOINT, DEVICE)
    pred_raw = predict_volume(model, img_volume, DEVICE)
    pred_clean = postprocess_volume(pred_raw)

    raw_voxels = (pred_raw > 0.5).sum()
    clean_voxels = pred_clean.sum()
    removed = raw_voxels - clean_voxels

    print(f"Patient: {stem}")
    print(f"Raw prediction voxels    : {int(raw_voxels):,}")
    print(f"After postprocessing     : {int(clean_voxels):,}")
    print(f"Noise voxels removed     : {int(removed):,}")
    print(f"Reduction                : {100*removed/max(raw_voxels,1):.1f}%")