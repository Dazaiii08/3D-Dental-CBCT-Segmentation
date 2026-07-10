"""
Ranks every test-set patient by their own mean Dice score, so you can pick
the cleanest-looking scan for your demo instead of guessing.

Run this from your activated venv:
    cd D:\\dental-segmentation
    python rank_test_patients.py

Does NOT modify anything in src/ or outputs/ — read-only, just reports.
"""

import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(r"D:\dental-segmentation")
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model import UNet          # noqa: E402
from dataset import get_file_splits  # noqa: E402

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw" / "CBCT_upload" / "images"
CHECKPOINT = BASE_DIR / "outputs" / "checkpoints" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def evaluate_patient(model, stem, batch_size=16):
    img = np.load(str(PROCESSED_DIR / f"{stem}_img.npy"))
    lbl = np.load(str(PROCESSED_DIR / f"{stem}_lbl.npy"))
    D = img.shape[2]
    dice_scores = []

    with torch.no_grad():
        for start in range(0, D, batch_size):
            end = min(start + batch_size, D)
            img_batch = img[:, :, start:end].transpose(2, 0, 1)[:, np.newaxis, :, :]
            lbl_batch = lbl[:, :, start:end].transpose(2, 0, 1)[:, np.newaxis, :, :]

            img_tensor = torch.tensor(img_batch, dtype=torch.float32).to(DEVICE)
            lbl_tensor = torch.tensor(lbl_batch, dtype=torch.float32).to(DEVICE)

            outputs = model(img_tensor)
            for i in range(outputs.shape[0]):
                dice_scores.append(dice_score(outputs[i], lbl_tensor[i]))

    return float(np.mean(dice_scores))


def main():
    print(f"Using device: {DEVICE}")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True))
    model.eval()

    _, _, test_stems = get_file_splits(str(PROCESSED_DIR))
    print(f"Test patients: {len(test_stems)}\n")

    results = []
    for stem in test_stems:
        try:
            mean_dice = evaluate_patient(model, stem)
            has_raw = (RAW_DIR / f"{stem}.nii").exists()
            results.append((stem, mean_dice, has_raw))
            print(f"  {stem:12s}  Dice: {mean_dice:.4f}  {'(raw .nii available)' if has_raw else '(no raw .nii found)'}")
        except FileNotFoundError:
            print(f"  {stem:12s}  skipped (processed .npy not found)")

    results.sort(key=lambda r: r[1], reverse=True)

    print("\n=== Ranked best-to-worst (only ones with a raw .nii you can upload) ===")
    uploadable = [r for r in results if r[2]]
    for stem, dice, _ in uploadable[:10]:
        print(f"  {stem:12s}  Dice: {dice:.4f}")

    if uploadable:
        best = uploadable[0]
        print(f"\nBest bet for your demo: {best[0]}.nii  (Dice {best[1]:.4f})")
        print(f"Path: {RAW_DIR / (best[0] + '.nii')}")
    else:
        print("\nNone of your test patients have a matching raw .nii in CBCT_upload/images —")
        print("you'll need to upload from the training/val set instead, or use a1028 as before.")


if __name__ == "__main__":
    main()
