import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet
from dataset import DentalDataset, get_file_splits


def dice_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def evaluate():
    PROCESSED_DIR = r"D:\dental-segmentation\data\processed"
    CHECKPOINT = r"D:\dental-segmentation\outputs\checkpoints\best_model.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_stems = get_file_splits(PROCESSED_DIR)
    print(f"Test patients: {len(test_stems)}")

    test_dataset = DentalDataset(PROCESSED_DIR, test_stems)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    all_dice, all_iou = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            for i in range(images.shape[0]):
                all_dice.append(dice_score(outputs[i], labels[i]))
                all_iou.append(iou_score(outputs[i], labels[i]))

    print(f"\n=== Test Set Results ===")
    print(f"Mean Dice Score : {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Mean IoU Score  : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"Median Dice     : {np.median(all_dice):.4f}")
    print(f"Min Dice        : {np.min(all_dice):.4f}")
    print(f"Max Dice        : {np.max(all_dice):.4f}")

    return np.mean(all_dice), np.mean(all_iou)


if __name__ == "__main__":
    evaluate()