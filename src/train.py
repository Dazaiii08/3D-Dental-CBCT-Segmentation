import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet
from dataset import DentalDataset, get_file_splits


def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(outputs, labels).item()
    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_dice += dice_score(outputs, labels).item()
    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    PROCESSED_DIR = r"D:\dental-segmentation\data\processed"
    CHECKPOINT_DIR = r"D:\dental-segmentation\outputs\checkpoints"
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    train_stems, val_stems, test_stems = get_file_splits(PROCESSED_DIR)
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")

    train_dataset = DentalDataset(PROCESSED_DIR, train_stems)
    val_dataset = DentalDataset(PROCESSED_DIR, val_stems)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_dice = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"Model saved! Best Dice: {best_dice:.4f}")

    print(f"\nTraining complete! Best Validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()