import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DentalDataset(Dataset):
    def __init__(self, processed_dir, file_stems, slice_axis=2):
        self.processed_dir = Path(processed_dir)
        self.slice_axis = slice_axis
        self.slices = []

        print("Indexing dataset...")
        for stem in file_stems:
            img_path = self.processed_dir / f"{stem}_img.npy"
            lbl_path = self.processed_dir / f"{stem}_lbl.npy"
            if not img_path.exists() or not lbl_path.exists():
                continue
            # Get shape without loading into RAM
            with open(str(img_path), 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
            n_slices = shape[self.slice_axis]
            for s in range(n_slices):
                self.slices.append((stem, s))

        print(f"Total slices: {len(self.slices)}")
        self._cache_stem = None
        self._cache_img = None
        self._cache_lbl = None

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        stem, slice_idx = self.slices[idx]

        if stem != self._cache_stem:
            self._cache_img = np.load(str(self.processed_dir / f"{stem}_img.npy"))
            self._cache_lbl = np.load(str(self.processed_dir / f"{stem}_lbl.npy"))
            self._cache_stem = stem

        image = self._cache_img[:, :, slice_idx].astype(np.float32)
        label = self._cache_lbl[:, :, slice_idx].astype(np.float32)

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())


def get_file_splits(processed_dir, train=0.7, val=0.15, test=0.15, seed=42):
    processed_dir = Path(processed_dir)
    stems = sorted(set(
        f.stem.replace('_img', '')
        for f in processed_dir.glob("*_img.npy")
    ))

    n = len(stems)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    train_end = int(n * train)
    val_end = int(n * (train + val))

    return (
        [stems[i] for i in indices[:train_end]],
        [stems[i] for i in indices[train_end:val_end]],
        [stems[i] for i in indices[val_end:]]
    )