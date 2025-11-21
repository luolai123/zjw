"""Dataset utilities for monocular navigation training."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Load RGB images and binary safety masks from a directory."""

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        mask_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted((self.root / "images").glob("*.png"))
        self.masks = sorted((self.root / "masks").glob("*.png"))
        if len(self.images) != len(self.masks):
            raise ValueError("Dataset images and masks must have the same length")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()
        return image, mask
