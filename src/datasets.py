from torch.utils.data import Dataset
from pathlib import Path
import json
import torch
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandScaleIntensityd, ToTensord
)


class BrainTumorDataset(Dataset):
    """
    Loads preprocessed BraTS samples and applies MONAI augmentations.
    """

    def __init__(self, split="train", augment=False):
        pre_dir = Path("data/preprocessed")
        with open(pre_dir / "splits.json") as f:
            self.splits = json.load(f)

        self.files = self.splits[split]
        self.augment = augment

        # --- Define transforms ---
        if self.augment:
            self.transforms = Compose([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            self.transforms = Compose([
                ToTensord(keys=["image", "label"]),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load preprocessed .pt file (contains image + label tensors)
        sample_path = Path(self.files[idx]).with_suffix(".pt")
        data = torch.load(sample_path)
        sample = {
            "image": data["image"],
            "label": data["label"],
        }

        # Apply transforms (if any)
        sample = self.transforms(sample)
        return sample["image"], sample["label"]
