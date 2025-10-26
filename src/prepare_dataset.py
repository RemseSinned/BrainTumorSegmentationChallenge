"""
prepare_dataset.py
==================

Preprocesses the Brain Tumor (MSD Task01) dataset:
- Loads .nii.gz images and labels
- Normalizes MRI volumes
- Splits into train/val/test sets (default 70/15/15)
- Saves preprocessed tensors (.pt) for PyTorch
- Generates splits.json for reproducibility
"""
import json
from pathlib import Path
import nibabel as nib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from configs.utils import load_config


cfg = load_config("../configs/paths.yaml")
RAW_DIR = Path(("../" + cfg["raw_dir"]))
PRE_DIR = Path(("../" + cfg["preprocessed_dir"]))
PRE_DIR.mkdir(parents=True, exist_ok=True)


def zscore_normalize(volume):
    """Z-score normalization over nonzero brain voxels."""
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume
    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std()
    return (volume - mean) / (std + 1e-8)


def process_case(image_path, label_path, save_path):
    """Load, normalize and save as .pt tensor."""
    img = nib.load(image_path)
    img_np = img.get_fdata().astype(dtype=np.float32)

    if img_np.ndim == 4:
        img_np = np.moveaxis(img_np, -1, 0)
    else:
        img_np = img_np[None, ...]

    img_np = np.stack([zscore_normalize(ch) for ch in img_np])

    img_t = torch.from_numpy(img_np).float()

    if label_path is not None and label_path.exists():
        label_np = nib.load(label_path).get_fdata().astype(dtype=np.int16)
        label_t = torch.from_numpy(label_np).long()
    else:
        label_t = None

    torch.save({"image": img_t, "label": label_t}, save_path)


def process_split(splits, img_list, label_list, split_name):
    for img_p, lbl_p in zip(img_list, label_list):
        save_path = PRE_DIR / split_name / f"{img_p.stem}.pt"
        print(f"[{split_name.upper()}] Processing {img_p.name}")
        process_case(img_p, lbl_p, save_path)
        splits[split_name].append(str(save_path.resolve()))
    return splits


def prepare_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)

    images = sorted((RAW_DIR / "imagesTr").glob("*.nii.gz"))
    labels = sorted((RAW_DIR / "labelsTr").glob("*.nii.gz"))

    assert len(images) == len(labels), "Mismatch between images and labels."

    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, train_size=train_ratio, test_size=(1 - train_ratio), random_state=seed
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed
    )

    splits = {"train": [], "val": [], "test": []}

    for split in splits.keys():
        (PRE_DIR / split).mkdir(parents=True, exist_ok=True)

    splits = process_split(splits, train_imgs, train_labels, "train")
    splits = process_split(splits, val_imgs, val_labels, "val")
    splits = process_split(splits, test_imgs, test_labels, "test")

    with open(PRE_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"Saved splits.json to: {PRE_DIR / 'splits.json'}")
