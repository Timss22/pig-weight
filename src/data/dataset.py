"""PyTorch Dataset for the segmentation stage.

Reads the (manifest + split) table produced by src/data/splits.py, filters
to rows with ground-truth masks (has_mask=True — only the rgb3394 subset
has masks), and yields RGB/mask tensor pairs with joint augmentation.

Ground-truth masks are used ONLY here, to supervise/evaluate segmentation.
Nothing downstream of segmentation reads mask_path — see CLAUDE.md.
"""

from __future__ import annotations

import os
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.data.splits import load_manifest_with_splits


def build_train_transform(aug_config: dict, image_size: tuple[int, int]) -> A.Compose:
    """Joint image+mask augmentation pipeline for training.

    `image_size` is (width, height). Color jitter uses generous ranges
    (config-driven) anticipating open-house lighting variability at
    eventual deployment, not just this dataset's controlled lighting.
    """
    width, height = image_size
    cj = aug_config["color_jitter"]
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)),
            A.Affine(
                rotate=(-aug_config.get("rotation_degrees", 15), aug_config.get("rotation_degrees", 15)),
                scale=tuple(aug_config.get("scale_range", [0.9, 1.1])),
                p=0.7,
            ),
            A.ColorJitter(
                brightness=cj.get("brightness", 0.4),
                contrast=cj.get("contrast", 0.4),
                saturation=cj.get("saturation", 0.3),
                hue=cj.get("hue", 0.1),
                p=0.8,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def build_eval_transform(image_size: tuple[int, int]) -> A.Compose:
    """Deterministic resize + normalize for val/test/inference — no augmentation."""
    width, height = image_size
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class PigSegmentationDataset(Dataset):
    """RGB + binary mask pairs for one split ("train"/"val"/"test").

    Only rows with has_mask=True are usable for segmentation supervision;
    rows without a mask are dropped regardless of `split`.
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        splits_path: str,
        split: str,
        transform: Optional[A.Compose] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")

        self.data_root = data_root
        merged = load_manifest_with_splits(manifest_path, splits_path)
        subset = merged[(merged["split"] == split) & (merged["has_mask"] == True)]  # noqa: E712
        if len(subset) == 0:
            raise ValueError(
                f"No masked rows found for split={split!r}. Check that the manifest "
                f"and splits files are aligned and the masked subset (rgb3394) has "
                f"coverage in this split."
            )
        self.rows = subset.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        image_path = os.path.join(self.data_root, row["image_path"])
        mask_path = os.path.join(self.data_root, row["mask_path"])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")
        # Binarize: masks are stored as {0, 255}.
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0).float()  # (1, H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "pig_id": int(row["pig_id"]),
            "subset": row["subset"],
            "weight_kg": float(row["weight_kg"]),
            "image_path": row["image_path"],
        }
