"""Zero-shot SAM2 segmentation baseline — comparison point for the trained U-Net.

Scope, deliberately narrow (see CLAUDE.md "Locked design decisions" for
why): pure zero-shot inference with sam2.1_hiera_base_plus, no training or
fine-tuning. This is NOT trying to reproduce the "SAM2-Pig" literature
setup (Applied Sciences 16(11), 5708, 2025), which trains a hybrid
SAM2+ViT dorsal-segmentation network — that would expand this phase's
scope into later-phase territory.

Prompting strategy: each PIGRGB-Weight image frames a single, roughly
centered pig from directly overhead. Rather than automatic (whole-image)
mask generation — which returns many candidate masks per image with no
obvious pig-vs-background selection rule — this harness gives SAM2 one
foreground point prompt at the image center and takes its own
highest-predicted-IoU mask. This is a standard zero-shot SAM2 usage
pattern and needs no dataset-specific tuning.

Requires the `sam2` package (see requirements.txt) and a downloaded
checkpoint — see README.md for the download command. Evaluation against
ground truth uses src.training.metrics (dice_coefficient / iou_score),
so this harness's numbers aren't final until those TODOs are implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.data.dataset import PigSegmentationDataset
from src.training.metrics import dice_coefficient, iou_score

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:  # pragma: no cover - exercised only when sam2 isn't installed
    build_sam2 = None
    SAM2ImagePredictor = None


def load_sam2_predictor(config: dict, device: torch.device) -> "SAM2ImagePredictor":
    if build_sam2 is None or SAM2ImagePredictor is None:
        raise ImportError(
            "The `sam2` package is required for the SAM2 baseline "
            "(pip install git+https://github.com/facebookresearch/sam2.git). "
            "See README.md for checkpoint download instructions."
        )
    sam2_cfg = config["sam2_baseline"]
    checkpoint_path = sam2_cfg["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint_path}. See README.md for the "
            f"download command for {sam2_cfg['checkpoint']}."
        )
    model = build_sam2(sam2_cfg["config_path"], checkpoint_path, device=str(device))
    return SAM2ImagePredictor(model)


def predict_mask_center_point(predictor: "SAM2ImagePredictor", image_rgb: np.ndarray) -> np.ndarray:
    """Run zero-shot SAM2 on one image using a single center-point prompt.

    `image_rgb`: (H, W, 3) uint8 RGB array. Returns a binary (H, W) mask
    (the SAM2 candidate with the highest predicted IoU).
    """
    predictor.set_image(image_rgb)
    height, width = image_rgb.shape[:2]
    center_point = np.array([[width // 2, height // 2]])
    center_label = np.array([1])  # foreground

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(np.uint8)


def evaluate_sam2_on_split(config: dict, split: str, device: torch.device) -> dict:
    """Run the SAM2 baseline over every masked image in `split` and report mean Dice/IoU.

    Uses the same test split (has_mask=True rows) the U-Net is evaluated
    on, for a fair comparison.
    """
    predictor = load_sam2_predictor(config, device)
    data_cfg = config["data"]

    dataset = PigSegmentationDataset(
        data_root=data_cfg["root"],
        manifest_path=data_cfg["manifest_path"],
        splits_path=data_cfg["splits_path"],
        split=split,
        transform=None,  # we need raw RGB pixels for SAM2, not normalized tensors
    )

    dice_scores, iou_scores = [], []
    for i in range(len(dataset)):
        row = dataset.rows.iloc[i]
        image_path = str(Path(data_cfg["root"]) / row["image_path"])
        mask_path = str(Path(data_cfg["root"]) / row["mask_path"])

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_binary = (gt_mask > 127).astype(np.float32)

        pred_mask = predict_mask_center_point(predictor, image_rgb)

        pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
        target_tensor = torch.from_numpy(gt_mask_binary).unsqueeze(0).unsqueeze(0).float()

        dice_scores.append(
            dice_coefficient(pred_tensor, target_tensor, pred_is_logits=False).item()
        )
        iou_scores.append(iou_score(pred_tensor, target_tensor, pred_is_logits=False).item())

    return {
        "mean_dice": float(np.mean(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "n_images": len(dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the zero-shot SAM2 baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    return parser.parse_args()


def main() -> None:
    from src.training.train import resolve_device  # local import avoids a training-module dependency at import time

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = resolve_device(args.device)

    results = evaluate_sam2_on_split(config, args.split, device)
    print(f"SAM2 zero-shot baseline on {args.split} split ({results['n_images']} images):")
    print(f"  mean Dice: {results['mean_dice']:.4f}")
    print(f"  mean IoU:  {results['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
