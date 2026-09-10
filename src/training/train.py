"""Segmentation training entry point.

Usage:
    python -m src.training.train --config configs/segmentation.yaml
    python -m src.training.train --config configs/segmentation.yaml --resume checkpoints/segmentation/last.pt
    python -m src.training.train --config configs/segmentation.yaml --device cpu

Epoch loop with AdamW + cosine LR schedule, DiceBCELoss, best-val-Dice and
last-epoch checkpointing, CSV + stdout logging, and early stopping. Note:
until src/models/unet.py's VanillaUNet.forward and src/training/metrics.py's
dice_coefficient/iou_score are implemented, this will run end-to-end only
with a non-"vanilla" encoder configured (metrics are needed for validation
regardless of encoder — see the TODOs in metrics.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import (
    PigSegmentationDataset,
    build_eval_transform,
    build_train_transform,
)
from src.models.unet import build_model
from src.training.losses import build_loss
from src.training.metrics import dice_coefficient, iou_score


def resolve_device(device_config: str) -> torch.device:
    """cuda | mps | cpu | auto -> a concrete torch.device.

    auto picks in priority order: CUDA, then Apple MPS, then CPU. Never
    hardcode a CUDA-only assumption elsewhere in the pipeline — this is the
    single place device selection happens.
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    aug_cfg = config["augmentation"]
    train_cfg = config["training"]
    image_size = tuple(data_cfg["image_size"])

    train_dataset = PigSegmentationDataset(
        data_root=data_cfg["root"],
        manifest_path=data_cfg["manifest_path"],
        splits_path=data_cfg["splits_path"],
        split="train",
        transform=build_train_transform(aug_cfg, image_size),
    )
    val_dataset = PigSegmentationDataset(
        data_root=data_cfg["root"],
        manifest_path=data_cfg["manifest_path"],
        splits_path=data_cfg["splits_path"],
        split="val",
        transform=build_eval_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """One pass over `loader`. If `optimizer` is given, trains; otherwise evaluates.

    Returns a dict with keys: loss, dice, iou (all epoch means).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(logits.detach(), masks).item()
            total_iou += iou_score(logits.detach(), masks).item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, epoch: int, best_val_dice: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_dice": best_val_dice,
        },
        path,
    )


def train(config: dict, resume_path: str | None = None, device_override: str | None = None) -> None:
    device = resolve_device(device_override or config.get("device", "auto"))
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(config)

    model = build_model(config).to(device)
    criterion = build_loss(config)

    train_cfg = config["training"]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0.0)
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    num_epochs = train_cfg["num_epochs"]
    scheduler_name = train_cfg.get("lr_scheduler")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    start_epoch = 0
    best_val_dice = -float("inf")
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint["best_val_dice"]
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    checkpoint_dir = train_cfg["checkpoint_dir"]
    log_path = train_cfg["log_path"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_is_new = not os.path.exists(log_path)

    patience = train_cfg.get("early_stopping_patience")
    epochs_without_improvement = 0

    with open(log_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        if log_is_new:
            writer.writerow(
                ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "seconds"]
            )

        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"- train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
                f"- val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
                f"({elapsed:.1f}s)"
            )
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["dice"],
                    train_metrics["iou"],
                    val_metrics["loss"],
                    val_metrics["dice"],
                    val_metrics["iou"],
                    f"{elapsed:.1f}",
                ]
            )
            log_file.flush()

            save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model, optimizer, epoch, best_val_dice)

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                epochs_without_improvement = 0
                save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), model, optimizer, epoch, best_val_dice)
                print(f"  New best val_dice: {best_val_dice:.4f} — saved checkpoint")
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"Early stopping: no val_dice improvement for {patience} epochs")
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pig segmentation U-Net")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu", "auto"], help="Override config.device"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config, resume_path=args.resume, device_override=args.device)


if __name__ == "__main__":
    main()
