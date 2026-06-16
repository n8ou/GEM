"""Entry point for custom training.

Usage:
    python scripts/train.py --backbone r50 --epochs 40 --batch-size 128
    python scripts/train.py --resume
"""
from __future__ import annotations

import argparse

from facecore.config import get_settings
from facecore.inference.factory import build_detector
from facecore.training.dataset import FaceFolderDataset
from facecore.training.trainer import TrainConfig, Trainer
from facecore.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a custom ArcFace model")
    p.add_argument("--backbone", default="r50", choices=["r18", "r34", "r50", "r100"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    device = resolve_device(settings.device)
    detector = build_detector(settings, device)

    train_ds = FaceFolderDataset(settings, detector, split="train")
    val_ds = FaceFolderDataset(settings, detector, split="val")
    if val_ds.num_classes == 0:  # no explicit val split -> reuse train manifest
        val_ds = train_ds

    cfg = TrainConfig(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )
    Trainer(settings, cfg, train_ds, val_ds, num_classes=train_ds.num_classes).fit()


if __name__ == "__main__":
    main()
