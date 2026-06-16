"""Lightweight, alignment-safe augmentation for embedding training.

We avoid geometric warps that would break landmark alignment; instead we use
photometric + occlusion-style augments that improve robustness in the wild.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_RNG = np.random.default_rng()  # seeded by trainer for reproducibility


def seed_augment(seed: int) -> None:
    global _RNG
    _RNG = np.random.default_rng(seed)


def random_photometric(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    if _RNG.random() < 0.5:  # brightness/contrast
        alpha = 1.0 + _RNG.uniform(-0.2, 0.2)
        beta = _RNG.uniform(-15, 15)
        out = out * alpha + beta
    if _RNG.random() < 0.3:  # gaussian noise
        out += _RNG.normal(0, 6, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def random_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1) if _RNG.random() < 0.5 else img


def random_erase(img: np.ndarray) -> np.ndarray:
    if _RNG.random() > 0.25:
        return img
    h, w = img.shape[:2]
    eh, ew = int(h * _RNG.uniform(0.1, 0.3)), int(w * _RNG.uniform(0.1, 0.3))
    y, x = _RNG.integers(0, h - eh), _RNG.integers(0, w - ew)
    img = img.copy()
    img[y : y + eh, x : x + ew] = _RNG.integers(0, 255)
    return img


def to_model_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> normalized RGB CHW float tensor in [-1, 1] (ArcFace convention)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()


def train_transform(img_bgr: np.ndarray) -> torch.Tensor:
    img_bgr = random_flip(img_bgr)
    img_bgr = random_photometric(img_bgr)
    img_bgr = random_erase(img_bgr)
    return to_model_tensor(img_bgr)
