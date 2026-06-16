"""Similarity-transform face alignment to the canonical ArcFace 112x112 template.

Aligning by 5 landmarks removes in-plane rotation/scale variance and is the
single biggest lever on embedding quality. Template is the standard ArcFace one.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as sk_transform

# Canonical 5-point template for 112x112 ArcFace input (x, y).
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

OUTPUT_SIZE = 112


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = OUTPUT_SIZE) -> np.ndarray:
    """Return an aligned, cropped face of shape (size, size, 3), BGR uint8."""
    if landmarks.shape != (5, 2):
        raise ValueError(f"expected 5x2 landmarks, got {landmarks.shape}")
    dst = _ARCFACE_TEMPLATE * (size / OUTPUT_SIZE)
    tform = sk_transform.SimilarityTransform()
    tform.estimate(landmarks.astype(np.float32), dst)
    matrix = tform.params[0:2, :]
    aligned = cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)
    return aligned
