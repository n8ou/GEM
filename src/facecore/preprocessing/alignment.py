"""Similarity-transform face alignment to the canonical ArcFace 112x112 template.

Aligning by 5 landmarks removes in-plane rotation/scale variance and is the
single biggest lever on embedding quality. Template is the standard ArcFace one.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as sk_transform

from facecore.domain.entities import BBox, DetectedFace

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


def align_by_bbox(
    image_bgr: np.ndarray, bbox: BBox, size: int = OUTPUT_SIZE, margin: float = 0.15
) -> np.ndarray:
    """Fallback alignment for box-only detectors (no landmarks).

    Takes a centered square crop around the box (expanded by ``margin``) and
    resizes to ``size``. Lacks landmark-based rotation normalization, so embedding
    quality is lower than :func:`align_face` — but keeps the pipeline working with
    detectors that emit only bounding boxes.
    """
    h, w = image_bgr.shape[:2]
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    half = max(bbox.width, bbox.height) * (1.0 + margin) / 2.0
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(w, int(round(cx + half)))
    y2 = min(h, int(round(cy + half)))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def align_detected_face(
    image_bgr: np.ndarray, face: DetectedFace, size: int = OUTPUT_SIZE
) -> np.ndarray:
    """Align a detected face: 5-point similarity transform if landmarks exist,
    else a bounding-box crop. Single entry point for all detector backends."""
    if face.landmarks is not None:
        return align_face(image_bgr, face.landmarks, size)
    return align_by_bbox(image_bgr, face.bbox, size)
