"""Sample-quality gates: corruption, blur, size, exposure.

Bad training samples poison embeddings far more than they help. These gates run
both at dataset-build time and at enrollment time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class QualityReport:
    ok: bool
    reason: str = ""
    blur_score: float = 0.0


def load_image_safe(path: Path) -> np.ndarray | None:
    """Decode an image without trusting the extension. Returns None if corrupt."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(
    image_bgr: np.ndarray,
    *,
    blur_threshold: float,
    min_size: int,
) -> QualityReport:
    if image_bgr is None or image_bgr.size == 0:
        return QualityReport(False, "empty_image")
    h, w = image_bgr.shape[:2]
    if h < min_size or w < min_size:
        return QualityReport(False, f"too_small_{w}x{h}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = variance_of_laplacian(gray)
    if blur < blur_threshold:
        return QualityReport(False, "blurry", blur)

    mean = float(gray.mean())
    if mean < 15 or mean > 240:  # crushed blacks / blown highlights
        return QualityReport(False, "bad_exposure", blur)

    return QualityReport(True, "ok", blur)
