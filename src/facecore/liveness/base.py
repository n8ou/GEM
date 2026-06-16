"""Liveness (anti-spoofing) interface — keeps the pipeline detector-agnostic."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from facecore.domain.entities import BBox


class LivenessDetector(ABC):
    """Contract: given the full BGR frame and a detected face box, return the
    probability in [0, 1] that the face is a *live* capture (not a printed photo
    or screen replay). The pipeline applies the accept/reject threshold."""

    @abstractmethod
    def score(self, image_bgr: np.ndarray, bbox: BBox) -> float:
        ...
