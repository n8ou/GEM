"""Detector interface — keeps the pipeline model-agnostic (RetinaFace/MTCNN/YOLO)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from facecore.domain.entities import DetectedFace


class FaceDetector(ABC):
    """Contract: take a BGR uint8 image, return detected faces with landmarks."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, max_faces: int | None = None) -> list[DetectedFace]:
        ...
