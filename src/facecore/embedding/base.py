"""Embedder interface — decouples recognition from the embedding model."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, aligned_faces_bgr: list[np.ndarray]) -> np.ndarray:
        """Batch-embed aligned 112x112 BGR faces -> (N, D) L2-normalized float32."""
