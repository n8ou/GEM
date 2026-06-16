"""Pure domain types — no framework dependencies. Stable contract across layers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A detected face: box, 5-point landmarks (eyes, nose, mouth corners)."""

    bbox: BBox
    landmarks: np.ndarray  # shape (5, 2), float32

    def is_valid(self, min_size: int) -> bool:
        return self.bbox.width >= min_size and self.bbox.height >= min_size


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    vector: np.ndarray  # shape (D,), L2-normalized float32
    bbox: BBox

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {self.vector.shape}")


@dataclass(slots=True)
class Identity:
    person_id: str
    similarity: float
    is_known: bool


@dataclass(slots=True)
class RecognitionResult:
    bbox: BBox
    identity: Identity
    embedding: np.ndarray = field(repr=False)
