"""Threshold-based identity matching with unknown-face rejection."""
from __future__ import annotations

import numpy as np

from facecore.domain.entities import Identity
from facecore.recognition.vector_store import FaissVectorStore


class Matcher:
    def __init__(self, store: FaissVectorStore, threshold: float) -> None:
        self._store = store
        self._threshold = threshold

    def identify(self, embedding: np.ndarray) -> Identity:
        results = self._store.search(embedding, top_k=1)
        if not results:
            return Identity(person_id="unknown", similarity=0.0, is_known=False)
        person_id, score = results[0]
        if score < self._threshold:
            return Identity(person_id="unknown", similarity=score, is_known=False)
        return Identity(person_id=person_id, similarity=score, is_known=True)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Direct 1:1 verification helper for two L2-normalized vectors."""
        return float(np.dot(a, b))
