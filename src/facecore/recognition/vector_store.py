"""FAISS-backed embedding database with persistent metadata.

We use an inner-product index on L2-normalized vectors, so the score == cosine
similarity. Each person can have multiple enrolled embeddings; identity score is
the max over that person's vectors (gallery matching).
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from facecore.logging_conf import get_logger

log = get_logger(__name__)


class FaissVectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path) -> None:
        self._dim = dim
        self._index_path = index_path
        self._meta_path = meta_path
        self._lock = threading.RLock()
        self._labels: list[str] = []  # row i -> person_id
        self._index = faiss.IndexFlatIP(dim)
        self._load()

    # --- persistence ---
    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._labels = json.loads(self._meta_path.read_text(encoding="utf-8"))
            log.info("Loaded index", extra={"extra_fields": {"n": len(self._labels)}})

    def save(self) -> None:
        with self._lock:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
            self._meta_path.write_text(json.dumps(self._labels), encoding="utf-8")

    # --- mutation ---
    def add(self, person_id: str, embeddings: np.ndarray) -> int:
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(f"expected (N, {self._dim}) embeddings")
        with self._lock:
            self._index.add(embeddings.astype(np.float32))
            self._labels.extend([person_id] * embeddings.shape[0])
        return embeddings.shape[0]

    # --- query ---
    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return per-person best similarity, sorted desc."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            q = query.reshape(1, -1).astype(np.float32)
            k = min(top_k * 4, self._index.ntotal)
            scores, idx = self._index.search(q, k)
        best: dict[str, float] = defaultdict(lambda: -1.0)
        for score, i in zip(scores[0], idx[0], strict=True):
            if i < 0:
                continue
            pid = self._labels[i]
            best[pid] = max(best[pid], float(score))
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    @property
    def size(self) -> int:
        return self._index.ntotal
