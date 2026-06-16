"""Unit tests for the matching/store core — no heavy models required."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from facecore.recognition.matcher import Matcher
from facecore.recognition.vector_store import FaissVectorStore


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)


def test_enroll_and_identify(tmp_path: Path) -> None:
    store = FaissVectorStore(8, tmp_path / "i.faiss", tmp_path / "m.json")
    alice = _unit(np.random.RandomState(1).randn(3, 8))
    store.add("alice", alice)
    matcher = Matcher(store, threshold=0.5)

    identity = matcher.identify(alice[0])
    assert identity.is_known
    assert identity.person_id == "alice"
    assert identity.similarity > 0.99


def test_unknown_rejected(tmp_path: Path) -> None:
    store = FaissVectorStore(8, tmp_path / "i.faiss", tmp_path / "m.json")
    store.add("alice", _unit(np.random.RandomState(1).randn(2, 8)))
    matcher = Matcher(store, threshold=0.95)
    stranger = _unit(np.random.RandomState(99).randn(1, 8))[0]
    assert not matcher.identify(stranger).is_known


def test_persistence_roundtrip(tmp_path: Path) -> None:
    idx, meta = tmp_path / "i.faiss", tmp_path / "m.json"
    s1 = FaissVectorStore(8, idx, meta)
    s1.add("bob", _unit(np.random.RandomState(2).randn(4, 8)))
    s1.save()
    s2 = FaissVectorStore(8, idx, meta)
    assert s2.size == 4


def test_cosine_symmetry() -> None:
    a = _unit(np.random.RandomState(3).randn(1, 8))[0]
    assert abs(Matcher.cosine(a, a) - 1.0) < 1e-5
