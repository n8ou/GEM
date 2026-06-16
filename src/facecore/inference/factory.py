"""Composition root: wire concrete implementations from settings.

Single place where we choose detector/embedder backends. Keeps the rest of the
code depending only on interfaces (Dependency Inversion).
"""
from __future__ import annotations

import torch

from facecore.config import Settings, get_settings
from facecore.detection.retinaface import RetinaFaceDetector
from facecore.embedding.arcface import OnnxArcFace, TorchArcFace
from facecore.embedding.base import FaceEmbedder
from facecore.inference.pipeline import RecognitionPipeline
from facecore.recognition.matcher import Matcher
from facecore.recognition.vector_store import FaissVectorStore
from facecore.utils.device import resolve_device


def _build_embedder(settings: Settings, device: torch.device) -> FaceEmbedder:
    onnx_path = settings.model_dir / f"{settings.embedder}.onnx"
    if onnx_path.exists():
        return OnnxArcFace(onnx_path, device)
    weights = settings.checkpoint_dir / "best.pt"
    backbone = settings.embedder.replace("arcface_", "")
    return TorchArcFace(backbone, weights, device)


def build_pipeline(settings: Settings | None = None) -> RecognitionPipeline:
    settings = settings or get_settings()
    device = resolve_device(settings.device)
    detector = RetinaFaceDetector(settings, device)
    embedder = _build_embedder(settings, device)
    store = FaissVectorStore(settings.embedding_dim, settings.index_path, settings.index_meta_path)
    matcher = Matcher(store, settings.match_threshold)
    return RecognitionPipeline(detector, embedder, matcher, settings.max_faces_per_image)


def build_store(settings: Settings | None = None) -> FaissVectorStore:
    settings = settings or get_settings()
    return FaissVectorStore(settings.embedding_dim, settings.index_path, settings.index_meta_path)
