"""Composition root: wire concrete implementations from settings.

Single place where we choose detector/embedder backends. Keeps the rest of the
code depending only on interfaces (Dependency Inversion).
"""
from __future__ import annotations

from pathlib import Path

import torch

from facecore.config import Settings, get_settings
from facecore.detection.base import FaceDetector
from facecore.detection.retinaface import RetinaFaceDetector
from facecore.detection.yolo import YoloFaceDetector
from facecore.embedding.arcface import OnnxArcFace, TorchArcFace
from facecore.embedding.base import FaceEmbedder
from facecore.inference.pipeline import RecognitionPipeline
from facecore.liveness.base import LivenessDetector
from facecore.liveness.silent_face import SilentFaceLiveness
from facecore.recognition.matcher import Matcher
from facecore.recognition.vector_store import FaissVectorStore
from facecore.utils.device import resolve_device

# Heavy backends (insightface / ultralytics) are lazy-imported inside each
# detector's __init__, so importing the classes here stays cheap.
_DETECTORS: dict[str, type[FaceDetector]] = {
    "retinaface": RetinaFaceDetector,
    "yolo": YoloFaceDetector,
}


def build_detector(settings: Settings, device: torch.device) -> FaceDetector:
    return _DETECTORS[settings.detector](settings, device)


def build_liveness(settings: Settings, device: torch.device) -> LivenessDetector | None:
    if not settings.liveness_enabled:
        return None
    return SilentFaceLiveness(Path(settings.model_dir) / settings.liveness_model_subdir, device)


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
    detector = build_detector(settings, device)
    embedder = _build_embedder(settings, device)
    liveness = build_liveness(settings, device)
    store = FaissVectorStore(settings.embedding_dim, settings.index_path, settings.index_meta_path)
    matcher = Matcher(store, settings.match_threshold)
    return RecognitionPipeline(
        detector, embedder, matcher, settings.max_faces_per_image,
        liveness=liveness, liveness_threshold=settings.liveness_threshold,
    )


def build_store(settings: Settings | None = None) -> FaissVectorStore:
    settings = settings or get_settings()
    return FaissVectorStore(settings.embedding_dim, settings.index_path, settings.index_meta_path)
