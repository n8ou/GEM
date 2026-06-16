"""End-to-end recognition pipeline. Stateless w.r.t. a single image/frame.

detect -> align -> batch-embed -> match. Designed for both single images and
batched/real-time use. The whole heavy stack (detector, embedder, store) is
constructed once and reused — see factory.build_pipeline().
"""
from __future__ import annotations

import numpy as np

from facecore.detection.base import FaceDetector
from facecore.domain.entities import RecognitionResult
from facecore.embedding.base import FaceEmbedder
from facecore.preprocessing.alignment import align_face
from facecore.recognition.matcher import Matcher


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: Matcher,
        max_faces: int,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._max_faces = max_faces

    def recognize(self, image_bgr: np.ndarray) -> list[RecognitionResult]:
        faces = self._detector.detect(image_bgr, max_faces=self._max_faces)
        if not faces:
            return []
        aligned = [align_face(image_bgr, f.landmarks) for f in faces]
        embeddings = self._embedder.embed(aligned)  # single batched call
        results: list[RecognitionResult] = []
        for face, emb in zip(faces, embeddings, strict=True):
            identity = self._matcher.identify(emb)
            results.append(RecognitionResult(bbox=face.bbox, identity=identity, embedding=emb))
        return results

    def embed_only(self, image_bgr: np.ndarray) -> np.ndarray:
        """Used for enrollment: return embeddings for all detected faces."""
        faces = self._detector.detect(image_bgr, max_faces=self._max_faces)
        if not faces:
            return np.empty((0, 512), dtype=np.float32)
        aligned = [align_face(image_bgr, f.landmarks) for f in faces]
        return self._embedder.embed(aligned)
