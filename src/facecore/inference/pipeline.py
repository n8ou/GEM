"""End-to-end recognition pipeline. Stateless w.r.t. a single image/frame.

detect -> align -> batch-embed -> match. Designed for both single images and
batched/real-time use. The whole heavy stack (detector, embedder, store) is
constructed once and reused — see factory.build_pipeline().
"""
from __future__ import annotations

import numpy as np

from facecore.detection.base import FaceDetector
from facecore.domain.entities import Identity, RecognitionResult
from facecore.embedding.base import FaceEmbedder
from facecore.liveness.base import LivenessDetector
from facecore.preprocessing.alignment import align_detected_face
from facecore.recognition.matcher import Matcher


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        matcher: Matcher,
        max_faces: int,
        liveness: LivenessDetector | None = None,
        liveness_threshold: float = 0.5,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._max_faces = max_faces
        self._liveness = liveness
        self._liveness_threshold = liveness_threshold

    def _liveness_check(self, image_bgr: np.ndarray, bbox) -> tuple[bool, float]:
        """(is_live, score). Live by default when anti-spoofing is disabled."""
        if self._liveness is None:
            return True, 1.0
        score = self._liveness.score(image_bgr, bbox)
        return score >= self._liveness_threshold, score

    @property
    def store(self):
        """The vector store the matcher queries — shared so enroll/recognize agree."""
        return self._matcher.store

    def recognize(self, image_bgr: np.ndarray) -> list[RecognitionResult]:
        faces = self._detector.detect(image_bgr, max_faces=self._max_faces)
        if not faces:
            return []
        # Liveness gate first — spoofs are never aligned/embedded/matched.
        live = [self._liveness_check(image_bgr, f.bbox) for f in faces]
        live_idx = [i for i, (ok, _) in enumerate(live) if ok]
        aligned = [align_detected_face(image_bgr, faces[i]) for i in live_idx]
        embs = self._embedder.embed(aligned) if aligned else np.empty((0, 512), dtype=np.float32)
        emb_by_idx = {idx: embs[k] for k, idx in enumerate(live_idx)}

        results: list[RecognitionResult] = []
        for i, face in enumerate(faces):
            is_live, score = live[i]
            if is_live:
                emb = emb_by_idx[i]
                identity = self._matcher.identify(emb)
            else:
                emb = np.zeros(512, dtype=np.float32)
                identity = Identity(person_id="spoof", similarity=0.0, is_known=False)
            results.append(
                RecognitionResult(bbox=face.bbox, identity=identity, embedding=emb,
                                  is_live=is_live, live_score=round(score, 4))
            )
        return results

    def embed_only(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enrollment: embeddings for live detected faces only (spoofs rejected)."""
        faces = self._detector.detect(image_bgr, max_faces=self._max_faces)
        if not faces:
            return np.empty((0, 512), dtype=np.float32)
        aligned = [
            align_detected_face(image_bgr, f)
            for f in faces
            if self._liveness_check(image_bgr, f.bbox)[0]
        ]
        if not aligned:
            return np.empty((0, 512), dtype=np.float32)
        return self._embedder.embed(aligned)
