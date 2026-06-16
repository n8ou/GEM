"""RetinaFace detector backed by InsightFace's ONNX models.

We wrap InsightFace because it ships well-validated, ONNX-exported RetinaFace
(`det_10g`) + ArcFace models, giving us reproducible accuracy and GPU/CPU EPs
out of the box. The wrapper isolates the rest of the codebase from the library.
"""
from __future__ import annotations

import numpy as np
import torch

from facecore.config import Settings
from facecore.detection.base import FaceDetector
from facecore.domain.entities import BBox, DetectedFace
from facecore.logging_conf import get_logger
from facecore.utils.device import onnx_providers

log = get_logger(__name__)


class RetinaFaceDetector(FaceDetector):
    def __init__(self, settings: Settings, device: torch.device) -> None:
        from insightface.app import FaceAnalysis  # lazy import: heavy + optional

        ctx_id = 0 if device.type == "cuda" else -1
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(settings.model_dir),
            providers=onnx_providers(device),
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=ctx_id, det_thresh=settings.detect_score_threshold)
        self._score_thr = settings.detect_score_threshold
        self._min_size = settings.min_face_size
        log.info("RetinaFace ready", extra={"extra_fields": {"ctx_id": ctx_id}})

    def detect(self, image_bgr: np.ndarray, max_faces: int | None = None) -> list[DetectedFace]:
        faces = self._app.get(image_bgr)
        out: list[DetectedFace] = []
        for f in faces:
            if f.det_score < self._score_thr:
                continue
            x1, y1, x2, y2 = f.bbox.astype(float)
            face = DetectedFace(
                bbox=BBox(x1, y1, x2, y2, float(f.det_score)),
                landmarks=np.asarray(f.kps, dtype=np.float32),
            )
            if face.is_valid(self._min_size):
                out.append(face)
        # Largest faces first — most relevant for single-subject flows.
        out.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        if max_faces is not None:
            out = out[:max_faces]
        return out
