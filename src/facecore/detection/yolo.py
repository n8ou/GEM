"""YOLO-Face detector backed by Ultralytics.

A drop-in alternative to :class:`RetinaFaceDetector`. We wrap an Ultralytics
YOLO *face* checkpoint (YOLOv8/YOLOv11-face) that emits a bounding box plus the
5 facial keypoints — left eye, right eye, nose, left/right mouth corner — in the
same order the ArcFace alignment template expects. That keypoint order is the
WIDERFACE/RetinaFace convention these face models are trained on, so aligned
crops are interchangeable with the RetinaFace path and the embedder is unchanged.

Only the detector swaps; the pipeline still depends solely on the
``FaceDetector`` interface (Dependency Inversion).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from facecore.config import Settings
from facecore.detection.base import FaceDetector
from facecore.domain.entities import BBox, DetectedFace
from facecore.logging_conf import get_logger

log = get_logger(__name__)


class YoloFaceDetector(FaceDetector):
    def __init__(self, settings: Settings, device: torch.device) -> None:
        from ultralytics import YOLO  # lazy import: heavy + optional

        weights = Path(settings.yolo_weights)
        if not weights.is_absolute():
            weights = settings.model_dir / weights
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLO-Face weights not found at '{weights}'. Download a 5-keypoint "
                "YOLO face checkpoint (e.g. yolov8n-face.pt) into model_dir, or set "
                "FACECORE_YOLO_WEIGHTS to its path."
            )

        self._device = "cuda:0" if device.type == "cuda" else "cpu"
        self._model = YOLO(str(weights))
        self._model.to(self._device)
        self._score_thr = settings.detect_score_threshold
        self._min_size = settings.min_face_size
        self._warned_no_kps = False
        log.info(
            "YOLO-Face ready",
            extra={"extra_fields": {"weights": weights.name, "device": self._device}},
        )

    def detect(self, image_bgr: np.ndarray, max_faces: int | None = None) -> list[DetectedFace]:
        # Ultralytics accepts a BGR numpy array (cv2 convention) directly.
        results = self._model.predict(
            image_bgr,
            conf=self._score_thr,
            device=self._device,
            verbose=False,
        )
        out: list[DetectedFace] = []
        for res in results:
            if res.boxes is None:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            # Pose/face checkpoints expose 5 keypoints; plain detect ones don't.
            kps = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
            for i, (box, score) in enumerate(zip(xyxy, scores, strict=False)):
                if score < self._score_thr:
                    continue
                landmarks: np.ndarray | None = None
                if kps is not None and kps[i].shape == (5, 2) and np.any(kps[i]):
                    landmarks = np.asarray(kps[i], dtype=np.float32)
                elif not self._warned_no_kps:
                    log.warning(
                        "YOLO model has no facial keypoints — using bounding-box "
                        "alignment (lower embedding quality than 5-point). Supply a "
                        "YOLO-face *pose* checkpoint for landmark alignment."
                    )
                    self._warned_no_kps = True
                x1, y1, x2, y2 = (float(v) for v in box)
                face = DetectedFace(
                    bbox=BBox(x1, y1, x2, y2, float(score)),
                    landmarks=landmarks,
                )
                if face.is_valid(self._min_size):
                    out.append(face)
        # Largest faces first — most relevant for single-subject flows.
        out.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        if max_faces is not None:
            out = out[:max_faces]
        return out
