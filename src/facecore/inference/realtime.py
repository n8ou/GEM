"""Real-time webcam recognition with frame-skipping and an FPS meter.

Detection is the cost driver, so we run the full pipeline every N frames and
draw cached boxes in between — keeps a smooth display while bounding GPU load.
"""
from __future__ import annotations

import time

import cv2

from facecore.domain.entities import RecognitionResult
from facecore.inference.factory import build_pipeline


def _draw(frame, results: list[RecognitionResult]) -> None:
    for r in results:
        x1, y1, x2, y2 = r.bbox.as_int_xyxy()
        known = r.identity.is_known
        color = (0, 200, 0) if known else (0, 0, 255)
        label = f"{r.identity.person_id} {r.identity.similarity:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_webcam(source: int | str = 0, detect_every: int = 3) -> None:
    pipeline = build_pipeline()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")

    frame_idx, fps, last = 0, 0.0, time.perf_counter()
    cached: list[RecognitionResult] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % detect_every == 0:
                cached = pipeline.recognize(frame)
            _draw(frame, cached)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            cv2.putText(frame, f"FPS {fps:4.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("facecore — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
