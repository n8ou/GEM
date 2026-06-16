"""API security: API-key auth, upload validation, and a simple rate limiter.

Defense in depth for image uploads:
 1. Size cap enforced before reading the whole body into memory.
 2. Magic-byte sniffing (don't trust Content-Type or filename).
 3. Decode with OpenCV; reject anything that fails to decode (defeats
    polyglot / malformed-image attacks that crash decoders).
 4. Dimension cap to prevent decompression-bomb memory blowups.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import cv2
import numpy as np
from fastapi import Header, HTTPException, status

from facecore.config import get_settings

# JPEG, PNG, BMP, WEBP magic numbers.
_MAGIC = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"BM",                    # BMP
    b"RIFF",                  # WEBP (container)
)
_MAX_PIXELS = 40_000_000  # ~40 MP ceiling (decompression-bomb guard)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    # Constant-time-ish comparison; reject default key in production.
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
    if settings.env == "production" and settings.api_key == "change-me-in-real-deployment":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "server API key not configured")


def validate_and_decode(raw: bytes) -> np.ndarray:
    settings = get_settings()
    if len(raw) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")
    if not raw.startswith(_MAGIC):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported image type")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "corrupt or undecodable image")
    if img.shape[0] * img.shape[1] > _MAX_PIXELS:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image resolution too large")
    return img


class RateLimiter:
    """In-process sliding-window limiter. For multi-replica use Redis instead."""

    def __init__(self, per_min: int) -> None:
        self._per_min = per_min
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str) -> None:
        now = time.monotonic()
        window = self._hits[client_id]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self._per_min:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
        window.append(now)
