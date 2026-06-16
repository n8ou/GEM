"""Provision the model artifacts the API needs, idempotently.

A fresh clone ships no weights (``artifacts/`` is gitignored), so deployment
needs a reproducible way to populate ``FACECORE_MODEL_DIR``:

  * YOLO11-pose face detector  -> ``<model_dir>/<FACECORE_YOLO_WEIGHTS>``
  * Pretrained ArcFace embedder -> ``<model_dir>/<FACECORE_EMBEDDER>.onnx``

The ArcFace weights come from InsightFace's ``buffalo_l`` bundle (downloaded via
the insightface package, same source the detector path already used), then the
``w600k_r50.onnx`` file is lifted to the model-dir root where the factory looks.

Run before first launch:  ``python scripts/fetch_models.py``
Safe to re-run: existing files are left untouched.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from facecore.config import get_settings
from facecore.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

# 5-keypoint YOLO-pose face model (zjykzj/YOLO11Face release v1.0.0).
YOLO_URL = (
    "https://github.com/zjykzj/YOLO11Face/releases/download/v1.0.0/yolo11n-pose_widerface.pt"
)

# Silent-Face anti-spoofing weights (minivision-ai, Apache-2.0).
_ANTISPOOF_BASE = (
    "https://raw.githubusercontent.com/minivision-ai/"
    "Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models"
)
ANTISPOOF_WEIGHTS = ("2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading", extra={"extra_fields": {"url": url, "dest": str(dest)}})
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (trusted release URL)
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def ensure_yolo(model_dir: Path, weights_name: str) -> None:
    dest = model_dir / weights_name
    if dest.is_file():
        log.info("YOLO weights present", extra={"extra_fields": {"path": str(dest)}})
        return
    if weights_name != "yolo11n-pose_widerface.pt":
        raise FileNotFoundError(
            f"{dest} missing and no known download URL for '{weights_name}'. "
            "Place the weights manually or set FACECORE_YOLO_WEIGHTS to yolo11n-pose_widerface.pt."
        )
    _download(YOLO_URL, dest)


def ensure_arcface(model_dir: Path, embedder: str) -> None:
    dest = model_dir / f"{embedder}.onnx"
    if dest.is_file():
        log.info("ArcFace ONNX present", extra={"extra_fields": {"path": str(dest)}})
        return
    if embedder != "w600k_r50":
        raise FileNotFoundError(
            f"{dest} missing. Auto-provisioning only knows InsightFace 'w600k_r50'; "
            f"export your own model to {dest} or set FACECORE_EMBEDDER=w600k_r50."
        )
    # Pull buffalo_l via insightface (downloads + extracts), then lift w600k_r50.onnx.
    from insightface.app import FaceAnalysis

    log.info("fetching buffalo_l via insightface (one-time, ~300MB)")
    FaceAnalysis(name="buffalo_l", root=str(model_dir), allowed_modules=["detection"])
    src = model_dir / "models" / "buffalo_l" / "w600k_r50.onnx"
    if not src.is_file():
        raise FileNotFoundError(f"expected {src} after buffalo_l download")
    shutil.copyfile(src, dest)
    log.info("ArcFace ready", extra={"extra_fields": {"path": str(dest)}})


def ensure_liveness(model_dir: Path, subdir: str) -> None:
    dest_dir = model_dir / subdir
    for name in ANTISPOOF_WEIGHTS:
        dest = dest_dir / name
        if dest.is_file():
            log.info("anti-spoof weight present", extra={"extra_fields": {"path": str(dest)}})
            continue
        _download(f"{_ANTISPOOF_BASE}/{name}", dest)


def main() -> int:
    configure_logging()
    s = get_settings()
    model_dir = Path(s.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ensure_yolo(model_dir, s.yolo_weights)
    ensure_arcface(model_dir, s.embedder)
    if s.liveness_enabled:
        ensure_liveness(model_dir, s.liveness_model_subdir)
    log.info("all models provisioned", extra={"extra_fields": {"model_dir": str(model_dir)}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
