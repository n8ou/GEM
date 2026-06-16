"""ArcFace embedder with two backends:

* `OnnxArcFace`  — loads an exported .onnx (InsightFace w600k_r50 or our trained
  model) and runs it through ONNXRuntime (GPU EP w/ CPU fallback). Default for
  production inference: fastest, no autograd overhead.
* `TorchArcFace` — wraps the trainable IResNet backbone for eval. Used when you
  just trained a custom model and haven't exported yet.

Both return L2-normalized embeddings so cosine similarity == dot product.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from facecore.embedding.base import FaceEmbedder
from facecore.embedding.backbone import build_backbone
from facecore.logging_conf import get_logger
from facecore.preprocessing.augmentation import to_model_tensor
from facecore.utils.device import onnx_providers

log = get_logger(__name__)


def _l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


class OnnxArcFace(FaceEmbedder):
    def __init__(self, onnx_path: Path, device: torch.device, batch_size: int = 32) -> None:
        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=onnx_providers(device))
        self._input = self._sess.get_inputs()[0].name
        self._batch = batch_size
        log.info("OnnxArcFace ready", extra={"extra_fields": {"providers": self._sess.get_providers()}})

    def embed(self, aligned_faces_bgr: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces_bgr:
            return np.empty((0, 512), dtype=np.float32)
        tensors = np.stack([to_model_tensor(f).numpy() for f in aligned_faces_bgr])
        outputs: list[np.ndarray] = []
        for i in range(0, len(tensors), self._batch):
            chunk = tensors[i : i + self._batch]
            outputs.append(self._sess.run(None, {self._input: chunk})[0])
        return _l2_normalize(np.concatenate(outputs, axis=0).astype(np.float32))


class TorchArcFace(FaceEmbedder):
    def __init__(self, backbone_name: str, weights: Path, device: torch.device, batch_size: int = 32):
        self._device = device
        self._batch = batch_size
        self._model = build_backbone(backbone_name).to(device).eval()
        state = torch.load(weights, map_location=device)
        self._model.load_state_dict(state["backbone"] if "backbone" in state else state)
        log.info("TorchArcFace ready", extra={"extra_fields": {"backbone": backbone_name}})

    @torch.inference_mode()
    def embed(self, aligned_faces_bgr: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces_bgr:
            return np.empty((0, 512), dtype=np.float32)
        tensors = torch.stack([to_model_tensor(f) for f in aligned_faces_bgr]).to(self._device)
        out: list[np.ndarray] = []
        for i in range(0, len(tensors), self._batch):
            with torch.autocast(self._device.type, enabled=self._device.type == "cuda"):
                emb = self._model(tensors[i : i + self._batch])
            out.append(emb.float().cpu().numpy())
        return _l2_normalize(np.concatenate(out, axis=0).astype(np.float32))
