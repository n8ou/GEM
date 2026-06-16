"""Export a trained IResNet backbone to ONNX with a dynamic batch axis.

The exported graph takes a normalized NCHW float tensor (the same preprocessing
as augmentation.to_model_tensor) and outputs raw embeddings. We verify parity
against PyTorch before declaring success.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from facecore.embedding.backbone import build_backbone
from facecore.logging_conf import get_logger

log = get_logger(__name__)


def export_to_onnx(
    backbone_name: str,
    weights_path: Path,
    output_path: Path,
    embedding_dim: int = 512,
    opset: int = 17,
    verify: bool = True,
) -> Path:
    model = build_backbone(backbone_name, embedding_dim).eval()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["backbone"] if "backbone" in state else state)

    dummy = torch.randn(1, 3, 112, 112)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info("Exported ONNX", extra={"extra_fields": {"path": str(output_path)}})

    if verify:
        _verify_parity(model, output_path, dummy)
    return output_path


def _verify_parity(model: torch.nn.Module, onnx_path: Path, dummy: torch.Tensor) -> None:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.inference_mode():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX parity check failed: max abs diff {max_diff}")
    log.info("ONNX parity OK", extra={"extra_fields": {"max_diff": max_diff}})
