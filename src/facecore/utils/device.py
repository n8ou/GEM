"""Device resolution with graceful GPU->CPU fallback."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from facecore.config import DevicePref
from facecore.logging_conf import get_logger

log = get_logger(__name__)

_cuda_dlls_registered = False


def _register_cuda_dlls() -> None:
    """Make ONNXRuntime's CUDA EP find the CUDA 12 / cuDNN 9 DLLs that ship with
    the PyTorch cu126 wheel. Without this, ORT logs a load error for
    `cublasLt64_12.dll`/cuDNN and silently falls back to CPU."""
    global _cuda_dlls_registered
    if _cuda_dlls_registered or sys.platform != "win32":
        return
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        os.add_dll_directory(str(torch_lib))  # py3.8+ Windows DLL search
        os.environ["PATH"] = f"{torch_lib}{os.pathsep}{os.environ.get('PATH', '')}"
    _cuda_dlls_registered = True


def resolve_device(pref: DevicePref) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        log.info("CUDA available — using GPU", extra={"extra_fields": {"gpu": torch.cuda.get_device_name(0)}})
        return torch.device("cuda")
    log.info("CUDA not available — falling back to CPU")
    return torch.device("cpu")


def onnx_providers(device: torch.device) -> list[str]:
    """ONNXRuntime execution providers, ordered by preference."""
    if device.type == "cuda":
        _register_cuda_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
