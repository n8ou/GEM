"""Export the best trained checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py --backbone r50
"""
from __future__ import annotations

import argparse

from facecore.config import get_settings
from facecore.export.onnx_export import export_to_onnx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="r50")
    p.add_argument("--checkpoint", default="best.pt")
    args = p.parse_args()

    settings = get_settings()
    out = settings.model_dir / f"{settings.embedder}.onnx"
    export_to_onnx(
        backbone_name=args.backbone,
        weights_path=settings.checkpoint_dir / args.checkpoint,
        output_path=out,
        embedding_dim=settings.embedding_dim,
    )
    print(f"Exported -> {out}")


if __name__ == "__main__":
    main()
