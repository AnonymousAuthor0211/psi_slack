"""
OpenCLIP embedder (any OpenCLIP backbone). Default: ViT-H/14 trained on LAION-2B.
"""

from __future__ import annotations

import torch

from .laion_embedder import LAIONEmbedder


class OpenCLIPEmbedder(LAIONEmbedder):
    """
    Same stack as LAIONEmbedder (open_clip), with ViT-H/14 defaults.

    Typical model_name:
      hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    """

    def __init__(
        self,
        model_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name=model_name, device=device, dtype=dtype)
