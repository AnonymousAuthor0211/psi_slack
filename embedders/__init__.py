from __future__ import annotations

from .clip_embedder import CLIPEmbedder, CLIPViTB16Embedder
from .siglip_embedder import SiglipEmbedder
from .blip_embedder import BLIPEmbedder
from .open_clip_embedder import OpenCLIPEmbedder
from .eva_clip_embedder import EVACLIPEmbedder

# CLAP pulls torchaudio (CUDA build must match torch). Lazy-load so image-only paths
# (e.g. LAION embedder: `from embedders.laion_embedder import LAIONEmbedder`) never import it.
__all__ = ["CLIPEmbedder", "CLIPViTB16Embedder", "CLAPEmbedder", "SiglipEmbedder", "SigLIP2Embedder",
           "BLIPEmbedder", "LAIONEmbedder", "OpenCLIPEmbedder", "EVACLIPEmbedder"]


def __getattr__(name: str):
    if name == "CLAPEmbedder":
        from .clap_embedder import CLAPEmbedder

        return CLAPEmbedder
    if name == "LAIONEmbedder":
        from .laion_embedder import LAIONEmbedder

        return LAIONEmbedder
    if name == "SigLIP2Embedder":
        from .siglip2_embedder import SigLIP2Embedder

        return SigLIP2Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")