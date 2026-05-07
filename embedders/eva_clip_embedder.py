"""
EVA-CLIP via Hugging Face transformers.
Default repo: microsoft/LLM2CLIP-EVA02-L-14-336.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel, CLIPModel, CLIPProcessor

# Recent transformers refuse torch.load(pickle) weights unless torch>=2.6; safetensors avoids that.
_LOAD_KW = {"use_safetensors": True}

# Hub repo that includes model.safetensors (works with torch<2.6 + current transformers).
_DEFAULT_SAFE_REPO = "townwish/EVACLIP-ViT-L-14-336px"


def _local_has_only_pickle_weights(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    has_bin = (p / "pytorch_model.bin").is_file()
    has_safe = (p / "model.safetensors").is_file()
    return has_bin and not has_safe


class EVACLIPEmbedder:
    """EVA-CLIP L/14 with L2-normalized embeddings (cosine = dot product)."""

    def __init__(
        self,
        model_name: str = _DEFAULT_SAFE_REPO,
        local_model_path: Optional[str] = None,
        processor_name: str = "openai/clip-vit-large-patch14-336",
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.model_name = model_name
        self.processor_name = processor_name

        path = model_name
        if local_model_path is not None:
            lp = Path(local_model_path)
            if (lp / "config.json").is_file():
                path = str(lp)
        if path == model_name:
            local = Path("base_model/eva_clip_l14")
            if (local / "config.json").is_file():
                path = str(local)

        logging.info("Loading EVA-CLIP model from %s", path)
        if _local_has_only_pickle_weights(path):
            raise RuntimeError(
                f"{path} has pytorch_model.bin but no model.safetensors (e.g. Microsoft LLM2CLIP snapshot). "
                "Do not mix weights across Hub repos. Remove and re-download so the first successful repo "
                f"is {_DEFAULT_SAFE_REPO} (includes model.safetensors):\n"
                "  rm -rf base_model/eva_clip_l14 && python base_model/download_model.py --eva-clip-l14\n"
                "Or install torch>=2.6 so pytorch_model.bin can load."
            )
        # Prefer CLIPModel/CLIPProcessor when the repo really is a vanilla CLIP. The
        # townwish EVA snapshot ships a custom EvaCLIPModel (config.json declares
        # architectures: ['EvaCLIPModel'], with modeling_evaclip.py). That model has
        # different vision tensor shapes than CLIP and exposes encode_image/encode_text
        # instead of get_image_features/get_text_features, so we route around CLIPModel
        # entirely when we detect it.
        self.model = None
        self.processor = None

        try:
            self.processor = CLIPProcessor.from_pretrained(path)
        except Exception as e:
            logging.warning(
                "Failed to load CLIPProcessor from %s (%s). Falling back to %s",
                path,
                e,
                processor_name,
            )
            self.processor = CLIPProcessor.from_pretrained(processor_name, use_fast=False)

        is_eva_custom = False
        try:
            cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            archs = getattr(cfg, "architectures", None) or []
            is_eva_custom = any("Eva" in a for a in archs)
        except Exception as e:
            logging.warning("Could not read config.architectures from %s (%s)", path, e)

        if is_eva_custom:
            logging.info("Detected custom EvaCLIPModel at %s; using AutoModel(trust_remote_code=True).", path)
            # The custom EvaCLIPModel registers several non-persistent buffers
            # (vision position_ids and per-layer RoPE freqs) that are NOT in the state
            # dict. Under transformers' load path these end up as uninitialized memory
            # (garbage indices for position_ids; NaN/Inf RoPE tables -> NaN features).
            # We explicitly recompute them after loading.
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **_LOAD_KW
            ).to(self.device)
            self._reset_eva_position_buffers()
        else:
            try:
                self.model = CLIPModel.from_pretrained(path, **_LOAD_KW).to(self.device)
            except Exception as e:
                logging.warning(
                    "Failed to load CLIPModel from %s (%s). Falling back to AutoModel(trust_remote_code=True).",
                    path,
                    e,
                )
                self.model = AutoModel.from_pretrained(
                    path, trust_remote_code=True, **_LOAD_KW
                ).to(self.device)
                self._reset_eva_position_buffers()

        has_get_features = hasattr(self.model, "get_image_features") and hasattr(
            self.model, "get_text_features"
        )
        has_encode = hasattr(self.model, "encode_image") and hasattr(self.model, "encode_text")
        if not (has_get_features or has_encode):
            raise RuntimeError(
                "Loaded EVA model exposes neither get_image_features/get_text_features "
                "nor encode_image/encode_text."
            )
        self._using_clip_api = has_get_features

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Robust config reads across CLIPModel and custom EVA configs.
        # (helper defined below as method)
        self.embedding_dim = int(getattr(self.model.config, "projection_dim", 1024))
        vision_cfg = getattr(self.model.config, "vision_config", None)
        text_cfg = getattr(self.model.config, "text_config", None)
        self.image_size = int(getattr(vision_cfg, "image_size", 336))
        self.context_length = int(getattr(text_cfg, "max_position_embeddings", 77))

    def _reset_eva_position_buffers(self) -> None:
        # The townwish EvaCLIPModel (modeling_evaclip.py) registers several non-persistent
        # buffers (`position_ids`, RoPE `freqs_cos`/`freqs_sin`) that are computed in
        # __init__ but NOT in the state dict. Under transformers' load path these buffers
        # end up holding uninitialized memory after from_pretrained. We recompute them
        # here using the same formulas as the upstream modeling file.
        ve = getattr(getattr(self.model, "vision_model", None), "embeddings", None)
        if ve is not None and hasattr(ve, "num_positions") and hasattr(ve, "position_ids"):
            ve.position_ids = torch.arange(ve.num_positions, device=ve.position_ids.device).expand((1, -1))

        te = getattr(getattr(self.model, "text_model", None), "embeddings", None)
        if te is not None and hasattr(te, "position_ids"):
            n = te.position_ids.shape[-1]
            te.position_ids = torch.arange(n, device=te.position_ids.device).expand((1, -1))

        vision_cfg = getattr(self.model.config, "vision_config", None)
        if vision_cfg is None:
            return
        seq_len = int(vision_cfg.image_size) // int(vision_cfg.patch_size)
        dim = int(vision_cfg.hidden_size) // int(vision_cfg.num_attention_heads) // 2
        rope_theta = float(getattr(vision_cfg, "rope_theta", 10000))
        pretrained_seq_len = int(getattr(vision_cfg, "pretrained_seq_len", seq_len))

        for module in self.model.modules():
            if not (hasattr(module, "freqs_cos") and hasattr(module, "freqs_sin")):
                continue
            target_device = module.freqs_cos.device
            target_dtype = module.freqs_cos.dtype
            t = torch.arange(seq_len, dtype=torch.float32) / seq_len * pretrained_seq_len
            freqs = 1.0 / (
                rope_theta
                ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            freqs = t.unsqueeze(-1) * freqs.unsqueeze(0)
            freqs = freqs.repeat_interleave(2, dim=-1)
            freqs = torch.cat(
                [
                    freqs.unsqueeze(1).expand(-1, seq_len, -1),
                    freqs.unsqueeze(0).expand(seq_len, -1, -1),
                ],
                dim=-1,
            )
            freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).to(device=target_device, dtype=target_dtype)
            freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).to(device=target_device, dtype=target_dtype)
            module.freqs_cos = freqs_cos
            module.freqs_sin = freqs_sin

    def encode_images(
        self,
        images: Union[List[Image.Image], List[str], List[Path]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                pil_batch = []
                for img in batch:
                    if isinstance(img, (str, Path)):
                        pil_batch.append(Image.open(img).convert("RGB"))
                    else:
                        pil_batch.append(img)
                inputs = self.processor(images=pil_batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self._using_clip_api:
                    feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                else:
                    feats = self.model.encode_image(pixel_values=inputs["pixel_values"])
                if normalize:
                    feats = F.normalize(feats, p=2, dim=-1)
                all_embs.append(feats.cpu().numpy().astype(np.float16))
        return np.concatenate(all_embs, axis=0)

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self._using_clip_api:
                    feats = self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                else:
                    feats = self.model.encode_text(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                if normalize:
                    feats = F.normalize(feats, p=2, dim=-1)
                all_embs.append(feats.cpu().numpy().astype(np.float16))
        return np.concatenate(all_embs, axis=0)

    def compute_similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        img = torch.from_numpy(image_features).to(self.device, dtype=self.dtype)
        txt = torch.from_numpy(text_features).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            return (img @ txt.T).cpu().numpy()

    def get_metadata(self) -> Dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "library": "transformers_eva_clip",
        }
