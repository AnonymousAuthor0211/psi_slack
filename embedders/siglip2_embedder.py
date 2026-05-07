"""
SigLIP2 embedder — wraps google/siglip2-so400m-patch14-384 (1152-dim).

Uses the same HuggingFace SiglipModel architecture as SigLIP v1 but with
the SO-400M checkpoint (27 layers, 1152 hidden, 384px images, 256K vocab).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
from typing import Dict, List, Union
from pathlib import Path
import logging


class SigLIP2Embedder:

    def __init__(self,
                 model_name: str = "google/siglip2-so400m-patch14-384",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 local_model_path: str = None):
        self.model_name = model_name
        self.dtype = dtype

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model_path = local_model_path or model_name

        local_dir = Path(model_path)
        if local_dir.exists() and (local_dir / "config.json").exists():
            logging.info(f"Loading SigLIP2 from local: {local_dir}")
        else:
            logging.info(f"Loading SigLIP2 from HuggingFace: {model_path}")

        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype,
                                                trust_remote_code=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        config = self.model.config
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.context_length = config.text_config.max_position_embeddings
        self.embed_dim = config.text_config.hidden_size

        logging.info(f"SigLIP2: dim={self.embed_dim}, img={self.image_size}px, "
                     f"patch={self.patch_size}, ctx={self.context_length}")

    def _to_pil(self, batch: torch.Tensor) -> List[Image.Image]:
        t = batch.detach().cpu()
        if t.dtype != torch.uint8:
            t = (t.clamp(0, 1) * 255).to(torch.uint8)
        return [Image.fromarray(x.permute(1, 2, 0).numpy()) for x in t]

    @staticmethod
    def _to_tensor(out):
        """Extract a plain tensor from model output (may be a NamedTuple)."""
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        return out[0] if isinstance(out, (tuple, list)) else out

    def encode_images(self, images, batch_size=32, normalize=True,
                      assume_preprocessed=False) -> np.ndarray:
        if isinstance(images, torch.Tensor) and not assume_preprocessed:
            images = self._to_pil(images)

        embeddings = []
        n_batches = (len(images) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                batch = images[i * batch_size:(i + 1) * batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                pv = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
                raw = self.model.get_image_features(pixel_values=pv)
                feats = self._to_tensor(raw)
                if normalize:
                    feats = F.normalize(feats, p=2, dim=1)
                embeddings.append(feats.cpu().numpy().astype(np.float32))
                del pv, feats, raw
        return np.concatenate(embeddings, axis=0)

    def encode_texts(self, texts: List[str], batch_size=32,
                     normalize=True) -> np.ndarray:
        embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                batch = texts[i * batch_size:(i + 1) * batch_size]
                inputs = self.processor(text=batch, return_tensors="pt",
                                        padding="max_length", truncation=True,
                                        max_length=self.context_length)
                ids = inputs.input_ids.to(self.device)
                mask = getattr(inputs, "attention_mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                    raw = self.model.get_text_features(input_ids=ids,
                                                       attention_mask=mask)
                else:
                    raw = self.model.get_text_features(input_ids=ids)
                feats = self._to_tensor(raw)
                if normalize:
                    feats = F.normalize(feats, p=2, dim=1)
                embeddings.append(feats.cpu().numpy().astype(np.float32))
                del ids, feats, raw
        return np.concatenate(embeddings, axis=0)

    def compute_similarity(self, image_features, text_features,
                           logit_scale=1.0) -> np.ndarray:
        img = torch.from_numpy(image_features).to(self.device, torch.float32)
        txt = torch.from_numpy(text_features).to(self.device, torch.float32)
        with torch.no_grad():
            sim = img @ txt.T
            if logit_scale != 1.0:
                sim = sim * logit_scale
        return sim.cpu().numpy()

    def get_metadata(self) -> Dict:
        return {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'context_length': self.context_length,
            'embed_dim': self.embed_dim,
            'dtype': str(self.dtype),
            'device': str(self.device),
            'unit_norm': True,
            'vMF_compatible': True,
        }
