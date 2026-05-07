"""
LAION CLIP embedder using OpenCLIP.
Model: CLIP-ViT-L-14 trained on LAION-2B (2B English subset)
Paper: https://arxiv.org/abs/2111.02114
"""

import open_clip
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from pathlib import Path


class LAIONEmbedder:
    """
    LAION CLIP embedder using OpenCLIP.
    Compatible with multi-GPU caching infrastructure.
    """
    
    def __init__(
        self, 
        model_name: str = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize LAION CLIP embedder.
        
        Args:
            model_name: Model identifier (default: LAION CLIP-ViT-L-14)
            device: Device to use (default: cuda)
            dtype: Data type (default: float32 for V100 compatibility)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        print(f"Loading LAION CLIP model from {model_name}...")
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move to device and set dtype
        self.model = self.model.to(device)
        if dtype == torch.float16:
            self.model = self.model.half()
        
        self.model.eval()

        self.embed_dim = self._infer_embed_dim()

        print(f"✓ Model loaded on {device}")
        print(f"  Embedding dimension: {self.embed_dim}")

    def _infer_embed_dim(self):
        """Infer the joint embedding dimension across OpenCLIP variants.

        Standard CLIP visual towers expose ``output_dim``, but TimmModel-wrapped
        towers (e.g. EVA-CLIP) do not. Fall back to the text projection shape,
        and finally to a dummy text encode.
        """
        visual = getattr(self.model, "visual", None)
        if visual is not None and hasattr(visual, "output_dim"):
            return visual.output_dim

        text_proj = getattr(self.model, "text_projection", None)
        if text_proj is not None:
            if hasattr(text_proj, "shape"):
                return int(text_proj.shape[-1])
            if hasattr(text_proj, "weight"):
                return int(text_proj.weight.shape[0])

        # Last-resort: run a single token through the text encoder.
        with torch.no_grad():
            dummy = self.tokenizer([""]).to(self.device)
            return int(self.model.encode_text(dummy).shape[-1])
    
    def encode_images(
        self, 
        images: Union[List[Image.Image], List[str], List[Path]], 
        batch_size: int = 64,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: List of PIL Images or paths to images
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            np.ndarray: Image embeddings [N, D]
        """
        all_embs = []
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # V100 doesn't support bfloat16
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Load and preprocess images
                processed_images = []
                for img in batch:
                    try:
                        # Load image if path
                        if isinstance(img, (str, Path)):
                            img = Image.open(img).convert("RGB")
                        
                        # Preprocess
                        processed_images.append(self.preprocess_val(img))
                    except Exception as e:
                        print(f"Warning: Failed to load image: {e}")
                        # Use blank image as placeholder
                        processed_images.append(
                            self.preprocess_val(Image.new("RGB", (224, 224), (0, 0, 0)))
                        )
                
                # Stack and move to device
                images_tensor = torch.stack(processed_images).to(self.device)
                if self.dtype == torch.float16:
                    images_tensor = images_tensor.half()
                
                # Encode
                image_features = self.model.encode_image(images_tensor)
                
                # Normalize if requested
                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_embs.append(image_features.cpu().float().numpy())
        
        return np.concatenate(all_embs, axis=0)
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 64,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text captions to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            np.ndarray: Text embeddings [N, D]
        """
        all_embs = []
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                text_tokens = self.tokenizer(batch_texts).to(self.device)
                
                # Encode
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize if requested
                if normalize:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_embs.append(text_features.cpu().float().numpy())
        
        return np.concatenate(all_embs, axis=0)
    
    def get_metadata(self) -> dict:
        """Return metadata about the embedder."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embed_dim,
            "device": self.device,
            "dtype": str(self.dtype),
            "library": "open_clip"
        }
    
    def compute_similarity(
        self, 
        image_embs: np.ndarray, 
        text_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embs: Image embeddings [N, D]
            text_embs: Text embeddings [M, D]
        
        Returns:
            np.ndarray: Similarity matrix [N, M]
        """
        return image_embs @ text_embs.T


# Standalone functions for backward compatibility
def embed_images(image_paths: Union[List[str], List[Path]], batch_size: int = 64) -> np.ndarray:
    """Standalone function - loads model each time (inefficient)."""
    embedder = LAIONEmbedder()
    return embedder.encode_images(image_paths, batch_size=batch_size)


def embed_texts(captions: List[str], batch_size: int = 64) -> np.ndarray:
    """Standalone function - loads model each time (inefficient)."""
    embedder = LAIONEmbedder()
    return embedder.encode_texts(captions, batch_size=batch_size)


def compute_similarity(image_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity."""
    return image_embs @ text_embs.T


if __name__ == "__main__":
    # Quick test
    print("\n=== Testing LAION CLIP Embedder ===")
    
    embedder = LAIONEmbedder()
    
    test_texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a beautiful sunset over the ocean"
    ]
    
    print(f"\nEncoding {len(test_texts)} test captions...")
    text_embs = embedder.encode_texts(test_texts)
    print(f"✓ Text embeddings shape: {text_embs.shape}")
    print(f"  Embedding norm (should be ~1.0): {np.linalg.norm(text_embs[0]):.4f}")
    
    # Text-to-text similarity
    sims = embedder.compute_similarity(text_embs, text_embs)
    print(f"\nText similarity matrix:\n{sims}")
    print(f"  Diagonal (self-similarity): {np.diag(sims)}")
    
    print(f"\nMetadata: {embedder.get_metadata()}")
    print("\n✓ Embedder ready to use!")
