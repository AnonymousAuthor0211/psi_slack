import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

class CLIPEmbedder:
    """
    OpenAI CLIP embedder with deterministic transforms and L2-normed outputs.
    Defaults to ViT-L/14@336px; use model_name=\"ViT-B/16\" for the base backbone.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-L/14@336px",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float16,
                 download_root: Optional[str] = None):
        """
        Initialize CLIP embedder.
        
        Args:
            model_name: CLIP model variant (e.g. ViT-L/14@336px, ViT-B/16)
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Model precision (fp16 for efficiency)
            download_root: If set, passed to ``clip.load`` (e.g. base_model/clip_vit_b16)
        """
        self.model_name = model_name
        self.dtype = dtype
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Load CLIP model
        load_kw = {}
        if download_root is not None:
            load_kw["download_root"] = download_root
        self.model, self.preprocess = clip.load(model_name, device=self.device, **load_kw)
        # Keep model in float32 for compatibility, but use dtype for text encoder
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get model info (resolution differs by variant, e.g. 224 for ViT-B/16)
        try:
            self.image_size = int(self.model.visual.input_resolution)
        except AttributeError:
            self.image_size = 224
        self.context_length = self.model.context_length
        
        logging.info(f"Loaded CLIP {model_name} on {self.device} with dtype {dtype}")
        logging.info(f"Image size: {self.image_size}, Context length: {self.context_length}")
    
    def encode_images(self, 
                     images: Union[List[Image.Image], torch.Tensor],
                     batch_size: int = 32,
                     normalize: bool = True) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: List of PIL Images or tensor of shape [N, C, H, W]
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings of shape [N, dim] as fp16 numpy array
        """
        if isinstance(images, list):
            # Process images in batches to avoid memory issues
            embeddings = []
            num_batches = (len(images) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(images))
                    batch_images = images[start_idx:end_idx]
                    
                    # Preprocess batch - CLIP visual encoder expects float32
                    image_tensors = torch.stack([
                        self.preprocess(img) for img in batch_images
                    ]).to(device=self.device, dtype=torch.float32)
                    
                    # Encode images
                    image_features = self.model.encode_image(image_tensors)
                    
                    # L2 normalize
                    if normalize:
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    embeddings.append(image_features.cpu().numpy().astype(np.float16))
                    
                    # Clear GPU memory
                    del image_tensors, image_features
                    torch.cuda.empty_cache()
            
            return np.concatenate(embeddings, axis=0)
        else:
            # Already tensor
            image_tensors = images.to(device=self.device, dtype=self.dtype)
            
            embeddings = []
            num_batches = (len(image_tensors) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(image_tensors))
                    batch = image_tensors[start_idx:end_idx]
                    
                    # Encode images
                    image_features = self.model.encode_image(batch)
                    
                    # L2 normalize
                    if normalize:
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    embeddings.append(image_features.cpu().numpy().astype(np.float16))
            
            return np.concatenate(embeddings, axis=0)
    
    def encode_texts(self, 
                    texts: List[str],
                    batch_size: int = 32,
                    normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings of shape [N, dim] as fp16 numpy array
        """
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize texts
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
                
                # Encode texts
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features.to(dtype=self.dtype)
                
                # L2 normalize
                if normalize:
                    text_features = F.normalize(text_features, p=2, dim=1)
                
                embeddings.append(text_features.cpu().numpy().astype(np.float16))
        
        return np.concatenate(embeddings, axis=0)
    
    def compute_similarity(self, 
                          image_features: np.ndarray, 
                          text_features: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text features.
        Assumes features are already L2-normalized.
        
        Args:
            image_features: [N, dim] image embeddings
            text_features: [M, dim] text embeddings
            
        Returns:
            Similarity matrix [N, M]
        """
        # Convert to tensors
        img_feats = torch.from_numpy(image_features).to(device=self.device, dtype=self.dtype)
        txt_feats = torch.from_numpy(text_features).to(device=self.device, dtype=self.dtype)
        
        # Compute similarity (dot product if L2-normed = cosine similarity)
        with torch.no_grad():
            similarity = img_feats @ txt_feats.T
            
        return similarity.cpu().numpy()
    
    def get_metadata(self) -> Dict:
        """Get model metadata for caching."""
        return {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'context_length': self.context_length,
            'dtype': str(self.dtype),
            'device': str(self.device)
        }


class CLIPViTB16Embedder(CLIPEmbedder):
    """OpenAI CLIP ViT-B/16 (224px). Uses local weights under base_model/clip_vit_b16 when present."""

    def __init__(
        self,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        download_root: Optional[str] = None,
    ):
        root = download_root
        if root is None:
            p = Path("base_model/clip_vit_b16")
            if p.is_dir():
                root = str(p)
        super().__init__(
            model_name="ViT-B/16",
            device=device,
            dtype=dtype,
            download_root=root,
        )