import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SiglipModel, SiglipProcessor
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

class SiglipEmbedder:
    """
    SigLIP embedder with unit-norm outputs compatible with vMF pipeline.
    Returns proper cosine similarities in [-1, 1] range.
    """
    
    def __init__(self, 
                 model_name: str = "google/siglip-base-patch16-256",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 local_model_path: str = None):
        """
        Initialize SigLIP embedder.
        
        Args:
            model_name: SigLIP model variant (default: base-patch16-256)
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Model precision (float32 recommended for SigLIP)
        """
        self.model_name = model_name
        self.dtype = dtype
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Determine model path - use local if available and valid, otherwise download
        if local_model_path is not None:
            model_path = local_model_path
            logging.info(f"Using specified local model: {model_path}")
        else:
            # Check if local SigLIP model exists and is valid
            local_path = Path("base_model/siglip")
            config_path = local_path / "config.json"
            
            if local_path.exists() and config_path.exists():
                # Check if it's actually a SigLIP model (not BLIP)
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Check if it's a SigLIP model
                    if config.get("model_type") == "siglip" or "siglip" in config.get("architectures", []):
                        model_path = str(local_path)
                        logging.info(f"Using local SigLIP model from: {model_path}")
                    else:
                        model_path = model_name  # Use Hugging Face model
                        logging.info(f"Local model is not SigLIP (found {config.get('model_type', 'unknown')}), using Hugging Face: {model_name}")
                except Exception as e:
                    model_path = model_name  # Use Hugging Face model
                    logging.info(f"Error reading local model config: {e}, using Hugging Face: {model_name}")
            else:
                model_path = model_name  # Use Hugging Face model
                logging.info(f"No local SigLIP model found, using Hugging Face: {model_name}")
        
        # Load SigLIP model in float32 (SigLIP prefers fp32)
        self.model = SiglipModel.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.processor = SiglipProcessor.from_pretrained(model_path, use_fast=True)
        
        # Get model info
        config = self.model.config
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.context_length = config.text_config.max_position_embeddings
        self.embed_dim = config.text_config.hidden_size
        
        # Keep a logit scale only for similarity matrices if needed (not for embeddings)
        self.logit_scale_for_logits = 1.0
        
        logging.info(f"Loaded SigLIP {model_name} on {self.device} with dtype {dtype}")
        logging.info(f"Image size: {self.image_size}, Context length: {self.context_length}")
        logging.info(f"Embedding dimension: {self.embed_dim}")
        logging.info("Embeddings will be unit-norm for vMF compatibility")
    
    def _tensor_batch_to_pil(self, batch: torch.Tensor) -> List[Image.Image]:
        """Convert CHW float tensors to PIL RGB uint8 images."""
        t = batch.detach().cpu()
        if t.ndim != 4 or t.shape[1] != 3:
            raise ValueError("Expected [N,3,H,W]")
        if t.dtype != torch.uint8:
            t = (t.clamp(0, 1) * 255).to(torch.uint8)
        return [Image.fromarray(x.permute(1,2,0).numpy()) for x in t]
    
    def encode_images(self, 
                     images: Union[List[Image.Image], torch.Tensor],
                     batch_size: int = 32,
                     normalize: bool = True,
                     assume_preprocessed: bool = False) -> np.ndarray:
        """
        Encode images to unit-norm embeddings.
        
        Args:
            images: List of PIL Images or tensor of shape [N, C, H, W]
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings (keep True for vMF)
            assume_preprocessed: If True, assume tensors are already processed by SigLIP processor.
                                If False (default), ALWAYS route through SigLIP processor.
            
        Returns:
            Unit-norm embeddings of shape [N, dim] as float32 numpy array
        """
        if isinstance(images, list) or not assume_preprocessed:
            # Always use processor (PIL list OR raw tensors)
            if isinstance(images, torch.Tensor) and not assume_preprocessed:
                images = self._tensor_batch_to_pil(images)
            
            embeddings = []
            num_batches = (len(images) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(images))
                    batch_images = images[start_idx:end_idx]
                    
                    # Preprocess batch using SigLIP processor
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
                    
                    # Encode images using SigLIP's get_image_features method
                    image_features = self.model.get_image_features(pixel_values=pixel_values)
                    
                    # L2 normalize to unit sphere (critical for vMF)
                    if normalize:
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    embeddings.append(image_features.cpu().numpy().astype(np.float32))
                    
                    # Clear GPU memory only if on CUDA
                    del pixel_values, image_features
            
            return np.concatenate(embeddings, axis=0)
        else:
            # Only use this path if you KNOW tensors are from SigLIP processor
            assert images.ndim == 4 and images.shape[-2:] == (self.image_size, self.image_size), \
                "assume_preprocessed=True expects SigLIP pixel_values-sized tensors"
            
            image_tensors = images.to(device=self.device, dtype=self.dtype)
            
            embeddings = []
            num_batches = (len(image_tensors) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(image_tensors))
                    batch = image_tensors[start_idx:end_idx]
                    
                    # Encode images using SigLIP's get_image_features method
                    image_features = self.model.get_image_features(pixel_values=batch)
                    
                    # L2 normalize to unit sphere (critical for vMF)
                    if normalize:
                        image_features = F.normalize(image_features, p=2, dim=1)
                    
                    embeddings.append(image_features.cpu().numpy().astype(np.float32))
            
            return np.concatenate(embeddings, axis=0)
    
    def encode_texts(self, 
                    texts: List[str],
                    batch_size: int = 32,
                    normalize: bool = True) -> np.ndarray:
        """
        Encode texts to unit-norm embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings (keep True for vMF)
            
        Returns:
            Unit-norm embeddings of shape [N, dim] as float32 numpy array
        """
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize texts using SigLIP processor with stable padding
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.context_length,
                )
                input_ids = inputs.input_ids.to(self.device)
                
                # SigLIP processor may return attention_mask; use it if available
                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    text_features = self.model.get_text_features(input_ids=input_ids)
                
                # L2 normalize to unit sphere (critical for vMF)
                if normalize:
                    text_features = F.normalize(text_features, p=2, dim=1)
                
                embeddings.append(text_features.cpu().numpy().astype(np.float32))
        
        return np.concatenate(embeddings, axis=0)
    
    def compute_similarity(self, 
                          image_features: np.ndarray, 
                          text_features: np.ndarray,
                          logit_scale: float = 1.0) -> np.ndarray:
        """
        Compute cosine similarity between unit-norm image and text features.
        
        Args:
            image_features: [N, dim] unit-norm image embeddings
            text_features: [M, dim] unit-norm text embeddings
            logit_scale: Optional scaling for similarity matrix (not embeddings!)
            
        Returns:
            Similarity matrix [N, M] with cosines in [-1, 1] range
        """
        # Convert to tensors (assumes inputs are unit-norm)
        img_feats = torch.from_numpy(image_features).to(device=self.device, dtype=torch.float32)
        txt_feats = torch.from_numpy(text_features).to(device=self.device, dtype=torch.float32)
        
        # Compute similarity matrix (true cosines in [-1, 1])
        with torch.no_grad():
            similarity = img_feats @ txt_feats.T
            
            # Apply logit scaling to similarity matrix if needed (not to vectors!)
            if logit_scale != 1.0:
                similarity = similarity * logit_scale
                
        return similarity.cpu().numpy()
    
    def get_metadata(self) -> Dict:
        """Get model metadata for caching."""
        return {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'context_length': self.context_length,
            'embed_dim': self.embed_dim,
            'dtype': str(self.dtype),
            'device': str(self.device),
            'unit_norm': True,
            'vMF_compatible': True
        }