import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

try:
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class BLIPEmbedder:
    """
    BLIP embedder for image-text retrieval with L2-normed outputs.
    Uses BlipForImageTextRetrieval to get properly aligned embeddings.
    """
    
    def __init__(self, 
                 model_path: str = "base_model/blip",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize BLIP embedder.
        
        Args:
            model_path: Path to local BLIP model directory
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Model precision (fp16 for efficiency)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
            
        self.model_path = model_path
        self.dtype = dtype
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logging.info(f"Loading BLIP for retrieval from {model_path}...")
        
        # Load BLIP model and processor from local path
        # Use BlipForImageTextRetrieval for proper retrieval embeddings
        self.model = BlipForImageTextRetrieval.from_pretrained(model_path)
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Loaded BLIP from {model_path} on {self.device} with dtype {dtype}")
        if torch.cuda.is_available():
            logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def encode_images(self, 
                     images: Union[List[Image.Image], torch.Tensor],
                     batch_size: int = 32,
                     normalize: bool = True) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: List of PIL Images or tensor
            batch_size: Batch size for processing
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings of shape [N, dim] as fp16 numpy array
        """
        if isinstance(images, torch.Tensor):
            raise ValueError("BLIP embedder expects PIL Images, not tensors")
            
        embeddings = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        with torch.inference_mode():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                
                # Preprocess batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                
                # Move to device
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype.is_floating_point:
                            processed_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                        else:
                            processed_inputs[k] = v.to(device=self.device)
                
                # Get vision embeddings using vision encoder + projection
                vision_outputs = self.model.vision_model(**processed_inputs)
                image_embeds = vision_outputs[0]  # [batch, seq_len, hidden_dim]
                # Take CLS token (first token) and project
                image_features = self.model.vision_proj(image_embeds[:, 0, :])
                
                # L2 normalize
                if normalize:
                    image_features = F.normalize(image_features, p=2, dim=1)
                
                embeddings.append(image_features.cpu().numpy().astype(np.float16))
                
                # Cleanup
                del inputs, processed_inputs, image_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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
        
        with torch.inference_mode():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Encode texts
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True)
                
                # Move to device
                processed_inputs = {}
                for k, v in inputs.items():
                    if k == 'input_ids':
                        processed_inputs[k] = v.to(device=self.device)  # Keep as Long
                    else:
                        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                            processed_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                        else:
                            processed_inputs[k] = v.to(device=self.device)
                
                # Get text embeddings using text encoder + projection
                text_outputs = self.model.text_encoder(**processed_inputs)
                text_embeds = text_outputs[0]  # [batch, seq_len, hidden_dim]
                # Take CLS token (first token) and project
                text_features = self.model.text_proj(text_embeds[:, 0, :])
                
                # L2 normalize
                if normalize:
                    text_features = F.normalize(text_features, p=2, dim=1)
                
                embeddings.append(text_features.cpu().numpy().astype(np.float16))
                
                # Cleanup
                del inputs, processed_inputs, text_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(embeddings, axis=0)
    
    def compute_similarity(self, 
                          image_features: np.ndarray, 
                          text_features: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text features.
        Assumes features are already L2-normalized.
        """
        img_feats = torch.from_numpy(image_features).to(device=self.device, dtype=self.dtype)
        txt_feats = torch.from_numpy(text_features).to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            similarity = img_feats @ txt_feats.T
            
        return similarity.cpu().numpy()
    
    def get_metadata(self) -> Dict:
        """Get model metadata for caching."""
        return {
            'model_path': self.model_path,
            'dtype': str(self.dtype),
            'device': str(self.device)
        }

