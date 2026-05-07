import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import warnings
import gc

try:
    from transformers import ClapModel, ClapProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CLAPEmbedder:
    """
    CLAP HTSAT-Base embedder with aggressive memory optimization.
    """
    
    def __init__(self, 
                 model_path: str = "base_model/clap-htsat-fused",
                 device: str = "auto",
                 dtype: torch.dtype = torch.float16,
                 target_sr: int = 48000,
                 max_duration: float = 10.0):  # Back to 10s, but process one at a time
        """
        Initialize CLAP embedder.
        
        Args:
            model_path: Path to local CLAP model directory
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Model precision (fp16 for efficiency)
            target_sr: Target sample rate for audio (48kHz for HTSAT-Base)
            max_duration: Maximum audio duration in seconds
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")
            
        self.model_path = model_path
        self.dtype = dtype
        self.target_sr = target_sr
        self.max_duration = max_duration
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Audio processing parameters
        self.max_samples = int(self.target_sr * self.max_duration)
        
        # Cache resampler to avoid recreating it every time
        self._resampler_cache = {}
        
        logging.info(f"Loading CLAP from {model_path}...")
        logging.info(f"Audio config: {target_sr} Hz, {self.max_duration}s ({self.max_samples} samples)")
        
        # Load CLAP model and processor from local path
        self.model = ClapModel.from_pretrained(model_path)
        self.processor = ClapProcessor.from_pretrained(model_path)
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Clear any existing cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Loaded CLAP from {model_path} on {self.device} with dtype {dtype}")
        logging.info(f"Target sample rate: {target_sr} Hz, Max duration: {self.max_duration}s")
        if torch.cuda.is_available():
            logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _resample_audio(self, 
                       audio: torch.Tensor, 
                       orig_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        Uses cached resampler to avoid recreating it every time.
        """
        if orig_sr == self.target_sr:
            return audio
            
        # Ensure audio is 2D [channels, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Use cached resampler to avoid recreating it
        cache_key = (orig_sr, self.target_sr)
        if cache_key not in self._resampler_cache:
            self._resampler_cache[cache_key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, 
                new_freq=self.target_sr
            )
        resampler = self._resampler_cache[cache_key]
        
        audio_resampled = resampler(audio)
        
        return audio_resampled
    
    def _preprocess_audio(self, 
                         audio: torch.Tensor, 
                         orig_sr: int) -> torch.Tensor:
        """
        Preprocess audio: resample, convert to mono, pad/truncate.
        Keeps audio on CPU throughout.
        """
        # Ensure audio is on CPU
        if audio.is_cuda:
            audio = audio.cpu()
        
        # Resample to target SR (on CPU)
        audio = self._resample_audio(audio, orig_sr)
        
        # Convert to mono if stereo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        # Pad or truncate to max_samples
        if audio.shape[1] > self.max_samples:
            audio = audio[:, :self.max_samples]
        elif audio.shape[1] < self.max_samples:
            padding = self.max_samples - audio.shape[1]
            audio = F.pad(audio, (0, padding))
        
        return audio  # Returns [1, max_samples] on CPU
    
    def encode_audios(self, 
                     audios: Union[List[torch.Tensor], torch.Tensor],
                     sample_rates: Union[List[int], int],
                     batch_size: int = 1,  # FORCE batch_size=1 for audio
                     normalize: bool = True) -> np.ndarray:
        """
        Encode audio to embeddings.
        CRITICAL: Always processes one audio at a time to avoid OOM.
        
        Args:
            audios: List of audio tensors or batched tensor
            sample_rates: Sample rate(s) for the audio(s)
            batch_size: IGNORED - always processes 1 at a time
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings of shape [N, dim] as fp16 numpy array
        """
        # FORCE batch_size=1 regardless of input
        actual_batch_size = 1
        
        if isinstance(audios, torch.Tensor) and audios.dim() == 3:
            audio_list = [audios[i] for i in range(audios.shape[0])]
        else:
            audio_list = audios
            
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates] * len(audio_list)
        
        embeddings = []
        
        # Process ONE audio at a time
        with torch.inference_mode():
            for idx in range(len(audio_list)):
                # Preprocess single audio on CPU
                audio_cpu = self._preprocess_audio(audio_list[idx], sample_rates[idx])
                audio_cpu = audio_cpu[0]  # [max_samples]
                
                # Process single audio through processor
                # CRITICAL: Pass as single item, not list, to minimize memory
                try:
                    inputs = self.processor(
                        audios=audio_cpu,  # Single audio tensor
                        return_tensors="pt",
                        padding=False,
                        sampling_rate=self.target_sr,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logging.error(f"OOM during processor (sample {idx})!")
                        torch.cuda.empty_cache()
                        gc.collect()
                    raise
                
                # Cleanup CPU audio immediately
                del audio_cpu
                
                # Move to device with minimal memory footprint
                try:
                    processed_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype.is_floating_point:
                                processed_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                            else:
                                processed_inputs[k] = v.to(device=self.device)
                    
                    # Cleanup CPU inputs
                    del inputs
                    gc.collect()
                    
                    # Forward pass
                    audio_features = self.model.get_audio_features(**processed_inputs)
                    
                    # Cleanup inputs immediately
                    del processed_inputs
                    
                    # L2 normalize
                    if normalize:
                        audio_features = F.normalize(audio_features, p=2, dim=1)
                    
                    # Move to CPU and store as numpy
                    embedding_np = audio_features.cpu().numpy().astype(np.float16)
                    embeddings.append(embedding_np)
                    
                    # Cleanup
                    del audio_features, embedding_np
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logging.error(f"OOM during forward/move (sample {idx})!")
                        torch.cuda.empty_cache()
                        gc.collect()
                    raise
                
                # Aggressive cleanup after EVERY sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Log progress every 100 samples
                if (idx + 1) % 100 == 0:
                    mem_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    logging.info(f"Processed {idx + 1}/{len(audio_list)} audios (GPU mem: {mem_used:.2f}GB)")
        
        return np.concatenate(embeddings, axis=0)
    
    def encode_texts(self, 
                    texts: List[str],
                    batch_size: int = 32,
                    normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        Text encoding is much more memory-efficient than audio.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (32 is safe for text)
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
                        processed_inputs[k] = v.to(device=self.device)
                    else:
                        processed_inputs[k] = v.to(device=self.device, dtype=self.dtype)
                
                text_features = self.model.get_text_features(**processed_inputs)
                
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
                          audio_features: np.ndarray, 
                          text_features: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between audio and text features.
        """
        aud_feats = torch.from_numpy(audio_features).to(device=self.device, dtype=self.dtype)
        txt_feats = torch.from_numpy(text_features).to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            similarity = aud_feats @ txt_feats.T
            
        return similarity.cpu().numpy()
    
    def get_metadata(self) -> Dict:
        """Get model metadata for caching."""
        return {
            'model_path': self.model_path,
            'target_sr': self.target_sr,
            'max_duration': self.max_duration,
            'max_samples': self.max_samples,
            'dtype': str(self.dtype),
            'device': str(self.device)
        }
