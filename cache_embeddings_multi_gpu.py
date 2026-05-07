#!/usr/bin/env python3
"""
Multi-GPU embedding caching script - supports multiple model types and datasets.
This script launches separate processes for each GPU to process different splits in parallel.

Supported model types:
- clip: CLIP for image-text
- siglip: SigLIP for image-text  
- clap: CLAP for audio-text

Examples:
  # CLIP for image-text
  python cache_embeddings_multi_gpu.py --model_type clip --datasets coco_captions,flickr30k
  
  # LAION sample (OpenCLIP LAION CLIP); writes embeddings_laion/laion_sample_*_{image,text}.npz
  python cache_embeddings_multi_gpu.py --model_type laion --datasets laion_sample \\
      --data_root datasets/dataset_experiment --output_dir embeddings_laion --num_gpus 1 --gpu_ids 0
  
  # SigLIP for image-text
  python cache_embeddings_multi_gpu.py --model_type siglip --model_path google/siglip-base-patch16-256
  
  # CLAP for audio-text
  python cache_embeddings_multi_gpu.py --model_type clap --datasets audiocaps,clotho --data_root ./datasets
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import os
import subprocess
import time
import threading
import queue

# Standalone bundle (`psi_slack/`): repo root is this file's directory (contains `embedders/`).
REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_gpu_script(
    gpu_id: int,
    tasks: list,
    data_root: str,
    output_dir: str,
    batch_size: int,
    model_type: str,
    model_path: str,
    index_workers: int,
    text_batch_size: int,
    shard_id: int = 0,
    num_shards: int = 1,
    repo_root: str | None = None,
) -> str:
    """Create a script for a specific GPU to process its assigned tasks."""
    root = repo_root if repo_root is not None else REPO_ROOT
    repo_root_repr = repr(root)

    # Determine embedder class and initialization based on model type
    if model_type == 'clip':
        embedder_import = "from embedders import CLIPEmbedder"
        embedder_init = f'embedder = CLIPEmbedder(device=device)'
        modality = 'image-text'
    elif model_type == 'siglip':
        embedder_import = "from embedders import SiglipEmbedder"
        embedder_init = f'embedder = SiglipEmbedder(model_name="{model_path}", device=device)'
        modality = 'image-text'
    elif model_type == 'laion':
        embedder_import = "from embedders.laion_embedder import LAIONEmbedder"
        embedder_init = f'embedder = LAIONEmbedder(model_name="{model_path}", device=device, dtype=torch.float32)'
        modality = 'image-text'
    elif model_type == 'clap':
        embedder_import = "from embedders import CLAPEmbedder"
        # Back to 10s - the embedder now processes one audio at a time which avoids OOM
        embedder_init = f'embedder = CLAPEmbedder(model_path="{model_path}", device=device, dtype=torch.float16, max_duration=10.0)'
        modality = 'audio-text'
    elif model_type == 'blip':
        embedder_import = "from embedders import BLIPEmbedder"
        embedder_init = f'embedder = BLIPEmbedder(model_path="{model_path}", device=device, dtype=torch.float16)'
        modality = 'image-text'
    elif model_type == 'clip_b16':
        embedder_import = "from embedders import CLIPViTB16Embedder"
        embedder_init = "embedder = CLIPViTB16Embedder(device=device)"
        modality = 'image-text'
    elif model_type == 'openclip_vit_h14':
        embedder_import = "from embedders import OpenCLIPEmbedder"
        embedder_init = (
            f'embedder = OpenCLIPEmbedder(model_name="{model_path}", device=device, dtype=torch.float32)'
        )
        modality = 'image-text'
    elif model_type == 'eva_clip_l14':
        # Route through open_clip (BAAI EVA-02 CLIP-L/14-336 via timm hub).
        # Route EVA-CLIP through OpenCLIP hub id (see embedders/open_clip_embedder.py).
        embedder_import = "from embedders import OpenCLIPEmbedder"
        embedder_init = (
            f'embedder = OpenCLIPEmbedder(model_name="{model_path}", device=device, dtype=torch.float32)'
        )
        modality = 'image-text'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate processing function based on modality
    if modality == 'image-text':
        process_func = '''
def process_split(dataset, dataset_name, split, embedder, output_dir, batch_size):
    """Process image-text dataset split."""
    from PIL import Image
    import io
    
    logging.info(f"Processing image-text split: {dataset_name}_{split}")
    
    images = []
    texts = []
    ids = []
    
    # Configure tqdm to update less frequently (every 1% or every 2 seconds, whichever is less frequent)
    total = len(dataset)
    pbar = tqdm(range(total), desc=f"Loading {dataset_name}_{split}", 
                miniters=max(100, total // 100), mininterval=2.0, maxinterval=10.0)
    for i in pbar:
        try:
            item = dataset[i]
            
            # Handle image loading
            if 'image' in item:
                if isinstance(item['image'], dict) and 'bytes' in item['image']:
                    image = Image.open(io.BytesIO(item['image']['bytes']))
                else:
                    image = item['image']
            elif 'filepath' in item and 'filename' in item:
                image_path = Path("datasets/coco") / item['filepath'] / item['filename']
                image = Image.open(image_path).convert('RGB')
            else:
                logging.warning(f"No image found in sample {{i}}, skipping...")
                continue
            
            # Handle text loading
            if 'text' in item:
                captions = item['text'] if isinstance(item['text'], list) else [item['text']]
            elif 'caption' in item:
                captions = item['caption'] if isinstance(item['caption'], list) else [item['caption']]
            elif 'sentences' in item:
                captions = item['sentences']
            else:
                logging.warning(f"No text found in sample {{i}}, skipping...")
                continue
            
            images.append(image)
            texts.extend(captions)
            
            base_id = f"{dataset_name}_{split}_{i}"
            for j in range(len(captions)):
                ids.append(f"{base_id}_cap{j}")
                
        except Exception as e:
            logging.warning(f"Error loading sample {i}: {e}")
            continue
    
    if len(images) == 0:
        logging.warning(f"No valid samples found for {dataset_name}_{split}")
        return
    
    logging.info(f"Loaded {len(images)} images, {len(texts)} texts")
    
    # Encode images
    logging.info(f"Encoding images (batch_size={batch_size})")
    image_embeddings = embedder.encode_images(images, batch_size=batch_size)
    
    # Save image embeddings
    image_output_path = output_dir / f"{dataset_name}_{split}_image.npz"
    np.savez_compressed(
        image_output_path,
        embeddings=image_embeddings,
        ids=[f"{dataset_name}_{split}_{i}" for i in range(len(images))],
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved image embeddings to {image_output_path}")
    
    # Encode texts
    logging.info(f"Encoding texts (batch_size={batch_size})")
    text_embeddings = embedder.encode_texts(texts, batch_size=batch_size)
    
    # Save text embeddings
    text_output_path = output_dir / f"{dataset_name}_{split}_text.npz"
    np.savez_compressed(
        text_output_path,
        embeddings=text_embeddings,
        ids=ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved text embeddings to {text_output_path}")

def process_file_based_split(dataset_name, split_path, split, embedder, output_dir, batch_size):
    """Process file-based dataset (raw images like Flickr30k or LAION)."""
    
    if dataset_name == 'flickr30k':
        process_flickr30k_split(split_path, split, embedder, output_dir, batch_size)
    elif dataset_name == 'laion_sample':
        process_laion_split(split_path, split, embedder, output_dir, batch_size)
    else:
        logging.warning(f"File-based processing not implemented for {dataset_name}")
        
def process_flickr30k_split(split_path, split, embedder, output_dir, batch_size):
    """Process Flickr30k split with raw images and captions.txt."""
    import csv
    from PIL import Image
    
    logging.info(f"Processing Flickr30k split: {split}")
    
    # Load captions from the original captions.txt
    captions_file = Path("datasets/flickr30k/flickr30k/captions.txt")
    if not captions_file.exists():
        logging.error(f"Captions file not found: {captions_file}")
        return
    
    # Parse captions
    image_to_captions = {}
    with open(captions_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['image']
            caption = row['caption']
            if image_name not in image_to_captions:
                image_to_captions[image_name] = []
            image_to_captions[image_name].append(caption)
    
    logging.info(f"Loaded {len(image_to_captions)} images with captions")
    
    # Get images in this split
    split_images = [f.name for f in split_path.glob("*.jpg")]
    logging.info(f"Found {len(split_images)} images in {split} split")
    
    # Collect data for this split
    images = []
    texts = []
    image_ids = []
    text_ids = []
    
    for i, image_name in enumerate(split_images):
        try:
            # Load image
            image_path = split_path / image_name
            image = Image.open(image_path).convert('RGB')
            
            # Get captions for this image
            if image_name in image_to_captions:
                captions = image_to_captions[image_name]
            else:
                logging.warning(f"No captions found for {image_name}, skipping...")
                continue
            
            images.append(image)
            texts.extend(captions)
            image_ids.append(f"flickr30k_{split}_{i}")
            
            # Create IDs for each caption
            for j in range(len(captions)):
                text_ids.append(f"flickr30k_{split}_{i}_cap{j}")
                
        except Exception as e:
            logging.warning(f"Error processing {image_name}: {e}")
            continue
    
    if len(images) == 0:
        logging.warning(f"No valid samples found for flickr30k_{split}")
        return
    
    logging.info(f"Loaded {len(images)} images, {len(texts)} texts")
    
    # Encode images
    logging.info(f"Encoding images (batch_size={batch_size})...")
    image_embeddings = embedder.encode_images(images, batch_size=batch_size, normalize=True)
    
    # Encode texts  
    logging.info(f"Encoding texts (batch_size={batch_size})...")
    text_embeddings = embedder.encode_texts(texts, batch_size=batch_size, normalize=True)
    
    logging.info(f"Encoded {len(image_embeddings)} images, {len(text_embeddings)} texts")
    
    # Save image embeddings
    image_output_path = output_dir / f"flickr30k_{split}_image.npz"
    np.savez_compressed(
        image_output_path,
        embeddings=image_embeddings,
        ids=image_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved image embeddings to {image_output_path}")
    
    # Save text embeddings
    text_output_path = output_dir / f"flickr30k_{split}_text.npz"
    np.savez_compressed(
        text_output_path,
        embeddings=text_embeddings,
        ids=text_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved text embeddings to {text_output_path}")

def process_laion_split(split_path, split, embedder, output_dir, batch_size):
    """LAION jpg+txt: index paths and captions only. Embedder loads images per batch (avoids RAM OOM).
    If NUM_SHARDS>1, only encodes this worker's contiguous shard and writes .part{SHARD_ID}.npz files."""
    from concurrent.futures import ThreadPoolExecutor
    
    logging.info(f"Processing LAION split: {split}")
    
    jpg_files = sorted(list(split_path.glob("*.jpg")))
    logging.info(f"Found {len(jpg_files)} images in {split} split")
    
    if len(jpg_files) == 0:
        logging.warning(f"No images found in {split_path}")
        return
    
    def _read_pair(jpg_path):
        txt_path = jpg_path.with_suffix(".txt")
        if not txt_path.exists():
            return None
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        except Exception:
            return None
        return (str(jpg_path.resolve()), caption)
    
    nw = max(1, int(INDEX_WORKERS))
    logging.info(f"Indexing captions with ThreadPoolExecutor (max_workers={nw})...")
    with ThreadPoolExecutor(max_workers=nw) as ex:
        raw = list(ex.map(_read_pair, jpg_files))
    pairs = [x for x in raw if x is not None]
    if len(pairs) < len(raw):
        logging.warning(f"Skipped {len(raw) - len(pairs)} items (missing/unreadable .txt)")
    paths = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]
    ntot = len(paths)
    ns = int(NUM_SHARDS)
    sid = int(SHARD_ID)
    if ns > 1:
        chunk = (ntot + ns - 1) // ns
        start = sid * chunk
        end = min(start + chunk, ntot)
        paths = paths[start:end]
        texts = texts[start:end]
        base = start
        logging.info(f"Shard {sid}/{ns}: global rows {start}:{end} of {ntot} (this worker: {len(paths)})")
    else:
        base = 0
    
    meta = embedder.get_metadata()
    suf = f".part{sid}" if ns > 1 else ""

    if len(paths) == 0:
        dim = int(meta.get("embedding_dim", 768))
        empty_np = np.zeros((0, dim), dtype=np.float32)
        image_output_path = output_dir / f"laion_sample_{split}_image{suf}.npz"
        text_output_path = output_dir / f"laion_sample_{split}_text{suf}.npz"
        np.savez_compressed(
            image_output_path,
            embeddings=empty_np,
            ids=np.array([], dtype=object),
            metadata=meta,
        )
        np.savez_compressed(
            text_output_path,
            embeddings=empty_np,
            ids=np.array([], dtype=object),
            metadata=meta,
        )
        logging.info(f"Saved empty shard npz: {image_output_path.name}")
        return

    image_ids = [f"laion_sample_{split}_{base + i}" for i in range(len(paths))]
    text_ids = [f"laion_sample_{split}_{base + i}_cap0" for i in range(len(paths))]
    logging.info(f"Indexed {len(paths)} valid jpg+txt pairs (global ids {base}..{base + len(paths) - 1})")

    logging.info(f"Encoding {len(paths)} images from paths (batched; no full-split PIL preload)")
    image_embeddings = embedder.encode_images(paths, batch_size=batch_size, normalize=True)

    image_output_path = output_dir / f"laion_sample_{split}_image{suf}.npz"
    np.savez_compressed(
        image_output_path,
        embeddings=image_embeddings,
        ids=image_ids,
        metadata=meta,
    )
    logging.info(f"Saved image embeddings to {image_output_path}")
    del image_embeddings

    tb = int(TEXT_BATCH_SIZE)
    logging.info(f"Encoding {len(texts)} texts (batch_size={tb})...")
    text_embeddings = embedder.encode_texts(texts, batch_size=tb, normalize=True)

    text_output_path = output_dir / f"laion_sample_{split}_text{suf}.npz"
    np.savez_compressed(
        text_output_path,
        embeddings=text_embeddings,
        ids=text_ids,
        metadata=meta,
    )
    logging.info(f"Saved text embeddings to {text_output_path}")
    del text_embeddings

    logging.info(f"Completed split {split}: saved image+text embeddings")
'''
    else:  # audio-text
        process_func = '''
def process_split(dataset, dataset_name, split, embedder, output_dir, batch_size):
    """Process audio-text dataset split - uses small GPU batches for efficiency while avoiding OOM."""
    import tempfile
    import shutil
    import gc
    
    logging.info(f"Processing audio-text split: {dataset_name}_{split}")
    
    total = len(dataset)
    
    # Process one audio at a time - CLAP with 240k samples is very memory-intensive
    # Even loading 5 audios into memory causes OOM
    audio_batch_size = 1  # Must be 1 to avoid OOM
    
    # Use smaller text batch size to avoid OOM
    text_batch_size = min(batch_size, 32)  # Text is less memory-intensive
    
    # Process one at a time, but save in small groups to reduce file I/O overhead
    # We'll process and save immediately, but accumulate file paths
    save_every_n = 10  # Save to disk every 10 processed audios (but process one at a time)
    
    # Use temporary files to save embeddings incrementally
    temp_dir = Path(tempfile.mkdtemp(prefix=f"embeddings_{dataset_name}_{split}_"))
    audio_chunk_files = []
    text_chunk_files = []
    all_audio_ids = []
    all_text_ids = []
    
    # Process in micro-batches: collect texts, encode together
    text_batch = []
    text_batch_ids = []
    
    try:
        # Process one audio at a time to minimize memory usage
        logging.info(f"Processing {total} samples one at a time (batch_size={audio_batch_size})...")
        
        # Accumulate embeddings in small batches before saving to reduce I/O
        audio_embeddings_buffer = []
        audio_ids_buffer = []
        buffer_count = 0
        
        for i in range(total):
            try:
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{total} samples...")
                
                item = dataset[i]
                if 'audio' not in item:
                    continue
                
                # Load audio - ONE at a time
                audio_data = item['audio']
                if isinstance(audio_data, dict):
                    waveform = torch.from_numpy(audio_data['array']).float()
                    sr = audio_data['sampling_rate']
                else:
                    waveform = torch.from_numpy(audio_data).float() if isinstance(audio_data, np.ndarray) else audio_data
                    sr = 48000
                
                audio_id = f"{dataset_name}_{split}_{i}"
                
                # Encode audio immediately (one at a time)
                audio_embedding = embedder.encode_audios(
                    [waveform], 
                    sample_rates=[sr], 
                    batch_size=audio_batch_size, 
                    normalize=True
                )
                
                # Add to buffer
                audio_embeddings_buffer.append(audio_embedding)
                audio_ids_buffer.append(audio_id)
                buffer_count += 1
                
                # Cleanup audio immediately
                del waveform, audio_data, audio_embedding
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Save buffer to disk when it reaches save_every_n
                if buffer_count >= save_every_n:
                    # Concatenate and save
                    chunk_embeddings = np.concatenate(audio_embeddings_buffer, axis=0)
                    audio_chunk_file = temp_dir / f"audio_chunk_{len(audio_chunk_files)}.npz"
                    np.savez_compressed(audio_chunk_file, embeddings=chunk_embeddings)
                    audio_chunk_files.append(audio_chunk_file)
                    all_audio_ids.extend(audio_ids_buffer)
                    
                    # Clear buffer
                    del audio_embeddings_buffer, chunk_embeddings
                    audio_embeddings_buffer = []
                    audio_ids_buffer = []
                    buffer_count = 0
                    gc.collect()
                
                # Collect text for batch encoding
                if 'caption' in item:
                    captions = item['caption'] if isinstance(item['caption'], list) else [item['caption']]
                    text_batch.extend(captions)
                    for j in range(len(captions)):
                        text_batch_ids.append(f"{dataset_name}_{split}_{i}_cap{j}")
                elif 'text' in item:
                    text_data = item['text'] if isinstance(item['text'], list) else [item['text']]
                    text_batch.extend(text_data)
                    for j in range(len(text_data)):
                        text_batch_ids.append(f"{dataset_name}_{split}_{i}_cap{j}")
                
                # Encode text batch when it reaches the batch size
                if len(text_batch) >= text_batch_size:
                    text_embeddings = embedder.encode_texts(
                        text_batch, 
                        batch_size=text_batch_size, 
                        normalize=True
                    )
                    
                    # Save text batch immediately to disk
                    text_chunk_file = temp_dir / f"text_batch_{len(text_chunk_files)}.npz"
                    np.savez_compressed(text_chunk_file, embeddings=text_embeddings)
                    text_chunk_files.append(text_chunk_file)
                    all_text_ids.extend(text_batch_ids)
                    
                    # Cleanup
                    del text_batch, text_embeddings
                    text_batch = []
                    text_batch_ids = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Save remaining audio buffer
        if len(audio_embeddings_buffer) > 0:
            chunk_embeddings = np.concatenate(audio_embeddings_buffer, axis=0)
            audio_chunk_file = temp_dir / f"audio_chunk_{len(audio_chunk_files)}.npz"
            np.savez_compressed(audio_chunk_file, embeddings=chunk_embeddings)
            audio_chunk_files.append(audio_chunk_file)
            all_audio_ids.extend(audio_ids_buffer)
            del audio_embeddings_buffer, chunk_embeddings
            gc.collect()
        
        # Encode remaining text batch
        if len(text_batch) > 0:
            text_embeddings = embedder.encode_texts(
                text_batch, 
                batch_size=text_batch_size, 
                normalize=True
            )
            text_chunk_file = temp_dir / f"text_batch_{len(text_chunk_files)}.npz"
            np.savez_compressed(text_chunk_file, embeddings=text_embeddings)
            text_chunk_files.append(text_chunk_file)
            all_text_ids.extend(text_batch_ids)
            del text_batch, text_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if len(all_audio_ids) == 0:
            logging.warning(f"No valid samples found for {dataset_name}_{split}")
            return
        
        # Load and concatenate chunks from disk (one at a time to minimize memory)
        logging.info(f"Loading and concatenating {len(audio_chunk_files)} audio embeddings from disk...")
        audio_embeddings_list = []
        for chunk_file in audio_chunk_files:
            chunk_data = np.load(chunk_file)
            audio_embeddings_list.append(chunk_data['embeddings'])
            chunk_data.close()
            chunk_file.unlink()  # Clean up immediately
        audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
        del audio_embeddings_list
        gc.collect()
        
        logging.info(f"Loading and concatenating {len(text_chunk_files)} text batches from disk...")
        if len(text_chunk_files) > 0:
            text_embeddings_list = []
            for chunk_file in text_chunk_files:
                chunk_data = np.load(chunk_file)
                text_embeddings_list.append(chunk_data['embeddings'])
                chunk_data.close()
                chunk_file.unlink()  # Clean up immediately
            text_embeddings = np.concatenate(text_embeddings_list, axis=0)
            del text_embeddings_list
        else:
            text_embeddings = np.array([])
        gc.collect()
        
        logging.info(f"Final counts: {len(audio_embeddings)} audio embeddings, {len(text_embeddings)} text embeddings")
        
        # Save final embeddings
        audio_output_path = output_dir / f"{dataset_name}_{split}_audio.npz"
        np.savez_compressed(
            audio_output_path,
            embeddings=audio_embeddings,
            ids=all_audio_ids,
            metadata=embedder.get_metadata()
        )
        logging.info(f"Saved audio embeddings to {audio_output_path}")
        
        # Save text embeddings
        text_output_path = output_dir / f"{dataset_name}_{split}_text.npz"
        np.savez_compressed(
            text_output_path,
            embeddings=text_embeddings,
            ids=all_text_ids,
            metadata=embedder.get_metadata()
        )
        logging.info(f"Saved text embeddings to {text_output_path}")
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
'''
    
    script_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, {repo_root_repr})

{embedder_import}
import logging
import datasets
import torch
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - GPU{gpu_id} - %(levelname)s - %(message)s')

# Initialize embedder for this GPU
device = "cuda:0"  # This will be the only visible GPU
{embedder_init}

# LAION: parallel caption indexing + optional larger text batch (GPU-bound encode_* already use CUDA)
INDEX_WORKERS = {index_workers}
TEXT_BATCH_SIZE = {text_batch_size}
SHARD_ID = {shard_id}
NUM_SHARDS = {num_shards}

{process_func}

# Process tasks
data_root = Path("{data_root}")
output_dir = Path("{output_dir}")
batch_size = {batch_size}
gpu_id = {gpu_id}

for dataset_name, split in {tasks}:
    try:
        split_path = data_root / dataset_name / split
        logging.info(f"GPU {{gpu_id}}: Processing {{dataset_name}} - {{split}}")
        
        if (split_path / 'dataset_info.json').exists():
            # HuggingFace dataset format (Arrow)
            dataset = datasets.load_from_disk(str(split_path))
            logging.info(f"GPU {{gpu_id}}: Loaded HuggingFace dataset: {{len(dataset)}} samples")
            
            process_split(dataset, dataset_name, split, embedder, output_dir, batch_size)
        else:
            # File-based dataset (raw images like Flickr30k)
            logging.info(f"GPU {{gpu_id}}: Processing file-based dataset: {{split_path}}")
            process_file_based_split(dataset_name, split_path, split, embedder, output_dir, batch_size)
            
    except Exception as e:
        logging.error(f"GPU {{gpu_id}}: Error processing {{dataset_name}} - {{split}}: {{e}}")
        import traceback
        traceback.print_exc()
        continue

logging.info(f"GPU {{gpu_id}}: Completed all tasks")
'''.format(
        repo_root_repr=repo_root_repr,
        embedder_import=embedder_import,
        embedder_init=embedder_init,
        process_func=process_func,
        data_root=data_root,
        output_dir=output_dir,
        batch_size=batch_size,
        gpu_id=gpu_id,
        tasks=tasks,
        index_workers=index_workers,
        text_batch_size=text_batch_size,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    
    return script_content


def _count_laion_jpg(data_root: Path, dataset_name: str, split: str) -> int:
    p = data_root / dataset_name / split
    if not p.exists():
        return 0
    return len(list(p.glob("*.jpg")))


def order_laion_splits_train_last(
    tasks: list,
    data_root: Path,
    dataset_name: str = "laion_sample",
) -> list:
    """Order splits by ascending size; train always last."""
    names = [s for d, s in tasks if d == dataset_name]
    if not names:
        return []
    sized = [(n, _count_laion_jpg(data_root, dataset_name, n)) for n in names]
    non_train = [(n, c) for n, c in sized if n != "train"]
    has_train = any(n == "train" for n, _ in sized)
    non_train.sort(key=lambda x: x[1])
    out = [n for n, _ in non_train]
    if has_train:
        out.append("train")
    return out


def merge_laion_shard_npz_files(output_dir: Path, split: str, num_shards: int) -> None:
    """Concatenate laion_sample_{split}_{image|text}.part{k}.npz into final .npz and delete parts."""
    out_dir = Path(output_dir)
    for kind in ("image", "text"):
        emb_parts = []
        id_parts = []
        meta = None
        for k in range(num_shards):
            path = out_dir / f"laion_sample_{split}_{kind}.part{k}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing shard file: {path}")
            d = np.load(path, allow_pickle=True)
            emb_parts.append(np.asarray(d["embeddings"]))
            id_parts.append(np.asarray(d["ids"], dtype=object))
            m = d["metadata"]
            if meta is None:
                meta = m.item() if isinstance(m, np.ndarray) and m.ndim == 0 else m
            d.close()
        embeddings = np.concatenate(emb_parts, axis=0)
        ids = np.concatenate(id_parts, axis=0)
        final = out_dir / f"laion_sample_{split}_{kind}.npz"
        np.savez_compressed(final, embeddings=embeddings, ids=ids, metadata=meta)
        logging.info(f"Merged {num_shards} shards -> {final.name} ({len(embeddings)} rows)")
        for k in range(num_shards):
            p = out_dir / f"laion_sample_{split}_{kind}.part{k}.npz"
            if p.exists():
                p.unlink()


def _wait_and_stream_gpu_processes(processes: list) -> None:
    """Wait for subprocess list of (process, gpu_id, script_path); print logs; remove scripts."""
    logging.info("\nWaiting for all GPU processes to complete...")
    logging.info("Monitoring all GPUs in parallel...\n")
    output_queue = queue.Queue()

    def monitor_process(process, gpu_id):
        try:
            for line in process.stdout:
                output_queue.put((gpu_id, line))
        except Exception as e:
            output_queue.put((gpu_id, f"Error reading output: {e}\n"))
        finally:
            process.wait()
            output_queue.put((gpu_id, None))

    monitor_threads = []
    for process, gpu_id, script_path in processes:
        thread = threading.Thread(target=monitor_process, args=(process, gpu_id), daemon=True)
        thread.start()
        monitor_threads.append((thread, gpu_id, script_path))

    completed = set()
    process_dict = {gpu_id: (process, script_path) for process, gpu_id, script_path in processes}

    while len(completed) < len(processes):
        try:
            gpu_id, line = output_queue.get(timeout=1.0)
            if line is None:
                completed.add(gpu_id)
                if gpu_id in process_dict:
                    process, script_path = process_dict[gpu_id]
                    process.wait()
                    if process.returncode == 0:
                        logging.info(f"\n✓ GPU {gpu_id} completed successfully")
                    else:
                        logging.error(f"\n✗ GPU {gpu_id} failed with return code {process.returncode}")
                    try:
                        os.remove(script_path)
                    except OSError:
                        pass
            else:
                print(f"[GPU {gpu_id}] {line}", end="", flush=True)
        except queue.Empty:
            continue

    for thread, gpu_id, script_path in monitor_threads:
        thread.join(timeout=5.0)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU embedding caching - supports CLIP, SigLIP, and CLAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLIP for COCO/Flickr30k
  python cache_embeddings_multi_gpu.py --model_type clip --datasets coco_captions,flickr30k
  
  # SigLIP for COCO/Flickr30k  
  python cache_embeddings_multi_gpu.py --model_type siglip --model_path google/siglip-base-patch16-256 --output_dir embeddings_siglip
  
  # CLAP for AudioCaps/Clotho
  python cache_embeddings_multi_gpu.py --model_type clap --datasets audiocaps,clotho --data_root ./datasets --output_dir embeddings_clap --batch_size 8
        """)
    
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['clip', 'siglip', 'laion', 'clap', 'blip',
                                'clip_b16', 'openclip_vit_h14', 'eva_clip_l14'],
                       help="Model type: clip, siglip, laion, clap, blip, clip_b16, openclip_vit_h14, eva_clip_l14")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Model path. Defaults: clip=auto, siglip=google/siglip-base-patch16-256, laion=hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K, clap=base_model/clap-htsat-fused")
    parser.add_argument("--datasets", type=str, default=None,
                       help="Comma-separated dataset names. Defaults: image-text=coco_captions,flickr30k, audio-text=audiocaps,clotho")
    parser.add_argument("--splits", type=str, default=None,
                       help="Comma-separated split names. Defaults: image-text=train,train_calib,val,test, audio-text=train,validation,test")
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_experiment",
                       help="Path to datasets directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for embeddings (default: embeddings_{model_type})")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for processing. Defaults: clip/siglip=128, clap=8")
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=None,
        help="LAION/CLIP: text encoder batch size (default: same as --batch_size). Often can be 256+ on V100.",
    )
    parser.add_argument(
        "--index_workers",
        type=int,
        default=None,
        help="LAION: parallel threads to read .txt captions (default: min(32, CPU count)). Speeds up indexing before GPU encode.",
    )
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="Comma-separated GPU IDs. Default: first --num_gpus visible CUDA ids (e.g., 0,1,2,3).")
    parser.add_argument(
        "--laion_schedule",
        type=str,
        choices=["sequential_sharded", "round_robin"],
        default=None,
        help="LAION: sequential_sharded = one split at a time, all GPUs shard that split (smallest splits first, train last); "
        "round_robin = legacy per-GPU task lists. Default: sequential_sharded for --model_type laion, else round_robin.",
    )

    args = parser.parse_args()
    
    setup_logging()
    
    # Set defaults based on model type
    if args.model_path is None:
        if args.model_type == 'clip':
            args.model_path = "auto"  # CLIPEmbedder handles this
        elif args.model_type == 'siglip':
            args.model_path = "google/siglip-base-patch16-256"
        elif args.model_type == 'laion':
            args.model_path = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        elif args.model_type == 'clap':
            args.model_path = "base_model/clap-htsat-fused"
        elif args.model_type == 'blip':
            args.model_path = "base_model/blip-itm-retrieval"
        elif args.model_type == 'clip_b16':
            args.model_path = "base_model/clip_vit_b16"
        elif args.model_type == 'openclip_vit_h14':
            args.model_path = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        elif args.model_type == 'eva_clip_l14':
            args.model_path = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
    
    if args.output_dir is None:
        args.output_dir = f"./embeddings_{args.model_type}"
    
    if args.batch_size is None:
        if args.model_type in ['clip', 'siglip', 'laion', 'blip', 'clip_b16', 'openclip_vit_h14', 'eva_clip_l14']:
            args.batch_size = 128
        else:  # clap
            args.batch_size = 8

    text_batch_size = args.text_batch_size if args.text_batch_size is not None else args.batch_size
    index_workers = args.index_workers if args.index_workers is not None else min(32, (os.cpu_count() or 8))
    
    # Determine modality
    modality = 'image-text' if args.model_type in [
        'clip', 'siglip', 'laion', 'blip', 'clip_b16', 'openclip_vit_h14', 'eva_clip_l14',
    ] else 'audio-text'
    
    # Set dataset defaults based on modality
    if args.datasets is None:
        if modality == 'image-text':
            datasets_list = ['coco_captions', 'flickr30k']
            # Adjust data_root for image-text if default is used
            if args.data_root == "./datasets/dataset_experiment":
                pass  # Keep default
        else:  # audio-text
            datasets_list = ['audiocaps', 'clotho']
            # Adjust data_root for audio-text if default is used
            if args.data_root == "./datasets/dataset_experiment":
                args.data_root = "./datasets"
    else:
        datasets_list = [d.strip() for d in args.datasets.split(',')]
    
    # Set split defaults based on modality
    if args.splits is None:
        if modality == 'image-text':
            splits = ['train', 'train_calib', 'val', 'test']
        else:  # audio-text
            splits = ['train', 'validation', 'test']
    else:
        splits = [s.strip() for s in args.splits.split(',')]
    
    # Parse GPU IDs (or auto-pick from visible CUDA devices).
    if args.gpu_ids is None:
        if torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            if n_visible == 0:
                raise ValueError("torch.cuda.is_available() is True but device_count() is 0.")
            if args.num_gpus > n_visible:
                logging.warning(
                    f"--num_gpus={args.num_gpus} but only {n_visible} visible CUDA device(s); "
                    f"using {n_visible} GPU(s)."
                )
                args.num_gpus = n_visible
            gpu_list = list(range(args.num_gpus))
        else:
            raise ValueError(
                "No CUDA devices visible and --gpu_ids was not provided. "
                "Set CUDA_VISIBLE_DEVICES and/or pass --gpu_ids explicitly."
            )
    else:
        gpu_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
    if len(gpu_list) < args.num_gpus:
        raise ValueError(
            f"--gpu_ids lists {len(gpu_list)} GPU(s) but --num_gpus is {args.num_gpus}. "
            f"Provide at least {args.num_gpus} comma-separated ids (e.g. 0,1,2,3,4,5 for 6 GPUs)."
        )
    if len(gpu_list) > args.num_gpus:
        logging.warning(
            f"Number of GPU IDs ({len(gpu_list)}) exceeds num_gpus ({args.num_gpus}); using first {args.num_gpus}."
        )
        gpu_list = gpu_list[: args.num_gpus]

    # Fail fast if user provided non-existent GPU ids (common cause of pod crashes / silent CUDA failures)
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        bad = [g for g in gpu_list if g < 0 or g >= n_visible]
        if bad:
            raise ValueError(
                f"Invalid --gpu_ids {bad} for this node: torch sees {n_visible} CUDA device(s) "
                f"(valid ids: 0..{n_visible - 1})."
            )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    logging.info("="*60)
    logging.info(f"Multi-GPU Embedding Caching")
    logging.info("="*60)
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Modality: {modality}")
    logging.info(f"Data root: {data_root}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Datasets: {datasets_list}")
    logging.info(f"Splits: {splits}")
    logging.info(f"Batch size: {args.batch_size}")
    if args.model_type == "laion":
        logging.info(f"LAION text_batch_size: {text_batch_size} | index_workers: {index_workers}")
    if args.laion_schedule is None:
        args.laion_schedule = "sequential_sharded" if args.model_type == "laion" else "round_robin"
    logging.info(f"Schedule: {args.laion_schedule}")
    logging.info(f"GPUs: {args.num_gpus} ({gpu_list})")
    logging.info("="*60)
    
    # Create all tasks
    tasks = []
    for dataset_name in datasets_list:
        for split in splits:
            split_path = data_root / dataset_name / split
            if split_path.exists():
                tasks.append((dataset_name, split))
            else:
                logging.warning(f"Split {split} not found for {dataset_name}, skipping...")
    
    if not tasks:
        raise ValueError("No valid dataset splits found! Check your data_root and dataset paths.")
    
    logging.info(f"Found {len(tasks)} splits to process: {tasks}")

    # LAION: one split at a time, all GPUs shard contiguous row ranges (train last)
    if args.model_type == "laion" and args.laion_schedule == "sequential_sharded":
        laion_tasks = [(d, s) for d, s in tasks if d == "laion_sample"]
        if not laion_tasks:
            raise ValueError(
                "sequential_sharded LAION requires laion_sample under data_root. "
                "Use --datasets laion_sample or set --laion_schedule round_robin."
            )
        extra = [(d, s) for d, s in tasks if d != "laion_sample"]
        if extra:
            logging.warning(f"Ignoring non-laion_sample tasks in sequential_sharded mode: {extra}")
        ordered = order_laion_splits_train_last(laion_tasks, data_root)
        if not ordered:
            raise ValueError("No laion_sample splits to process.")
        logging.info(f"LAION sequential order (small→large, train last): {ordered}")
        for si, split in enumerate(ordered):
            logging.info("\n" + "=" * 60)
            logging.info(f"LAION sequential split {split} ({si + 1}/{len(ordered)})")
            logging.info("=" * 60)
            task_one = [("laion_sample", split)]
            processes = []
            for gpu_idx in range(args.num_gpus):
                gpu_id = gpu_list[gpu_idx]
                script_content = create_gpu_script(
                    gpu_id,
                    task_one,
                    str(data_root),
                    str(output_dir),
                    args.batch_size,
                    args.model_type,
                    args.model_path,
                    index_workers,
                    text_batch_size,
                    shard_id=gpu_idx,
                    num_shards=args.num_gpus,
                )
                script_path = f"temp_gpu_{args.model_type}_{gpu_id}_{split}_shard{gpu_idx}.py"
                with open(script_path, "w") as f:
                    f.write(script_content)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                processes.append((process, gpu_id, script_path))
                logging.info(f"Launched GPU {gpu_id} shard {gpu_idx}/{args.num_gpus} (PID: {process.pid})")
            time.sleep(2)
            logging.info("\nVerifying all processes are running...")
            for process, gpu_id, _ in processes:
                if process.poll() is None:
                    logging.info(f"  ✓ GPU {gpu_id} process (PID: {process.pid}) is running")
                else:
                    logging.error(f"  ✗ GPU {gpu_id} process exited early with code {process.returncode}")
            _wait_and_stream_gpu_processes(processes)
            if args.num_gpus > 1:
                merge_laion_shard_npz_files(output_dir, split, args.num_gpus)
        logging.info("\n" + "=" * 60)
        logging.info("All LAION sequential splits completed!")
        logging.info("=" * 60)
        return

    # Distribute tasks across GPUs (round_robin)
    gpu_tasks = [[] for _ in range(args.num_gpus)]
    for i, task in enumerate(tasks):
        gpu_tasks[i % args.num_gpus].append(task)

    logging.info(f"\nDistributed {len(tasks)} tasks across {args.num_gpus} GPUs:")
    for i, gpu_task_list in enumerate(gpu_tasks):
        if gpu_task_list:
            logging.info(f"  GPU {gpu_list[i]}: {len(gpu_task_list)} tasks - {gpu_task_list}")

    # Launch processes for each GPU
    processes = []
    for gpu_idx, gpu_task_list in enumerate(gpu_tasks):
        if gpu_task_list:
            # Create a temporary script for this GPU
            script_content = create_gpu_script(
                gpu_list[gpu_idx],
                gpu_task_list,
                str(data_root),
                str(output_dir),
                args.batch_size,
                args.model_type,
                args.model_path,
                index_workers,
                text_batch_size,
                shard_id=0,
                num_shards=1,
            )
            script_path = f"temp_gpu_{args.model_type}_{gpu_list[gpu_idx]}.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Launch process for this GPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_list[gpu_idx])
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append((process, gpu_list[gpu_idx], script_path))
            
            logging.info(f"Launched GPU {gpu_list[gpu_idx]} process (PID: {process.pid})")
    
    # Verify all processes are still running
    time.sleep(2)  # Give processes time to start
    logging.info("\nVerifying all processes are running...")
    for process, gpu_id, script_path in processes:
        if process.poll() is None:
            logging.info(f"  ✓ GPU {gpu_id} process (PID: {process.pid}) is running")
        else:
            logging.error(f"  ✗ GPU {gpu_id} process (PID: {process.pid}) has already exited with code {process.returncode}")

    _wait_and_stream_gpu_processes(processes)

    logging.info("\n" + "="*60)
    logging.info("All GPU processes completed!")
    logging.info("="*60)

if __name__ == "__main__":
    main()
