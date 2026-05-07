#!/usr/bin/env python3
"""
Standalone Embedding Caching Script for Control-TTR.

Supports multiple model types and datasets:
- CLIP, SigLIP, LAION, BLIP for image-text
- CLAP for audio-text

Examples:
  # Single GPU - CLIP for COCO/Flickr30k
  python cache_embeddings.py --model_type clip --datasets coco_captions,flickr30k
  
  # Single GPU - SigLIP
  python cache_embeddings.py --model_type siglip --datasets flickr30k
  
  # Single GPU - CLAP for audio
  python cache_embeddings.py --model_type clap --datasets audiocaps,clotho --data_root ./datasets

  # Multi-GPU - CLIP
  python cache_embeddings.py --model_type clip --multi_gpu --num_gpus 4 --gpu_ids 0,1,2,3
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import os
import gc

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_embedder(model_type: str, model_path: str, device: str):
    """Initialize the appropriate embedder based on model type."""
    
    if model_type == 'clip':
        from embedders import CLIPEmbedder
        return CLIPEmbedder(device=device)
    
    elif model_type == 'siglip':
        from embedders import SiglipEmbedder
        return SiglipEmbedder(model_name=model_path, device=device)
    
    elif model_type == 'laion':
        from embedders import LAIONEmbedder
        return LAIONEmbedder(model_name=model_path, device=device, dtype=torch.float32)
    
    elif model_type == 'blip':
        from embedders import BLIPEmbedder
        return BLIPEmbedder(model_path=model_path, device=device, dtype=torch.float16)
    
    elif model_type == 'clap':
        from embedders import CLAPEmbedder
        return CLAPEmbedder(model_path=model_path, device=device, dtype=torch.float16, max_duration=10.0)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_image_text_split(dataset, dataset_name: str, split: str, 
                             embedder, output_dir: Path, batch_size: int):
    """Process image-text dataset split."""
    from PIL import Image
    import io
    
    logging.info(f"Processing image-text split: {dataset_name}_{split}")
    
    images = []
    texts = []
    image_ids = []
    text_ids = []
    
    total = len(dataset)
    pbar = tqdm(range(total), desc=f"Loading {dataset_name}_{split}", 
                miniters=max(100, total // 100), mininterval=2.0)
    
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
                # COCO format
                image_path = Path("datasets/coco") / item['filepath'] / item['filename']
                image = Image.open(image_path).convert('RGB')
            else:
                logging.warning(f"No image found in sample {i}, skipping...")
                continue
            
            # Handle text loading
            if 'text' in item:
                captions = item['text'] if isinstance(item['text'], list) else [item['text']]
            elif 'caption' in item:
                captions = item['caption'] if isinstance(item['caption'], list) else [item['caption']]
            elif 'sentences' in item:
                captions = item['sentences']
            else:
                logging.warning(f"No text found in sample {i}, skipping...")
                continue
            
            images.append(image)
            texts.extend(captions)
            
            image_id = f"{dataset_name}_{split}_{i}"
            image_ids.append(image_id)
            for j in range(len(captions)):
                text_ids.append(f"{image_id}_cap{j}")
                
        except Exception as e:
            logging.warning(f"Error loading sample {i}: {e}")
            continue
    
    if len(images) == 0:
        logging.warning(f"No valid samples found for {dataset_name}_{split}")
        return
    
    logging.info(f"Loaded {len(images)} images, {len(texts)} texts")
    
    # Encode images
    logging.info(f"Encoding images (batch_size={batch_size})")
    image_embeddings = embedder.encode_images(images, batch_size=batch_size, normalize=True)
    
    # Save image embeddings
    image_output_path = output_dir / f"{dataset_name}_{split}_image.npz"
    np.savez_compressed(
        image_output_path,
        embeddings=image_embeddings,
        ids=image_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved image embeddings to {image_output_path}")
    
    # Encode texts
    logging.info(f"Encoding texts (batch_size={batch_size})")
    text_embeddings = embedder.encode_texts(texts, batch_size=batch_size, normalize=True)
    
    # Save text embeddings
    text_output_path = output_dir / f"{dataset_name}_{split}_text.npz"
    np.savez_compressed(
        text_output_path,
        embeddings=text_embeddings,
        ids=text_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved text embeddings to {text_output_path}")
    
    # Cleanup
    del images, texts, image_embeddings, text_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_flickr30k_split(split_path: Path, split: str, embedder, 
                           output_dir: Path, batch_size: int):
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
    
    for i, image_name in enumerate(tqdm(split_images, desc=f"Loading {split}")):
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
    
    # Save embeddings
    image_output_path = output_dir / f"flickr30k_{split}_image.npz"
    np.savez_compressed(
        image_output_path,
        embeddings=image_embeddings,
        ids=image_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved image embeddings to {image_output_path}")
    
    text_output_path = output_dir / f"flickr30k_{split}_text.npz"
    np.savez_compressed(
        text_output_path,
        embeddings=text_embeddings,
        ids=text_ids,
        metadata=embedder.get_metadata()
    )
    logging.info(f"Saved text embeddings to {text_output_path}")


def process_audio_text_split(dataset, dataset_name: str, split: str,
                            embedder, output_dir: Path, batch_size: int):
    """Process audio-text dataset split (memory-efficient, one audio at a time)."""
    import tempfile
    import shutil
    
    logging.info(f"Processing audio-text split: {dataset_name}_{split}")
    
    total = len(dataset)
    audio_batch_size = 1  # Process one audio at a time to avoid OOM
    text_batch_size = min(batch_size, 32)
    save_every_n = 10
    
    temp_dir = Path(tempfile.mkdtemp(prefix=f"embeddings_{dataset_name}_{split}_"))
    audio_chunk_files = []
    text_chunk_files = []
    all_audio_ids = []
    all_text_ids = []
    
    text_batch = []
    text_batch_ids = []
    audio_embeddings_buffer = []
    audio_ids_buffer = []
    buffer_count = 0
    
    try:
        logging.info(f"Processing {total} samples one at a time...")
        
        for i in range(total):
            try:
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{total} samples...")
                
                item = dataset[i]
                if 'audio' not in item:
                    continue
                
                # Load audio
                audio_data = item['audio']
                if isinstance(audio_data, dict):
                    waveform = torch.from_numpy(audio_data['array']).float()
                    sr = audio_data['sampling_rate']
                else:
                    waveform = torch.from_numpy(audio_data).float() if isinstance(audio_data, np.ndarray) else audio_data
                    sr = 48000
                
                audio_id = f"{dataset_name}_{split}_{i}"
                
                # Encode audio
                audio_embedding = embedder.encode_audios(
                    [waveform], sample_rates=[sr], batch_size=audio_batch_size, normalize=True
                )
                
                audio_embeddings_buffer.append(audio_embedding)
                audio_ids_buffer.append(audio_id)
                buffer_count += 1
                
                del waveform, audio_data, audio_embedding
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Save buffer to disk periodically
                if buffer_count >= save_every_n:
                    chunk_embeddings = np.concatenate(audio_embeddings_buffer, axis=0)
                    audio_chunk_file = temp_dir / f"audio_chunk_{len(audio_chunk_files)}.npz"
                    np.savez_compressed(audio_chunk_file, embeddings=chunk_embeddings)
                    audio_chunk_files.append(audio_chunk_file)
                    all_audio_ids.extend(audio_ids_buffer)
                    
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
                
                # Encode text batch
                if len(text_batch) >= text_batch_size:
                    text_embeddings = embedder.encode_texts(text_batch, batch_size=text_batch_size, normalize=True)
                    
                    text_chunk_file = temp_dir / f"text_batch_{len(text_chunk_files)}.npz"
                    np.savez_compressed(text_chunk_file, embeddings=text_embeddings)
                    text_chunk_files.append(text_chunk_file)
                    all_text_ids.extend(text_batch_ids)
                    
                    del text_batch, text_embeddings
                    text_batch = []
                    text_batch_ids = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                logging.warning(f"Error processing sample {i}: {e}")
                continue
        
        # Save remaining buffers
        if len(audio_embeddings_buffer) > 0:
            chunk_embeddings = np.concatenate(audio_embeddings_buffer, axis=0)
            audio_chunk_file = temp_dir / f"audio_chunk_{len(audio_chunk_files)}.npz"
            np.savez_compressed(audio_chunk_file, embeddings=chunk_embeddings)
            audio_chunk_files.append(audio_chunk_file)
            all_audio_ids.extend(audio_ids_buffer)
        
        if len(text_batch) > 0:
            text_embeddings = embedder.encode_texts(text_batch, batch_size=text_batch_size, normalize=True)
            text_chunk_file = temp_dir / f"text_batch_{len(text_chunk_files)}.npz"
            np.savez_compressed(text_chunk_file, embeddings=text_embeddings)
            text_chunk_files.append(text_chunk_file)
            all_text_ids.extend(text_batch_ids)
        
        if len(all_audio_ids) == 0:
            logging.warning(f"No valid samples found for {dataset_name}_{split}")
            return
        
        # Load and concatenate chunks
        logging.info(f"Loading and concatenating {len(audio_chunk_files)} audio chunks...")
        audio_embeddings_list = []
        for chunk_file in audio_chunk_files:
            chunk_data = np.load(chunk_file)
            audio_embeddings_list.append(chunk_data['embeddings'])
            chunk_data.close()
            chunk_file.unlink()
        audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
        
        logging.info(f"Loading and concatenating {len(text_chunk_files)} text chunks...")
        if len(text_chunk_files) > 0:
            text_embeddings_list = []
            for chunk_file in text_chunk_files:
                chunk_data = np.load(chunk_file)
                text_embeddings_list.append(chunk_data['embeddings'])
                chunk_data.close()
                chunk_file.unlink()
            text_embeddings = np.concatenate(text_embeddings_list, axis=0)
        else:
            text_embeddings = np.array([])
        
        logging.info(f"Final: {len(audio_embeddings)} audio, {len(text_embeddings)} text embeddings")
        
        # Save final embeddings
        audio_output_path = output_dir / f"{dataset_name}_{split}_audio.npz"
        np.savez_compressed(
            audio_output_path,
            embeddings=audio_embeddings,
            ids=all_audio_ids,
            metadata=embedder.get_metadata()
        )
        logging.info(f"Saved audio embeddings to {audio_output_path}")
        
        text_output_path = output_dir / f"{dataset_name}_{split}_text.npz"
        np.savez_compressed(
            text_output_path,
            embeddings=text_embeddings,
            ids=all_text_ids,
            metadata=embedder.get_metadata()
        )
        logging.info(f"Saved text embeddings to {text_output_path}")
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_file_based_split(dataset_name: str, split_path: Path, split: str,
                            embedder, output_dir: Path, batch_size: int):
    """Process file-based dataset (raw images like Flickr30k)."""
    if dataset_name == 'flickr30k':
        process_flickr30k_split(split_path, split, embedder, output_dir, batch_size)
    else:
        logging.warning(f"File-based processing not implemented for {dataset_name}")


def cache_embeddings_single_gpu(
    model_type: str,
    model_path: str,
    datasets_list: list,
    splits: list,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    device: str = "cuda:0"
):
    """Cache embeddings using a single GPU."""
    
    modality = 'image-text' if model_type in ['clip', 'siglip', 'laion', 'blip'] else 'audio-text'
    
    logging.info(f"Initializing {model_type} embedder on {device}...")
    embedder = get_embedder(model_type, model_path, device)
    
    for dataset_name in datasets_list:
        for split in splits:
            split_path = data_root / dataset_name / split
            
            if not split_path.exists():
                logging.warning(f"Split {split} not found for {dataset_name}, skipping...")
                continue
            
            logging.info(f"Processing {dataset_name} - {split}")
            
            try:
                if (split_path / 'dataset_info.json').exists():
                    # HuggingFace dataset format
                    import datasets as hf_datasets
                    dataset = hf_datasets.load_from_disk(str(split_path))
                    logging.info(f"Loaded HuggingFace dataset: {len(dataset)} samples")
                    
                    if modality == 'image-text':
                        process_image_text_split(dataset, dataset_name, split, embedder, output_dir, batch_size)
                    else:
                        process_audio_text_split(dataset, dataset_name, split, embedder, output_dir, batch_size)
                else:
                    # File-based dataset
                    logging.info(f"Processing file-based dataset: {split_path}")
                    process_file_based_split(dataset_name, split_path, split, embedder, output_dir, batch_size)
                    
            except Exception as e:
                logging.error(f"Error processing {dataset_name} - {split}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logging.info("Embedding caching completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone embedding caching for Control-TTR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLIP for COCO/Flickr30k
  python cache_embeddings.py --model_type clip --datasets coco_captions,flickr30k
  
  # SigLIP for Flickr30k  
  python cache_embeddings.py --model_type siglip --datasets flickr30k
  
  # CLAP for AudioCaps/Clotho
  python cache_embeddings.py --model_type clap --datasets audiocaps,clotho --data_root ./datasets
        """)
    
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=['clip', 'siglip', 'laion', 'clap', 'blip'],
                       help="Model type: clip, siglip, laion, clap, or blip")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Model path/name. Defaults based on model_type")
    parser.add_argument("--datasets", type=str, default=None,
                       help="Comma-separated dataset names")
    parser.add_argument("--splits", type=str, default=None,
                       help="Comma-separated split names")
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_experiment",
                       help="Path to datasets directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: embeddings_{model_type})")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default: 128 for image, 8 for audio)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Set defaults based on model type
    modality = 'image-text' if args.model_type in ['clip', 'siglip', 'laion', 'blip'] else 'audio-text'
    
    if args.model_path is None:
        defaults = {
            'clip': "auto",
            'siglip': "google/siglip-base-patch16-256",
            'laion': "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            'clap': "base_model/clap-htsat-fused",
            'blip': "base_model/blip-itm-retrieval",
        }
        args.model_path = defaults[args.model_type]
    
    if args.output_dir is None:
        args.output_dir = f"./embeddings_{args.model_type}"
    
    if args.batch_size is None:
        args.batch_size = 128 if modality == 'image-text' else 8
    
    if args.datasets is None:
        if modality == 'image-text':
            datasets_list = ['coco_captions', 'flickr30k']
        else:
            datasets_list = ['audiocaps', 'clotho']
    else:
        datasets_list = [d.strip() for d in args.datasets.split(',')]
    
    if args.splits is None:
        if modality == 'image-text':
            splits = ['train', 'train_calib', 'val', 'test']
        else:
            splits = ['train', 'validation', 'test']
    else:
        splits = [s.strip() for s in args.splits.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    logging.info("=" * 60)
    logging.info("Control-TTR Embedding Caching")
    logging.info("=" * 60)
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Modality: {modality}")
    logging.info(f"Data root: {data_root}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Datasets: {datasets_list}")
    logging.info(f"Splits: {splits}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Device: {args.device}")
    logging.info("=" * 60)
    
    cache_embeddings_single_gpu(
        model_type=args.model_type,
        model_path=args.model_path,
        datasets_list=datasets_list,
        splits=splits,
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
