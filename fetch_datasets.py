#!/usr/bin/env python3
"""
Dataset Downloader for Control-TTR.

Downloads and organizes datasets needed for reproducing experiments:
- COCO Captions (Karpathy split)
- Flickr30k
- AudioCaps (for audio-text experiments)
- Clotho (for audio-text experiments)

Examples:
  # Download all image-text datasets
  python fetch_datasets.py --dataset all
  
  # Download only COCO
  python fetch_datasets.py --dataset coco --output-dir ./datasets
  
  # Download Flickr30k
  python fetch_datasets.py --dataset flickr30k
  
  # Organize existing Flickr30k from downloaded files
  python fetch_datasets.py --organize-existing ./flickr30k/dataset_flickr30k.json \
      --flickr30k-folder ./flickr30k/Images --target-dir ./datasets/flickr30k
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import json
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_coco_captions(output_dir: Path) -> None:
    """Download COCO2017 captions dataset (Karpathy split)."""
    from datasets import DatasetDict, load_dataset
    
    logger.info("Downloading COCO captions dataset (Karpathy split)")

    try:
        # Load the dataset
        dataset = load_dataset("yerevann/coco-karpathy")
        
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["validation"] 
        test_dataset = dataset["test"]

        # Create dataset dict
        sampled_dataset = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        # Save to disk
        coco_dir = output_dir / "coco_captions"
        coco_dir.mkdir(parents=True, exist_ok=True)
        sampled_dataset.save_to_disk(str(coco_dir))

        logger.info(f"Saved COCO captions to {coco_dir}")
        logger.info("COCO captions dataset info:")
        for split, data in sampled_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download COCO captions: {e}")
        raise


def organize_existing_flickr30k(json_file_path: str, flickr30k_folder: str, 
                                 target_dir: Optional[str] = None) -> None:
    """
    Organize existing Flickr30k dataset from JSON into train/test/val folders.
    
    Args:
        json_file_path: Path to dataset_flickr30k.json (Karpathy format)
        flickr30k_folder: Folder containing the raw images
        target_dir: Where to create train/val/test folders
    """
    import shutil
    
    logger.info("Organizing existing Flickr30k dataset into train/test/val folders")
    
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create directories
        if target_dir is None:
            target_dir = Path(flickr30k_folder)
        else:
            target_dir = Path(target_dir)
        
        train_dir = target_dir / "train"
        val_dir = target_dir / "val"
        test_dir = target_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created directories: {train_dir}, {val_dir}, {test_dir}")
        logger.info(f"Images will be searched in: {flickr30k_folder}")
        
        if "images" not in data:
            raise ValueError("No 'images' key found in JSON file")
        
        images = data["images"]
        logger.info(f"Processing {len(images)} images in Karpathy format")
        
        processed_count = 0
        for img in images:
            split = img.get("split", "train")
            filename = img.get("filename", "")
            
            if not filename:
                continue
            
            # Determine target directory
            if split == "train":
                split_target_dir = train_dir
            elif split == "val":
                split_target_dir = val_dir
            elif split == "test":
                split_target_dir = test_dir
            else:
                split_target_dir = train_dir  # Default
            
            # Find the image file
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '']:
                potential_file = Path(flickr30k_folder) / f"{filename}{ext}" if ext else Path(flickr30k_folder) / filename
                if potential_file.exists():
                    image_file = potential_file
                    break
            
            if image_file and image_file.exists():
                target_path = split_target_dir / image_file.name
                if not target_path.exists():
                    shutil.copy2(image_file, target_path)
                    processed_count += 1
                    
                    if processed_count % 1000 == 0:
                        logger.info(f"Processed {processed_count} images...")
            else:
                logger.warning(f"Image file not found for: {filename}")
        
        # Count files
        train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.jpeg")))
        val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.jpeg")))
        test_count = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.jpeg")))
        
        logger.info(f"Organization completed!")
        logger.info(f"  Train: {train_count} images")
        logger.info(f"  Val: {val_count} images")
        logger.info(f"  Test: {test_count} images")
        
    except Exception as e:
        logger.error(f"Failed to organize Flickr30k data: {e}")
        raise


def download_flickr30k(output_dir: Path, target_dir: Optional[str] = None) -> None:
    """Download Flickr30k dataset from GitHub release."""
    import subprocess
    import os
    
    logger.info("Downloading Flickr30k dataset")

    try:
        flickr_dir = output_dir / "flickr30k"
        flickr_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(flickr_dir)
        
        try:
            # Download the dataset parts
            logger.info("Downloading Flickr30k dataset parts...")
            subprocess.run(["wget", "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"], check=True)
            subprocess.run(["wget", "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"], check=True)
            subprocess.run(["wget", "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"], check=True)
            
            # Combine and extract
            logger.info("Combining parts and extracting...")
            subprocess.run(["cat", "flickr30k_part00", "flickr30k_part01", "flickr30k_part02"], 
                          stdout=open("flickr30k.zip", "wb"), check=True)
            subprocess.run(["unzip", "-q", "flickr30k.zip", "-d", "."], check=True)
            
            # Cleanup
            subprocess.run(["rm", "flickr30k.zip", "flickr30k_part00", "flickr30k_part01", "flickr30k_part02"], check=True)
            
            logger.info("Dataset downloaded and extracted successfully")
            
        finally:
            os.chdir(original_cwd)
        
        # Find and process the JSON file
        json_files = list(flickr_dir.rglob("*.json"))
        main_json = None
        for json_file in json_files:
            if "flickr30k" in json_file.name.lower() or "dataset" in json_file.name.lower():
                main_json = json_file
                break
        
        if main_json:
            logger.info(f"Found main JSON file: {main_json}")
            organize_existing_flickr30k(str(main_json), str(flickr_dir), target_dir=target_dir)
        
        logger.info(f"Flickr30k processing completed in {flickr_dir}")

    except Exception as e:
        logger.error(f"Failed to download Flickr30k: {e}")
        raise


def download_audiocaps(output_dir: Path) -> None:
    """Download AudioCaps dataset."""
    from datasets import load_dataset
    
    logger.info("Downloading AudioCaps dataset")
    
    try:
        dataset = load_dataset("d0rj/audiocaps")
        
        audiocaps_dir = output_dir / "audiocaps"
        audiocaps_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(audiocaps_dir))
        
        logger.info(f"Saved AudioCaps to {audiocaps_dir}")
        logger.info("AudioCaps dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
            
    except Exception as e:
        logger.error(f"Failed to download AudioCaps: {e}")
        raise


def download_clotho(output_dir: Path) -> None:
    """Download Clotho dataset."""
    from datasets import load_dataset
    
    logger.info("Downloading Clotho dataset")
    
    try:
        dataset = load_dataset("d0rj/clotho")
        
        clotho_dir = output_dir / "clotho"
        clotho_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(clotho_dir))
        
        logger.info(f"Saved Clotho to {clotho_dir}")
        logger.info("Clotho dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
            
    except Exception as e:
        logger.error(f"Failed to download Clotho: {e}")
        raise


def convert_to_experiment_format(input_dir: Path, output_dir: Path) -> None:
    """
    Convert downloaded datasets to the experiment format with train/train_calib/val/test splits.
    
    This creates the structure expected by the experiments:
    - train: Training data
    - train_calib: Calibration split (subset of train for hyperparameter tuning)
    - val: Validation data  
    - test: Test data
    """
    from datasets import load_from_disk, Dataset
    
    logger.info("Converting datasets to experiment format...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process COCO
    coco_input = input_dir / "coco_captions"
    if coco_input.exists():
        logger.info("Converting COCO captions...")
        coco_output = output_dir / "coco_captions"
        coco_output.mkdir(exist_ok=True)
        
        dataset = load_from_disk(str(coco_input))
        
        # Save train
        (coco_output / "train").mkdir(exist_ok=True)
        dataset["train"].save_to_disk(str(coco_output / "train"))
        
        # Create train_calib (first 5000 samples from train)
        train_calib = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        (coco_output / "train_calib").mkdir(exist_ok=True)
        train_calib.save_to_disk(str(coco_output / "train_calib"))
        
        # Save val
        (coco_output / "val").mkdir(exist_ok=True)
        dataset["validation"].save_to_disk(str(coco_output / "val"))
        
        # Save test
        (coco_output / "test").mkdir(exist_ok=True)
        dataset["test"].save_to_disk(str(coco_output / "test"))
        
        logger.info(f"  Saved COCO to {coco_output}")
    
    # Process AudioCaps
    audiocaps_input = input_dir / "audiocaps"
    if audiocaps_input.exists():
        logger.info("Converting AudioCaps...")
        audiocaps_output = output_dir / "audiocaps"
        audiocaps_output.mkdir(exist_ok=True)
        
        dataset = load_from_disk(str(audiocaps_input))
        
        for split in dataset.keys():
            split_dir = audiocaps_output / split
            split_dir.mkdir(exist_ok=True)
            dataset[split].save_to_disk(str(split_dir))
        
        logger.info(f"  Saved AudioCaps to {audiocaps_output}")
    
    # Process Clotho  
    clotho_input = input_dir / "clotho"
    if clotho_input.exists():
        logger.info("Converting Clotho...")
        clotho_output = output_dir / "clotho"
        clotho_output.mkdir(exist_ok=True)
        
        dataset = load_from_disk(str(clotho_input))
        
        for split in dataset.keys():
            split_dir = clotho_output / split
            split_dir.mkdir(exist_ok=True)
            dataset[split].save_to_disk(str(split_dir))
        
        logger.info(f"  Saved Clotho to {clotho_output}")
    
    logger.info("Conversion completed!")


def download_all_datasets(output_dir: Path) -> None:
    """Download all available datasets."""
    logger.info(f"Downloading all datasets to {output_dir}")
    
    try:
        download_coco_captions(output_dir)
        download_flickr30k(output_dir)
        logger.info("All image-text datasets downloaded successfully!")
        logger.info("Note: For audio datasets, use --dataset audiocaps or --dataset clotho")
    except Exception as e:
        logger.error(f"Failed to download all datasets: {e}")
        raise


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Download datasets for Control-TTR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all image-text datasets
  python fetch_datasets.py --dataset all --output-dir ./datasets
  
  # Download only COCO
  python fetch_datasets.py --dataset coco
  
  # Download Flickr30k
  python fetch_datasets.py --dataset flickr30k
  
  # Download audio datasets
  python fetch_datasets.py --dataset audiocaps
  python fetch_datasets.py --dataset clotho
  
  # Organize existing Flickr30k from downloaded files
  python fetch_datasets.py --organize-existing ./flickr30k/dataset_flickr30k.json \\
      --flickr30k-folder ./flickr30k/Images --target-dir ./datasets/flickr30k
  
  # Convert to experiment format (after downloading)
  python fetch_datasets.py --convert-to-experiment --input-dir ./datasets --output-dir ./datasets/dataset_experiment
        """)
    
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("./datasets"),
        help="Output directory for datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--dataset", 
        choices=["coco", "flickr30k", "audiocaps", "clotho", "all"],
        default="all",
        help="Dataset to download (default: all)"
    )
    parser.add_argument(
        "--organize-existing", 
        type=str,
        help="Organize existing Flickr30k dataset from JSON file"
    )
    parser.add_argument(
        "--target-dir", 
        type=str,
        default=None,
        help="Target directory for organized train/val/test folders"
    )
    parser.add_argument(
        "--flickr30k-folder", 
        type=str,
        default="./flickr30k_raw",
        help="Folder containing existing Flickr30k images"
    )
    parser.add_argument(
        "--convert-to-experiment",
        action="store_true",
        help="Convert downloaded datasets to experiment format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory for conversion (default: same as output-dir)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle organizing existing dataset
    if args.organize_existing:
        logger.info("Organizing existing Flickr30k dataset...")
        
        if not Path(args.organize_existing).exists():
            logger.error(f"JSON file not found: {args.organize_existing}")
            exit(1)
        
        if not Path(args.flickr30k_folder).exists():
            logger.error(f"Flickr30k folder not found: {args.flickr30k_folder}")
            exit(1)
        
        target_dir = args.target_dir if args.target_dir else args.flickr30k_folder
        organize_existing_flickr30k(args.organize_existing, args.flickr30k_folder, target_dir)
        return

    # Handle conversion to experiment format
    if args.convert_to_experiment:
        input_dir = args.input_dir if args.input_dir else args.output_dir
        convert_to_experiment_format(input_dir, args.output_dir / "dataset_experiment")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir.absolute()}")

    try:
        if args.dataset == "coco":
            download_coco_captions(args.output_dir)
        elif args.dataset == "flickr30k":
            download_flickr30k(args.output_dir, args.target_dir)
        elif args.dataset == "audiocaps":
            download_audiocaps(args.output_dir)
        elif args.dataset == "clotho":
            download_clotho(args.output_dir)
        elif args.dataset == "all":
            download_all_datasets(args.output_dir)
        
        logger.info("Dataset download completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
