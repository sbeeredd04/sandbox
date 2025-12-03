#!/usr/bin/env python3
"""
Organize ImageNet validation set into class folders for PyTorch ImageFolder.

This script reads the LOC_val_solution.csv file and creates a directory structure
where validation images are organized by their synset class ID:

/val/
  n01440764/
    ILSVRC2012_val_00012345.JPEG
    ...
  n01443537/
    ILSVRC2012_val_00023456.JPEG
    ...
  ...

This allows using torchvision.datasets.ImageFolder for easy data loading.
"""

import os
import shutil
import csv
from pathlib import Path
from tqdm import tqdm
import argparse


def organize_val_set(imagenet_root, dry_run=False):
    """
    Organize ImageNet validation images into class folders.
    
    Args:
        imagenet_root: Path to imagenet directory containing ILSVRC/ and LOC_val_solution.csv
        dry_run: If True, only print what would be done without actually moving files
    """
    imagenet_root = Path(imagenet_root)
    
    # Paths
    val_solution_csv = imagenet_root / 'LOC_val_solution.csv'
    val_images_dir = imagenet_root / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val'
    
    # Check if files exist
    if not val_solution_csv.exists():
        raise FileNotFoundError(f"Solution file not found: {val_solution_csv}")
    
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
    
    # Check if already organized (contains subdirectories)
    existing_subdirs = [d for d in val_images_dir.iterdir() if d.is_dir()]
    if existing_subdirs:
        print(f"Validation set already appears to be organized ({len(existing_subdirs)} class folders found).")
        response = input("Re-organize anyway? This will move all images back to flat structure first. (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        
        # Flatten the structure first
        print("Flattening existing structure...")
        for class_dir in tqdm(existing_subdirs, desc="Moving images to flat structure"):
            for img_file in class_dir.glob('*.JPEG'):
                target = val_images_dir / img_file.name
                if not dry_run:
                    shutil.move(str(img_file), str(target))
            if not dry_run:
                class_dir.rmdir()
    
    # Parse the validation solution file
    print(f"Reading validation solution from {val_solution_csv}...")
    image_to_class = {}
    
    with open(val_solution_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 2:
                continue
            
            image_id = row[0]  # e.g., ILSVRC2012_val_00048981
            prediction_string = row[1]  # e.g., "n03995372 85 1 499 272"
            
            # Extract synset ID (first token in prediction string)
            parts = prediction_string.strip().split()
            if parts:
                synset_id = parts[0]  # e.g., n03995372
                image_to_class[image_id] = synset_id
    
    print(f"Found {len(image_to_class)} image-to-class mappings")
    
    # Create class directories and organize images
    class_dirs = set(image_to_class.values())
    print(f"Creating {len(class_dirs)} class directories...")
    
    if not dry_run:
        for synset_id in class_dirs:
            class_dir = val_images_dir / synset_id
            class_dir.mkdir(exist_ok=True)
    
    # Move images to their respective class folders
    print("Organizing images by class...")
    moved_count = 0
    missing_count = 0
    
    for image_id, synset_id in tqdm(image_to_class.items(), desc="Moving images"):
        # Image filename
        image_filename = f"{image_id}.JPEG"
        src_path = val_images_dir / image_filename
        dst_path = val_images_dir / synset_id / image_filename
        
        if src_path.exists():
            if dry_run:
                print(f"Would move: {src_path} -> {dst_path}")
            else:
                shutil.move(str(src_path), str(dst_path))
            moved_count += 1
        else:
            missing_count += 1
            if dry_run or missing_count <= 10:  # Only show first 10 missing
                print(f"Warning: Image not found: {src_path}")
    
    print(f"\nOrganization complete!")
    print(f"  Moved: {moved_count} images")
    print(f"  Missing: {missing_count} images")
    print(f"  Classes: {len(class_dirs)} folders")
    print(f"\nValidation set organized at: {val_images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Organize ImageNet validation set into class folders'
    )
    parser.add_argument(
        '--imagenet_root',
        type=str,
        default='/scratch/sbeeredd/imagenet',
        help='Path to imagenet directory (default: /scratch/sbeeredd/imagenet)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    args = parser.parse_args()
    
    organize_val_set(args.imagenet_root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
