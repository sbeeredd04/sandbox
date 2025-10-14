"""
UCF Sports Action Dataset utilities with OWL-ViT + SAM preprocessing pipeline.

This module provides:
- UCF Sports dataset loading with Deep Lake
- One-time preprocessing with OWL-ViT object detection + SAM segmentation
- Automatic filtering of images without detections
- Preprocessed dataset saved to disk for fast training
- Efficient DataLoader that reads preprocessed images

Preprocessing Pipeline:
1. Load image from Deep Lake
2. Run OWL-ViT object detection with class-specific text queries
3. Filter out images with no detections (skip to next image)
4. Run SAM segmentation on detected bounding box
5. Apply mask to remove background
6. Apply transforms and save to disk with label

Training Pipeline:
1. Check if preprocessed data exists
2. If not, run preprocessing pipeline
3. Load preprocessed images from disk in __getitem__
"""

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import random
import os
import pickle
from tqdm import tqdm
import json
import pandas as pd
import csv

# Import OWL-ViT and SAM
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import SamPredictor, sam_model_registry


# ============================================================================
# Class Mapping Helper Functions (Module-level for pickle compatibility)
# ============================================================================

def get_ucf_class_mappings(deeplake_ds, group_similar=True):
    """
    Get class mappings for UCF Sports dataset.
    
    Args:
        deeplake_ds: Deep Lake dataset
        group_similar: Whether to group similar action classes
        
    Returns:
        Tuple of class mappings (varies based on group_similar flag)
    """
    original_class_names = deeplake_ds.labels.info['class_names']
    
    if group_similar:
        # Define grouping rules - map original class names to grouped class names
        grouping_rules = {
            'Golf-Swing-Front': 'Golf-Swing',
            'Golf-Swing-Side': 'Golf-Swing', 
            'Golf-Swing-Back': 'Golf-Swing',
            'Kicking-Front': 'Kicking',
            'Kicking-Side': 'Kicking',
            'Swing-SideAngle': 'Swing-SideAngle',
            'Swing-Bench': 'Swing-Bench',
            'Walk-Front': 'Walk',
            'Run-Side': 'Run',
            'SkateBoarding-Front': 'SkateBoarding',
            'Diving-Side': 'Diving',
            'Lifting': 'Lifting',
            'Riding-Horse': 'Riding-Horse'
        }
        
        # Create mappings
        original_label_to_id = {name: idx for idx, name in enumerate(original_class_names)}
        original_id_to_label = {idx: name for idx, name in enumerate(original_class_names)}
        
        # Create grouped class names (unique only)
        grouped_class_names = list(set(grouping_rules.values()))
        grouped_class_names.sort()  # Sort for consistency
        
        # Create grouped class mappings
        grouped_label_to_id = {name: idx for idx, name in enumerate(grouped_class_names)}
        grouped_id_to_label = {idx: name for idx, name in enumerate(grouped_class_names)}
        
        # Create mapping from original ID to grouped ID
        original_to_grouped_id = {}
        for orig_id, orig_name in original_id_to_label.items():
            grouped_name = grouping_rules[orig_name]
            grouped_id = grouped_label_to_id[grouped_name]
            original_to_grouped_id[orig_id] = grouped_id
        
        # Create reverse mapping from grouped ID to list of original IDs
        grouped_to_original_ids = {}
        for grouped_id, grouped_name in grouped_id_to_label.items():
            original_ids = []
            for orig_id, orig_name in original_id_to_label.items():
                if grouping_rules[orig_name] == grouped_name:
                    original_ids.append(orig_id)
            grouped_to_original_ids[grouped_id] = original_ids
        
        print(f"✓ Original classes: {len(original_class_names)} -> Grouped classes: {len(grouped_class_names)}")
        print(f"✓ Grouped class names: {grouped_class_names}")
        
        return (grouped_class_names, original_to_grouped_id, grouped_to_original_ids, 
                grouped_id_to_label, grouping_rules)
    
    else:
        # Original behavior without grouping
        label_name_to_id = {name: idx for idx, name in enumerate(original_class_names)}
        label_id_to_name = {idx: name for idx, name in enumerate(original_class_names)}
        
        return original_class_names, label_name_to_id, label_id_to_name


def get_ucf_sports_transforms(image_size=256):
    """
    Get transforms for UCF Sports dataset (applied AFTER preprocessing).
    
    Args:
        image_size: Target image size (default 224 for most vision models)
        
    Returns:
        torchvision transforms composition
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform


# ============================================================================
# OWL-ViT Text Queries
# ============================================================================

def get_owlvit_text_queries():
    """
    Get text queries for OWL-ViT detection for each action class.
    
    Returns:
        Dictionary mapping class names to list of text queries
    """
    text_queries_mapping = {
        'Diving': ['human', 'person', 'people', 'body', 'human body'],
        'Golf-Swing': ['human', 'person', 'people', 'body', 'human body'],
        'Kicking': ['human', 'person', 'people', 'body', 'human body'],
        'Lifting': ['human', 'person', 'people', 'body', 'human body'],
        'Riding-Horse': ['human', 'person', 'people', 'body', 'human body'],
        'Run': ['human', 'person', 'people', 'body', 'human body'],
        'SkateBoarding': ['human', 'person', 'people', 'body', 'human body'],
        'Swing-Bench': ['human', 'person', 'people', 'body', 'human body'],
        'Swing-SideAngle': ['human', 'person', 'people', 'body', 'human body'],
        'Walk': ['human', 'person', 'people', 'body', 'human body']
    }
    return text_queries_mapping


# ============================================================================
# Model Initialization Functions
# ============================================================================

def initialize_owlvit_model(device_id=8):
    """
    Initialize OWL-ViT model on specific GPU device.
    
    Args:
        device_id: GPU device ID to use (default 8)
        
    Returns:
        Tuple of (model, processor, device)
    """
    print(f"\n{'='*80}")
    print(f"Initializing OWL-ViT model on GPU {device_id}...")
    print(f"{'='*80}")
    
    # Load model and processor
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    # Move to specific GPU device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"✓ OWL-ViT model loaded on {device}")
    return model, processor, device


def initialize_sam_model(device_id=1, model_type="vit_h"):
    """
    Initialize SAM model on specific GPU device.
    
    Args:
        device_id: GPU device ID to use (default 9)
        model_type: SAM model type ('vit_b', 'vit_l', or 'vit_h')
        
    Returns:
        SAM predictor object
    """
    print(f"\n{'='*80}")
    print(f"Initializing SAM model ({model_type}) on GPU {device_id}...")
    print(f"{'='*80}")
    
    # Get checkpoint path
    checkpoint_paths = {
        "vit_h": "checkpoints/SAM/sam_vit_h_4b8939.pth",
        "vit_l": "checkpoints/SAM/sam_vit_l_0b3195.pth",
        "vit_b": "checkpoints/SAM/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_paths[model_type])
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
    
    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    
    # Move to specific GPU device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    sam = sam.to(device)
    
    predictor = SamPredictor(sam)
    print(f"✓ SAM model loaded on {device}")
    
    return predictor


# ============================================================================
# Preprocessing Functions
# ============================================================================

def detect_and_segment_image(image_np, class_name, owlvit_model, owlvit_processor, owlvit_device, sam_predictor, text_queries_mapping, confidence_threshold=0.1):

    # Convert to PIL for OWL-ViT
    pil_image = Image.fromarray(image_np, mode='RGB')
    
    # Get text queries for this class
    text_queries = text_queries_mapping.get(class_name, ['human', 'person', 'people'])
    
    # Run OWL-ViT detection
    inputs = owlvit_processor(text=text_queries, images=pil_image, return_tensors="pt").to(owlvit_device)
    
    with torch.no_grad():
        outputs = owlvit_model(**inputs)
    
    # Post-process to get boxes
    target_sizes = torch.Tensor([pil_image.size[::-1]]).to(owlvit_device)  # (height, width)
    results = owlvit_processor.post_process(outputs=outputs, target_sizes=target_sizes)
    
    # Extract results
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    
    # Filter by confidence threshold
    valid_indices = scores >= confidence_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    
    # Check if any detections found
    if len(boxes) == 0:
        return None, False
    
    # Run SAM segmentation on first detected box
    sam_predictor.set_image(image_np)
    input_box = np.array(boxes[0])  # Use highest confidence box
    
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
    )
    
    # Apply mask to remove background
    if len(masks) > 0:
        mask = masks[0]  # Take first mask
        
        # Create masked image (background = black)
        masked_image_np = image_np.copy().astype(np.float32)
        for c in range(3):
            masked_image_np[:, :, c] = masked_image_np[:, :, c] * mask
        
        masked_image_np = masked_image_np.astype(np.uint8)
        return masked_image_np, True
    
    return None, False


def preprocess_and_save_dataset(deeplake_ds, save_dir, csv_save_path, class_mappings, use_grouped_classes=True, owlvit_device_id=0, sam_device_id=1, image_size=256):

    print(f"\n{'='*80}")
    print(f"PREPROCESSING UCF SPORTS DATASET")
    print(f"{'='*80}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving preprocessed images to {save_dir}")
    
    # Initialize models on specific GPUs
    owlvit_model, owlvit_processor, owlvit_device = initialize_owlvit_model(owlvit_device_id)
    sam_predictor = initialize_sam_model(sam_device_id)
    
    # Get text queries
    text_queries_mapping = get_owlvit_text_queries()
    
    # Unpack class mappings
    if use_grouped_classes:
        class_names, original_to_grouped_id, grouped_to_original_ids, grouped_id_to_label, _ = class_mappings
        label_id_to_name = grouped_id_to_label
    else:
        class_names, _, label_id_to_name = class_mappings
    
    # Process each image
    total_samples = len(deeplake_ds)
    processed_indices = []
    dropped_count = 0
    
    print(f"\n{'='*80}")
    print(f"Processing {total_samples} images...")
    print(f"{'='*80}\n")
    
    # Create data directory if it doesn't exist and open CSV file for writing
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    with open(csv_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_idx', 'save_path', 'label', 'original_label'])
        
        for idx in tqdm(range(total_samples), desc="Preprocessing images"):
            sample = deeplake_ds[idx]
            image = sample.images.numpy()
            original_label = int(sample.labels.numpy()[0])
                        
            # Convert to grouped label if needed
            if use_grouped_classes:
                label = original_to_grouped_id.get(original_label, -1)
                if label == -1:
                    print(f" Warning: Could not map original label {original_label}")
                    continue
            else:
                label = original_label
            
            # Get class name
            class_name = label_id_to_name.get(label, "Unknown")
            
            # Run detection and segmentation
            masked_image_np, success = detect_and_segment_image(image, class_name, owlvit_model, owlvit_processor, owlvit_device, sam_predictor, text_queries_mapping)
            
            if not success or masked_image_np is None:
                dropped_count += 1
                continue
            
            # Save preprocessed image
            save_path = os.path.join(save_dir, f"img_{idx:05d}_label_{label}.png")
            Image.fromarray(masked_image_np).save(save_path)
            
            # Record this index as successfully processed
            processed_indices.append({
                'original_idx': idx,
                'save_path': save_path,
                'label': label,
                'original_label': original_label,
                'class_name': class_name
            })
            
            # Write to csv (now within the same with block)
            writer.writerow([idx, save_path, label, original_label, class_name])

    
    # Save metadata
    metadata = {
        'processed_indices': processed_indices,
        'dropped_count': dropped_count,
        'total_original': total_samples,
        'label_id_to_name': label_id_to_name,
        'class_names': class_names,
        'use_grouped_classes': use_grouped_classes
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save human-readable stats
    stats_path = os.path.join(save_dir, 'preprocessing_stats.json')
    stats = {
        'total_original_images': total_samples,
        'successfully_processed': len(processed_indices),
        'dropped_images': dropped_count,
        'drop_rate_percent': (dropped_count / total_samples) * 100,
        'label_id_to_name': label_id_to_name,
        'class_names': class_names
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Total original images: {total_samples}")
    print(f"✓ Successfully processed: {len(processed_indices)}")
    print(f"✗ Dropped (no detection): {dropped_count}")
    print(f"✗ Drop rate: {(dropped_count / total_samples) * 100:.1f}%")
    print(f"✓ Saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    return processed_indices, dropped_count, label_id_to_name

# ============================================================================
# Dataset Class
# ============================================================================


class UCFSportsDataset(data.Dataset):
    """
    UCF Sports Action Dataset that uses preprocessed images.
    
    Images are preprocessed once with OWL-ViT + SAM and saved to disk.
    __getitem__ simply loads preprocessed images from disk.
    
    Features:
    - Support for category exclusion via exclude_categories parameter
    - Grouped or original class labels
    - Stratified train/test split
    """
    
    def __init__(self, deeplake_ds, split='train', transform=None, use_grouped_classes=True, data_dir='./data/ucf_preprocessed', force_preprocess=False, owlvit_device_id=0, sam_device_id=1, exclude_categories=None, use_csv=True):
        """
        Initialize UCF Sports dataset with preprocessing.
        
        Args:
            deeplake_ds: Deep Lake dataset
            split: 'train' or 'test'
            transform: Transforms to apply when loading images
            use_grouped_classes: Whether to group similar action classes
            data_dir: Directory to save/load preprocessed data
            force_preprocess: Force reprocessing even if preprocessed data exists
            owlvit_device_id: GPU device for OWL-ViT during preprocessing
            sam_device_id: GPU device for SAM during preprocessing
            exclude_categories: List of category names to exclude from the dataset
        """
        self.deeplake_ds = deeplake_ds
        self.split = split
        self.transform = transform or get_ucf_sports_transforms()
        self.use_grouped_classes = use_grouped_classes
        self.data_dir = data_dir
        self.exclude_categories = exclude_categories or []
        self.use_csv = use_csv
        self.csv_save_path = './data/dataset.csv'
        
        print(f"\n{'='*80}")
        print(f"Initializing UCF Sports Dataset - {split.upper()} split")
        print(f"{'='*80}")
        
        if self.exclude_categories:
            print(f"Excluding categories: {self.exclude_categories}")
        
        # Get class mappings
        self.class_mappings = get_ucf_class_mappings(deeplake_ds, group_similar=use_grouped_classes)
        
        if use_grouped_classes:
            self.class_names, self.original_to_grouped_id, _, self.label_id_to_name, _ = self.class_mappings
        else:
            self.class_names, _, self.label_id_to_name = self.class_mappings
        
        # Create mapping from class name to ID for exclusion
        self.class_name_to_id = {name: idx for idx, name in self.label_id_to_name.items()}
        
        # Get IDs of categories to exclude
        self.exclude_category_ids = []
        for category in self.exclude_categories:
            if category in self.class_name_to_id:
                self.exclude_category_ids.append(self.class_name_to_id[category])
            else:
                print(f"Warning: Category '{category}' not found in dataset, ignoring.")
        
        print(f"✓ Number of classes: {len(self.class_names)}")
        print(f"✓ Class names: {self.class_names}")
        if self.exclude_category_ids:
            print(f"✓ Excluding class IDs: {self.exclude_category_ids}")
        
        # Check if preprocessing needed
        metadata_path = os.path.join(data_dir, 'metadata.pkl')
        csv_path = os.path.join(data_dir, 'dataset.csv')

        # Check if preprocessing is needed
        need_preprocessing = force_preprocess or not os.path.exists(metadata_path)

        if need_preprocessing:
            print(f"\nPreprocessed data not found or force_preprocess=True")
            print(f"Starting preprocessing pipeline...")
            
            # Run preprocessing
            processed_indices, dropped_count, label_id_to_name = preprocess_and_save_dataset( deeplake_ds, data_dir, self.csv_save_path, self.class_mappings, use_grouped_classes, owlvit_device_id, sam_device_id)
            
            self.processed_indices = processed_indices
            self.label_id_to_name = label_id_to_name
            
            # After preprocessing, both PKL and CSV should exist
            print(f"Preprocessing complete. Created both PKL and CSV files.")
        else:
            # No preprocessing needed, load from existing files based on preference
            if self.use_csv and os.path.exists(csv_path):
                # Load from CSV file
                print(f"Loading preprocessed data from CSV: {csv_path}")
                self.processed_indices = []
                
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.processed_indices.append({
                            'original_idx': int(row['original_idx']),
                            'save_path': row['save_path'],
                            'label': int(row['label']),
                            'original_label': int(row['original_label'])
                        })
            else:
                # Load from PKL file
                print(f"Loading preprocessed data from PKL: {metadata_path}")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.processed_indices = metadata['processed_indices']
                self.label_id_to_name = metadata['label_id_to_name']
            
            print(f"Loaded {len(self.processed_indices)} preprocessed images")
        
        # Filter out excluded categories
        if self.exclude_category_ids:
            original_count = len(self.processed_indices)
            self.processed_indices = [
                item for item in self.processed_indices 
                if item['label'] not in self.exclude_category_ids
            ]
            filtered_count = original_count - len(self.processed_indices)
            print(f"Filtered out {filtered_count} images from excluded categories")
        
        # Create stratified train/test split (90/10)
        self._create_train_test_split()
        
        print(f"\n{'='*80}")
        print(f"Dataset initialized successfully")
        print(f"{'='*80}\n")
    
    def _create_train_test_split(self):
        """Create stratified train/test split based on labels."""
        # Group indices by label
        label_to_indices = defaultdict(list)
        for idx, item in enumerate(self.processed_indices):
            label = item['label']
            label_to_indices[label].append(idx)
        
        # Split each class 90/10
        train_indices = []
        test_indices = []
        
        print(f"\n{'='*80}")
        print(f"Creating stratified train/test split (90/10)")
        print(f"{'='*80}")
        
        for label, indices in sorted(label_to_indices.items()):
            n_samples = len(indices)
            n_train = int(0.9 * n_samples)
            
            # Shuffle indices for this class
            indices_shuffled = indices.copy()
            random.shuffle(indices_shuffled)
            
            train_indices.extend(indices_shuffled[:n_train])
            test_indices.extend(indices_shuffled[n_train:])
            
            label_name = self.label_id_to_name.get(label, "Unknown")
            print(f"  {label_name:20s} (ID: {label:2d}): {n_train:3d} train, {n_samples - n_train:3d} test  (total: {n_samples:3d})")
        
        # Shuffle again
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        # Assign to this dataset
        if self.split == 'train':
            self.indices = train_indices
        else:
            self.indices = test_indices
        
        print(f"\n{self.split.capitalize()} split: {len(self.indices)} samples")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution for current split."""
        distribution = defaultdict(int)
        for idx in self.indices:
            label = self.processed_indices[idx]['label']
            distribution[label] += 1
        
        print(f"\nClass distribution ({self.split} split):")
        for label, count in sorted(distribution.items()):
            label_name = self.label_id_to_name.get(label, "Unknown")
            percentage = (count / len(self.indices)) * 100
            print(f"  {label_name:20s} (ID: {label:2d}): {count:3d} samples ({percentage:5.1f}%)")
    
    def __len__(self):
        """Return number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Load preprocessed image and label from disk.
        
        Args:
            idx: Index in the current split
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get actual index in processed_indices
        actual_idx = self.indices[idx]
        item = self.processed_indices[actual_idx]
        
        # Load image from disk
        image_path = item['save_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = item['label']
        
        return image, label
    
    def get_class_name(self, label_id):
        """Convert label ID to class name."""
        return self.label_id_to_name.get(label_id, "Unknown")
    
    def get_num_classes(self):
        """Get number of classes."""
        return len(self.class_names)
    
    def get_all_class_names(self):
        """Get all class names."""
        return self.class_names.copy()


# ============================================================================
# DataLoader Functions
# ============================================================================

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, **kwargs):
    """
    Create a DataLoader for preprocessed dataset.
    
    Args:
        dataset: UCFSportsDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )


def create_ucf_sports_datasets(deeplake_ds, data_dir='./data/ucf_preprocessed', 
                              transform=None, use_grouped_classes=True,
                              force_preprocess=False, owlvit_device_id=0, 
                              sam_device_id=1, exclude_categories=None, use_csv=True):
    """
    Create train and test UCFSportsDataset instances with optional category exclusion.
    
    Args:
        deeplake_ds: Deep Lake dataset
        data_dir: Directory to save/load preprocessed data
        transform: Transforms to apply when loading images
        use_grouped_classes: Whether to group similar action classes
        force_preprocess: Force reprocessing even if preprocessed data exists
        owlvit_device_id: GPU device for OWL-ViT during preprocessing
        sam_device_id: GPU device for SAM during preprocessing
        exclude_categories: List of category names to exclude from the dataset
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Create transform if not provided
    if transform is None:
        transform = get_ucf_sports_transforms()
    
    # Create train dataset
    train_dataset = UCFSportsDataset(
        deeplake_ds, 
        split='train', 
        transform=transform, 
        use_grouped_classes=use_grouped_classes,
        data_dir=data_dir,
        use_csv=use_csv,
        force_preprocess=force_preprocess,
        owlvit_device_id=owlvit_device_id,
        sam_device_id=sam_device_id,
        exclude_categories=exclude_categories
    )
    
    # Create test dataset (using same preprocessing)
    test_dataset = UCFSportsDataset(
        deeplake_ds, 
        split='test', 
        transform=transform, 
        use_grouped_classes=use_grouped_classes,
        data_dir=data_dir,
        use_csv=use_csv,
        force_preprocess=False,  # Don't reprocess for test set
        owlvit_device_id=owlvit_device_id,
        sam_device_id=sam_device_id,
        exclude_categories=exclude_categories
    )
    
    return train_dataset, test_dataset

## Do not transform while preprocssing the dataset... 