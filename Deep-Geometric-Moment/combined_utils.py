#simple dataloader module for combined dataset

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import random
import os
import csv 
import pandas as pd

def get_transforms(image_size=256):
    """Get transforms for combined dataset"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return transform


class CombinedDataset(data.Dataset):
    """
    Combined Dataset (UCF Sports + Olympic Action) that loads preprocessed images.
    
    CSV format:
    save_path,class_name
    ./data/ucf_preprocessed/img_00000_label_6.png,SkateBoarding
    """
    
    def __init__(self, csv_path, split='train', transform=None, train_ratio=0.9, random_seed=42):
        """
        Initialize Combined Dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file with save_path and class_name columns
            split: 'train' or 'test'
            transform: Transforms to apply when loading images
            train_ratio: Ratio of data to use for training (default 0.8)
            random_seed: Random seed for reproducible splits
        """
        self.csv_path = csv_path
        self.split = split
        self.transform = transform or get_transforms()
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        print(f"\n{'='*80}")
        print(f"Initializing Combined Dataset - {split.upper()} split")
        print(f"{'='*80}")
        print(f"CSV path: {csv_path}")
        
        # Load data from CSV
        self._load_from_csv()
        
        # Create class name to label ID mapping
        self._create_class_mapping()
        
        # Create train/test split
        self._create_train_test_split()
        
        print(f"✓ Combined dataset initialized successfully")
        print(f"✓ Total classes: {len(self.class_names)}")
        print(f"✓ {split.capitalize()} samples: {len(self.indices)}")
    
    def _load_from_csv(self):
        """Load data from CSV file."""
        self.data = []
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'save_path': row['save_path'],
                    'class_name': row['class_name']
                })
        
        print(f"Loaded {len(self.data)} samples from CSV")
    
    def _create_class_mapping(self):
        """Create mapping between class names and label IDs."""
        # Get unique class names and sort them for consistency
        unique_classes = sorted(set(item['class_name'] for item in self.data))
        
        # Create bidirectional mappings
        self.class_name_to_id = {name: idx for idx, name in enumerate(unique_classes)}
        self.label_id_to_name = {idx: name for idx, name in enumerate(unique_classes)}
        self.class_names = unique_classes
        
        print(f"\nClass mapping created:")
        for name, idx in sorted(self.class_name_to_id.items(), key=lambda x: x[1]):
            print(f"  {name:30s} -> ID: {idx:2d}")
    
    def _create_train_test_split(self):
        """Create stratified train/test split by class."""
        random.seed(self.random_seed)
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, item in enumerate(self.data):
            class_name = item['class_name']
            class_indices[class_name].append(idx)
        
        # Shuffle indices within each class
        for class_name in class_indices:
            random.shuffle(class_indices[class_name])
        
        # Split each class
        train_indices = []
        test_indices = []
        
        for class_name, indices in class_indices.items():
            split_point = int(len(indices) * self.train_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])
        
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
            class_name = self.data[idx]['class_name']
            label_id = self.class_name_to_id[class_name]
            distribution[label_id] += 1
        
        print(f"\nClass distribution ({self.split} split):")
        for label_id, count in sorted(distribution.items()):
            label_name = self.label_id_to_name[label_id]
            percentage = (count / len(self.indices)) * 100
            print(f"  {label_name:30s} (ID: {label_id:2d}): {count:4d} samples ({percentage:5.1f}%)")
    
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
        # Get actual index in data
        actual_idx = self.indices[idx]
        item = self.data[actual_idx]
        
        # Load image from disk
        image_path = item['save_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label ID from class name
        label = self.class_name_to_id[item['class_name']]
        
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


def create_combined_datasets(csv_path, transform=None, train_ratio=0.8, random_seed=42):
    """
    Create train and test datasets from combined CSV.
    
    Args:
        csv_path: Path to combined CSV file
        transform: Transforms to apply (default: get_transforms())
        train_ratio: Ratio of data for training (default: 0.8)
        random_seed: Random seed for reproducible splits (default: 42)
    
    Returns:
        trainset, testset
    """
    transform = transform or get_transforms()
    
    trainset = CombinedDataset(
        csv_path=csv_path,
        split='train',
        transform=transform,
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    testset = CombinedDataset(
        csv_path=csv_path,
        split='test',
        transform=transform,
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    return trainset, testset


def create_combined_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, **kwargs):
    """
    Create a DataLoader for combined dataset.
    
    Args:
        dataset: CombinedDataset instance
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
    