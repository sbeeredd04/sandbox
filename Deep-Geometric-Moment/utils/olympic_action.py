"""
Olympic Action Dataset utilities with OWL-ViT + SAM preprocessing pipeline.

This module provides:
- Olympic Action dataset loading with video frame extraction
- One-time preprocessing with OWL-ViT object detection + SAM segmentation
- Automatic filtering of frames without detections
- Preprocessed dataset saved to disk for fast training
- Efficient DataLoader that reads preprocessed frames

Preprocessing Pipeline:
1. Extract frames from .seq video files
2. Run OWL-ViT object detection with class-specific text queries
3. Filter out frames with no detections (skip to next frame)
4. Run SAM segmentation on detected bounding box
5. Apply mask to remove background
6. Save preprocessed frames to disk with label

Training Pipeline:
1. Check if preprocessed data exists
2. If not, run preprocessing pipeline
3. Load preprocessed frames from disk in __getitem__
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import pickle
import csv
import json
from tqdm import tqdm
from collections import defaultdict
import warnings
import sys
from contextlib import contextmanager

# Import OWL-ViT and SAM
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import SamPredictor, sam_model_registry

# Suppress OpenCV warnings for video processing
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

# Context manager to suppress stderr temporarily
@contextmanager
def suppress_stderr():
    """Suppress stderr output temporarily (useful for noisy video processing)"""
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr


# ============================================================================
# Model Initialization Functions
# ============================================================================

def initialize_owlvit_model(device_id=0):
    """
    Initialize OWL-ViT model on specific GPU device.
    
    Args:
        device_id: GPU device ID to use
        
    Returns:
        Tuple of (model, processor, device)
    """
    print(f"\n{'='*80}")
    print(f"Initializing OWL-ViT model on GPU {device_id}...")
    print(f"{'='*80}")
    
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"✓ OWL-ViT model loaded on {device}")
    return model, processor, device


def initialize_sam_model(device_id=1, model_type="vit_h"):
    """
    Initialize SAM model on specific GPU device.
    
    Args:
        device_id: GPU device ID to use
        model_type: SAM model type ('vit_b', 'vit_l', or 'vit_h')
        
    Returns:
        SAM predictor object
    """
    print(f"\n{'='*80}")
    print(f"Initializing SAM model ({model_type}) on GPU {device_id}...")
    print(f"{'='*80}")
    
    checkpoint_paths = {
        "vit_h": "checkpoints/SAM/sam_vit_h_4b8939.pth",
        "vit_l": "checkpoints/SAM/sam_vit_l_0b3195.pth",
        "vit_b": "checkpoints/SAM/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_path = checkpoint_paths[model_type]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    sam = sam.to(device)
    
    predictor = SamPredictor(sam)
    print(f"✓ SAM model loaded on {device}")
    
    return predictor


# ============================================================================
# Text Queries for OWL-ViT
# ============================================================================

def get_owlvit_text_queries():
    """
    Get text queries for OWL-ViT detection for each sport action class.
    
    Returns:
        Dictionary mapping class names to list of text queries
    """
    # Generic queries for human detection in sports
    text_queries_mapping = {
        'default': ['human', 'person', 'people', 'athlete', 'body', 'human body']
    }
    return text_queries_mapping


# ============================================================================
# Preprocessing Functions
# ============================================================================

def extract_frames_from_video(video_path, num_frames=8):
    """
    Extract frames from .seq video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of frames as numpy arrays (RGB)
    """
    frames = []
    
    try:
        with suppress_stderr():
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Get total frame count
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames > 0:
                    # Sample frames uniformly
                    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                    
                    for frame_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame_rgb)
                else:
                    # Fallback: read sequentially
                    frame_count = 0
                    while frame_count < num_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame_rgb)
                            frame_count += 1
                    
                    cap.release()
    except:
        pass
        
    # Handle case with fewer frames than requested
    if len(frames) > 0 and len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.extend(frames[:min(len(frames), num_frames - len(frames))])
    
    return frames[:num_frames]


def detect_and_segment_frame(frame_np, class_name, owlvit_model, owlvit_processor, 
                             owlvit_device, sam_predictor, text_queries_mapping, 
                             confidence_threshold=0.1):
    """
    Detect and segment human in a single frame using OWL-ViT + SAM.
    
    Args:
        frame_np: Frame as numpy array (RGB)
        class_name: Class name for text queries
        owlvit_model: OWL-ViT model
        owlvit_processor: OWL-ViT processor
        owlvit_device: Device for OWL-ViT
        sam_predictor: SAM predictor
        text_queries_mapping: Mapping of class names to text queries
        confidence_threshold: Confidence threshold for detection
        
    Returns:
        Tuple of (masked_frame, success_flag)
    """
    # Convert to PIL for OWL-ViT
    pil_image = Image.fromarray(frame_np, mode='RGB')
    
    # Get text queries
    text_queries = text_queries_mapping.get(class_name, text_queries_mapping['default'])
    
    # Run OWL-ViT detection
    inputs = owlvit_processor(text=text_queries, images=pil_image, return_tensors="pt").to(owlvit_device)
    
    with torch.no_grad():
        outputs = owlvit_model(**inputs)
    
    # Post-process to get boxes
    target_sizes = torch.Tensor([pil_image.size[::-1]]).to(owlvit_device)
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
    sam_predictor.set_image(frame_np)
    input_box = np.array(boxes[0])
    
    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
    )
    
    # Apply mask to remove background
    if len(masks) > 0:
        mask = masks[0]
        
        # Create masked image (background = black)
        masked_frame_np = frame_np.copy().astype(np.float32)
        for c in range(3):
            masked_frame_np[:, :, c] = masked_frame_np[:, :, c] * mask
        
        masked_frame_np = masked_frame_np.astype(np.uint8)
        return masked_frame_np, True
    
    return None, False


def preprocess_and_save_dataset(root_dir, save_dir, owlvit_device_id=0, 
                                sam_device_id=1, frames_per_video=8, image_size=256):
    """
    Preprocess Olympic Action dataset with OWL-ViT + SAM and save to disk.
    
    Args:
        root_dir: Root directory of Olympic Action dataset
        save_dir: Directory to save preprocessed frames
        owlvit_device_id: GPU device for OWL-ViT
        sam_device_id: GPU device for SAM
        frames_per_video: Number of frames to extract per video
        image_size: Target image size
        
    Returns:
        Tuple of (processed_indices, dropped_count, label_id_to_name)
    """
    print(f"\n{'='*80}")
    print(f"PREPROCESSING OLYMPIC ACTION DATASET")
    print(f"{'='*80}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving preprocessed frames to {save_dir}")
    
    # Initialize models
    owlvit_model, owlvit_processor, owlvit_device = initialize_owlvit_model(owlvit_device_id)
    sam_predictor = initialize_sam_model(sam_device_id)
    
    # Get text queries
    text_queries_mapping = get_owlvit_text_queries()
    
    # Get all sport categories (classes)
    classes = sorted([d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    label_id_to_name = {idx: cls for cls, idx in class_to_idx.items()}
    
    print(f"✓ Found {len(classes)} classes: {classes}")
    
    # Collect all video files
    video_files = []
    video_labels = []
    
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        seq_files = [f for f in os.listdir(class_dir) if f.endswith('.seq')]
        
        for seq_file in seq_files:
            video_files.append(os.path.join(class_dir, seq_file))
            video_labels.append(class_to_idx[class_name])
    
    print(f"✓ Found {len(video_files)} total videos")
    
    # Process each video and extract frames
    processed_indices = []
    dropped_count = 0
    total_frames_extracted = 0
    
    print(f"\n{'='*80}")
    print(f"Processing {len(video_files)} videos...")
    print(f"{'='*80}\n")
    
    # Open CSV file for writing
    csv_path = os.path.join(save_dir, 'dataset.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_idx', 'frame_idx', 'save_path', 'label', 'video_name'])
        
        for video_idx, (video_path, label) in enumerate(tqdm(zip(video_files, video_labels), desc="Preprocessing videos", total=len(video_files))):
           
            # Extract frames from video
            frames = extract_frames_from_video(video_path, frames_per_video)
            
            if len(frames) == 0:
                dropped_count += frames_per_video
                continue
            
            # Get class name
            class_name = label_id_to_name[label]
            video_name = os.path.basename(video_path)
            
            # Process each frame with OWL-ViT + SAM
            for frame_idx, frame in enumerate(frames):
                # Run detection and segmentation
                masked_frame, success = detect_and_segment_frame(
                    frame, class_name, owlvit_model, owlvit_processor, 
                    owlvit_device, sam_predictor, text_queries_mapping
                )
                
                if not success or masked_frame is None:
                    dropped_count += 1
                    continue
                
                # Save preprocessed frame
                save_path = os.path.join(save_dir, f"video_{video_idx:05d}_frame_{frame_idx:02d}_label_{label}.png")
                Image.fromarray(masked_frame).save(save_path)
                
                # Record this frame as successfully processed
                processed_indices.append({
                    'video_idx': video_idx,
                    'frame_idx': frame_idx,
                    'save_path': save_path,
                    'label': label,
                    'video_name': video_name,
                    'class_name': class_name
                })
                
                # Write to CSV
                writer.writerow([video_idx, frame_idx, save_path, label, video_name, class_name])
                total_frames_extracted += 1
    
    # Save metadata as pickle
    metadata = {
        'processed_indices': processed_indices,
        'dropped_count': dropped_count,
        'total_videos': len(video_files),
        'total_frames_extracted': total_frames_extracted,
        'label_id_to_name': label_id_to_name,
        'class_names': classes,
        'frames_per_video': frames_per_video
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save human-readable stats
    stats_path = os.path.join(save_dir, 'preprocessing_stats.json')
    stats = {
        'total_videos': len(video_files),
        'total_frames_extracted': total_frames_extracted,
        'dropped_frames': dropped_count,
        'drop_rate_percent': (dropped_count / (total_frames_extracted + dropped_count) * 100) if (total_frames_extracted + dropped_count) > 0 else 0,
        'label_id_to_name': label_id_to_name,
        'class_names': classes
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Total videos: {len(video_files)}")
    print(f"✓ Successfully processed frames: {total_frames_extracted}")
    print(f"✗ Dropped frames (no detection): {dropped_count}")
    if (total_frames_extracted + dropped_count) > 0:
        print(f"✗ Drop rate: {(dropped_count / (total_frames_extracted + dropped_count)) * 100:.1f}%")
    print(f"✓ Saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    return processed_indices, dropped_count, label_id_to_name


# ============================================================================
# Transform Functions
# ============================================================================

def get_olympic_transforms(image_size=256):
    """
    Get transforms for Olympic Action dataset (applied AFTER preprocessing).
    
    Args:
        image_size: Target image size
        
    Returns:
        torchvision transforms composition
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


# ============================================================================
# Preprocessed Dataset Class
# ============================================================================

class OlympicAction(Dataset):
    """
    Olympic Action Dataset that uses preprocessed frames.
    
    Frames are preprocessed once with OWL-ViT + SAM and saved to disk.
    __getitem__ simply loads preprocessed frames from disk.
    """
    
    def __init__(self, root_dir, split='train', transform=None, data_dir='./data/olympic_preprocessed',
                 force_preprocess=False, owlvit_device_id=0, sam_device_id=1, 
                 frames_per_video=8, use_csv=False, csv_path=None):
        """
        Initialize Olympic Action dataset with preprocessing.
        
        Args:
            root_dir: Root directory of original Olympic Action dataset
            split: 'train' or 'test'
            transform: Transforms to apply when loading frames
            data_dir: Directory to save/load preprocessed data
            force_preprocess: Force reprocessing even if preprocessed data exists
            owlvit_device_id: GPU device for OWL-ViT during preprocessing
            sam_device_id: GPU device for SAM during preprocessing
            frames_per_video: Number of frames to extract per video
            use_csv: Whether to use CSV file as data source
            csv_path: Path to CSV file (if use_csv is True)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or get_olympic_transforms()
        self.data_dir = data_dir
        self.frames_per_video = frames_per_video
        self.use_csv = use_csv
        self.csv_path = csv_path
        
        print(f"\n{'='*80}")
        print(f"Initializing Olympic Action Dataset - {split.upper()} split")
        print(f"{'='*80}")
        
        # Check if preprocessing needed
        metadata_path = os.path.join(data_dir, 'metadata.pkl')
        print(f"Preprocess is needed: {force_preprocess or not os.path.exists(metadata_path)}")
        
        
        if not self.use_csv:
            if force_preprocess or not os.path.exists(metadata_path):
                print(f"\nPreprocessed data not found or force_preprocess=True")
                print(f"Starting preprocessing pipeline...")
                
                # Run preprocessing
                processed_indices, dropped_count, label_id_to_name = preprocess_and_save_dataset(
                    root_dir, data_dir, owlvit_device_id, sam_device_id, frames_per_video
                )
                
                self.processed_indices = processed_indices
                self.label_id_to_name = label_id_to_name
            else:
                # Load preprocessed metadata from PKL
                print(f"Loading preprocessed data from {data_dir}")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.processed_indices = metadata['processed_indices']
                self.label_id_to_name = metadata['label_id_to_name']
                
                print(f"Loaded {len(self.processed_indices)} preprocessed frames")
        else:
            
            if force_preprocess or not os.path.exists(self.csv_path):
                print(f"\nPreprocessed data not found or force_preprocess=True")
                print(f"Starting preprocessing pipeline...")
                
                # Run preprocessing
                processed_indices, dropped_count, label_id_to_name = preprocess_and_save_dataset(
                    root_dir, data_dir, owlvit_device_id, sam_device_id, frames_per_video
                )
                
                self.processed_indices = processed_indices
                self.label_id_to_name = label_id_to_name

            # Load from CSV
            if not self.csv_path:
                self.csv_path = os.path.join(data_dir, 'dataset.csv')
            
            print(f"Loading dataset from CSV file: {self.csv_path}")
            self.processed_indices = []
            
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.processed_indices.append({
                        'video_idx': int(row['video_idx']),
                        'frame_idx': int(row['frame_idx']),
                        'save_path': row['save_path'],
                        'label': int(row['label']),
                        'video_name': row['video_name']
                    })
            
            # Load label mapping from metadata
            metadata_path = os.path.join(data_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.label_id_to_name = metadata['label_id_to_name']
            else:
                # Create basic label mapping
                unique_labels = set(item['label'] for item in self.processed_indices)
                self.label_id_to_name = {label: f"class_{label}" for label in unique_labels}
            
            print(f"Loaded {len(self.processed_indices)} preprocessed frames from CSV")
        
        # Get class names
        self.class_names = sorted(set(self.label_id_to_name.values()))
        
        print(f"✓ Number of classes: {len(self.class_names)}")
        print(f"✓ Class names: {self.class_names}")
        
        # Create stratified train/test split (80/20)
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
        
        # Split each class 80/20
        train_indices = []
        test_indices = []
        
        print(f"\n{'='*80}")
        print(f"Creating stratified train/test split (80/20)")
        print(f"{'='*80}")
        
        for label, indices in sorted(label_to_indices.items()):
            n_samples = len(indices)
            n_train = int(0.8 * n_samples)
            
            # Shuffle indices for this class
            indices_shuffled = indices.copy()
            random.shuffle(indices_shuffled)
            
            train_indices.extend(indices_shuffled[:n_train])
            test_indices.extend(indices_shuffled[n_train:])
            
            label_name = self.label_id_to_name.get(label, "Unknown")
            print(f"  {label_name:25s} (ID: {label:2d}): {n_train:4d} train, {n_samples - n_train:4d} test  (total: {n_samples:4d})")
        
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
            print(f"  {label_name:25s} (ID: {label:2d}): {count:4d} samples ({percentage:5.1f}%)")
    
    def __len__(self):
        """Return number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Load preprocessed frame and label from disk.
        
        Args:
            idx: Index in the current split
            
        Returns:
            Tuple of (frame_tensor, label)
        """
        # Get actual index in processed_indices
        actual_idx = self.indices[idx]
        item = self.processed_indices[actual_idx]
        
        # Load frame from disk
        frame_path = item['save_path']
        frame = Image.open(frame_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            frame = self.transform(frame)
        
        label = item['label']
        
        return frame, label
    
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

def create_preprocessed_dataloaders(root_dir, data_dir='./data/olympic_preprocessed',
                                   batch_size=32, num_workers=4, frames_per_video=8,
                                   force_preprocess=False, owlvit_device_id=0, 
                                   sam_device_id=1, use_csv=False):
    """
    Create train and test dataloaders using preprocessed data.
    
    Args:
        root_dir: Root directory of original Olympic Action dataset
        data_dir: Directory to save/load preprocessed data
        batch_size: Batch size
        num_workers: Number of data loading workers
        frames_per_video: Number of frames to extract per video
        force_preprocess: Force reprocessing even if preprocessed data exists
        owlvit_device_id: GPU device for OWL-ViT during preprocessing
        sam_device_id: GPU device for SAM during preprocessing
        use_csv: Whether to use CSV file as data source
    
    Returns:
        train_loader, test_loader, num_classes
    """
    
    # create data_dir if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    transform = get_olympic_transforms()
    
    train_dataset = OlympicAction(
        root_dir=root_dir,
        split='train',
        transform=transform,
        data_dir=data_dir,
        force_preprocess=force_preprocess,
        owlvit_device_id=owlvit_device_id,
        sam_device_id=sam_device_id,
        frames_per_video=frames_per_video,
        use_csv=use_csv
    )
    
    test_dataset = OlympicAction(
        root_dir=root_dir,
        split='test',
        transform=transform,
        data_dir=data_dir,
        force_preprocess=False,  # Don't reprocess for test set
        owlvit_device_id=owlvit_device_id,
        sam_device_id=sam_device_id,
        frames_per_video=frames_per_video,
        use_csv=use_csv
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    num_classes = train_dataset.get_num_classes()
    
    print(f"Created preprocessed dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print(f"Number of classes: {num_classes}")
    
    return train_loader, test_loader, num_classes


# For backward compatibility
def create_olympic_action_dataloaders(root_dir, batch_size=16, num_workers=4, num_frames_per_video=8):
    """Backward compatibility function - uses preprocessed pipeline"""
    return create_preprocessed_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        frames_per_video=num_frames_per_video
    )


def get_olympic_action_transforms():
    """Backward compatibility function"""
    return get_olympic_transforms()
