import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class OlympicActionDataset(Dataset):
    """
    Simple Olympic Action Dataset for extracting frames from .seq files
    """
    
    def __init__(self, root_dir, split='train', transform=None, frames_per_video=8):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Get all sport categories (classes)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all video files
        self.video_files = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            seq_files = [f for f in os.listdir(class_dir) if f.endswith('.seq')]
            
            for seq_file in seq_files:
                self.video_files.append(os.path.join(class_dir, seq_file))
                self.labels.append(self.class_to_idx[class_name])
        
        # Create train/test split (80/20)
        self._create_split()
        
        print(f"Olympic Action {split}: {len(self.video_files)} videos, {len(self.classes)} classes")
        
        # Show detailed dataset distribution
        self._show_dataset_distribution()
    
    def _create_split(self):
        """Create simple train/test split"""
        # Group by class
        class_videos = {}
        for i, (video, label) in enumerate(zip(self.video_files, self.labels)):
            if label not in class_videos:
                class_videos[label] = []
            class_videos[label].append(i)
        
        # Split each class 80/20
        split_indices = []
        for label, indices in class_videos.items():
            random.shuffle(indices)
            split_point = int(0.8 * len(indices))
            if self.split == 'train':
                split_indices.extend(indices[:split_point])
            else:
                split_indices.extend(indices[split_point:])
        
        # Update video files and labels
        self.video_files = [self.video_files[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]
    
    def _extract_frames(self, video_path):
        """Extract frames from .seq file"""
        frames = []
        
        try:
            # Try OpenCV first
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = 0
                while frame_count < self.frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                        frame_count += 1
                cap.release()
        except:
            pass
        
        # If no frames extracted, return empty list
        if len(frames) == 0:
            return []
        
        # Ensure we have the right number of frames
        if len(frames) < self.frames_per_video:
            # Repeat frames if not enough
            while len(frames) < self.frames_per_video:
                frames.extend(frames[:min(len(frames), self.frames_per_video - len(frames))])
        elif len(frames) > self.frames_per_video:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames[:self.frames_per_video]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        # Skip if no frames extracted
        if len(frames) == 0:
            return None
        
        # Convert frames to tensors
        processed_frames = []
        for frame in frames:
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame.astype(np.uint8))
            
            # Apply transforms
            if self.transform:
                tensor_frame = self.transform(pil_frame)
            else:
                # Default: resize and normalize to 256x256
                tensor_frame = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(pil_frame)
            
            processed_frames.append(tensor_frame)
        
        # Stack frames
        frames_tensor = torch.stack(processed_frames)
        
        return frames_tensor, label
    
    def _show_dataset_distribution(self):
        """Show detailed dataset distribution for debugging"""
        print(f"\n=== OLYMPIC ACTION DATASET DISTRIBUTION ({self.split.upper()}) ===")
        
        # Count samples per class
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Calculate statistics
        total_videos = len(self.labels)
        counts = list(class_counts.values())
        avg_count = sum(counts) / len(counts)
        min_count = min(counts)
        max_count = max(counts)
        
        print(f"Total {self.split} videos: {total_videos}")
        print(f"Average videos per class: {avg_count:.1f}")
        print(f"Min videos per class: {min_count}")
        print(f"Max videos per class: {max_count}")
        print(f"Class balance ratio (min/max): {min_count/max_count:.3f}")
        
        if min_count / max_count < 0.5:
            print("⚠️  WARNING: Significant class imbalance detected!")
        else:
            print("✓ Classes are reasonably balanced")
        
        print(f"\nDetailed breakdown:")
        print(f"{'Class':<25} {'Videos':<8} {'Percentage':<12} {'Estimated Frames'}")
        print("-" * 60)
        
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / total_videos) * 100
            est_frames = count * self.frames_per_video  # Estimated total frames
            print(f"{class_name:<25} {count:<8} {percentage:<8.1f}%    ~{est_frames:<8}")
        
        print("=" * 60)
        print(f"Estimated total frames in {self.split} set: ~{total_videos * self.frames_per_video}")
        print("=" * 60 + "\n")


def get_transforms():
    """Get simple transforms for training and testing - using 256x256 for model compatibility"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def collate_fn(batch):
    """Custom collate function to handle None values"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    return torch.utils.data.default_collate(batch)


def create_dataloaders(root_dir, batch_size=16, num_workers=4, frames_per_video=8):
    """
    Create simple train and test dataloaders
    
    Args:
        root_dir: Path to olympic_sports dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        frames_per_video: Number of frames to extract per video
    
    Returns:
        train_loader, test_loader, num_classes
    """
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = OlympicActionDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        frames_per_video=frames_per_video
    )
    
    test_dataset = OlympicActionDataset(
        root_dir=root_dir,
        split='test',
        transform=test_transform,
        frames_per_video=frames_per_video
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    num_classes = len(train_dataset.classes)
    
    print(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print(f"Number of classes: {num_classes}")
    
    return train_loader, test_loader, num_classes


# For backward compatibility
def create_olympic_action_dataloaders(root_dir, batch_size=16, num_workers=4, num_frames_per_video=8):
    """Backward compatibility function"""
    return create_dataloaders(root_dir, batch_size, num_workers, num_frames_per_video)


def get_olympic_action_transforms():
    """Backward compatibility function"""
    return get_transforms()