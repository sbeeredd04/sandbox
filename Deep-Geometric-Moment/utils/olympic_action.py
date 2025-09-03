import os
import struct
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import defaultdict


def extract_frames_from_seq(seq_path, max_frames=10):
    """
    Simple function to extract frames from a .seq file.
    Returns a list of RGB numpy arrays.
    """
    frames = []
    try:
        with open(seq_path, 'rb') as f:
            # Try to read as video file first (more reliable)
            cap = cv2.VideoCapture(seq_path)
            if cap.isOpened():
                frame_count = 0
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                        frame_count += 1
                cap.release()
            
            # If OpenCV fails, try manual parsing
            if len(frames) == 0:
                frames = _parse_seq_manual(seq_path, max_frames)
                
    except Exception as e:
        print(f"Warning: Could not extract frames from {os.path.basename(seq_path)}: {e}")
    
    return frames


def _parse_seq_manual(seq_path, max_frames=10):
    """
    Manual parsing of .seq files as fallback.
    """
    frames = []
    try:
        with open(seq_path, 'rb') as f:
            # Skip header (try different header sizes)
            for header_size in [1024, 2048, 4096]:
                f.seek(header_size)
                frame_count = 0
                temp_frames = []
                
                while frame_count < max_frames:
                    # Try to read frame size
                    size_bytes = f.read(4)
                    if len(size_bytes) < 4:
                        break
                    
                    try:
                        frame_size = struct.unpack('<I', size_bytes)[0]
                        if frame_size > 10 * 1024 * 1024 or frame_size < 100:
                            continue
                        
                        frame_data = f.read(frame_size)
                        if len(frame_data) < frame_size:
                            continue
                        
                        # Try to decode as JPEG
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            temp_frames.append(frame_rgb)
                            frame_count += 1
                            
                    except (struct.error, cv2.error):
                        continue
                
                if temp_frames:
                    frames = temp_frames
                    break
                    
    except Exception:
        pass
    
    return frames


def visualize_frames(frames, title="Frames", max_display=4):
    """
    Simple function to visualize a list of frames.
    """
    import matplotlib.pyplot as plt
    
    if not frames:
        print(f"No frames to display for {title}")
        return
    
    num_frames = min(len(frames), max_display)
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
    
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        frame = frames[i]
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {i+1}\n{frame.shape[1]}x{frame.shape[0]}')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_dataset_info(root_dir):
    """
    Simple function to get information about the Olympic dataset.
    """
    if not os.path.exists(root_dir):
        return None
    
    sport_categories = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
    
    info = {
        'num_classes': len(sport_categories),
        'classes': sport_categories,
        'files_per_class': {}
    }
    
    for sport in sport_categories:
        sport_dir = os.path.join(root_dir, sport)
        seq_files = [f for f in os.listdir(sport_dir) if f.endswith('.seq')]
        info['files_per_class'][sport] = len(seq_files)
    
    return info


class OlympicActionDataset(Dataset):
    """
    Olympic Sports Action Dataset
    
    This dataset loads .seq files from the Olympic Sports dataset and extracts frames.
    The .seq files are Norpix sequence files containing JPEG frames.
    Modified to return individual frames instead of video sequences.
    """
    
    def __init__(self, root_dir, split='train', transform=None, num_frames_per_video=16):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.num_frames_per_video = num_frames_per_video
        
        # Get all sport categories
        self.sport_categories = sorted([d for d in os.listdir(root_dir) 
                                      if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {sport: idx for idx, sport in enumerate(self.sport_categories)}
        
        # Collect all individual frames with their labels
        self.frame_data = []  # List of (frame, label) tuples
        self.video_files = []
        self.labels = []
        
        print(f"Loading Olympic Action Dataset - {split} split...")
        print("Extracting all frames from videos...")
        
        for sport in self.sport_categories:
            sport_dir = os.path.join(root_dir, sport)
            sport_files = [f for f in os.listdir(sport_dir) if f.endswith('.seq')]
            
            for video_file in sport_files:
                video_path = os.path.join(sport_dir, video_file)
                self.video_files.append(video_path)
                self.labels.append(self.class_to_idx[sport])
        
        # Create train/test split (80/20) and extract frames
        self._create_split_and_extract_frames()
        
        print(f"Olympic Action Dataset - {split} split:")
        print(f"Total individual frames: {len(self.frame_data)}")
        print(f"Number of classes: {len(self.sport_categories)}")
        print(f"Classes: {self.sport_categories}")
        
    def _create_split_and_extract_frames(self):
        """Create stratified train/test split and extract frames"""
        
        #split the data set first
        self._create_split()
        self._extract_frames()
        
    def _extract_frames(self):
        """Extract frames from videos"""
        for i, video_path in enumerate(self.video_files):
            frames = self._read_seq_file(video_path)
            self.frame_data.extend(frames)
            self.labels.append(self.labels[i])
        
    def _create_split(self):
        """Create stratified train/test split"""
        # Group videos by class
        class_to_videos = defaultdict(list)
        for i, (video_path, label) in enumerate(zip(self.video_files, self.labels)):
            class_to_videos[label].append(i)
        
        # Split each class 80/20
        train_indices = []
        test_indices = []
        
        for class_idx, video_indices in class_to_videos.items():
            n_train = int(0.8 * len(video_indices))
            random.shuffle(video_indices)
            train_indices.extend(video_indices[:n_train])
            test_indices.extend(video_indices[n_train:])
        
        # Shuffle the indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        if self.split == 'train':
            self.indices = train_indices
        else:  # test
            self.indices = test_indices
        
        # Update video_files and labels to only include split indices
        self.video_files = [self.video_files[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]
    
    def _read_seq_file(self, seq_path):
        """
        Read frames from .seq file using the simple extraction function.
        """
        return extract_frames_from_seq(seq_path, max_frames=self.num_frames_per_video)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Read frames from .seq file
        frames = self._read_seq_file(video_path)
        
        # Skip videos with no extractable frames
        if len(frames) == 0:
            # Return None to indicate this sample should be skipped
            return None
        
        # Sample frames uniformly
        if len(frames) > self.num_frames_per_video:
            indices = np.linspace(0, len(frames) - 1, self.num_frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames_per_video:
            # Repeat frames cyclically if not enough frames
            while len(frames) < self.num_frames_per_video:
                frames.extend(frames[:min(len(frames), self.num_frames_per_video - len(frames))])
        
        # Convert frames to PIL Images and apply transforms
        processed_frames = []
        for frame in frames:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame.astype(np.uint8))
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(pil_image)
                processed_frames.append(transformed)
            else:
                # If no transforms, convert PIL to tensor manually
                frame_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
                processed_frames.append(frame_tensor)
        
        # Stack frames into a tensor - ensure all frames are tensors
        if processed_frames and isinstance(processed_frames[0], torch.Tensor):
            frames_tensor = torch.stack(processed_frames)
        else:
            # Fallback: convert any remaining PIL images to tensors
            tensor_frames = []
            for frame in processed_frames:
                if isinstance(frame, torch.Tensor):
                    tensor_frames.append(frame)
                else:
                    # Convert PIL to tensor
                    frame_array = np.array(frame)
                    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                    tensor_frames.append(frame_tensor)
            frames_tensor = torch.stack(tensor_frames)
        
        return frames_tensor, label


def get_olympic_action_transforms():
    """
    Get transforms for Olympic Action dataset
    
    Based on the .seq file analysis, the original dimensions are 480x360.
    We'll resize to 224x224 to match standard model inputs.
    """
    transform_train = transforms.Compose([``
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return transform_train, transform_test


def olympic_collate_fn(batch):
    """
    Custom collate function to handle None values (skipped samples).
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Use default collate for valid samples
    return torch.utils.data.default_collate(batch)


def olympic_collate_fn_training(batch):
    """
    Custom collate function for training that ensures we always return valid batches.
    If all samples are None, we'll retry with a fallback mechanism.
    """
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    # If we have no valid samples, this shouldn't happen often with the robust extraction
    # but we'll handle it gracefully
    if len(valid_batch) == 0:
        print("Warning: Empty batch encountered, this may indicate extraction issues")
        return None
    
    # Use default collate for valid samples
    return torch.utils.data.default_collate(valid_batch)


def create_olympic_action_dataloaders(root_dir, batch_size=32, num_workers=4, num_frames_per_video=16):
    """
    Create train and test dataloaders for Olympic Action dataset with GPU optimization.
    
    Args:
        root_dir: Path to the olympic_sports dataset directory
        batch_size: Batch size for training and testing
        num_workers: Number of data loading workers
        num_frames_per_video: Number of frames to extract per video
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data  
        num_classes: Number of sport classes (16 for Olympic Action)
    """
    transform_train, transform_test = get_olympic_action_transforms()
    
    # Create datasets
    train_dataset = OlympicActionDataset(
        root_dir=root_dir, 
        split='train', 
        transform=transform_train,
        num_frames_per_video=num_frames_per_video
    )
    
    test_dataset = OlympicActionDataset(
        root_dir=root_dir, 
        split='test', 
        transform=transform_test,
        num_frames_per_video=num_frames_per_video
    )
    
    # Optimize for GPU usage
    pin_memory = torch.cuda.is_available()
    if pin_memory:
        print("Using GPU optimizations: pin_memory=True")
    
    # Adjust num_workers based on system capabilities
    if num_workers == 0:
        print("Warning: Using num_workers=0 may slow down data loading")
    elif torch.cuda.is_available() and num_workers > 8:
        print("Info: Consider reducing num_workers if you encounter GPU memory issues")
    
    # Create dataloaders with GPU optimizations and custom collate function
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': pin_memory,
        'collate_fn': olympic_collate_fn,
    }
    
    # Add multiprocessing options only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs.update({
            'num_workers': num_workers,
            'persistent_workers': True,
            'prefetch_factor': 2
        })
    else:
        dataloader_kwargs['num_workers'] = 0
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,
        **dataloader_kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False,
        **dataloader_kwargs
    )
    
    print(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print(f"GPU optimizations: pin_memory={pin_memory}, persistent_workers={num_workers > 0}")
    
    return train_loader, test_loader, len(train_dataset.sport_categories)


def get_olympic_dataloader_iterator(dataloader):
    """
    Generator that yields only valid (non-None) batches from the dataloader.
    This ensures training loops don't have to handle None batches.
    """
    for batch in dataloader:
        if batch is not None:
            yield batch
        else:
            print("Skipping empty batch (no valid frames extracted)")
            continue


# For compatibility with existing code
def get_olympic_action_transforms_legacy():
    """Legacy function name for compatibility"""
    return get_olympic_action_transforms()


if __name__ == "__main__":
    # Test the dataset
    root_dir = "./data/olympic_sports"
    
    if os.path.exists(root_dir):
        # Test dataset info
        info = get_dataset_info(root_dir)
        if info:
            print(f"Dataset Info:")
            print(f"  Classes: {info['num_classes']}")
            print(f"  Sports: {info['classes'][:5]}...")  # Show first 5
        
        # Test frame extraction
        print("\nTesting frame extraction:")
        for sport in info['classes'][:3]:  # Test first 3 sports
            sport_dir = os.path.join(root_dir, sport)
            seq_files = [f for f in os.listdir(sport_dir) if f.endswith('.seq')]
            if seq_files:
                seq_path = os.path.join(sport_dir, seq_files[0])
                frames = extract_frames_from_seq(seq_path, max_frames=3)
                print(f"  {sport}: {len(frames)} frames extracted")
        
        # Test dataloaders
        train_loader, test_loader, num_classes = create_olympic_action_dataloaders(
            root_dir, batch_size=2, num_workers=0, num_frames_per_video=8
        )
        
        print(f"\nDataLoader Info:")
        print(f"  Number of classes: {num_classes}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test loading a batch
        print("\nTesting batch loading:")
        for batch_idx, batch in enumerate(train_loader):
            if batch is not None:  # Skip empty batches
                frames, labels = batch
                print(f"  Batch {batch_idx}: frames shape: {frames.shape}, labels: {labels}")
            else:
                print(f"  Batch {batch_idx}: Empty batch (all samples skipped)")
            if batch_idx >= 2:  # Test only first few batches
                break
    else:
        print(f"Dataset directory {root_dir} not found!")
