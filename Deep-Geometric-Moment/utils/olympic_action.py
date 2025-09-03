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


class OlympicActionDataset(Dataset):
    """
    Olympic Sports Action Dataset
    
    This dataset loads .seq files from the Olympic Sports dataset and extracts frames.
    The .seq files are Norpix sequence files containing JPEG frames.
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
        
        # Collect all video files with their labels
        self.video_files = []
        self.labels = []
        
        for sport in self.sport_categories:
            sport_dir = os.path.join(root_dir, sport)
            for video_file in os.listdir(sport_dir):
                if video_file.endswith('.seq'):
                    self.video_files.append(os.path.join(sport_dir, video_file))
                    self.labels.append(self.class_to_idx[sport])
        
        # Create train/test split (80/20)
        self._create_split()
        
        #print(f"Olympic Action Dataset - {split} split:")
        #print(f"Total videos: {len(self.video_files)}")
        #print(f"Number of classes: {len(self.sport_categories)}")
        #print(f"Classes: {self.sport_categories}")
        
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
        Read a Norpix .seq file and extract JPEG frames with improved error handling
        
        Based on analysis of the Olympic Sports dataset .seq files:
        - Norpix sequence format with UTF-16 header
        - Dimensions at offset 0x220 and 0x224 (little-endian)
        - Frame data starts at offset 0x400
        - Each frame: 4-byte size + JPEG data
        """
        frames = []
        
        try:
            with open(seq_path, 'rb') as f:
                # Read full header (1024 bytes)
                header = f.read(1024)
                if len(header) < 1024:
                    #print(f"Warning: Header too short in {seq_path}, trying fallback")
                    return self._read_video_file_fallback(seq_path)
                
                # Check for Norpix signature (UTF-16 encoded "Norpix seq")
                if header[0:2] != b'\xed\xfe':
                    # Try to use as regular video file
                    return self._read_video_file_fallback(seq_path)
                
                # Parse dimensions from header (little-endian)
                try:
                    width = struct.unpack('<I', header[0x220:0x224])[0]
                    height = struct.unpack('<I', header[0x224:0x228])[0]
                    
                    # Validate dimensions
                    if not (10 <= width <= 10000 and 10 <= height <= 10000):
                        #print(f"Warning: Invalid dimensions {width}x{height} ({hex(width)}x{hex(height)}), using defaults")
                        width, height = 480, 270
                    else:
                        #print(f"Parsing {os.path.basename(seq_path)}: {width}x{height}")
                        pass
                except struct.error:
                    width, height = 480, 270

                
                # Seek to start of frame data (offset 0x400)
                f.seek(0x400)
                
                frame_count = 0
                valid_frames = 0
                max_frames = 100  # Reduced limit to avoid getting stuck
                consecutive_errors = 0
                max_consecutive_errors = 10
                
                while frame_count < max_frames and consecutive_errors < max_consecutive_errors:
                    # Read frame size (4 bytes, little-endian)
                    size_bytes = f.read(4)
                    if len(size_bytes) < 4:
                        break
                    
                    try:
                        frame_size = struct.unpack('<I', size_bytes)[0]
                        
                        # Sanity check frame size (should be reasonable)
                        if frame_size > 10 * 1024 * 1024 or frame_size < 100:  # 10MB max, 100 bytes min
                            consecutive_errors += 1
                            continue
                            
                    except struct.error:
                        consecutive_errors += 1
                        continue
                    
                    # Read frame data
                    frame_data = f.read(frame_size)
                    if len(frame_data) < frame_size:
                        # If we got some frames already, stop here instead of failing completely
                        if valid_frames > 0:
                            break
                        consecutive_errors += 1
                        continue
                    
                    # Try to decode even if JPEG header is missing/corrupted
                    try:
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                            valid_frames += 1
                            consecutive_errors = 0  # Reset error counter on success
                            
                            # Stop after getting at least one good frame to avoid excessive processing
                            if valid_frames >= 1:
                                break
                        else:
                            consecutive_errors += 1
                    except Exception:
                        consecutive_errors += 1
                        continue
                    
                    frame_count += 1
                
                if valid_frames > 0:
                    #print(f"Successfully extracted {valid_frames} frames from {os.path.basename(seq_path)}")
                    pass
                elif consecutive_errors >= max_consecutive_errors:
                    #print(f"Too many consecutive errors in {os.path.basename(seq_path)}, trying fallback")
                    return self._read_video_file_fallback(seq_path)
                        
        except Exception as e:
            #print(f"Error reading seq file {seq_path}: {e}, trying fallback")
            return self._read_video_file_fallback(seq_path)
        
        # If no frames extracted, try fallback
        if len(frames) == 0:
            return self._read_video_file_fallback(seq_path)
        
        return frames
    
    def _read_video_file_fallback(self, video_path):
        """
        Fallback method to read .seq files as regular video files using OpenCV
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            frame_count = 0
            max_frames = 10  # Only extract a few frames for efficiency
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Validate frame
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
                
                # Stop after getting at least one good frame
                if len(frames) >= 1:
                    break
            
            cap.release()
            if len(frames) > 0:
                #print(f"Fallback: extracted {len(frames)} frames from {os.path.basename(video_path)}")
                pass
        except Exception as e:
            #print(f"Fallback method failed for {video_path}: {e}")
            pass
        return frames
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Read frames from .seq file
        frames = self._read_seq_file(video_path)
        
        if len(frames) == 0:
            # Create a synthetic frame with some pattern to indicate missing data
            # Use a gradient pattern instead of pure black for better feature learning
            synthetic_frame = np.zeros((270, 480, 3), dtype=np.uint8)
            # Add a diagonal gradient pattern
            for i in range(270):
                for j in range(480):
                    synthetic_frame[i, j] = [min(255, i + j), min(255, abs(i - j)), min(255, i * j // 100)]
            frames = [synthetic_frame]
            #print(f"Warning: Using synthetic frame for {os.path.basename(video_path)}")
        
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
    transform_train = transforms.Compose([
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


def create_olympic_action_dataloaders(root_dir, batch_size=32, num_workers=4, num_frames_per_video=16):
    """
    Create train and test dataloaders for Olympic Action dataset with GPU optimization
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
        ##print("Using GPU optimizations: pin_memory=True")
        pass
    
    # Adjust num_workers based on system capabilities
    if num_workers == 0:
        #print("Warning: Using num_workers=0 may slow down data loading")
        pass
    elif torch.cuda.is_available() and num_workers > 8:
        #print("Info: Consider reducing num_workers if you encounter GPU memory issues")
        pass
    
    # Create dataloaders with GPU optimizations
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': pin_memory,
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
    
    #print(f"Created dataloaders: {len(train_loader)} train batches, {len(test_loader)} test batches")
    #print(f"GPU optimizations: pin_memory={pin_memory}, persistent_workers={num_workers > 0}")
    
    return train_loader, test_loader, len(train_dataset.sport_categories)


# For compatibility with existing code
def get_olympic_action_transforms_legacy():
    """Legacy function name for compatibility"""
    return get_olympic_action_transforms()


if __name__ == "__main__":
    # Test the dataset
    root_dir = "./data/olympic_sports"
    
    if os.path.exists(root_dir):
        train_loader, test_loader, num_classes = create_olympic_action_dataloaders(
            root_dir, batch_size=2, num_workers=0, num_frames_per_video=8
        )
        
        #print(f"Number of classes: {num_classes}")
        #print(f"Train batches: {len(train_loader)}")
        #print(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for batch_idx, (frames, labels) in enumerate(train_loader):
            #print(f"Batch {batch_idx}: frames shape: {frames.shape}, labels: {labels}")
            if batch_idx >= 2:  # Test only first few batches
                break
    else:
        #print(f"Dataset directory {root_dir} not found!")
        pass
