"""
UCF Sports Action Dataset utilities with OWL-ViT + SAM pipeline integration.

This module provides:
- UCF Sports dataset loading with Deep Lake
- OWL-ViT object detection for human/action detection
- SAM segmentation for background masking
- Automatic filtering of images without bounding box detections
- Logging of dropped images statistics
- CUDA multiprocessing-safe model initialization

Key changes from YOLO-World version:
- Replaced YOLO-World with OWL-ViT for better text-based object detection
- Added automatic image filtering when no bounding boxes are detected
- Removed debugging code for cleaner production use
- Added statistics tracking for dropped images
- Implemented lazy CUDA initialization to avoid multiprocessing conflicts

CUDA Multiprocessing Fix:
- Models are initialized lazily in each process to avoid "Cannot re-initialize CUDA" errors
- Thread-safe initialization with proper error handling and CPU fallback
- Works with DataLoader multiprocessing (num_workers > 0)
"""

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from collections import defaultdict
import random
import cv2
import os
import sys
import threading

# Import OWL-ViT and SAM
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import SamPredictor, sam_model_registry

def get_ucf_class_mappings(deeplake_ds, group_similar=True):

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
            'Lifting': 'Lifting',  # Keep as is
            'Riding-Horse': 'Riding-Horse'  # Keep as is
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
        
        def get_class_name(original_label_id):
            """Convert original label ID to grouped class name"""
            if original_label_id in original_to_grouped_id:
                grouped_id = original_to_grouped_id[original_label_id]
                return grouped_id_to_label.get(grouped_id, "Unknown")
            return "Unknown"
        
        def get_grouped_class_id(original_label_id):
            """Convert original label ID to grouped class ID"""
            return original_to_grouped_id.get(original_label_id, -1)
        
        print(f"Original classes: {len(original_class_names)} -> Grouped classes: {len(grouped_class_names)}")
        print(f"Grouped class names: {grouped_class_names}")
        print(f"Original to grouped mapping: {original_to_grouped_id}")
        
        return (grouped_class_names, original_to_grouped_id, grouped_to_original_ids, 
                get_class_name, get_grouped_class_id, grouping_rules)
    
    else:
        # Original behavior without grouping
        label_name_to_id = {name: idx for idx, name in enumerate(original_class_names)}
        label_id_to_name = {idx: name for idx, name in enumerate(original_class_names)}
        
        def get_class_name(label_id):
            """Convert label ID to class name"""
            return label_id_to_name.get(label_id, "Unknown")
        
        return original_class_names, label_name_to_id, label_id_to_name, get_class_name

def get_ucf_sports_transforms():
    """Get UCF Sports Action dataset transforms with aggressive augmentation for 224x224"""
    # More aggressive transforms similar to CIFAR-10 but adapted for 224x224
    transforms1 = transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomAffine(90, translate=(0.2, 0.2), scale=(0.6, 1.3))
    ]), p=0.4)
    
    transforms2 = transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.25)
    ]), p=0.3)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 first
        # transforms2,  # Color jitter
        # transforms1,  # Affine transforms
        # transforms.RandomCrop(224, padding=28),  # Random crop to 224x224 with padding (equivalent to padding=4 for 32x32)
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    return transform

class UCFSportsDataset(data.Dataset):
    """UCF Sports Action Dataset using Deep Lake with optional class grouping"""
    def __init__(self, deeplake_ds, split='train', transform=None, use_grouped_classes=True):
        self.deeplake_ds = deeplake_ds
        self.transform = transform
        self.split = split
        self.use_grouped_classes = use_grouped_classes
        
        # Initialize models lazily to avoid CUDA multiprocessing issues
        self.owlvit_model = None
        self.owlvit_processor = None
        self.device = None
        self._model_initialized = False
        
        # Initialize SAM model lazily to avoid CUDA multiprocessing issues
        self.sam_model = None
        self.predictor = None
        self._sam_initialized = False
        
        # Logging for dropped images
        self.dropped_images_count = 0
        self.total_images_processed = 0
    
        # Create stratified train/test split
        total_samples = len(self.deeplake_ds)
        print(f"Total samples: {total_samples}")
        
        # Get original class names from Deep Lake dataset
        self.original_class_names = self.deeplake_ds.labels.info['class_names']
        print(f"Original class names from dataset: {self.original_class_names}")
        
        # Set up class mappings based on whether we're using grouped classes
        if self.use_grouped_classes:
            # Get grouped class mappings
            (self.class_names, self.original_to_grouped_id, self.grouped_to_original_ids, 
             self.get_grouped_class_name, self.get_grouped_class_id, self.grouping_rules) = get_ucf_class_mappings(
                deeplake_ds, group_similar=True)
            
            print(f"Using GROUPED classes for training: {self.class_names}")
            print(f"Reduced from {len(self.original_class_names)} to {len(self.class_names)} classes")
            
            # Create mappings for grouped classes
            self.label_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}
            self.label_id_to_name = {idx: name for idx, name in enumerate(self.class_names)}
            
        else:
            # Use original classes
            self.class_names = self.original_class_names
            self.label_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}
            self.label_id_to_name = {idx: name for idx, name in enumerate(self.class_names)}
            print(f"Using ORIGINAL classes for training: {self.class_names}")
        
        print(f"Label name to ID mapping: {self.label_name_to_id}")
        print(f"Label ID to name mapping: {self.label_id_to_name}")
        
        # Get all labels first to create stratified split
        all_original_labels = []
        all_labels = []  # Will store grouped labels if using grouped classes
        
        for i in range(total_samples):
            sample = self.deeplake_ds[i]
            original_label = int(sample.labels.numpy()[0])
            all_original_labels.append(original_label)
            
            if self.use_grouped_classes:
                grouped_label = self.original_to_grouped_id.get(original_label, -1)
                all_labels.append(grouped_label)
            else:
                all_labels.append(original_label)
            
        # print(f"All original labels: {all_original_labels}")
        if self.use_grouped_classes:
            print(f"All grouped labels: {all_labels}")
        
        # Create stratified indices based on the labels we're actually using for training
        label_to_indices = defaultdict(list)
        for idx in range(total_samples):
            label = all_labels[idx]
            label_to_indices[label].append(idx)
        
        print(f"Label to indices: {label_to_indices}")
        
        # Print statistics for each class
        print(f"\nClass distribution (for training):")
        for label_id, indices in label_to_indices.items():
            if label_id in self.label_id_to_name:
                label_name = self.label_id_to_name[label_id]
                print(f"  {label_name} (ID: {label_id}): {len(indices)} samples")
            else:
                print(f"  Unknown class (ID: {label_id}): {len(indices)} samples")
        
        # Split each class 90/10
        train_indices = []
        test_indices = []
        
        for label, indices in label_to_indices.items():
            n_train = int(0.9 * len(indices))
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
            
            label_name = self.label_id_to_name[label]
            print(f"  {label_name}: {n_train} train, {len(indices) - n_train} test")
        
        # Shuffle the indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        if self.split == 'train':
            self.indices = train_indices
            print(f"\nTrain split: {len(self.indices)} samples")
        else:  # test
            self.indices = test_indices
            print(f"\nTest split: {len(self.indices)} samples")
        
        # Print class distribution after indices are assigned
        distribution = self.get_class_distribution()
        print(f"Class distribution: {distribution}")
        
        # Create text queries for OWL-ViT based on our dataset classes
        self.text_queries_mapping = {
            'Diving': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing diving'],
            'Golf-Swing': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing golf swing'],
            'Kicking': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing kicking'],
            'Lifting': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing lifting'],
            'Riding-Horse': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing riding horse'],
            'Run': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing running'],
            'SkateBoarding': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing skateboarding'],
            'Swing-Bench': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing swing bench'],
            'Swing-SideAngle': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing swing side angle'],
            'Walk': ['human', 'person', 'people', 'body', 'human body', 'human being', 'doing walking']
        }
        print(f"OWL-ViT text queries prepared for {len(self.text_queries_mapping)} classes")
    
    def _initialize_owlvit(self):
        """Lazily initialize OWL-ViT model to avoid CUDA multiprocessing issues"""
        if not self._model_initialized:
            with threading.Lock():  # Thread-safe initialization
                if not self._model_initialized:  # Double-check after acquiring lock
                    print("Initializing OWL-ViT model...")
                    self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
                    self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
                    
                    # Use GPU if available, but handle CUDA initialization carefully
                    if torch.cuda.is_available():
                        try:
                            self.device = torch.device("cuda")
                            self.owlvit_model = self.owlvit_model.to(self.device)
                            print(f"OWL-ViT model loaded on GPU: {self.device}")
                        except RuntimeError as e:
                            if "Cannot re-initialize CUDA" in str(e):
                                print("CUDA re-initialization error detected, falling back to CPU")
                                self.device = torch.device("cpu")
                                self.owlvit_model = self.owlvit_model.to(self.device)
                            else:
                                raise e
                    else:
                        self.device = torch.device("cpu")
                        print("CUDA not available, using CPU")
                    
                    self.owlvit_model.eval()
                    self._model_initialized = True
    
    def _initialize_sam(self):
        """Lazily initialize SAM model to avoid CUDA multiprocessing issues"""
        if not self._sam_initialized:
            with threading.Lock():  # Thread-safe initialization
                if not self._sam_initialized:  # Double-check after acquiring lock
                    print("Initializing SAM model...")
                    model_type = "vit_b"
                    if model_type == "vit_h":
                        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_h_4b8939.pth")
                    elif model_type == "vit_l":
                        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_l_0b3195.pth")
                    else:  # vit_b
                        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_b_01ec64.pth")
                                
                    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                    
                    # Use GPU if available, but handle CUDA initialization carefully
                    if torch.cuda.is_available():
                        try:
                            sam.cuda()
                            print(f"SAM model loaded on GPU")
                        except RuntimeError as e:
                            if "Cannot re-initialize CUDA" in str(e):
                                print("CUDA re-initialization error detected for SAM, falling back to CPU")
                                sam.cpu()
                            else:
                                raise e
                    else:
                        sam.cpu()
                        print("CUDA not available for SAM, using CPU")
                    
                    self.predictor = SamPredictor(sam)
                    self._sam_initialized = True
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Initialize models lazily to avoid CUDA multiprocessing issues
        self._initialize_owlvit()
        self._initialize_sam()
        
        # Get the actual index from our split indices
        actual_idx = self.indices[idx]
        
        # Get image and label from Deep Lake dataset
        sample = self.deeplake_ds[actual_idx]      
        
        image = sample.images.numpy()
        original_label = int(sample.labels.numpy()[0])
        
        # Convert to grouped label if using grouped classes
        if self.use_grouped_classes:
            # Convert original label to grouped label
            grouped_label = self.original_to_grouped_id.get(original_label, -1)
            if grouped_label == -1:
                print(f"Warning: Could not map original label {original_label} to grouped label")
            label = grouped_label
        else:
            label = original_label
        
        pil_image = Image.fromarray(image, mode='RGB')
        original_image = pil_image.resize((256, 256))
        original_image_np = np.array(original_image)
        
        # Get class name for OWL-ViT
        class_name = self.get_class_name(label)
        
        # Get text queries for this class
        text_queries = self.text_queries_mapping.get(class_name, ['human', 'person', 'people'])
        
        # Process image and text inputs with OWL-ViT
        inputs = self.owlvit_processor(text=text_queries, images=original_image, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)
        
        # Post-process to get boxes in original image coordinates
        target_sizes = torch.Tensor([original_image.size[::-1]]).to(self.device)  # (height, width)
        results = self.owlvit_processor.post_process(outputs=outputs, target_sizes=target_sizes)
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"].cpu().numpy()
        
        # Filter by confidence threshold
        confidence_threshold = 0.1
        valid_indices = scores >= confidence_threshold
        
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        
        # Check if any bounding boxes were detected
        if len(boxes) == 0:
            # No bounding boxes detected - skip this image
            self.dropped_images_count += 1
            self.total_images_processed += 1
            
            if self.total_images_processed % 100 == 0:
                print(f"Dropped {self.dropped_images_count}/{self.total_images_processed} images ({(100*self.dropped_images_count/self.total_images_processed):.1f}%) due to no bounding box detection")
            
            # Return None to indicate this sample should be skipped
            return None
        
        # Use SAM model to segment the image using the first detected bounding box
        self.predictor.set_image(original_image_np)
        input_box = np.array(boxes[0])  # Use first detected box
        
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
        )
        
        # Convert mask to proper input format
        if len(masks) > 0:
            mask = masks[0]  # Take first mask
            
            # Create masked image where background is zeroed out
            masked_image_np = original_image_np.copy().astype(np.float32)
            # Apply mask to each channel - zero out background
            for c in range(3):
                masked_image_np[:, :, c] = masked_image_np[:, :, c] * mask
            
            # Convert back to uint8
            masked_image_np = masked_image_np.astype(np.uint8)
            
            # Convert masked image to PIL and apply transforms
            masked_image_pil = Image.fromarray(masked_image_np)
            transformed_image = self.transform(masked_image_pil)
        else:
            # If no mask found, use original image
            transformed_image = self.transform(pil_image)
        
        self.total_images_processed += 1
        return transformed_image, label, original_image_np
    
    def get_class_name(self, label_id):
        """Convert label ID to class name"""
        return self.label_id_to_name.get(label_id, "Unknown")
    
    def get_class_id(self, label_name):
        """Convert class name to label ID"""
        return self.label_name_to_id.get(label_name, -1)
    
    def get_all_class_names(self):
        """Get all class names in order"""
        return self.class_names.copy()
    
    def get_class_distribution(self):
        """Get class distribution for current split"""
        distribution = defaultdict(int)
        for idx in self.indices:
            sample = self.deeplake_ds[idx]
            original_label = int(sample.labels.numpy()[0])
            
            if self.use_grouped_classes:
                # Convert to grouped label
                grouped_label = self.original_to_grouped_id.get(original_label, -1)
                distribution[grouped_label] += 1
            else:
                distribution[original_label] += 1
                
        return dict(distribution)
    
    def get_num_classes(self):
        """Get the number of classes for model architecture"""
        return len(self.class_names)
    
    def get_dropped_images_stats(self):
        """Get statistics about dropped images"""
        if self.total_images_processed == 0:
            return {"dropped": 0, "total": 0, "percentage": 0.0}
        
        percentage = (self.dropped_images_count / self.total_images_processed) * 100
        return {
            "dropped": self.dropped_images_count,
            "total": self.total_images_processed,
            "percentage": percentage
        }


def collate_fn_skip_none(batch):
    """Custom collate function that filters out None values (dropped images)"""
    # Filter out None values (dropped images)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # Return empty batch if all images were dropped
        return None, None, None
    
    # Separate the components
    images, labels, original_images = zip(*batch)
    
    # Convert to tensors
    images = torch.stack(images, 0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels, original_images


def setup_multiprocessing_for_cuda():
    """
    Setup multiprocessing to use 'spawn' method for CUDA compatibility.
    Call this at the beginning of your main script before any DataLoader usage.
    """
    import multiprocessing as mp
    
    try:
        # Try to set start method to 'spawn' for CUDA compatibility
        mp.set_start_method('spawn', force=True)
        print("✓ Multiprocessing set to 'spawn' method for CUDA compatibility")
        return True
    except RuntimeError as e:
        if "context has already been set" in str(e):
            current_method = mp.get_start_method()
            if current_method == 'spawn':
                print("✓ Multiprocessing already set to 'spawn' method")
                return True
            else:
                print(f"⚠️  Multiprocessing already initialized with '{current_method}' method")
                print("   CUDA multiprocessing may not work properly. Consider restarting the script.")
                return False
        else:
            print(f"⚠️  Could not set multiprocessing to 'spawn': {e}")
            return False


def create_cuda_safe_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, **kwargs):
    """
    Create a DataLoader that's safe for CUDA multiprocessing.
    
    Args:
        dataset: The dataset to create DataLoader for
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for no multiprocessing)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader with proper CUDA multiprocessing setup
    """
    # Setup multiprocessing for CUDA if using multiple workers
    if num_workers > 0:
        setup_multiprocessing_for_cuda()
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_skip_none,
        **kwargs
    )