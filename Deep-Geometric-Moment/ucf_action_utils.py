import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image
from collections import defaultdict
import random
import os


def get_ucf_sports_transforms():
    """Get UCF Sports Action dataset transforms with aggressive augmentation for 224x224"""
    # More aggressive transforms similar to CIFAR-10 but adapted for 224x224
    transforms1 = transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomAffine(90, translate=(0.2, 0.2), scale=(0.6, 1.3))
    ]), p=0.4)
    
    transforms2 = transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.25)
    ]), p=0.3)
    
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 first
        transforms2,  # Color jitter
        transforms1,  # Affine transforms
        transforms.RandomCrop(224, padding=28),  # Random crop to 224x224 with padding (equivalent to padding=4 for 32x32)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        
        #save the image to local bef
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize for testing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return transform_train, transform_test


class UCFSportsDataset(data.Dataset):
    """UCF Sports Action Dataset using Deep Lake"""
    def __init__(self, deeplake_ds, split='train', transform=None, debug=True):
        self.deeplake_ds = deeplake_ds
        self.transform = transform
        self.split = split
        self.debug = debug
        self.debug_counter = 0
        
        # Create debug directory
        if self.debug:
            self.debug_dir = f"debug_images_{split}"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"Debug mode enabled. Images will be saved to: {self.debug_dir}")
        
        # Create stratified train/test split
        total_samples = len(deeplake_ds)
        
        # Get all labels first to create stratified split
        all_labels = []
        for i in range(total_samples):
            sample = deeplake_ds[i]
            label = int(sample.labels.numpy()[0])
            all_labels.append(label)
        
        # Create stratified indices
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            label_to_indices[label].append(idx)
        
        # Split each class 80/20
        train_indices = []
        test_indices = []
        
        for label, indices in label_to_indices.items():
            n_train = int(0.8 * len(indices))
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
        
        # Shuffle the indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        if split == 'train':
            self.indices = train_indices
        else:  # test
            self.indices = test_indices
    
    def _get_label_distribution(self):
        """Get distribution of labels in current split"""
        label_counts = {}
        for i, sample in enumerate(self.deeplake_ds):
            #for all samples
            label = int(sample.labels.numpy()[0])
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our split indices
        actual_idx = self.indices[idx]
        
        # Get image and label from Deep Lake dataset
        sample = self.deeplake_ds[actual_idx]        
        
        # Extract image and label
        image = sample.images.numpy()
        label = int(sample.labels.numpy()[0])
        
        # Handle different image formats from Deep Lake
        if len(image.shape) == 4:  # (1, C, H, W) format
            image = image.squeeze(0)
        elif len(image.shape) == 3:  # (C, H, W) format
            pass
        else:
            image = image.squeeze()
        
        # Ensure we have a 3D tensor (C, H, W)
        if len(image.shape) != 3:
            raise ValueError(f"Unexpected image shape after processing: {image.shape}")
        
        # Convert CHW to HWC for PIL
        image = image.transpose(1, 2, 0)
        
        # Handle different data types and normalize to uint8
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        try:
            image = Image.fromarray(image, mode='RGB')
        except Exception as e:
            print(f"Error converting image to PIL: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            transformed_image = self.transform(image)
        else:
            # Convert to tensor if no transforms
            transform_to_tensor = transforms.ToTensor()
            transformed_image = transform_to_tensor(image)
        
        return transformed_image, label
