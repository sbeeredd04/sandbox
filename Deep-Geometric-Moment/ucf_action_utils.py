import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import random
import cv2

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
    """UCF Sports Action Dataset using Deep Lake"""
    def __init__(self, deeplake_ds, split='train', transform=None):
        self.deeplake_ds = deeplake_ds
        self.transform = transform
        self.split = split
    
        # Create stratified train/test split
        total_samples = len(self.deeplake_ds)
        print(f"Total samples: {total_samples}")
        
        # Get class names from Deep Lake dataset
        self.class_names = self.deeplake_ds.labels.info['class_names']
        print(f"Class names from dataset: {self.class_names}")
        
        # Create label name to ID mapping
        self.label_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.label_id_to_name = {idx: name for idx, name in enumerate(self.class_names)}
        
        print(f"Label name to ID mapping: {self.label_name_to_id}")
        print(f"Label ID to name mapping: {self.label_id_to_name}")
        
        # Get all labels first to create stratified split
        all_labels = []
        for i in range(total_samples):
            sample = self.deeplake_ds[i]
            label = int(sample.labels.numpy()[0])
            all_labels.append(label)
            
        print(f"All labels: {all_labels}")
        
        # Create stratified indices
        label_to_indices = defaultdict(list)
        for idx in range(total_samples):
            label = all_labels[idx]
            label_to_indices[label].append(idx)
        
        print(f"Label to indices: {label_to_indices}")
        
        # Print statistics for each class
        print(f"\nClass distribution:")
        for label_id, indices in label_to_indices.items():
            label_name = self.label_id_to_name[label_id]
            print(f"  {label_name} (ID: {label_id}): {len(indices)} samples")
        
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
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our split indices
        actual_idx = self.indices[idx]
        
        # Get image and label from Deep Lake dataset
        sample = self.deeplake_ds[actual_idx]      
        
        image = sample.images.numpy()
        label = int(sample.labels.numpy()[0])
        
        pil_image = Image.fromarray(image, mode='RGB')

        original_image = pil_image.resize((256, 256))
        original_image = np.array(original_image)
        
        transformed_image = self.transform(pil_image)

        return transformed_image, label, original_image
    
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
            label = int(sample.labels.numpy()[0])
            distribution[label] += 1
        return dict(distribution)