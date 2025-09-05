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
        
        # Get all labels first to create stratified split
        all_labels = []
        for i in range(total_samples):
            sample = self.deeplake_ds[i]
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
        
        if self.split == 'train':
            self.indices = train_indices
        else:  # test
            self.indices = test_indices
    
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
