import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import random
import cv2

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
            
        print(f"All original labels: {all_original_labels}")
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
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
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