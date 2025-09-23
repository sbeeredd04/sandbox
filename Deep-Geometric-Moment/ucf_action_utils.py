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

# Import YOLO-World and SAM
from ultralytics import YOLO
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
        
        # Initialize YOLO-World model
        self.yolo_model = YOLO('yolov8s-world.pt')
        
        # Initialize SAM model
        model_type = "vit_b"
        if model_type == "vit_h":
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_h_4b8939.pth")
        elif model_type == "vit_l":
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_l_0b3195.pth")
        else:  # vit_b
            checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "SAM", "sam_vit_b_01ec64.pth")
                    
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.cuda()
        self.predictor = SamPredictor(sam)
        
        # Debug settings
        self.debug_save_count = 0
        self.max_debug_saves = 0
        self.debug_dir = "debug_pipeline"
        os.makedirs(self.debug_dir, exist_ok=True)
    
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
        
        # Set YOLO-World classes based on our dataset classes
        print(f"Setting YOLO-World classes: {self.class_names}")
        self.yolo_model.set_classes(self.class_names)
    
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
        original_image_np = np.array(original_image)
        
        # Get class name for GroundingDINO
        class_name = self.get_class_name(label)
        
        # Debug saving flag - save first few samples for debugging
        should_debug = self.debug_save_count < self.max_debug_saves
        debug_idx = self.debug_save_count if should_debug else -1
        
        if should_debug:
            self.debug_save_count += 1
            # Step 1: Save original image
            cv2.imwrite(f"{self.debug_dir}/step1_original_{debug_idx}_{class_name}.jpg", 
                       cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))
            print(f"Debug {debug_idx}: Saved original image for {class_name}")
        
        # Apply YOLO-World processing
        # Save image temporarily for YOLO-World (it expects file path or PIL image)
        temp_image_path = f"temp_image_{idx}.jpg"
        original_image.save(temp_image_path)
        
        # Use YOLO-World to detect objects
        results = self.yolo_model.predict(temp_image_path, verbose=False)
        
        # Extract bounding boxes from YOLO-World results
        boxes = []
        confidences = []
        class_names_detected = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                # Convert from YOLO format (x1, y1, x2, y2) to required format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(box.conf[0].cpu().numpy()))
                class_id = int(box.cls[0].cpu().numpy())
                class_names_detected.append(self.yolo_model.names[class_id])
        
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        if should_debug:
            # Step 2: Save YOLO-World detection results
            if len(boxes) > 0:
                # Draw bounding boxes on image for visualization
                annotated_image_np = original_image_np.copy()
                for i, (box, conf, cls_name) in enumerate(zip(boxes, confidences, class_names_detected)):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image_np, f"{cls_name}: {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imwrite(f"{self.debug_dir}/step2_yoloworld_{debug_idx}_{class_name}.jpg", 
                           cv2.cvtColor(annotated_image_np, cv2.COLOR_RGB2BGR))
                print(f"Debug {debug_idx}: Found {len(boxes)} boxes for {class_name}: {class_names_detected}")
            else:
                print(f"Debug {debug_idx}: No boxes found for {class_name}")
        
        # Use SAM model to segment the image
        self.predictor.set_image(original_image_np)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(boxes[0]) if len(boxes) > 0 else None,  # Convert to numpy array and handle case where no boxes found
        )
        
        # Convert mask to proper input format
        if len(masks) > 0:
            mask = masks[0]  # Take first mask
            
            if should_debug:
                # Step 3: Save SAM mask
                cv2.imwrite(f"{self.debug_dir}/step3_sam_mask_{debug_idx}_{class_name}.jpg", 
                           (mask * 255).astype(np.uint8))
                print(f"Debug {debug_idx}: Saved SAM mask for {class_name}")
            
            # Create masked image where background is zeroed out
            masked_image_np = original_image_np.copy().astype(np.float32)
            # Apply mask to each channel - zero out background
            for c in range(3):
                masked_image_np[:, :, c] = masked_image_np[:, :, c] * mask
            
            # Convert back to uint8
            masked_image_np = masked_image_np.astype(np.uint8)
            
            if should_debug:
                # Step 4: Save masked image (background zeroed out)
                cv2.imwrite(f"{self.debug_dir}/step4_masked_bg_zero_{debug_idx}_{class_name}.jpg", 
                           cv2.cvtColor(masked_image_np, cv2.COLOR_RGB2BGR))
                print(f"Debug {debug_idx}: Saved masked image (background zeroed) for {class_name}")
            
            # Convert masked image to PIL and apply transforms
            masked_image_pil = Image.fromarray(masked_image_np)
            transformed_image = self.transform(masked_image_pil)
            
            if should_debug:
                # Step 5: Save final transformed tensor as image for visualization
                # Convert tensor back to image format for saving
                tensor_for_save = transformed_image.clone()
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor_for_save = tensor_for_save * std + mean
                tensor_for_save = torch.clamp(tensor_for_save, 0, 1)
                
                # Convert to numpy and save
                final_image_np = (tensor_for_save.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"{self.debug_dir}/step5_final_tensor_{debug_idx}_{class_name}.jpg", 
                           cv2.cvtColor(final_image_np, cv2.COLOR_RGB2BGR))
                print(f"Debug {debug_idx}: Saved final transformed tensor for {class_name}")
        else:
            # If no mask found, use original image
            transformed_image = self.transform(pil_image)
            
            if should_debug:
                print(f"Debug {debug_idx}: No mask found, using original image for {class_name}")
                # Save original as final since no mask was applied
                cv2.imwrite(f"{self.debug_dir}/step4_no_mask_{debug_idx}_{class_name}.jpg", 
                           cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR))

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