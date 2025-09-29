#!/usr/bin/env python3
"""
Example script demonstrating the new UCF Sports preprocessing pipeline.

This shows how to:
1. Load the dataset with automatic preprocessing
2. Inspect preprocessed data
3. Train a model efficiently
"""

import torch
import deeplake
from ucf_action_utils import (
    UCFSportsDataset, 
    get_ucf_sports_transforms, 
    create_dataloader,
    get_ucf_class_mappings
)

# ============================================================================
# Example 1: Basic Dataset Creation with Auto-Preprocessing
# ============================================================================

def example_basic_usage():
    """Most common use case - automatic preprocessing on first run."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with Auto-Preprocessing")
    print("="*80)
    
    # Load Deep Lake dataset
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    
    # Get transforms
    transform = get_ucf_sports_transforms()
    
    # Create dataset (automatically preprocesses if needed)
    train_dataset = UCFSportsDataset(
        ds,
        split='train',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed',
        force_preprocess=False,  # Only preprocess if data doesn't exist
        owlvit_device_id=8,      # OWL-ViT on GPU 8
        sam_device_id=9          # SAM on GPU 9
    )
    
    # Create dataloader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    print(f"\n✓ Dataset created successfully!")
    print(f"✓ Number of training samples: {len(train_dataset)}")
    print(f"✓ Number of classes: {train_dataset.get_num_classes()}")
    print(f"✓ Class names: {train_dataset.get_all_class_names()}")
    
    # Test loading a batch
    print(f"\nTesting batch loading...")
    for images, labels in train_loader:
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Sample class: {train_dataset.get_class_name(labels[0].item())}")
        break
    
    return train_dataset, train_loader


# ============================================================================
# Example 2: Force Reprocessing
# ============================================================================

def example_force_reprocess():
    """Force reprocessing even if preprocessed data exists."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Force Reprocessing")
    print("="*80)
    
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    transform = get_ucf_sports_transforms()
    
    # Force reprocessing
    train_dataset = UCFSportsDataset(
        ds,
        split='train',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed_v2',
        force_preprocess=True,  # Force reprocessing
        owlvit_device_id=8,
        sam_device_id=9
    )
    
    print(f"\n✓ Reprocessing complete!")
    return train_dataset


# ============================================================================
# Example 3: Using Class Mappings
# ============================================================================

def example_class_mappings():
    """Explore class mappings and groupings."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Class Mappings")
    print("="*80)
    
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    
    # Get class mappings
    class_mappings = get_ucf_class_mappings(ds, group_similar=True)
    (grouped_class_names, original_to_grouped_id, grouped_to_original_ids, 
     grouped_id_to_label, grouping_rules) = class_mappings
    
    print(f"\nOriginal to Grouped Mapping:")
    print(f"{'Original ID':<12} {'Original Name':<25} {'Grouped ID':<12} {'Grouped Name'}")
    print("-" * 80)
    
    original_class_names = ds.labels.info['class_names']
    for orig_id, orig_name in enumerate(original_class_names):
        grouped_id = original_to_grouped_id[orig_id]
        grouped_name = grouped_id_to_label[grouped_id]
        print(f"{orig_id:<12} {orig_name:<25} {grouped_id:<12} {grouped_name}")
    
    print(f"\nGrouped Class Distribution:")
    for grouped_id, grouped_name in sorted(grouped_id_to_label.items()):
        original_ids = grouped_to_original_ids[grouped_id]
        original_names = [original_class_names[i] for i in original_ids]
        print(f"  {grouped_name}: {', '.join(original_names)}")


# ============================================================================
# Example 4: Creating Train and Test Sets
# ============================================================================

def example_train_test_split():
    """Create both train and test datasets."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Train/Test Split")
    print("="*80)
    
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    transform = get_ucf_sports_transforms()
    
    # Train dataset
    train_dataset = UCFSportsDataset(
        ds,
        split='train',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed'
    )
    
    # Test dataset (uses same preprocessed data)
    test_dataset = UCFSportsDataset(
        ds,
        split='test',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed',
        force_preprocess=False  # Don't reprocess for test set
    )
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Total samples: {len(train_dataset) + len(test_dataset)}")
    print(f"✓ Train/Test ratio: {len(train_dataset) / len(test_dataset):.1f}")
    
    return train_loader, test_loader


# ============================================================================
# Example 5: Inspecting Preprocessed Data
# ============================================================================

def example_inspect_data():
    """Load and inspect individual preprocessed samples."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Inspecting Preprocessed Data")
    print("="*80)
    
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    transform = get_ucf_sports_transforms()
    
    dataset = UCFSportsDataset(
        ds,
        split='train',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed'
    )
    
    print(f"\nInspecting first 5 samples:")
    for i in range(min(5, len(dataset))):
        image, label = dataset[i]
        class_name = dataset.get_class_name(label)
        print(f"  Sample {i}: shape={image.shape}, label={label}, class={class_name}")


# ============================================================================
# Example 6: Simple Training Loop
# ============================================================================

def example_training_loop():
    """Demonstrate a simple training loop with preprocessed data."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Simple Training Loop")
    print("="*80)
    
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
    transform = get_ucf_sports_transforms()
    
    # Create dataset and dataloader
    train_dataset = UCFSportsDataset(
        ds,
        split='train',
        transform=transform,
        use_grouped_classes=True,
        data_dir='./data/ucf_preprocessed'
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    
    print(f"\nSimulating training loop (1 epoch, 3 batches)...")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 3:  # Only show 3 batches
            break
        
        # Simulate training
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")
        print(f"  Classes in batch: {[train_dataset.get_class_name(l.item()) for l in labels[:3]]}")
        
        # In real training, you would:
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
    
    print(f"\n✓ Training loop demonstration complete!")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("UCF SPORTS PREPROCESSING PIPELINE - EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Basic usage (most common)
        train_dataset, train_loader = example_basic_usage()
        
        # Example 3: Class mappings
        example_class_mappings()
        
        # Example 4: Train/test split
        train_loader, test_loader = example_train_test_split()
        
        # Example 5: Inspect data
        example_inspect_data()
        
        # Example 6: Training loop
        example_training_loop()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run actual training: python main.py --dataset ucf_sports --epochs 150")
        print("  2. Inspect preprocessed images in: ./data/ucf_preprocessed/")
        print("  3. Check preprocessing stats: ./data/ucf_preprocessed/preprocessing_stats.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
