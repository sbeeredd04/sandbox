#!/bin/bash

# verify_imagenet.sh - Verify ImageNet dataset installation

echo "======================================"
echo "ImageNet Dataset Verification Script"
echo "======================================"
echo ""

# Set the base directory - try multiple locations
if [ -n "$1" ]; then
    IMAGENET_DIR="$1"
elif [ -d "/scratch/sbeeredd/sandbox/imagenet" ]; then
    IMAGENET_DIR="/scratch/sbeeredd/sandbox/imagenet"
elif [ -d "$HOME/sandbox/imagenet" ]; then
    IMAGENET_DIR="$HOME/sandbox/imagenet"
elif [ -d "./imagenet" ]; then
    IMAGENET_DIR="./imagenet"
else
    echo "ERROR: Could not find ImageNet directory"
    echo "Usage: $0 [path_to_imagenet]"
    echo "Checked locations:"
    echo "  /scratch/sbeeredd/sandbox/imagenet"
    echo "  $HOME/sandbox/imagenet"
    echo "  ./imagenet"
    exit 1
fi

echo "Checking ImageNet directory: $IMAGENET_DIR"
echo ""

# Check if directory exists
if [ ! -d "$IMAGENET_DIR" ]; then
    echo "ERROR: ImageNet directory not found at $IMAGENET_DIR"
    exit 1
fi

echo "✓ ImageNet directory exists"
echo ""

# Expected counts from the code
EXPECTED_TRAIN=1281167
EXPECTED_VAL=50000
EXPECTED_TRAIN_SYNSETS=1000
EXPECTED_VAL_SYNSETS=1000

# Check training directory
echo "--- TRAINING SET ---"
if [ -d "$IMAGENET_DIR/train" ]; then
    echo "✓ Training directory exists"
    
    # Count synsets (subdirectories)
    train_synsets=$(find "$IMAGENET_DIR/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Synset folders: $train_synsets (expected: $EXPECTED_TRAIN_SYNSETS)"
    
    # Count images
    echo "  Counting JPEG images (this may take a while)..."
    train_images=$(find "$IMAGENET_DIR/train" -type f -name "*.JPEG" 2>/dev/null | wc -l)
    echo "  Training images: $train_images (expected: $EXPECTED_TRAIN)"
    
    if [ "$train_images" -eq "$EXPECTED_TRAIN" ]; then
        echo "  ✓ Training set complete!"
    elif [ "$train_images" -gt 0 ]; then
        echo "  ⚠ WARNING: Found $train_images images, expected $EXPECTED_TRAIN"
        missing=$((EXPECTED_TRAIN - train_images))
        echo "  Missing: $missing images"
    else
        echo "  ✗ ERROR: No training images found"
    fi
    
    # Show sample synset
    sample_synset=$(find "$IMAGENET_DIR/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)
    if [ ! -z "$sample_synset" ]; then
        sample_count=$(find "$sample_synset" -type f -name "*.JPEG" 2>/dev/null | wc -l)
        echo "  Sample synset: $(basename $sample_synset) with $sample_count images"
    fi
else
    echo "✗ Training directory NOT found"
fi

echo ""
echo "--- VALIDATION SET ---"
if [ -d "$IMAGENET_DIR/val" ]; then
    echo "✓ Validation directory exists"
    
    # Count synsets (subdirectories)
    val_synsets=$(find "$IMAGENET_DIR/val" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Synset folders: $val_synsets (expected: $EXPECTED_VAL_SYNSETS)"
    
    # Count images
    echo "  Counting JPEG images..."
    val_images=$(find "$IMAGENET_DIR/val" -type f -name "*.JPEG" 2>/dev/null | wc -l)
    echo "  Validation images: $val_images (expected: $EXPECTED_VAL)"
    
    if [ "$val_images" -eq "$EXPECTED_VAL" ]; then
        echo "  ✓ Validation set complete!"
    elif [ "$val_images" -gt 0 ]; then
        echo "  ⚠ WARNING: Found $val_images images, expected $EXPECTED_VAL"
        missing=$((EXPECTED_VAL - val_images))
        echo "  Missing: $missing images"
    else
        echo "  ✗ ERROR: No validation images found"
    fi
    
    # Show sample synset
    sample_synset=$(find "$IMAGENET_DIR/val" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)
    if [ ! -z "$sample_synset" ]; then
        sample_count=$(find "$sample_synset" -type f -name "*.JPEG" 2>/dev/null | wc -l)
        echo "  Sample synset: $(basename $sample_synset) with $sample_count images"
    fi
else
    echo "✗ Validation directory NOT found"
fi

echo ""
echo "--- DIRECTORY STRUCTURE ---"
echo "Expected structure:"
echo "  imagenet/"
echo "  ├── train/"
echo "  │   ├── n01440764/  (synset folder)"
echo "  │   │   ├── n01440764_*.JPEG"
echo "  │   ├── n01443537/"
echo "  │   │   └── ..."
echo "  │   └── ... (1000 synset folders total)"
echo "  └── val/"
echo "      ├── n01440764/"
echo "      │   ├── ILSVRC2012_val_*.JPEG"
echo "      ├── n01443537/"
echo "      │   └── ..."
echo "      └── ... (1000 synset folders total)"
echo ""

echo "--- DISK USAGE ---"
du -sh "$IMAGENET_DIR" 2>/dev/null
du -sh "$IMAGENET_DIR/train" 2>/dev/null
du -sh "$IMAGENET_DIR/val" 2>/dev/null

echo ""
echo "--- ADDITIONAL FILES ---"
# Check for metadata files
if [ -f "$IMAGENET_DIR/dataset.json" ]; then
    echo "✓ dataset.json found"
fi

# List top-level files
echo "Files in $IMAGENET_DIR:"
ls -lh "$IMAGENET_DIR" 2>/dev/null | grep -v "^d" | awk '{if (NF>=9) print "  " $9, "(" $5 ")"}'

echo ""
echo "======================================"
echo "Verification Complete"
echo "======================================"