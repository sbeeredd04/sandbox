# UCF Sports Dataset Preprocessing Update Summary

## Overview
The UCF Sports Action dataset processing has been completely rewritten to use a **preprocessing-first** architecture instead of on-the-fly processing.

## Key Changes

### 1. **ucf_action_utils.py** - Complete Rewrite

#### Architecture Change
- **Before**: Images processed during training in `__getitem__` (slow, repeated work)
- **After**: All images preprocessed once and saved to disk, `__getitem__` loads from disk (fast!)

#### New Preprocessing Pipeline
1. Load all images from Deep Lake dataset
2. Run OWL-ViT object detection with class-specific text queries
3. Filter out images with no detections
4. Run SAM segmentation on detected bounding boxes
5. Apply mask to remove background
6. Save preprocessed images to `./data/ucf_preprocessed/`
7. Save metadata (class mappings, statistics, etc.)

#### GPU Device Assignment
- **OWL-ViT**: GPU device 8 (`cuda:8`)
- **SAM**: GPU device 9 (`cuda:9`)
- No more CUDA multiprocessing conflicts!

#### New Functions

**Helper Functions (Class Mappings)**
- `get_class_name_mapping()` - Convert original label ID to grouped class name
- `get_grouped_class_id_mapping()` - Convert original label ID to grouped class ID
- `get_class_name_simple()` - Convert label ID to class name
- `get_ucf_class_mappings()` - Get class groupings and mappings
- `get_ucf_sports_transforms()` - Image transforms for preprocessed data
- `get_owlvit_text_queries()` - Text queries for OWL-ViT detection

**Model Initialization**
- `initialize_owlvit_model(device_id=8)` - Initialize OWL-ViT on specific GPU
- `initialize_sam_model(device_id=9)` - Initialize SAM on specific GPU

**Preprocessing**
- `detect_and_segment_image()` - Run detection + segmentation on single image
- `preprocess_and_save_dataset()` - Main preprocessing function

**Dataset Class**
- `UCFSportsDataset` - Simplified dataset that loads preprocessed images
  - Automatic preprocessing detection
  - Stratified train/test splits (90/10)
  - Comprehensive logging and statistics
  - Class distribution reporting

**DataLoader**
- `create_dataloader()` - Simple dataloader creation (no more CUDA workarounds needed)

#### Removed Functions
- ❌ `collate_fn_skip_none` - No longer needed
- ❌ `setup_multiprocessing_for_cuda` - No longer needed
- ❌ `create_cuda_safe_dataloader` - Replaced with `create_dataloader`
- ❌ Lazy model initialization code - Models only initialized during preprocessing

---

### 2. **train_utils.py** - Updated for Preprocessed Data

#### Changes to `train_ucf_sports()`
- **Data unpacking**: Changed from `(inputs, targets, image)` to `(inputs, targets)`
- **Image logging**: Updated to use preprocessed images only (no original images)
- **Class names**: Now logs class names instead of just IDs
- **Efficiency**: Limits image logging to 4 images per batch maximum

#### Changes to `test_ucf_sports()`
- **Data unpacking**: Changed from `(inputs, targets, image)` to `(inputs, targets)`
- **Image logging**: Updated to use preprocessed images only
- **Class names**: Now logs class names instead of just IDs
- **Efficiency**: Limits image logging to 4 images per batch maximum

#### New Logging Format
- Shows preprocessed image (background removed)
- Shows IMGR1-4 visualizations
- Includes class names in captions
- More informative labels

---

### 3. **main.py** - Updated Imports and Dataset Creation

#### Import Changes
```python
# Before
from ucf_action_utils import UCFSportsDataset, get_ucf_sports_transforms, setup_multiprocessing_for_cuda, create_cuda_safe_dataloader

# After
from ucf_action_utils import UCFSportsDataset, get_ucf_sports_transforms, create_dataloader
```

#### Removed Code
- Removed `setup_multiprocessing_for_cuda()` call

#### New Command-Line Arguments
- `--ucf-data-dir` - Directory for preprocessed data (default: `./data/ucf_preprocessed`)
- `--ucf-force-preprocess` - Force reprocessing even if data exists
- `--ucf-owlvit-device` - GPU device for OWL-ViT (default: 8)
- `--ucf-sam-device` - GPU device for SAM (default: 9)

#### Updated Dataset Creation
```python
# Create train dataset with preprocessing options
trainset = UCFSportsDataset(
    ds, 
    split='train', 
    transform=transform, 
    use_grouped_classes=True,
    data_dir=args.ucf_data_dir,
    force_preprocess=args.ucf_force_preprocess,
    owlvit_device_id=args.ucf_owlvit_device,
    sam_device_id=args.ucf_sam_device
)

# Create dataloader (no special CUDA handling needed)
train_loader = create_dataloader(
    trainset, 
    batch_size=args.train_batch, 
    shuffle=True, 
    num_workers=args.workers
)
```

---

## Usage Examples

### Basic Training (with automatic preprocessing)
```bash
python main.py \
    --dataset ucf_sports \
    --train-batch 32 \
    --test-batch 32 \
    --epochs 150 \
    --lr 0.001 \
    --workers 4 \
    --gpu-id 0,1
```

### Force Repreprocessing
```bash
python main.py \
    --dataset ucf_sports \
    --ucf-force-preprocess \
    --ucf-owlvit-device 8 \
    --ucf-sam-device 9 \
    --train-batch 32 \
    --epochs 150
```

### Custom Preprocessing Directory
```bash
python main.py \
    --dataset ucf_sports \
    --ucf-data-dir ./data/my_custom_preprocessed \
    --train-batch 32 \
    --epochs 150
```

### Inference on Preprocessed Model
```bash
python main.py \
    --model-path ./checkpoint/model_best.pth.tar \
    --image-path ./test_images/ \
    --dataset ucf_sports
```

---

## Benefits of New Architecture

### Performance
✅ **Much faster training** - No on-the-fly OWL-ViT + SAM processing  
✅ **Consistent preprocessing** - All images processed the same way  
✅ **Faster iteration** - Preprocessed data reused across multiple training runs  

### Reliability
✅ **No CUDA conflicts** - Models only run during preprocessing phase  
✅ **Better error handling** - Preprocessing errors caught before training  
✅ **Reproducible results** - Same preprocessed data every time  

### Debugging
✅ **Comprehensive logging** - Statistics at every step  
✅ **Dataset inspection** - Can manually inspect preprocessed images  
✅ **Class distribution** - Clear reporting of data splits  

### Scalability
✅ **Parallel processing ready** - Can easily add multiprocessing to preprocessing  
✅ **Storage efficient** - Preprocessed images saved as compressed PNG  
✅ **Flexible** - Easy to swap out detection/segmentation models  

---

## File Structure

```
./data/ucf_preprocessed/
├── img_00000_label_0.png          # Preprocessed image (background removed)
├── img_00001_label_3.png
├── ...
├── metadata.pkl                    # Python pickle with all metadata
└── preprocessing_stats.json        # Human-readable statistics
```

### Metadata Contents
- `processed_indices` - List of successfully processed images
- `dropped_count` - Number of images filtered out (no detection)
- `total_original` - Total images in original dataset
- `label_id_to_name` - Label ID to class name mapping
- `class_names` - List of class names
- `use_grouped_classes` - Whether grouped classes were used

---

## Class Grouping

The dataset groups similar action classes together to reduce the number of classes:

### Grouped Classes (10 total)
1. **Diving** - Diving-Side
2. **Golf-Swing** - Golf-Swing-Front, Golf-Swing-Side, Golf-Swing-Back
3. **Kicking** - Kicking-Front, Kicking-Side
4. **Lifting** - Lifting
5. **Riding-Horse** - Riding-Horse
6. **Run** - Run-Side
7. **SkateBoarding** - SkateBoarding-Front
8. **Swing-Bench** - Swing-Bench
9. **Swing-SideAngle** - Swing-SideAngle
10. **Walk** - Walk-Front

This reduces from **13 original classes to 10 grouped classes**.

---

## Expected Preprocessing Output

```
================================================================================
PREPROCESSING UCF SPORTS DATASET
================================================================================

================================================================================
Initializing OWL-ViT model on GPU 8...
================================================================================
✓ OWL-ViT model loaded on cuda:8

================================================================================
Initializing SAM model (vit_b) on GPU 9...
================================================================================
✓ SAM model loaded on cuda:9

================================================================================
Processing 200 images...
================================================================================
Processing images: 100%|████████████████| 200/200 [02:15<00:00,  1.48it/s]

================================================================================
PREPROCESSING COMPLETE
================================================================================
✓ Total original images: 200
✓ Successfully processed: 187
✗ Dropped (no detection): 13
✗ Drop rate: 6.5%
✓ Saved to: ./data/ucf_preprocessed
================================================================================
```

---

## Notes

- Preprocessing happens **only once** on first run
- Subsequent runs load from disk (very fast!)
- Use `--ucf-force-preprocess` to reprocess
- Preprocessing uses GPUs 8 and 9 by default
- Training can use any GPUs via `--gpu-id`
- All class mappings preserved and saved to disk
