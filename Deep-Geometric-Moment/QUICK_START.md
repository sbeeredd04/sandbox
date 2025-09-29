# UCF Sports Action - Quick Start Guide

## 🚀 Quick Start

### 1. First Time Setup (with preprocessing)

```bash
# Basic training run - will automatically preprocess on first run
python main.py \
    --dataset ucf_sports \
    --train-batch 32 \
    --test-batch 32 \
    --epochs 150 \
    --lr 0.001 \
    --workers 4 \
    --gpu-id 0,1 \
    --ucf-owlvit-device 8 \
    --ucf-sam-device 9
```

**What happens:**
1. ✅ Loads UCF Sports dataset from Deep Lake
2. ✅ Runs OWL-ViT detection on GPU 8
3. ✅ Runs SAM segmentation on GPU 9
4. ✅ Saves preprocessed images to `./data/ucf_preprocessed/`
5. ✅ Starts training on GPU 0,1

**Time estimate:**
- Preprocessing: ~5-10 minutes (one time only!)
- Training per epoch: ~30 seconds

---

### 2. Subsequent Training Runs (fast!)

```bash
# All subsequent runs are MUCH faster - loads from disk
python main.py \
    --dataset ucf_sports \
    --train-batch 32 \
    --epochs 150 \
    --gpu-id 0,1
```

**What happens:**
1. ✅ Detects existing preprocessed data
2. ✅ Loads from disk (very fast!)
3. ✅ Starts training immediately

**Time estimate:**
- Dataset loading: ~1 second
- Training per epoch: ~30 seconds

---

### 3. Force Reprocessing

```bash
# If you want to reprocess the data
python main.py \
    --dataset ucf_sports \
    --ucf-force-preprocess \
    --train-batch 32 \
    --epochs 150
```

---

## 📊 Understanding the Output

### Preprocessing Output
```
================================================================================
PREPROCESSING UCF SPORTS DATASET
================================================================================

✓ Original classes: 13 -> Grouped classes: 10
✓ Grouped class names: ['Diving', 'Golf-Swing', 'Kicking', ...]

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
Processing images: 100%|████████████| 200/200 [02:15<00:00,  1.48it/s]

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

### Training Output
```
================================================================================
Initializing UCF Sports Dataset - TRAIN split
================================================================================
✓ Loading preprocessed data from ./data/ucf_preprocessed
✓ Loaded 187 preprocessed images

================================================================================
Creating stratified train/test split (90/10)
================================================================================
  Diving        (ID:  0):  15 train,   2 test  (total:  17)
  Golf-Swing    (ID:  1):  20 train,   3 test  (total:  23)
  Kicking       (ID:  2):  18 train,   2 test  (total:  20)
  ...

✓ Train split: 168 samples

Class distribution (train split):
  Diving        (ID:  0):  15 samples ( 8.9%)
  Golf-Swing    (ID:  1):  20 samples (11.9%)
  ...
```

---

## 📁 File Structure After Preprocessing

```
./data/ucf_preprocessed/
├── img_00000_label_0.png          # Preprocessed image
├── img_00001_label_3.png
├── img_00002_label_1.png
├── ...
├── metadata.pkl                    # Python pickle with metadata
└── preprocessing_stats.json        # Human-readable statistics
```

---

## 🔧 Command-Line Options

### Required Options
```bash
--dataset ucf_sports               # Use UCF Sports dataset
```

### Training Options
```bash
--train-batch 32                   # Training batch size (default: 128)
--test-batch 32                    # Test batch size (default: 100)
--epochs 150                       # Number of epochs (default: 150)
--lr 0.001                         # Learning rate (default: 0.1)
--workers 4                        # Number of data loading workers (default: 4)
```

### GPU Options
```bash
--gpu-id 0,1                       # GPUs for training (e.g., "0,1,2,3")
--ucf-owlvit-device 8             # GPU for OWL-ViT during preprocessing (default: 8)
--ucf-sam-device 9                # GPU for SAM during preprocessing (default: 9)
```

### Preprocessing Options
```bash
--ucf-data-dir ./data/my_folder   # Custom preprocessed data directory
--ucf-force-preprocess            # Force reprocessing even if data exists
```

### Other Options
```bash
--checkpoint ./my_checkpoint      # Directory to save checkpoints
--resume ./checkpoint/xxx.pth.tar # Resume from checkpoint
--evaluate                        # Evaluation only (no training)
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Test Run (5 epochs)
```bash
python main.py --dataset ucf_sports --epochs 5 --train-batch 16
```

### Use Case 2: Full Training Run
```bash
python main.py \
    --dataset ucf_sports \
    --epochs 150 \
    --train-batch 32 \
    --lr 0.001 \
    --workers 4 \
    --gpu-id 0,1,2,3
```

### Use Case 3: Resume Training
```bash
python main.py \
    --dataset ucf_sports \
    --resume ./checkpoint/checkpoint.pth.tar \
    --epochs 200
```

### Use Case 4: Evaluation Only
```bash
python main.py \
    --dataset ucf_sports \
    --evaluate \
    --resume ./checkpoint/model_best.pth.tar
```

### Use Case 5: Inference on New Images
```bash
python main.py \
    --model-path ./checkpoint/model_best.pth.tar \
    --image-path ./test_images/ \
    --dataset ucf_sports
```

---

## 📈 Expected Performance

### Preprocessing (One Time)
- **Time**: ~5-10 minutes (depends on GPU)
- **Disk space**: ~50-100 MB (187 images as PNG)
- **Success rate**: ~93-95% (some images filtered due to no detection)

### Training (Per Epoch)
- **Time**: ~30 seconds (batch size 32, 4 workers)
- **GPU memory**: ~2-3 GB per GPU
- **Accuracy**: ~70-80% (depends on hyperparameters)

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory" during preprocessing
**Solution:** Use smaller GPUs or reduce batch size
```bash
--ucf-owlvit-device 0  # Use a different GPU
```

### Problem: "Preprocessed data not found"
**Solution:** Let it preprocess automatically, or force reprocess
```bash
--ucf-force-preprocess
```

### Problem: "Too many dropped images"
**Solution:** Check detection threshold in `ucf_action_utils.py` line 398
```python
confidence_threshold = 0.1  # Lower this to keep more images
```

### Problem: Training is slow
**Solution:** Increase number of workers
```bash
--workers 8  # More workers = faster data loading
```

---

## 📚 Additional Resources

- **Full documentation**: See `PREPROCESSING_UPDATE_SUMMARY.md`
- **Example code**: Run `./example_ucf_sports_usage.py`
- **Preprocessing stats**: Check `./data/ucf_preprocessed/preprocessing_stats.json`

---

## ✅ Checklist

Before running:
- [ ] Have access to GPUs 8 and 9 (for preprocessing)
- [ ] Have sufficient disk space (~100 MB)
- [ ] Installed required packages: `torch`, `transformers`, `segment-anything`, `deeplake`

After first run:
- [ ] Check preprocessing stats in `./data/ucf_preprocessed/preprocessing_stats.json`
- [ ] Verify class distribution looks reasonable
- [ ] Inspect a few preprocessed images manually

During training:
- [ ] Monitor WandB for training progress
- [ ] Check GPU utilization
- [ ] Save best checkpoint

---

## 🎓 Key Concepts

1. **Preprocessing**: OWL-ViT detects objects, SAM segments them, background is removed
2. **Class Grouping**: 13 original classes grouped into 10 classes (e.g., Golf-Swing-Front → Golf-Swing)
3. **Stratified Split**: Each class split 90/10 for train/test
4. **Disk Caching**: Preprocessed images saved to disk, loaded during training (very fast!)

---

Happy Training! 🚀
