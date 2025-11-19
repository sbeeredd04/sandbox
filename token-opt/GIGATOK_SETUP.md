# GigaTok Setup Instructions

## Quick Start

GigaTok pretrained models are automatically downloaded from Google Drive when you first use the wrapper.

### Requirements

Install the `gdown` package for automatic checkpoint downloads:

```bash
pip install gdown
```

### Usage

```python
from tto.gigatok_wrapper import GigaTokWrapper

# Auto-downloads pretrained weights on first run
model = GigaTokWrapper(config_name='BL256')  # 622M parameters

# Or use the larger model
model = GigaTokWrapper(config_name='XLXXL256')  # 3B parameters
```

## Available Models

| Model | Parameters | Codebook Size | Download Size | rFID | LPIPS |
|-------|-----------|---------------|---------------|------|-------|
| BL256 | 622M | 16,384 | ~500MB | 0.81 | 0.2059 |
| XLXXL256 | 2.9B | 16,384 | ~3GB | 0.79 | 0.1947 |

## Manual Download (if auto-download fails)

If automatic download fails:

### BL256 Model
1. Visit: https://drive.google.com/file/d/1VnBu4aj0N7lFa1_wKBdTgLNZmaNgFvwi/view
2. Download the file
3. Save as: `pretrained/gigatok/VQ_BL256_e200.pt`

### XLXXL256 Model
1. Visit: https://drive.google.com/file/d/1HK_bV_zklLfGmIHGE4gMwjfhLLKi6Z3G/view
2. Download the file
3. Save as: `pretrained/gigatok/VQ_XLXXL256_e300.pt`

Then use:
```python
model = GigaTokWrapper(
    config_name='BL256',
    checkpoint_path='pretrained/gigatok/VQ_BL256_e200.pt'
)
```

## Troubleshooting

### Error: "gdown library required for automatic download"

**Solution:** Install gdown:
```bash
pip install gdown
```

### Error: "invalid load key, '<'"

This means the download got an HTML page instead of the checkpoint file.

**Solution:**
1. Install gdown: `pip install gdown`
2. Or manually download the checkpoint (see above)

### Error: "Downloaded file does not appear to be a valid checkpoint"

The download may have been corrupted.

**Solution:**
1. Delete the corrupted file in `pretrained/gigatok/`
2. Re-run your code to retry the download
3. Or manually download the checkpoint

## For Colab Users

The notebook automatically installs gdown:
```python
%pip install -q jaxtyping open_clip_torch omegaconf timm safetensors huggingface_hub gdown
```

## Expected Behavior

On first run, you should see:
```
Using predefined config: BL256
  Base encoder, Large decoder, 256 tokens (622M params)
  
No checkpoint provided, auto-downloading pretrained weights...
Downloading BL256 checkpoint from Google Drive...
  File ID: 1VnBu4aj0N7lFa1_wKBdTgLNZmaNgFvwi
  Destination: pretrained/gigatok/VQ_BL256_e200.pt
  This may take a few minutes (file is large ~500MB-3GB)...
Downloading...
From: https://drive.google.com/uc?id=1VnBu4aj0N7lFa1_wKBdTgLNZmaNgFvwi
To: pretrained/gigatok/VQ_BL256_e200.pt
100%|██████████| 500M/500M [01:23<00:00, 6.01MB/s]
  Download completed successfully using gdown
  Validating downloaded checkpoint...
  Checkpoint validated successfully

Loading checkpoint from pretrained/gigatok/VQ_BL256_e200.pt...
  Using EMA weights
Checkpoint loaded successfully
```

On subsequent runs:
```
Using cached checkpoint: pretrained/gigatok/VQ_BL256_e200.pt
```

## Performance Expectations

With pretrained weights, you should see:
- **PSNR > 25 dB** for reconstructed images
- **MSE < 0.01** for reconstruction error
- Visually accurate reconstructions preserving fine details

Without pretrained weights (random initialization):
- Reconstructions will be garbage/noise
- This is expected behavior - VQ-VAE models require training

## References

- Paper: [GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters](https://arxiv.org/abs/2504.08736)
- Project Page: https://silentview.github.io/GigaTok/
- Original Repo: https://github.com/FoundationVision/GigaTok

