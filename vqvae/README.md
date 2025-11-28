# Vector Quantized Variational Autoencoder

This is a PyTorch implementation of the vector quantized variational autoencoder (https://arxiv.org/abs/1711.00937) with support for Deep Geometric Moments (DGM) auxiliary loss.

You can find the author's [original implementation in Tensorflow here](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py) with [an example you can run in a Jupyter notebook](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb).

## Installing Dependencies

To install dependencies, create a conda or virtual environment with Python 3 and then run `pip install -r requirements.txt`.

## Dataset Support

This implementation supports multiple datasets:
- **CIFAR10**: Small 32x32 RGB images (10 classes)
- **CIFAR100**: Small 32x32 RGB images (100 classes)
- **ImageNet**: Large-scale dataset with 1000 classes
  - **IMAGENET_VAL**: Validation set only (50K images) - for quick experiments
  - **IMAGENET_FULL**: Full training set (1.28M images) + validation set

### ImageNet Setup

For detailed instructions on setting up and using ImageNet, see **[IMAGENET_SETUP.md](IMAGENET_SETUP.md)**.

Quick start:
1. Extract ImageNet dataset to `/scratch/sbeeredd/imagenet/`
2. Organize validation set: `python organize_imagenet_val.py`
3. Train: `bash train_imagenet_val.sh` or `bash train_imagenet_full_baseline.sh` 

## Running the VQ VAE

### Quick Start

For CIFAR10:
```bash
bash train_cifar10_baseline.sh  # Baseline training
bash train_cifar10.sh           # With DGM loss
```

For ImageNet:
```bash
bash train_imagenet_val.sh            # Val set, baseline
bash train_imagenet_val_dgm.sh        # Val set, with DGM
bash train_imagenet_full_baseline.sh  # Full training, baseline
bash train_imagenet_full_dgm.sh       # Full training, with DGM
```

### Manual Execution

To run the VQ-VAE manually: `python3 main.py [options]`. Make sure to include the `-save` flag if you want to save your model.

**Core Parameters:**
```python
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--data_root", type=str, default=None)
parser.add_argument("--image_size", type=int, default=None)
```

**DGM Loss Parameters:**
```python
parser.add_argument("--use_dgm_loss", action="store_true")
parser.add_argument("--dgm_model_path", type=str, default="...")
parser.add_argument("--dgm_loss_weight", type=float, default=0.01)
parser.add_argument("--dgm_loss_type", type=str, default='mse')
parser.add_argument("--dgm_model_type", type=str, default='imagenet')
parser.add_argument("--dgm_arch", type=str, default='resnet34')
parser.add_argument("--dgm_image_size", type=int, default=256)
```

**Logging Parameters:**
```python
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="vqvae-dgm")
parser.add_argument("--wandb_run_name", type=str, default=None)
```

## Models

The VQ VAE has the following fundamental model components:

1. An `Encoder` class which defines the map `x -> z_e`
2. A `VectorQuantizer` class which transform the encoder output into a discrete one-hot vector that is the index of the closest embedding vector `z_e -> z_q`
3. A `Decoder` class which defines the map `z_q -> x_hat` and reconstructs the original image

The Encoder / Decoder classes are convolutional and inverse convolutional stacks, which include Residual blocks in their architecture [see ResNet paper](https://arxiv.org/abs/1512.03385). The residual models are defined by the `ResidualLayer` and `ResidualStack` classes.

These components are organized in the following folder structure:

```
models/
    - decoder.py -> Decoder
    - encoder.py -> Encoder
    - quantizer.py -> VectorQuantizer
    - residual.py -> ResidualLayer, ResidualStack
    - vqvae.py -> VQVAE
```

## Deep Geometric Moments (DGM) Loss

This implementation supports an optional auxiliary loss based on Deep Geometric Moments (DGM). The DGM loss helps preserve spatial structure and global statistics during reconstruction by matching features from a pretrained classifier.

**How it works:**
1. A pretrained ResNet model (trained on ImageNet or CIFAR) extracts features from both original and reconstructed images
2. The loss compares geometric moment representations between original and reconstruction
3. This encourages the VQVAE to preserve important visual structure beyond pixel-level similarity

**Benefits:**
- Better preservation of spatial structure
- Improved visual quality
- More semantically meaningful latent codes

**Usage:**
```bash
python main.py \
    --dataset IMAGENET_VAL \
    --use_dgm_loss \
    --dgm_model_path /path/to/pretrained/model.pth \
    --dgm_loss_weight 0.01 \
    --dgm_model_type imagenet \
    --dgm_arch resnet34
```

The DGM model checkpoint should be available at:
```
/scratch/sbeeredd/sandbox/Deep-Geometric-Moment/checkpoints/res34_model_best.pth.tar
```

## PixelCNN - Sampling from the VQ VAE latent space 

To sample from the latent space, we fit a PixelCNN over the latent pixel values `z_ij`. The trick here is recognizing that the VQ VAE maps an image to a latent space that has the same structure as a 1 channel image. For example, if you run the default VQ VAE parameters you'll RGB map images of shape `(32,32,3)` to a latent space with shape `(8,8,1)`, which is equivalent to an 8x8 grayscale image. Therefore, you can use a PixelCNN to fit a distribution over the "pixel" values of the 8x8 1-channel latent space.

To train the PixelCNN on latent representations, you first need to follow these steps:

1. Train the VQ VAE on your dataset of choice
2. Use saved VQ VAE parameters to encode your dataset and save discrete latent space representations with `np.save` API. In the `quantizer.py` this is the `min_encoding_indices` variable. 
3. Specify path to your saved latent space dataset in `utils.load_latent_block` function.
4. Run the PixelCNN script

To run the PixelCNN, simply type 

`python pixelcnn/gated_pixelcnn.py`

as well as any parameters (see the argparse statements). The default dataset is `LATENT_BLOCK` which will only work if you have trained your VQ VAE and saved the latent representations.
