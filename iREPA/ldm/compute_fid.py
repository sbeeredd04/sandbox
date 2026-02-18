#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
from PIL import Image
from pathlib import Path


def ensure_pytorch_fid():
    """Install pytorch-fid if not available."""
    try:
        import pytorch_fid
        return True
    except ImportError:
        print("pytorch-fid not found. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pytorch-fid"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓ Successfully installed pytorch-fid")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install pytorch-fid: {e}")
            print("Please install manually: pip install pytorch-fid")
            return False


def download_reference_stats(output_path="VIRTUAL_imagenet256_labeled.npz"):
    """Download ImageNet-256 reference statistics."""
    if os.path.exists(output_path):
        print(f"✓ Reference stats already exist: {output_path}")
        return output_path
    
    url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
    print(f"Downloading ImageNet-256 reference statistics...")
    print(f"  URL: {url}")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        print(f"  Please download manually from: {url}")
        return None


def extract_npz_to_images(npz_path, output_dir, num_samples=None):
    """
    Extract images from .npz file to a temporary directory.
    
    Args:
        npz_path: Path to .npz file containing samples
        output_dir: Directory to save extracted images
        num_samples: Number of samples to extract (None = all)
    
    Returns:
        Number of images extracted
    """
    print(f"\nExtracting samples from {npz_path}...")
    
    # Load npz file
    data = np.load(npz_path)
    
    # Get samples array (usually stored as 'arr_0')
    if 'arr_0' in data:
        samples = data['arr_0']
    elif 'samples' in data:
        samples = data['samples']
    else:
        # Try the first key
        samples = data[list(data.keys())[0]]
    
    total_samples = len(samples)
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)
    
    print(f"  Total samples in npz: {total_samples}")
    print(f"  Extracting: {num_samples} samples")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    for idx in range(num_samples):
        if idx % 5000 == 0 and idx > 0:
            print(f"    Extracted {idx}/{num_samples} images...")
        
        img_array = samples[idx]
        
        # Ensure uint8 range [0, 255]
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f"{idx:06d}.png"))
    
    print(f"✓ Extracted {num_samples} images to {output_dir}")
    return num_samples


def convert_reference_npz_to_pytorch_fid_format(input_npz, output_npz):
    """
    Convert OpenAI's reference .npz format to pytorch-fid compatible format.
    
    OpenAI format: contains image samples as 'arr_0'
    pytorch-fid format: contains statistics as 'mu' and 'sigma'
    
    This function extracts images and computes statistics using pytorch-fid.
    """
    if os.path.exists(output_npz):
        print(f"✓ pytorch-fid compatible reference stats exist: {output_npz}")
        return output_npz
    
    print(f"\nConverting reference npz to pytorch-fid format...")
    
    # Extract reference images to temp directory
    temp_ref_dir = tempfile.mkdtemp(prefix="fid_ref_")
    try:
        extract_npz_to_images(input_npz, temp_ref_dir)
        
        # Use pytorch-fid to compute and save statistics
        print(f"  Computing reference statistics with pytorch-fid...")
        cmd = [
            sys.executable, "-m", "pytorch_fid",
            "--save-stats",
            temp_ref_dir,
            output_npz
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ Failed to compute reference stats:")
            print(result.stderr)
            return None
        
        print(f"✓ Saved pytorch-fid reference stats: {output_npz}")
        return output_npz
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_ref_dir, ignore_errors=True)


def compute_fid_pytorch(sample_npz, reference_npz, batch_size=50, device="cuda", 
                        dims=2048, num_workers=4):
    """
    Compute FID score using pytorch-fid library.
    
    Args:
        sample_npz: Path to generated samples .npz file
        reference_npz: Path to reference dataset .npz file (pytorch-fid format)
        batch_size: Batch size for InceptionV3 feature extraction
        device: Device to use (cuda or cpu)
        dims: Dimensionality of Inception features (2048 for final pool)
        num_workers: Number of dataloader workers
    
    Returns:
        FID score
    """
    print("\n" + "="*60)
    print("Computing FID with pytorch-fid")
    print("="*60)
    print(f"Sample NPZ:     {sample_npz}")
    print(f"Reference NPZ:  {reference_npz}")
    print(f"Device:         {device}")
    print(f"Batch size:     {batch_size}")
    print(f"Feature dims:   {dims}")
    print("="*60 + "\n")
    
    # Create temp directories for image extraction
    temp_sample_dir = tempfile.mkdtemp(prefix="fid_samples_")
    
    try:
        # Extract sample images
        num_samples = extract_npz_to_images(sample_npz, temp_sample_dir)
        
        print(f"\nRunning pytorch-fid...")
        print(f"  This will take 5-10 minutes for 50K samples...")
        
        # Build pytorch-fid command
        cmd = [
            sys.executable, "-m", "pytorch_fid",
            temp_sample_dir,
            reference_npz,
            "--batch-size", str(batch_size),
            "--device", device,
            "--dims", str(dims),
        ]
        
        if num_workers > 0:
            cmd.extend(["--num-workers", str(num_workers)])
        
        print(f"  Command: {' '.join(cmd)}\n")
        
        # Run pytorch-fid
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ pytorch-fid failed:")
            print(result.stderr)
            return None
        
        # Parse FID score from output
        output = result.stdout
        print(output)
        
        # Extract FID value
        for line in output.split('\n'):
            if 'FID:' in line:
                try:
                    fid_value = float(line.split('FID:')[1].strip())
                    return fid_value
                except (IndexError, ValueError):
                    pass
        
        print("✗ Could not parse FID score from output")
        return None
        
    finally:
        # Cleanup temp directories
        shutil.rmtree(temp_sample_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID score using pytorch-fid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sample-npz", type=str, required=True,
        help="Path to generated samples .npz file"
    )
    parser.add_argument(
        "--ref-npz", type=str, default="VIRTUAL_imagenet256_labeled.npz",
        help="Path to reference .npz file (will be downloaded if missing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for InceptionV3 feature extraction"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--dims", type=int, default=2048, choices=[64, 192, 768, 2048],
        help="Dimensionality of Inception features"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("FID Computation using pytorch-fid")
    print("="*60)
    
    # Ensure pytorch-fid is installed
    if not ensure_pytorch_fid():
        sys.exit(1)
    
    # Check sample file exists
    if not os.path.exists(args.sample_npz):
        print(f"✗ Sample file not found: {args.sample_npz}")
        sys.exit(1)
    
    # Download reference stats if needed
    ref_npz_original = args.ref_npz
    if not os.path.exists(ref_npz_original):
        ref_npz_original = download_reference_stats(ref_npz_original)
        if ref_npz_original is None:
            sys.exit(1)
    
    # Convert reference to pytorch-fid format
    ref_npz_pytorch = ref_npz_original.replace(".npz", "_pytorch_fid.npz")
    ref_npz_pytorch = convert_reference_npz_to_pytorch_fid_format(
        ref_npz_original, ref_npz_pytorch
    )
    
    if ref_npz_pytorch is None:
        print("✗ Failed to prepare reference statistics")
        sys.exit(1)
    
    # Compute FID
    fid_score = compute_fid_pytorch(
        args.sample_npz,
        ref_npz_pytorch,
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        num_workers=args.num_workers
    )
    
    if fid_score is not None:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"  FID Score: {fid_score:.2f}")
        print("="*60)
        
        # Print comparison to paper
        print("\nExpected FID for iREPA @ 100K steps (from paper Table 4):")
        print("  - CFG=1.0 (no guidance):  FID ≈ 16.9")
        print("  - CFG=2.0 (with guidance): FID ≈ 5.15")
        print("\nExpected FID for iREPA @ 400K steps:")
        print("  - CFG=1.0:  FID ≈ 7.5")
        print("  - CFG=2.0:  FID ≈ 1.93")
    else:
        print("\n✗ FID computation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()