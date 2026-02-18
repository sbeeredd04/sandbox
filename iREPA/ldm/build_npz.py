#!/usr/bin/env python3
"""
Recovery script: Build .npz from existing sample PNGs after a crashed eval run.
This avoids re-running the expensive 9+ hour sample generation.

Usage:
    python build_npz.py --sample-dir samples/<folder_name> --num-samples 50000
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """Builds a single .npz file from a folder of .png samples."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        path = f"{sample_dir}/{i:06d}.png"
        if not os.path.exists(path):
            print(f"WARNING: Missing sample {path}")
            continue
        sample_pil = Image.open(path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3), \
        f"Expected shape ({num}, H, W, 3), got {samples.shape}"
    
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description="Build .npz from sample folder")
    parser.add_argument("--sample-dir", type=str, required=True,
                        help="Path to folder containing .png samples")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of samples to include in npz")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove the sample folder after building npz")
    args = parser.parse_args()
    
    if not os.path.isdir(args.sample_dir):
        print(f"ERROR: Directory not found: {args.sample_dir}")
        return
    
    # Count existing files
    png_count = len([f for f in os.listdir(args.sample_dir) if f.endswith('.png')])
    print(f"Found {png_count} PNG files in {args.sample_dir}")
    
    if png_count < args.num_samples:
        print(f"WARNING: Only {png_count} files found, but {args.num_samples} requested")
        print(f"  Proceeding with {min(png_count, args.num_samples)} samples")
    
    npz_path = create_npz_from_sample_folder(args.sample_dir, args.num_samples)
    
    if args.cleanup:
        import shutil
        shutil.rmtree(args.sample_dir)
        print(f"Cleaned up sample folder: {args.sample_dir}")
    
    print(f"\nNPZ file ready: {npz_path}")
    print(f"To compute FID:")
    print(f"  python compute_fid.py --sample-npz {npz_path}")


if __name__ == "__main__":
    main()
