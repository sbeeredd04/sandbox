#!/usr/bin/env python3
import os
import sys
import argparse
import random
from pathlib import Path

import spinflow.dataset.frodo_helpers as fh

def main(input_file, output_dir, full_file, train_file, val_file, test_file, mini, seed=42):
    # Set seed for reproducibility
    random.seed(seed)

    if os.path.isdir(input_file):
        samples = [ f for f in os.listdir(input_file) if f.endswith('.h5') and not f.endswith('.tmp.h5') ]

        sample_ids = [fh.get_frodo_id(Path(sample)) for sample in samples]

        # Unpack tuple to string
        lines = [ " ".join(str(s) for s in sid) for sid in sample_ids ]
        input_dir = input_file
    else:
        input_dir = os.path.dirname(input_file)

        # Load lines from input file
        with open(input_file, 'r') as f:
            lines = f.read().splitlines()
    # Shuffle the data
    random.shuffle(lines)

    if mini:
        lines = lines[:100]
        output_dir = output_dir + '_mini'
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = output_dir + '_full'
        os.makedirs(output_dir, exist_ok=True)
    n = len(lines)
    
    # Compute split sizes
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    # The rest goes to test (which might be slightly more than 10% if rounding down)
    n_test = n - n_train - n_val

    # Partition the data
    full_lines = lines[:]
    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train+n_val]
    test_lines = lines[n_train+n_val:]

    # Sort each split for readability
    def sort_key(line):
        parts = line.split()
        subparts = parts[1].split('_')
        return (str(parts[0]), str(subparts[0]), str(subparts[1]), str(parts[2]))
    
    save_files = [full_file, train_file, val_file, test_file]
    save_lines = [full_lines, train_lines, val_lines, test_lines]
    for file, lines in zip(save_files, save_lines):
        lines = sorted(lines, key=sort_key)

        # Ensure that input file exists
        input_file_paths = [fh.set_frodo_path(input_dir, *line.split(' ')) for line in lines]
        no_exist_paths = [p for p in input_file_paths if not p.exists()]
        if len(no_exist_paths) > 0:
            print(f"Warning: The following files do not exist and will be skipped: {no_exist_paths}", file=sys.stderr)
            raise ValueError("Some input files do not exist. Please check the input directory or file.")

        file = Path(output_dir) / file
        with open(file, 'w') as f:
            f.write("\n".join(lines))
    
    print(f"Total samples: {n}")
    print(f"Full samples: {len(full_lines)}")
    print(f"Training samples: {len(train_lines)}")
    print(f"Validation samples: {len(val_lines)}")
    print(f"Test samples: {len(test_lines)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split a dataset text file into train (80%), validation (10%), and test (10%) splits."
    )
    parser.add_argument("--input", default="./data/frodo8k/spinflow_processed/h5files", help="Path to the input directory or .txt file with splits.")
    parser.add_argument("--output", required=True, help="Directory to save the output splits.")
    parser.add_argument("--full", default="full.txt", help="Output full split file.")
    parser.add_argument("--train", default="train.txt", help="Output training split file.")
    parser.add_argument("--val", default="val.txt", help="Output validation split file.")
    parser.add_argument("--test", default="test.txt", help="Output test split file.")
    parser.add_argument("--mini", action="store_true", help="Use a small dataset for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    main(args.input, args.output, args.full, args.train, args.val, args.test, args.mini, args.seed)
