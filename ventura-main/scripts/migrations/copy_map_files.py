#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from typing import List

"""
python scripts/migrations/copy_map_files.py ./data/frodobots8k_processed_halfsampling ./data/frodobots8k_processed --subdirs maps full_raw.txt
"""
def copy_subdirs(input_dir: Path, output_dir: Path, subdirs: List[str]) -> None:
    """
    For every top-level *ride* directory inside ``input_dir``:
        • Look for each item in ``subdirs`` **relative to that ride dir**.
          An item may be either
              – a directory  → copy its entire tree, or
              – a file       → copy just that file.
        • The copied content is written to ``output_dir`` while preserving the
          path portion that comes after ``input_dir``.
    """
    input_dir  = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for ride_dir in input_dir.iterdir():
        if not ride_dir.is_dir():
            continue
        if "output_rides_" not in ride_dir.name:
            print(f"[warn] {ride_dir} is not an output-ride dir; skipping.")
            continue

        for sub in subdirs:
            src_path = ride_dir / sub  # may be dir or file
            if src_path.is_dir():
                # recurse over every file in that directory
                for file_path in src_path.rglob('*'):
                    if file_path.is_file():
                        rel = file_path.relative_to(input_dir)
                        dest = output_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest)
                        print(f"Copied dir-file {file_path} → {dest}")
            elif src_path.is_file():
                rel  = src_path.relative_to(input_dir)
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest)
                print(f"Copied file {src_path} → {dest}")
            else:
                print(f"[warn] {src_path} not found; skipping.")

def main():
    p = argparse.ArgumentParser(
        description="Copy specified subdirectories from a set of ride folders, preserving structure."
    )
    p.add_argument("input_dir",  type=Path,
                   help="Root folder containing output_rides_{ride_id} directories")
    p.add_argument("output_dir", type=Path,
                   help="Where to mirror & copy the files")
    p.add_argument("--subdirs", "-s", nargs="+", required=True,
                   help="Names of subdirectories to copy (e.g. maps)")
    args = p.parse_args()

    copy_subdirs(args.input_dir, args.output_dir, args.subdirs)

if __name__ == "__main__":
    main()
