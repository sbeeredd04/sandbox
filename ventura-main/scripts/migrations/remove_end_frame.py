#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

def rename_seq_dirs(root: Path):
    """
    Walks under `root/output_rides_*/*/seq_*_*` and renames each
    seq_{start}_{end} --> seq_{start}.
    """
    pattern = re.compile(r"^seq_(\d+)_(\d+)$")
    count = 0

    for ride_root in root.glob("output_rides_*"):
        if not ride_root.is_dir():
            continue
        for ride_dir in ride_root.glob("ride_*"):
            if not ride_dir.is_dir():
                continue
            for seq_dir in ride_dir.glob("seq_*_*"):
                if not seq_dir.is_dir():
                    continue

                m = pattern.match(seq_dir.name)
                if not m:
                    print(f"  ❌ skipping unexpected folder name: {seq_dir}", file=sys.stderr)
                    continue

                start = m.group(1)
                new_name = f"seq_{start}"
                new_path = seq_dir.parent / new_name

                if new_path.exists():
                    print(f"  ⚠️  target already exists, skipping: {new_path}", file=sys.stderr)
                    continue

                print(f"  Renaming: {seq_dir.name} → {new_name}")
                seq_dir.rename(new_path)
                count += 1

    print(f"Done: renamed {count} folders.")

def main():
    p = argparse.ArgumentParser(
        description="Rename seq_{start}_{end} → seq_{start} under output_rides_*"
    )
    p.add_argument(
        "root",
        type=Path,
        help="Root directory containing output_rides_* subfolders"
    )
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"Error: {args.root} is not a directory", file=sys.stderr)
        sys.exit(1)

    rename_seq_dirs(args.root)

if __name__ == "__main__":
    main()
