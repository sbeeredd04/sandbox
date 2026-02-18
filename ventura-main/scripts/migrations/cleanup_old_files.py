#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


def delete_target_files(root: Path, target_name: str, *, dry_run: bool = False) -> None:
    """
    Recursively search under `root/output_rides_*/*/seq_*`
    (and any already-renamed seq_* folders) and delete every file whose
    *basename* equals `target_name`.

    If `dry_run=True`, only report what would be deleted.
    """
    seq_pat = re.compile(r"^seq_\d+(?:_\d+)?$")      # seq_123  or  seq_123_456
    matched = 0
    deleted = 0

    for ride_root in root.glob("output_rides_*"):
        if not ride_root.is_dir():
            continue
        for ride_dir in ride_root.glob("ride_*"):
            if not ride_dir.is_dir():
                continue
            for seq_dir in ride_dir.glob("seq_*"):
                if not seq_dir.is_dir() or not seq_pat.match(seq_dir.name):
                    continue

                # recurse and find matching files
                for file_path in seq_dir.rglob(target_name):
                    if file_path.is_file():
                        rel = file_path.relative_to(root)
                        if dry_run:
                            print(f"  🤔 Would delete: {rel}")
                        else:
                            print(f"  🗑 Deleting:    {rel}")
                            file_path.unlink(missing_ok=True)
                            deleted += 1
                        matched += 1

    if dry_run:
        print(f"Dry-run complete: {matched} file(s) match '{target_name}'.")
    else:
        print(f"Done: deleted {deleted} file(s) named '{target_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively delete every file whose name matches TARGET_NAME "
            "inside seq_* folders under output_rides_*."
        )
    )
    parser.add_argument("root", type=Path,
                        help="Root directory containing output_rides_* sub-folders")
    parser.add_argument("target_name", type=str,
                        help="Exact file name to delete (e.g. unwanted.txt)")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show what would be deleted without deleting anything")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Error: {args.root} is not a directory", file=sys.stderr)
        sys.exit(1)

    delete_target_files(args.root, args.target_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
