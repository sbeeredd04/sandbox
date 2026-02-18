#!/usr/bin/env python3
import argparse
import sys
import shutil
from pathlib import Path

def parse_list_file(txt_path: Path) -> dict[str, set[int]]:
    """
    Returns: { 'output_rides_.../ride_..._0': {2960, 2962, ...}, ... }
    """
    mapping: dict[str, set[int]] = {}
    cur_key: str | None = None

    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # New block header (ride dir)
            if line.startswith("output_rides_"):
                cur_key = line.rstrip("/")      # keep relative path as-is
                mapping.setdefault(cur_key, set())
                continue
            # Frame indices under current block
            if cur_key is not None:
                try:
                    frame = int(line)
                except ValueError:
                    # Ignore non-integer garbage lines quietly
                    continue
                mapping[cur_key].add(frame)
            # If we saw frames before any header, ignore them
    return mapping

def ensure_within(base: Path, target: Path) -> bool:
    """Safety: only delete inside base directory."""
    try:
        base_r = base.resolve()
        targ_r = target.resolve()
    except FileNotFoundError:
        # If it doesn't exist, we still want to validate the intended path
        base_r = base.resolve()
        targ_r = (base / target).resolve() if not target.is_absolute() else target
    return str(targ_r).startswith(str(base_r))

def main():
    ap = argparse.ArgumentParser(
        description="Delete seq_{frame} dirs listed under each output_rides.../ride... block."
    )
    ap.add_argument("data_dir", type=Path, help="Root directory that contains the ride subdirs")
    ap.add_argument("list_file", type=Path, help="Text file in the given format")
    # Dry-run enabled by default; disable with --no-dry-run (Python 3.9+)
    try:
        from argparse import BooleanOptionalAction
        ap.add_argument("--dry-run", default=True, action=BooleanOptionalAction,
                        help="Preview deletions without removing anything (default: on)")
    except Exception:
        ap.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview deletions without removing anything (default: on)")
        ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually delete")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser()
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data_dir '{data_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(2)

    if not args.list_file.exists():
        print(f"ERROR: list_file '{args.list_file}' not found.", file=sys.stderr)
        sys.exit(2)

    plan = parse_list_file(args.list_file)
    if not plan:
        print("Nothing to do (no blocks found).")
        return

    total = 0
    missing = 0
    print(("DRY RUN" if args.dry_run else "EXECUTE") + f": scanning {len(plan)} ride block(s)\n")

    for rel_ride, frames in plan.items():
        ride_dir = (data_dir / rel_ride).expanduser()
        print(f"[{rel_ride}]  ({len(frames)} seq dirs)")
        for frame in sorted(frames):
            seq_dir = ride_dir / f"seq_{frame}"
            # safety: do not delete outside base
            if not ensure_within(data_dir, seq_dir):
                print(f"  ! SKIP (unsafe path): {seq_dir}")
                continue
            if not seq_dir.exists():
                print(f"  - MISSING: {seq_dir}")
                missing += 1
                continue
            if not seq_dir.is_dir():
                print(f"  - NOT A DIR: {seq_dir}")
                continue

            total += 1
            if args.dry_run:
                print(f"  - WOULD DELETE: {seq_dir}")
            else:
                try:
                    shutil.rmtree(seq_dir)
                    print(f"  - DELETED: {seq_dir}")
                except Exception as e:
                    print(f"  ! ERROR deleting {seq_dir}: {e}")

        print()  # blank line between blocks

    print(f"Done. {'Planned' if args.dry_run else 'Performed'} deletions: {total}. Missing: {missing}.")

if __name__ == "__main__":
    main()
