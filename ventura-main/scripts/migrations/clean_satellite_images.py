import os
from pathlib import Path
import hickle as hkl

from tqdm import tqdm

def prune_satellite_files(
    root_dir: str | Path,
    image_key: str,
    target_shape: tuple[int, ...],
    dry_run: bool = True
) -> list[Path]:
    """
    Recursively scan for `satellite_infos.h5` files under `output_rides_*/*/seq_*`,
    open each with hickle, check that `data[image_key].shape == target_shape`,
    and delete any that don’t match. In dry‐run mode, only report.

    Args:
        root_dir     : root directory containing output_rides_*/
        image_key    : key in the h5 under which the satellite image is stored
        target_shape : desired shape tuple; e.g. (3, 256, 256)
        dry_run      : if True, only print which files would be deleted

    Returns:
        List of Path objects to files that were (or would be) deleted.
    """
    root = Path(root_dir)
    pattern = "output_rides_*/ride_*_*_*/seq_*/satellite_info.h5"
    to_delete = []
    h5_paths = list(root.rglob(pattern))

    for h5_path in tqdm(h5_paths, desc="Scanning files"):
        try:
            data = hkl.load(str(h5_path))
        except Exception as e:
            print(f"[ERROR] failed to load {h5_path}: {e}")
            continue

        if image_key not in data:
            print(f"[WARN] key '{image_key}' not found in {h5_path}, skipping")
            continue

        arr = data[image_key]
        shape = arr.shape
        if shape != target_shape:
            print(f"[MISMATCH] {h5_path} has shape {shape}, target is {target_shape}")
            to_delete.append(h5_path)
    import pdb; pdb.set_trace()  # Debugging breakpoint to inspect `to_delete`
    # report or delete
    if dry_run:
        print("\nDry run mode: the above files would be deleted.")
    else:
        for p in to_delete:
            try:
                os.remove(p)
                print(f"[DELETED] {p}")
            except Exception as e:
                print(f"[ERROR] failed to delete {p}: {e}")

    return to_delete

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune satellite images that do not match the target shape."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing output_rides_* directories.",
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="satellite_image",
        help="Key in the h5 file under which the satellite image is stored.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only report files to be deleted without actually deleting them.",
    )

    args = parser.parse_args()
    target_shape = (576, 1024, 3)
    prune_satellite_files(args.root_dir, args.image_key, target_shape, args.dry_run)