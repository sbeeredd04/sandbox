#!/usr/bin/env python3
# selective_sync.py
"""
Pick M rows from a split-file, locate the matching ride_*/ folders under
<SRC>/h5files, and rsync just those folders to a remote or local <DST>.
"""
import argparse, random, subprocess, sys, tempfile, os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np

from spinflow.dataset.frodo_helpers import set_frodo_dir

# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Randomly select split rows and rsync the matching ride_* folders.")
    p.add_argument("--src",   required=True,
                   help="Source root; it must contain folders.")
    p.add_argument("--dst",   required=True,
                   help="Destination path for rsync (local dir or user@host:/path).")
    p.add_argument("--dataset", default="engine", help="Dataset format to synv [raw, h5, engine]")
    p.add_argument("--split", required=True,
                   help="Path to the split file (three columns: ride drive date).")
    p.add_argument("--num",   type=int, required=True,
                   help="How many rows / ride-folders to copy.")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed (reproducible).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rsync command without executing.")
    return p.parse_args()

# --------------------------------------------------------------------- #
def read_split(fp: Path) -> List[Tuple[str,str,str]]:
    """Return list of (ride_id, drive_id, date) tuples."""
    rows = []
    with fp.open() as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 3:
                print(f"Skip malformed line: {ln}", file=sys.stderr)
                continue
            rows.append(tuple(parts))
    if not rows:
        sys.exit("No valid rows found in split file.")
    return rows

def read_csv_or_empty(path, **kwargs):
    """
    Try to read a CSV at `path` with pandas.read_csv.  
    If the file is empty (or has no parsable lines), return an empty DataFrame.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        # no data: return empty DataFrame (or you could return None here)
        return pd.DataFrame()

def read_engine_split(split_fp: Path) -> List[Tuple[str,str,str]]:
    """
    Read the engine split file, which has a different format.
    The file is expected to have three columns: ride_name, start_frame, end_frame
    """
    df = None
    if not split_fp.is_file(): 
        ride_dirs = [p for p in split_fp.glob("output_rides_*") if p.is_dir()]
        if not ride_dirs:
            sys.exit(f"No ride directories found in {split_fp}.")
        df_list = []
        for ride_dir in ride_dirs:
            split_path = ride_dir / "full_raw.txt"
            df = read_csv_or_empty(split_path, sep=",")
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_csv(split_fp, sep=",")

    if df.shape[1] != 3:
        sys.exit(f"Expected 3 columns in the split file, got {df.shape[1]}.")

    return df['ride_name'].tolist()

def choose(rows: List[Tuple[str,str,str]], k: int, seed: int):
    rng = random.Random(seed)
    if k > len(rows):
        print(f"Requested {k} rows but split only has {len(rows)}. "
              f"Using all rows.", file=sys.stderr)
        k = len(rows)
    elif k <= 0:
        # Select all rows if k is zero or negative
        return rows
    rng.shuffle(rows)
    return rows[:k]

# --------------------------------------------------------------------- #
def build_h5file_names(rows: List[Tuple[str,str,str]]) -> List[str]:
    """Return ride_{ride}_{drive}_{date} for each row."""
    return [f"h5files/ride_{r}_{d}_{t}.h5" for (r,d,t) in rows]

def write_list_file(root: Path, files: List[str]) -> str:
    """
    rsync --files-from expects *relative* paths *from* the directory we give it.
    We'll copy entire files (trailing / keeps rsync semantics).
    """
    tmp = tempfile.NamedTemporaryFile("w+", delete=False)
    for rel in files:
        full = root / rel
        if full.is_dir():
            # use os.walk instead of rglob for speed
            for dirpath, _, filenames in os.walk(full):
                for fname in filenames:
                    rf = Path(dirpath) / fname
                    relpath = rf.relative_to(root).as_posix()
                    tmp.write(relpath + "\n")
        else:
            tmp.write(Path(rel).as_posix() + "\n")
    tmp.flush()
    return tmp.name
    # tmp = tempfile.NamedTemporaryFile("w+", delete=False)
    # for file in files:
    #     full_path = root / Path(file)
    #     if full_path.is_dir() and not file.endswith("/"):
    #         file += "/"
    #     tmp.write(f"{file}\n")
    # tmp.flush()
    # return tmp.name

def build_subdir_names(rows: List[Tuple[str,str,str]]) -> List[str]:
    return [f"output_rides_{r}/ride_{d}_{t}" for (r,d,t) in rows]

def build_engine_names(rows: List[str]) -> List[str]:
    """
    For the engine dataset, we assume the folder structure is:
    output_rides_{ride_name}/ride_{ride_name}
    """
    return [str(set_frodo_dir("", *r.split(" "))) for r in rows]

# --------------------------------------------------------------------- #
# 
# python scripts/aws/sync_frodo.py --src /robodata/public_datasets/frodobots8k \
#   --dst ec2-user@52.13.84.237:/data/public_datasets/frodo8k \
#   --split ./data/frodo8k/splits/spinflow_full/full.txt --num 200

# Sync splits from engine
# python scripts/aws/sync_frodo.py --src ./data/frodobots8k --dst ec2-user@52.13.84.237:/data/public_datasets/frodobots8k --split ./data/frodobots8k_processed/output_rides_1/full_raw.txt --num 200 --dataset engine
def main():
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    split_fp = Path(args.split).expanduser().resolve()

    # 1.  Pick the rows
    if args.dataset == "h5":
        rows_all   = read_split(split_fp)
        rows_chosen = choose(rows_all, args.num, args.seed)
        sync_files = build_h5file_names(rows_chosen)
    elif args.dataset == "raw":
        rows_all   = read_split(split_fp)
        rows_chosen = choose(rows_all, args.num, args.seed)
        sync_files = build_subdir_names(rows_chosen)
    elif args.dataset == "engine":
        row_all = read_engine_split(split_fp)
        rows_chosen = choose(row_all, args.num, args.seed)
        sync_files = build_engine_names(rows_chosen)
    else:
        raise ValueError(f"Unknown dataset format: {args.dataset}")

    # 2.  Make sure the folders exist
    missing = [sync_file for sync_file in sync_files if not (src_root / sync_file).exists()]
    if missing:
        print("WARNING: the following ride-folders were not found and will be skipped:",
              *missing, sep="\n  ", file=sys.stderr)
        sync_files = [f for f in sync_files if f not in missing]
    # Convert to set to remove duplicates
    sync_files = list(set(sync_files))

    ride_txt_relpaths = []
    # collect all unique ride_root directories
    ride_roots = { (src_root / rel).parent for rel in sync_files }
    exts = {".txt", ".csv", ".graphml"}
    for ride_root in ride_roots:
        # os.walk is far faster than Path.glob/rglob
        for dirpath, _, filenames in os.walk(ride_root):
            for fname in filenames:
                if Path(fname).suffix in exts:
                    full = Path(dirpath) / fname
                    ride_txt_relpaths.append(full.relative_to(src_root).as_posix())

    ride_txt_relpaths = list(set(ride_txt_relpaths))   # de-duplicate

    txt_files_from = None
    if ride_txt_relpaths:
        txt_files_from = write_list_file(src_root, ride_txt_relpaths)

    # import pdb; pdb.set_trace()
    if not sync_files:
        sys.exit("No folders to sync — exiting.")

    # 3.  rsync
    files_from = write_list_file(src_root, sync_files)

    rsync_cmd = [
        "rsync", "-azvhP", "--stats"
    ]

    if args.dry_run:
        print("Running in dry-run mode; no files will be copied.")
        rsync_cmd.append("--dry-run")

    rsync_cmd.extend([
        "--files-from", files_from, str(src_root) + "/", args.dst
    ])

    print(f"Syncing {len(sync_files)} ride folders from {src_root} to {args.dst}...")
    print("Command:", " ".join(rsync_cmd))
    
    subprocess.check_call(rsync_cmd)
    if txt_files_from:
        rsync_txt_cmd = rsync_cmd.copy()
        # replace the last command-specific bits
        rsync_txt_cmd[rsync_txt_cmd.index("--files-from") + 1] = txt_files_from
        print("Syncing ride-level .txt files …")
        print("Command:", " ".join(rsync_txt_cmd))
        subprocess.check_call(rsync_txt_cmd)

    os.remove(files_from)
    if txt_files_from:
        os.remove(txt_files_from)

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
