#!/usr/bin/env python3
"""
split_by_ride.py

Group rows by ride-ID (the first token of ride_name) and write each group to
    chunk_<ride_id>.txt   (tab-separated, with headers)

Usage
-----
    python split_by_ride.py INPUT_TXT OUT_DIR
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED = ["ride_name", "start_frame", "end_frame"]


def read_table(path: Path) -> pd.DataFrame:
    """Try whitespace-delimited first, fall back to comma CSV."""
    return pd.read_csv(path, sep=",", dtype=str)


def sanitize(name: str) -> str:
    """Make ride_id safe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split INPUT_TXT into chunk_<ride_id>.txt files "
                    "containing ride_name, start_frame, end_frame."
    )
    ap.add_argument("input_txt", type=Path, help="Source text file")
    ap.add_argument("out_dir",   type=Path, help="Directory to store chunk files")
    args = ap.parse_args()

    if not args.input_txt.is_file():
        sys.exit(f"Error: {args.input_txt} does not exist or is not a file")

    df = read_table(args.input_txt)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Error: missing columns: {', '.join(missing)}")

    # keep only required columns and create ride_id
    df = df[REQUIRED].copy()
    df["ride_id"] = df["ride_name"].str.split().str[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ride_id, chunk in df.groupby("ride_id", sort=False):
        out_file = args.out_dir / f"output_rides_{sanitize(ride_id)}" / "depth_filtered.txt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        chunk.drop(columns="ride_id").to_csv(
            out_file, sep=",", index=False, header=True
        )
        print(f"✔  Wrote {len(chunk):>5} rows → {out_file.name}")

    print(f"All chunks saved in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
