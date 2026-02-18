#!/usr/bin/env python3
import multiprocessing, os
import argparse, yaml
from pathlib import Path
from typing import Dict, List
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from functools import partial

import subprocess, shlex
import pandas as pd
import s3fs, fsspec
from scripts.fai.process_utils import inspect_and_stream
from scripts.utils.log_utils import logging
from scripts.preprocessing.run_engine import (
    construct_filters
)

from spinflow.dataset.frodo_helpers import get_frodo_raw_id

DEBUG_MODE=False

# ------------------ helpers ----------------------------------------------
def load_cfg(fp: Path) -> dict:
    with fp.open() as f:
        return yaml.safe_load(f)

def _bag_to_s3_key(filename: str, root: str, robots: List[str]) -> str | None:
    """
    Convert 'aih_seattle_2025-05-21_ferrite2_2025-05-21-16-56-00_test.bag'
    → 'aih/seattle/2025-05-21/ferrite2/aih_seattle_2025-05-21_ferrite2_2025-05-21-16-56-00_test.bag'
    """
    name = Path(filename).name
    stem = name[:-4] if name.endswith(".bag") else name
    parts = stem.split("_")

    # locate robot/device token
    try:
        idx_robot = next(i for i, p in enumerate(parts) if p in robots)
    except StopIteration:
        return None                      # skip unknown devices

    if idx_robot < 2:                    # need at least dir0 + subdir2
        return None

    dir0     = parts[0]
    subdir2  = parts[idx_robot - 1]
    subdir1  = "_".join(parts[1:idx_robot - 1]) or None
    robot_name = parts[idx_robot]

    path_parts = [dir0]
    if subdir1:
        path_parts.append(subdir1)
    path_parts.extend([subdir2, parts[idx_robot], name])

    key = "/".join(path_parts[:-2])
    bag_dir = f"{key}/{robot_name}_{'_'.join(parts[idx_robot + 1:])}"

    return bag_dir if bag_dir.startswith(root) else f"{root.rstrip('/')}/{bag_dir}"


def sessions_from_csv(
    csv_fp: Path,
    bucket: str,
    root: str,
    sites: List[str],
    robots: List[str],
) -> List[str]:
    """
    Original behaviour + extra Foxglove lookup:
      For each unique 'Dataset Name' in the CSV, run the Foxglove one‑liner
      to find matching .bag filenames, convert each to an S3 key, and return
      the full list (sorted, bucket‑relative).
    """
    csv_files = list(csv_fp.glob("*.csv")) if csv_fp.is_dir() else [csv_fp]

    sess: list[str] = []

    # ── existing CSV parsing (unchanged) ────────────────────────────────
    for fp in csv_files:
        df = pd.read_csv(fp, keep_default_na=False)
        df.columns = df.columns.str.strip()

        # Remove empty dataset name columns
        df = df[df["Dataset Name"].notna() & (df["Dataset Name"] != "")]

        # user‑specific row filtering logic (uncomment / edit if needed)
        # mask = (
        #     df["Site"].str.lower().isin([s.lower() for s in sites]) &
        #     (df["Health"].str.strip().str.lower() == "good")
        # )
        # for s3 in df.loc[mask, "S3"]:
        #     ...

        # ── NEW: supplement with Foxglove search ───────────────────────
        for dataset in df["Dataset Name"].unique():
            # Check if the "S3 Path is populated" for this dataset
            if df.loc[df["Dataset Name"] == dataset, "S3 Path"].any():
                s3_path = df.loc[df["Dataset Name"] == dataset, "S3 Path"].iloc[0]
                if len(s3_path.strip()) > 0:
                    # remove the bucket prefix if present
                    if s3_path.startswith(f"s3://{bucket}/"):
                        s3_path = s3_path[len(f"s3://{bucket}/"):]
                    sess.append(s3_path)
                    continue

            # safe‑quote the pattern for bash
            pat = shlex.quote(dataset)

            cmd = (
                f"foxglove data imports list --format json | "
                f"jq -r --arg pat {pat} "
                f"'map(select(.filename|contains($pat)))[] | .filename'"
            )

            try:
                proc = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Foxglove query failed for '{dataset}': {e}")
                continue

            for line in proc.stdout.strip().splitlines():
                key = _bag_to_s3_key(line.strip(), root, robots)
                if key:
                    s3_path = f"{key}"
                    sess.append(s3_path)

    # aih/seattle_250528_163305/2025-05-29/ferrite2_2025-05-29-14-57-00_test

    return sorted(set(sess))

def build_topics(cfg_topics: dict) -> Dict[str, Dict[str, str]]:
    return cfg_topics
    # return {
    #     k: {
    #         "ros_topic": v["ros_topic"], 
    #         "save_prefix": v["save_prefix"]
    #     }
    #     for k, v in cfg_topics.items()
    # }

def run_session(sess: str, bucket: str, cache_dir: str, save_root: str, cfg_topics: dict, filters: dict) -> bool:
    try:
        robot = Path(sess).name.split("_")[0]
        topics_dict = build_topics(cfg_topics[robot])

        fs_cached = fsspec.filesystem(
            "filecache",
            target_protocol="s3",
            cache_storage=cache_dir,
            default_fill_cache=False,
            target_options={"anon": False},
        )

        remote = f"s3://{bucket}/{sess}"
        logging.info(f"[PID {os.getpid()}] → {remote}")

        df = inspect_and_stream(fs_cached, remote, save_root, robot, topics_dict, filters)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except Exception as exc:
        logging.error(f"Failed session {sess}: {exc}")
        return None

# ------------------ main --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cfg", default="scripts/fai/config/urban.yaml")
    ap.add_argument("--csv", default="./data/fai_raw/lhy_meta/",
                    help="CSV listing sessions (default: robot_matching_all.csv)")
    args = ap.parse_args()

    cfg       = load_cfg(Path(args.cfg))
    bucket    = cfg["s3_bucket"]
    root      = cfg["s3_path"].rstrip("/")
    sites     = cfg["sites"]
    robots    = cfg["robots"]
    cache_dir = cfg.get("s3_cache", "/tmp/s3_bag_cache")
    save_root = cfg.get("root_dir", "./data/fai_raw")
    filters   = construct_filters(cfg)

    # ▼ derive sessions straight from CSV
    sessions = sessions_from_csv(Path(args.csv), bucket, root, sites, robots)
    if not sessions:
        logging.info("No sessions matched CSV + YAML filters — exiting.")
        return
    
    fs_cached = fsspec.filesystem(
        "filecache",
        target_protocol="s3",
        cache_storage=cache_dir,
        default_fill_cache=False,
        target_options={"anon": False},
    )
    if DEBUG_MODE:
        dfs = [run_session(s, bucket, cache_dir, save_root, cfg["topics"], filters)
           for s in sessions]
        dfs = [d for d in dfs if d is not None]
    else:
        n_jobs = cfg.get("n_jobs", 8)

        with tqdm(total=len(sessions), desc="Sessions") as pbar:
            dfs = []
            for df in Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(run_session)(s, bucket, cache_dir, save_root, cfg["topics"], filters)
                    for s in sessions):
                if df is not None:
                    dfs.append(df)
                pbar.update(1)
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        # global manifest (unchanged)
        combined.to_csv(Path(save_root) / "all_metadata.csv", index=False)
        from collections import defaultdict

        rides: dict[str, list[dict]] = defaultdict(list)
        for row in combined.to_dict("records"):
            ride_dir = str(Path(row["video"]).parent.parent)   # directory that holds the MP4
            rides[ride_dir].append(row)

        for ride_dir, rows in rides.items():
            txt_path = Path(ride_dir) / "ride_manifest.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with txt_path.open("w") as f:
                f.write("ride_name,child_dt,video,odometry,controls\n")
                for r in rows:
                    ride_info = get_frodo_raw_id(r["video"])
                    ride_name = " ".join(ride_info)
                    f.write(f"{ride_name},{r['child_dt']},{r['video']},{r['odometry']},{r['controls']}\n")

        logging.info(f"✅  {len(dfs)}/{len(sessions)} sessions processed "
                    f"({len(combined)} rows total, {len(rides)} ride manifests)")

if __name__ == "__main__":
    main()
