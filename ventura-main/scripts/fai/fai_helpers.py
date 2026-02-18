from typing import Tuple
from pathlib import Path

def get_fai_s3id(s3_path: str) -> Tuple[str, str, str]:
    """
    Extract (robot_name, parent_datetime, child_datetime) from an ai4h-S3 key.

    Returns
    -------
    (robot_name, parent_dt, child_dt)  where all dt strings are in the
    format 'YYYY-MM-DD-HH-MM-SS'.
    """
    # Strip optional URI scheme so Path treats it like a normal posix path
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]

    parts = Path(s3_path).parts
    if len(parts) < 3:
        raise ValueError("S3 key too short to contain expected segments.")

    parent_dir = parts[-3]
    parent_fields = parent_dir.split("_")
    if len(parent_fields) < 2:
        raise ValueError(f"Malformed parent directory: {parent_dir}")

    robot_name, parent_dt = parent_fields[0], parent_fields[1]
    child_dt = Path(parts[-1]).stem.split("_")[-2]

    return robot_name, parent_dt, child_dt
