import re
import s3fs
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rosbags.highlevel import AnyReader

from scripts.fai.fai_helpers import (
    get_fai_s3id
)

""" BEGIN ROS HELPERS"""

_TS_FMT   = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"   # YYYY-MM-DD-HH-MM-SS
_CHUNK_RX = rf"{_TS_FMT}_(\d+)\.bag$"                 # …_<chunk>.bag

def key_to_bags(
    remote_path: str,
    robot_name: str,
    topics: List[str],
) -> Dict[str, List[str]]:
    """
    Return {key_seg : [bag_paths]} where **every** filename matches

        <robot_name>_<key_seg>_<YYYY-MM-DD-HH-MM-SS>_<chunk>.bag
    """
    fs       = s3fs.S3FileSystem(anon=False)
    bag_root = f"{remote_path}/rosbag"
    all_objs = fs.ls(bag_root, detail=False)

    matches: Dict[str, List[str]] = {k: [] for k in topics}

    for key_seg in topics:
        # compile once per key_seg for speed
        full_rx = re.compile(
            rf"^{re.escape(robot_name)}_{re.escape(key_seg)}_{_CHUNK_RX}"
        )

        for obj in all_objs:
            filename = Path(obj).name
            if full_rx.match(filename):
                matches[key_seg].append(obj)

    return matches
    

def get_ros_type(bag_file: str | Path, topic: str) -> Optional[str]:
    """
    Return the fully-qualified ROS message type for *topic* inside *bag_file*.
    """
    bag_file = Path(bag_file)
    try:
        with AnyReader([bag_file]) as reader:
            # `reader.topics` is populated from the bag header(s) only
            info = reader.topics.get(topic)
            if info is not None:
                return info.msgtype          # e.g. "nav_msgs/Odometry"
    except FileNotFoundError:
        raise
    return None

def _chunk_index(s3_path: str) -> int:
    """Return the trailing chunk index (…_<idx>.bag → idx)."""
    return int(Path(s3_path).stem.split("_")[-1])

def group_bag_chunks(paths: List[str]) -> Dict[str, List[str]]:
    """
    Group bag paths into contiguous chunks.

    A new group starts every time a file with chunk-index 0 is encountered.
    The key for each group is that file’s *child* timestamp
    (format YYYY-MM-DD-HH-MM-SS).

    Parameters
    ----------
    paths : list[str]
        Unordered list of bag S3 keys.

    Returns
    -------
    dict[str, list[str]]
        { chunk_start_child_timestamp : [paths_in_chronological_order] }
    """
    # ── extract child-timestamp, idx, and sort chronologically ────────────
    records = []
    for p in paths:
        _, _, child_ts = get_fai_s3id(p)
        dt = datetime.strptime(child_ts, "%Y-%m-%d-%H-%M-%S")
        idx = _chunk_index(p)
        records.append((dt, child_ts, idx, p))

    records.sort(key=lambda x: x[0])      # earliest → latest

    # ── scan & group ───────────────────────────────────────────────────────
    groups: Dict[str, List[str]] = {}
    current_key = None

    for _, ts, idx, path in records:
        if idx == 0 or current_key is None:
            current_key = ts
            groups[current_key] = []
        groups[current_key].append(path)
    return groups

def print_bag_info(bag_path: str):
    """
    Print summary info for a ROS1 or ROS2 bag, with zero Python loops over messages.
    """

    topics_list = []
    with AnyReader([Path(bag_path)]) as reader:
        # overall metadata
        print(f"Start time: {reader.start_time} ns")
        print(f"End   time: {reader.end_time} ns")
        print(f"Duration:   {reader.duration} ns")
        print(f"Total msgs: {reader.message_count}")  # :contentReference[oaicite:0]{index=0}

        print("\nTopics:")
        for topic, info in reader.topics.items():
            print(f"  {topic:<30}  msgs: {info.msgcount:>6}  type: {info.msgtype}")  # :contentReference[oaicite:1]{index=1}
            topics_list.append(topic)
    print("\n")
    return topics_list

def stream_messages(bag_local_path: str, topics: str):
    """
    Print every message on `topic` from this bag,
    using AnyReader (ROS1 or ROS2) and its deserialize API.
    """
    print(f"\n--- streaming '{topics}' ---")
    msg_cache = { k: [] for k in topics }
    with AnyReader([Path(bag_local_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            print(f"No connections found for topic(s) '{topics}'")
            return

        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            msg_cache[connection.topic].append((timestamp, msg))

    return msg_cache