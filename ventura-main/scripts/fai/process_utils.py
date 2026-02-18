"""
Lean helpers to pull RGB sequences and odometry data out of a ROS bag
and save them immediately to disk.

• RGB   →  MP4 + HDF5 (timestamps + frame-shape metadata)
• Odom  →  tab-separated TXT (human-readable) + pandas DataFrame in-memory
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Any, Union
from collections import defaultdict
from collections import OrderedDict

import cv2
from imageio import get_writer
import numpy as np
import pandas as pd
import hickle as hkl
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from pytransform3d.transform_manager import TransformManager

from scripts.utils.log_utils import logging
from scripts.fai.bag_helpers import (
    key_to_bags,
    get_ros_type,
    group_bag_chunks,
    print_bag_info
)
from scripts.preprocessing.run_engine import (
    apply_velocity_filter,
    apply_odometry_filter
)
from scripts.utils.time_utils import (
    find_contiguous_true_intervals
)
from scripts.fai.fai_helpers import (
    get_fai_s3id
)
from spinflow.dataset.frodo_helpers import (
    get_frodo_id,
    set_frodo_dir,
)

SAVE_VIDEO = True # set to false to speed up processing

# ───────────────────────── generic helpers ──────────────────────────
def _local_bag_path(fs, s3_path: str) -> str:
    """Open the S3 file via filecache and return the local cache filename."""
    f = fs.open(s3_path, "rb")
    return f.name

def _topic_to_filename(topic: str, ext: str | None = None) -> Path:
    """"""
    topic = topic.strip()                      # trim accidental spaces
    segs  = [s for s in topic.lstrip("/").split("/") if s]

    if len(segs) < 2:                          # nothing to parse
        return Path(topic.lstrip("/").replace("/", "_") + ext)

    sensor = segs[1]                           # usually the sensor label

    # special-case: Ouster IMU   →  "ouster_imu_data"
    if sensor == "ouster" and (len(segs) >= 3 and segs[2] == "imu"):
        sensor = "ouster_imu_data"

    if ext is None:
        return f"{sensor}"
    return f"{sensor}.{ext}"

def _flatten_ros_msg(msg, prefix: str = "", out: Dict[str, Any] | None = None):
    """Recursively copy all numeric ROS fields into a flat dict."""
    if out is None:
        out = {}
    for name in getattr(msg, "__slots__", []):
        val = getattr(msg, name)
        key = f"{prefix}{name}"
        if isinstance(val, (int, float, bool)):
            out[key] = val
        elif hasattr(val, "__slots__"):
            _flatten_ros_msg(val, f"{key}_", out)
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, float, bool)):
            out[key] = list(val)
    return out

def _camera_info_to_yaml(info: Dict[str, Any], stream) -> None:
    """
    Write a ROS-style CameraInfo mapping to `stream` (file or stdout),
    with D/K/R/P as inline [ … ] sequences.
    """
    # 1) Build a CommentedMap for insertion‐order safety and ruamel hooks
    m = CommentedMap()
    m["image_width"]      = int(info["width"])
    m["image_height"]     = int(info["height"])
    m["distortion_model"] = info["distortion_model"]
    m["frame_id"]         = info["frame_id"]

    # 2) Wrap each of your lists in a CommentedSeq and force flow style
    for key in ("D", "K", "R", "P"):
        arr = np.array(info[key], dtype=float).reshape(-1).tolist()
        seq = CommentedSeq(arr)
        seq.fa.set_flow_style()         # ← this makes [a, b, c, …]
        m[key] = seq

    # 3) Dump with ruamel.yaml
    yaml = YAML()
    yaml.default_flow_style = False    # block style for maps
    yaml.dump(m, stream)

def _transforms_to_yaml(
    transforms: Dict[str, List[float]],
    stream
) -> None:
    """
    Write a mapping of frame-pair keys to 16-element transform lists,
    using inline sequences. Cast values to native Python floats.
    """
    m = CommentedMap()
    for key, flat in transforms.items():
        flat_py = [float(val) for val in flat]
        seq = CommentedSeq(flat_py)
        seq.fa.set_flow_style()
        m[key] = seq

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.dump(m, stream)

def build_transforms(
    bag: Union[str, Path],
    topics: List[str],
) -> TransformManager:
    """
    Read all TF messages on *any* of `topics` in `bag` (e.g. ['/tf', '/tf_static']),
    build a TransformManager containing every parent->child edge.
    Note: parent_frame == header.frame_id, child_frame == child_frame_id.
    """
    tm = TransformManager()
    with AnyReader([Path(bag)]) as reader:
        # collect every connection whose topic is in our list
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            raise ValueError(f"No TF topics {topics} found in {bag}")

        # iterate them all
        for conn in conns:
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                for tf in msg.transforms:
                    parent = tf.header.frame_id
                    child  = tf.child_frame_id
                    t = tf.transform.translation
                    q = tf.transform.rotation

                    # build 4×4 homogeneous matrix:
                    # rotation first
                    R_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                    T = np.eye(4, dtype=float)
                    T[:3, :3] = R_mat
                    T[:3,  3] = [t.x, t.y, t.z]

                    """
                    ROS uses transform from child to parent,
                    so we need to invert it for TransformManager.
                    """
                    tm.add_transform(
                        to_frame=parent,
                        from_frame=child,
                        A2B=T
                    )
    return tm

# ───────────────────────── RGB processor ────────────────────────────

def process_rgb(
    bags: List[str],
    out_dir: str,
    topics: List[str],
    save_prefixes: List[str],
    seq: str,
    fps: int = 15,
    process_bag: bool = True
) -> Dict[str, str]:
    """
    Aggregate frames from *bags* on *topics* into ONE near-lossless MP4 **and**
    a lighter “preview” MP4, plus <sensor>_timestamps.csv.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-topic state ──────────────────────────────────────────────
    artefacts       = {t: {}   for t in topics}
    writers         = {t: None for t in topics}   # RGB / near-lossless
    writers_lossy   = {t: None for t in topics}   # NEW – browser preview
    cam_ts          = {t: []   for t in topics}
    infos           = {t: {"timestamps": [], "camera_info": []} for t in topics}

    # ── iterate through bag files exactly like before ───────────────
    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conns = {c.topic: c for c in reader.connections if c.topic in topics}
            if not conns:
                continue
            conn_to_topic = {c: t for t, c in conns.items()}

            for conn, _, raw in reader.messages(connections=list(conns.values())):
                topic = conn_to_topic[conn]
                msg   = reader.deserialize(raw, conn.msgtype)
                ts    = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                # (CameraInfo branch unchanged) --------------------------------
                if conn.msgtype.endswith("CameraInfo"):
                    intr = {
                        "K": msg.K.tolist(), "D": msg.D.tolist(), "R": msg.R.tolist(),
                        "P": msg.P.tolist(), "width": msg.width, "height": msg.height,
                        "distortion_model": msg.distortion_model,
                        "frame_id": msg.header.frame_id,
                    }
                    infos[topic]["camera_info"].append(intr)
                    infos[topic]["timestamps"].append(ts)
                    continue

                # (Image / CompressedImage decode unchanged) ------------------
                if conn.msgtype.endswith("CompressedImage"):
                    img = cv2.imdecode(
                        np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif conn.msgtype.endswith("Image"):
                    img = np.frombuffer(msg.data, np.uint8).reshape(
                        msg.height, msg.width, -1
                    )
                    if msg.encoding in ("bgr8", "bgr"):
                        img = img[..., ::-1]
                else:
                    continue

                cam_ts[topic].append(ts)

                if not process_bag or not SAVE_VIDEO:
                    continue

                # ── create writers the first time we see this topic ─────────
                if writers[topic] is None:
                    prefix        = save_prefixes[topics.index(topic)]
                    lossless_path = out_dir / f"{prefix}.mp4"
                    lossy_path    = out_dir / f"{prefix}_lossy.mp4"   # <── NEW

                    # 1) near-lossless 4 : 4 : 4 RGB master
                    writers[topic] = get_writer(
                        lossless_path, fps=fps,
                        codec="libx264rgb",         # keeps true RGB, no subsampling
                        pixelformat="rgb24",
                        ffmpeg_params=[
                            "-crf", "10",            # mathematically lossless
                            "-preset", "slow",
                            "-movflags", "+faststart"
                        ]
                    )
                    # 2) lighter 4 : 2 : 0 preview (unchanged quality line)
                    writers_lossy[topic] = get_writer(
                        lossy_path, fps=fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        ffmpeg_params=[
                            "-crf", "18",
                            "-preset", "veryfast",
                            "-movflags", "+faststart"
                        ]
                    )

                # ── write the frame to both files ───────────────────────────
                writers[topic].append_data(img)
                writers_lossy[topic].append_data(img)

    # ── finalise & save artefacts (minor additions only) ─────────────────────
    for topic, writer in writers.items():
        prefix       = save_prefixes[topics.index(topic)]
        lossless_mp4 = out_dir / f"{prefix}.mp4"
        lossy_mp4    = out_dir / f"{prefix}_lossy.mp4"
        csv_path     = out_dir / f"{prefix}_timestamps_{seq}.csv"

        if writer:
            writer.close()
        if writers_lossy[topic]:
            writers_lossy[topic].close()

        if cam_ts[topic]:
            pd.DataFrame({
                "timestamp": np.asarray(cam_ts[topic], dtype=np.float64)
            }).to_csv(csv_path, index=False)

            artefacts[topic].update({
                "video"        : str(lossless_mp4),   # keep key name for back-compat
                "video_lossy"  : str(lossy_mp4),      # NEW – preview copy
                "timestamps"   : str(csv_path),
            })

        if infos[topic]["camera_info"]:
            caminfo_path = out_dir / f"{prefix}_{seq}.yaml"
            with caminfo_path.open("w") as f:
                _camera_info_to_yaml(infos[topic]["camera_info"][0], f)
            artefacts[topic]["camera_info"] = str(caminfo_path)

    return artefacts

# ───────────────────────── odometry processor ───────────────────────
def process_odometry(
    bags: List[str],
    topic: str,
    out_dir: str | Path,
    save_prefix: str,
    save_file: bool = True,
):
    """
    Collapse *bags* on *topic* (nav_msgs/Odometry) into a single CSV:

        ts,x,y,z,qw,qx,qy,qz   (comma-separated, ts in **seconds**)

    Returns
    -------
    {"csv": <path>, "df": pandas.DataFrame}
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / _topic_to_filename(topic, "csv")
    rows: list[dict[str, float]] = []

    # ── gather rows across all bags ───────────────────────────────────────
    for bag in bags:
        with AnyReader([Path(bag)]) as reader:
            conn = next((c for c in reader.connections if c.topic == topic), None)
            if conn is None:
                continue

            for _, bag_ts, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)

                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # nav_msgs/Odometry -> msg.pose.pose.position / orientation
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                rows.append(
                    {
                        "timestamp": ts,      # nanoseconds → seconds
                        "x":  pos.x,
                        "y":  pos.y,
                        "z":  pos.z,
                        "qw": ori.w,             # note: qw FIRST
                        "qx": ori.x,
                        "qy": ori.y,
                        "qz": ori.z,
                    }
                )

    if not rows:
        return None

    # ── write CSV ---------------------------------------------------------
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if save_file:
        df.to_csv(csv_path, index=False)            # comma-separated by default

    return {"csv": str(csv_path), "df": df}

def process_controls(
    bags: List[str],
    topic: str,
    out_dir: str | Path,
    save_prefix: str,
    save_file: bool = True,
):
    """
    Collapse geometry-msgs control commands (Twist / TwistStamped / …)
    into one CSV with

        ts,linear_x,linear_y,linear_z,angular_x,angular_y,angular_z,(other…)

    Returns {"csv": <path>, "df": DataFrame}.
    """
    out_dir = Path(out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{save_prefix}.csv"

    rows: list[dict[str, float]] = []
    for bag in bags:
        with AnyReader([Path(bag)]) as r:
            conn = next((c for c in r.connections if c.topic == topic), None)
            if conn is None:
                continue

            for _, ts_ns, raw in r.messages(connections=[conn]):
                msg = r.deserialize(raw, conn.msgtype)
                # Handle Twist vs. TwistStamped transparently
                twist = msg.twist if hasattr(msg, "twist") else msg
                # Compute flat ground speed and angular velocity
                linear_magnitude = np.sqrt(
                    twist.linear.x ** 2 +
                    twist.linear.y ** 2
                )
                angular_magnitude = twist.angular.z
                row = {
                    "linear": linear_magnitude,
                    "angular": angular_magnitude,
                    "linear_x":  twist.linear.x,
                    "linear_y":  twist.linear.y,
                    "linear_z":  twist.linear.z,
                    "angular_x": twist.angular.x,
                    "angular_y": twist.angular.y,
                    "angular_z": twist.angular.z,
                    "timestamp": ts_ns * 1e-9,
                }
                rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if save_file:
        df.to_csv(csv_path, index=False)

    return {"csv": str(csv_path), "df": df}

def process_tf(
    bag: str | Path,
    topics: List[str],
    frame_pairs: List[Tuple[str, str]],
) -> Dict[str, str]:
    """
    Build a TransformManager from `bag`/`topic`, then for each (src, tgt)
    in `frame_pairs`, lookup the 4×4 transform matrix (skipping pairs
    that cannot be connected). Store into a dict and write to YAML.

    Returns
    -------
    {"yaml": <path_to_yaml>}
    """
    # build the transform graph
    tm = build_transforms(bag, topics)

    # collect transforms in a dict: 'src tgt' -> flat 16-element list
    transforms = {}
    for src, tgt in frame_pairs:
        try:
            """
            ROS uses transform from parent to child
            so we need to invert it for TransformManager.
            """
            T = tm.get_transform(
                from_frame=tgt,
                to_frame=src
            )
        except Exception:
            continue

        key = f"{src} {tgt}"
        transforms[key] = list(T.flatten())

    return {"transforms": transforms}

def check_session_done(sess_dir: str | Path,
                       topics: Dict[str, Dict[str, Any]]) -> bool:
    """
    Return **True** if *sess_dir* already contains at least one file whose
    name starts with **every** expected ``save_prefix`` listed in *topics*.

    A ``save_prefix`` may be a single string or a list/tuple (one per image
    topic).  The check is intentionally loose – any extension/suffix after
    the prefix (``.mp4``, ``_lossy.mp4``, ``.csv`` …) is accepted.
    """
    sess_dir = Path(sess_dir)
    if not sess_dir.exists():
        return False

    for spec in topics.values():
        pref = spec["save_prefix"]
        prefs = pref if isinstance(pref, (list, tuple)) else [pref]

        # each prefix must match at least one file
        for p in prefs:
            found_files = list(sess_dir.glob(f"{p}*"))
            if not found_files:
                return False
    return True

# ───────────────────────── orchestrator  ────────────────────────────
def inspect_and_stream(
    fs,
    remote_dir: str,
    save_root_dir: str,
    robot_name: str,
    topics_dict: Dict[str, Dict[str, str]],
    filters_dict: Dict[str, Any] | None = None
):
    """
    For every child-timestamp chunk:
        * write ONE MP4 per image topic (with `<child_dt>` suffix)
        * write ONE odometry CSV for the whole parent mission
        * emit missions.csv listing   child_dt, video, odometry
    """
    # find all bags grouped by sensor key
    bags_dict = key_to_bags(remote_dir, robot_name, list(topics_dict.keys()))
    if not bags_dict:
        logging.info("No matching bags.")
        return False

    # -------------------- process odometry first (only once) -------------
    odom_key   = next(k for k, v in topics_dict.items() if v["save_prefix"] == "odometry")
    odom_paths = bags_dict.pop(odom_key, [])
    assert odom_paths, "No odometry bags found"

    robot, parent_dt, odom_dt = get_fai_s3id(odom_paths[0])
    try:
        local_odom = [_local_bag_path(fs, p) for p in odom_paths]
    except Exception as e:
        logging.error(f"Failed to open odometry bags: {e}")
        return False

    odom_art = process_odometry(
        local_odom,
        topics_dict[odom_key]["ros_topic"],
        "",
        topics_dict[odom_key]["save_prefix"],
        save_file=False
    )
    assert odom_art, "Failed to generate odometry CSV"
    odom_df = odom_art["df"]

    # -------------------- process static TFs (if any) -------------------
    tf_static_key = next((k for k, v in topics_dict.items() if v
                        ["save_prefix"] == "tf_static"), None)
    static_tf_paths = bags_dict.pop(tf_static_key, [])
    assert static_tf_paths, "No static TF bags found"
    try:
        local_static_tf = [_local_bag_path(fs, p) for p in static_tf_paths]
    except Exception as e:
        logging.error(f"Failed to open static TF bags: {e}")
        return False

    tf_art = process_tf(
        local_static_tf[0],
        topics_dict[tf_static_key]["ros_topic"],
        topics_dict[tf_static_key]["frames"]
    )
    assert tf_art, "Failed to generate static TF YAML"
    static_transforms = tf_art["transforms"]

    # -------------------- process control commands (if any) -------------
    ctrl_key = next(k for k, v in topics_dict.items()
                    if v["save_prefix"] == "control")
    ctrl_paths = bags_dict.pop(ctrl_key, [])
    assert ctrl_paths, "No controls bags found"
    try:
        local_ctrl = [_local_bag_path(fs, p) for p in ctrl_paths]
    except Exception as e:
        logging.error(f"Failed to open control bags: {e}")
        return False
    ctrl_art = process_controls(
        local_ctrl,
        topics_dict[ctrl_key]["ros_topic"],
        "",                                  # temp dir; we’ll copy later
        topics_dict[ctrl_key]["save_prefix"],
        save_file=False,
    )
    assert ctrl_art, "Failed to create controls CSV"
    ctrl_df = ctrl_art["df"]

    # -------------------- apply controls filter --------------------------
    if filters_dict:
        for fkey, fdict in filters_dict.items():
            if fkey == "velocity_filter":
                # Format for cross compatibility: create contorls np array with lin and ang velocityies
                data_dict = {
                    "controls": np.column_stack(
                        (ctrl_df["linear"], ctrl_df["angular"], ctrl_df["timestamp"])
                    )
                }
                mask = apply_velocity_filter(data_dict, fdict)
                subsequences = find_contiguous_true_intervals(mask, window_len=fdict['params']['min_length'])
                if len(subsequences) == 0:
                    logging.warning(f"Failed {fkey} filter: no valid subsequences found.")
                    return False
            elif fkey == "odometry_filter":
                data_dict = {
                    "odometry": odom_df.to_numpy(),
                    "timestamps": odom_df["timestamp"].to_numpy()
                }
                mask = apply_odometry_filter(data_dict, fdict)
                subsequences = find_contiguous_true_intervals(mask, window_len=fdict['params']['horizon_frames'])
                if len(subsequences) == 0:
                    logging.warning(f"Failed {fkey} filter: no valid subsequences found.")
                    return False

    # -------------------- image topics and camera infos per child_dt ----------------------
    ride_results = defaultdict(dict)      # child_dt → {topic: artefact}
    metadata_rows = []

    for key_seg, s3_paths in bags_dict.items():              # loop sensors
        ros_topics   = topics_dict[key_seg]["ros_topic"]
        save_prefixes = topics_dict[key_seg]["save_prefix"]
        
        for chunk_idx, (child_dt, chunk) in enumerate(group_bag_chunks(s3_paths).items()):
            out_dir = set_frodo_dir(
                save_root_dir, parent_dt, robot, child_dt, chunk_idx
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            odom_path = out_dir / f"{topics_dict[odom_key]['save_prefix']}_data_{robot}.csv"
            ctrl_path = out_dir / f"{topics_dict[ctrl_key]['save_prefix']}_data_{robot}.csv"
            tf_path = out_dir / f"{topics_dict[tf_static_key]['save_prefix']}_{robot}.yaml"

            process_bag = True
            if check_session_done(out_dir, topics_dict):
                logging.info(f"✓  {out_dir} already processed – skipping")
                process_bag = False

            local_bags = [_local_bag_path(fs, p) for p in chunk]

            # print_bag_info(local_bags[0])
            # skip if topic absent in first bag
            for topic in ros_topics:
                if get_ros_type(local_bags[0], topic) is None:
                    logging.warning(f"Topic {topic} not found in {local_bags[0]}")
                    continue

            art = process_rgb(
                local_bags, out_dir, ros_topics,
                save_prefixes, seq=robot, fps=15, process_bag=process_bag
            )
            if not art:
                logging.warning(f"No frames found for {ros_topics} in {child_dt}")
                continue

            # Only save these if requested
            odom_df.to_csv(odom_path, index=False)
            ctrl_df.to_csv(ctrl_path, index=False)
            with tf_path.open("w") as f:
                _transforms_to_yaml(static_transforms, f)
            
            # Assume the first topic is the main video
            metadata_rows.append(
                {
                    "child_dt": child_dt,
                    "video":    art[ros_topics[0]]["video"],
                    "odometry": odom_path,
                    "controls": ctrl_path,
                    "tf_static": tf_path,
                    "robot":    robot_name,
                }
            )

    # -------------------- metadata.csv -----------------------------------
    metadata_df = pd.DataFrame(metadata_rows)
    # metadata_path = Path(save_root_dir) / "ride_manifest.csv"
    # metadata_df.to_csv(metadata_path, index=False)
    return metadata_df