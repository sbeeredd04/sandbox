
import s3fs
import fsspec
from pathlib import Path
import numpy as np
import cv2, imageio.v3 as iio                     # imageio‑v3 is lightweight
from rosbags.highlevel import AnyReader
from tqdm.auto import tqdm 

def _local_path(fs, s3_path: str) -> str:
    """Open the S3 file via filecache and return the local cache filename."""
    f = fs.open(s3_path, "rb")
    return f.name

def bags_to_video(
    s3_paths,                    # iterable[str | Path]
    topic,                        # '/robot/camera_front/image_raw/compressed'
    out_mp4="output.mp4",
    fps=15,
    quality=7,                    # 0‑10, lower = better quality (≈ CRF 22)
):
    """
    Aggregate *all* CompressedImage frames from multiple ROS2 bags
    into a single H.264 MP4.

    Returns the output video path.
    """
    writer = None
    n_frames = 0

    for s3_path in tqdm(s3_paths, desc="bags"):
        local_bag = _local_path(fs, s3_path)
        print("Processing bag:", local_bag)
        with AnyReader([Path(local_bag)]) as r:
            conn = next((c for c in r.connections if c.topic == topic), None)
            if conn is None:
                continue

            for _, _, raw in r.messages(connections=[conn]):
                msg = r.deserialize(raw, conn.msgtype)
                # --- CompressedImage → RGB numpy ---------------------------
                img = cv2.imdecode(
                    np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
                )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # lazy‑init the video writer when first frame arrives
                h, w = img.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))                    
                n_frames += 1

    if writer:
        writer.release()
    else:
        raise RuntimeError(f"No frames for topic '{topic}' found in supplied bags.")

    print(f"✔️  Wrote {n_frames} frames → {out_mp4}")
    return Path(out_mp4)

def key_to_bags(root_path, robot_name, topics):
    bag_path = f"{root_path}/rosbag"
    fs = s3fs.S3FileSystem(anon=False)
    all_objs = fs.ls(str(bag_path), detail=False)

    matches = []
    for key_seg in topics:
        prefix = f"{robot_name}_{key_seg}_"
        for obj in all_objs:
            fname = Path(obj).name
            if fname.startswith(prefix) and fname.endswith(".bag"):
                matches.append(obj)
    return matches

ROBOT_NAME = "ferrite2"
S3_BAG_CACHE_DIR = "/tmp/s3_bag_cache"

s3_bucket = "ai4h-datasets"
# s3_path = f"aih/seattle_250528_163305/2025-05-30/{ROBOT_NAME}_2025-05-30-14-58-00_test"
s3_path = f"aih/madero_rainforest/2025-05-23/{ROBOT_NAME}_2025-05-23-12-29-00_test"
s3_full_path = f"s3://{s3_bucket}/{s3_path}"

fs = fsspec.filesystem(
    "filecache",
    target_protocol="s3",
    cache_storage=S3_BAG_CACHE_DIR,
    default_fill_cache=False,       # only cache bytes you read
    target_options={"anon": False},
)

# ---------------- example usage ------------------------------------
topics_dict = {
    "front_rgb": {
        "ros_topic": f"/{ROBOT_NAME}/camera_front/color/image_raw/compressed",
        "s3_key": "vision_xavier"
    },
}
s3_keys  = [v["s3_key"]  for v in topics_dict.values()]
s3_topics = [v["ros_topic"] for v in topics_dict.values()]
bags = key_to_bags(s3_full_path, ROBOT_NAME, s3_keys)
bags = [Path(bag) for bag in bags]

bags_to_video(
    bags,
    s3_topics[0],
    out_mp4="front_cam.mp4",
)