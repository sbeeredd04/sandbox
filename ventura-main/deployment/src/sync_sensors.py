#!/usr/bin/env python3
"""
Synchronize camera + lidar with ApproximateTimeSynchronizer and republish
to "<camera_topic>_synchronized" and "<cloud_topic>_synchronized".

Assumes camera is sensor_msgs/CompressedImage (change type if you use raw Image).
"""

import argparse
import yaml
import rospy
import numpy as np
from copy import deepcopy

from sensor_msgs.msg import CompressedImage, PointCloud2
# If your camera is raw, use:
# from sensor_msgs.msg import Image as CompressedImage  # and update help text above

from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer


def load_robot_config(path: str):
    """Load robot_config.yaml and resolve {ROBOT_NAME} substitutions."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    robot_name = cfg.get("robot_name", "")
    # replace all string values that contain {ROBOT_NAME}
    def _subst(v):
        if isinstance(v, str):
            return v.replace("{ROBOT_NAME}", robot_name)
        if isinstance(v, dict):
            return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_subst(x) for x in v]
        return v
    return _subst(cfg)


def main():
    pa = argparse.ArgumentParser(description="Synchronize camera+lidar and republish with /synchronized suffix.")
    pa.add_argument("--robot_config", default="deployment/config/robot_convoi.yaml",
                    help="Path to robot_config.yaml that contains camera_topic and cloud_topic")
    pa.add_argument("--slop", type=float, default=0.30,
                    help="ApproximateTimeSynchronizer allowed stamp difference (seconds)")
    pa.add_argument("--ats_queue", type=int, default=30,
                    help="ApproximateTimeSynchronizer internal queue size")
    pa.add_argument("--sub_queue", type=int, default=5,
                    help="Per-topic subscriber queue size")
    pa.add_argument("--buff_mb", type=int, default=16,
                    help="Subscriber buff_size in MB (helps with large PointCloud2)")
    pa.add_argument("--align_timestamps", action="store_true",
                    help="If set, publish both messages with the IMAGE timestamp (copies messages)")
    args = pa.parse_args()

    cfg = load_robot_config(args.robot_config)
    cam_topic   = cfg.get("camera_topic", None)
    cloud_topic = cfg.get("cloud_topic", None)
    if not cam_topic or not cloud_topic:
        raise ValueError("robot_config.yaml must contain both 'camera_topic' and 'cloud_topic'.")

    out_cam_topic   = f"{cam_topic}/synchronized"
    out_cloud_topic = f"{cloud_topic}/synchronized"

    rospy.init_node("sync_sensors", anonymous=False)
    rospy.loginfo(f"[sync] camera  : {cam_topic}  -> {out_cam_topic}")
    rospy.loginfo(f"[sync] lidar   : {cloud_topic} -> {out_cloud_topic}")
    rospy.loginfo(f"[sync] slop={args.slop:.3f}s  ats_queue={args.ats_queue}  sub_queue={args.sub_queue}  "
                  f"buff={args.buff_mb}MB  align_ts={args.align_timestamps}")

    # Publishers
    pub_cam   = rospy.Publisher(out_cam_topic,   CompressedImage, queue_size=10)
    pub_cloud = rospy.Publisher(out_cloud_topic, PointCloud2,     queue_size=10)

    buff_size = (1 << 20) * int(max(1, args.buff_mb))

    # Subscribers for ATS (message_filters.Subscriber forwards kwargs to rospy.Subscriber)
    img_sub   = MFSubscriber(cam_topic,   CompressedImage,
                             queue_size=args.sub_queue, buff_size=buff_size, tcp_nodelay=True)
    cloud_sub = MFSubscriber(cloud_topic, PointCloud2,
                             queue_size=args.sub_queue, buff_size=buff_size, tcp_nodelay=True)

    ats = ApproximateTimeSynchronizer([img_sub, cloud_sub],
                                      queue_size=args.ats_queue,
                                      slop=args.slop,
                                      allow_headerless=False)

    # State for basic rate/latency logging
    last_it = None
    last_pub = rospy.Time.now()

    def on_sync(img_msg: CompressedImage, cloud_msg: PointCloud2):
        nonlocal last_it, last_pub
        it = img_msg.header.stamp.to_sec()
        ct = cloud_msg.header.stamp.to_sec()
        dt = abs(ct - it)

        if args.align_timestamps:
            # Deep-copy only if we are modifying stamps (avoid mutating incoming messages)
            img_out = deepcopy(img_msg)
            cloud_out = deepcopy(cloud_msg)
            img_out.header.stamp = img_msg.header.stamp
            cloud_out.header.stamp = img_msg.header.stamp  # align to image time
        else:
            img_out = img_msg
            cloud_out = cloud_msg

        pub_cam.publish(img_out)
        pub_cloud.publish(cloud_out)

        # Light logging
        now = rospy.Time.now()
        if last_it is not None and (now - last_pub).to_sec() > 1.0:
            rospy.loginfo(f"[sync] paired dt={dt:.3f}s  image_ts={it:.3f}  cloud_ts={ct:.3f}")
            last_pub = now
        last_it = it

    ats.registerCallback(on_sync)

    rospy.loginfo("[sync] ready; spinning…")
    rospy.spin()


if __name__ == "__main__":
    main()
