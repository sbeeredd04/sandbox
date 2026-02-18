#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bag → Rerun (.rrd) using the SAME Hydra policy robot config.

Expected Hydra config (same one you pasted), plus a tiny "io" group:
  io:
    bag_path: /abs/path/to/file.bag
    out_dir: /abs/path/to/outdir              # default: rerun_out
    write_blueprint: true                     # optional
    use_model: true                           # run your model & log plan/waypoints

Run examples:
  python bag_to_rerun_hydra.py io.bag_path:=/data/ride.bag robot_config:=deployment/config/robot.yaml
  python bag_to_rerun_hydra.py io.bag_path:=/data/ride.bag io.out_dir:=/tmp/rerun io.use_model:=false
"""

import os
import torch
import pathlib
from typing import Dict, Optional, Tuple, Iterable

import heapq
from bisect import bisect_left
import numpy as np
import cv2
import rosbag
import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from nav_msgs.msg import Path
import sensor_msgs.point_cloud2 as pc2

import rerun as rr
import rerun.blueprint as rrb
# from rerun.blueprint import Blueprint, Horizontal, Spatial3DView, Spatial2DView
# HAS_BLUEPRINT = True

import hydra
from omegaconf import DictConfig, OmegaConf

# ──────────────────────────────────────────────────────────────────────────────
# (Optional) your model bits — only used if io.use_model == true
try:
    from scripts.inference.build_model import build_model
    from deployment.src.utils import (
        ImagePreprocessor,
        apply_intrinsics_to_image,
    )
    HAVE_MODEL = True
except Exception:
    HAVE_MODEL = False

ROOT = "/world"
BASE = f"{ROOT}/base_link"
CAM  = f"{BASE}/camera"
IMG  = f"{CAM}/image"
DEVICE = "cuda" if HAVE_MODEL else "cpu"
SEED = 42
NUM_WAYPOINTS = 50  # fallback if your model expects it
DEFAULT_GOAL = "Continue straight ahead on the paved path. Keep a safe distance from any loose clutter."

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
def _resolve_robot_placeholders(d: Dict, robot_name: str) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = v.replace("{ROBOT_NAME}", robot_name)
        else:
            out[k] = v
    return out

def _quat_xyzw_to_rot(q: Iterable[float]) -> np.ndarray:
    """Return 3x3 rotation from (x,y,z,w)."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

def _intrinsics_from_caminfo(msg: CameraInfo) -> Dict:
    fx, fy = msg.K[0], msg.K[4]
    cx, cy = msg.K[2], msg.K[5]
    return dict(width=msg.width, height=msg.height, fx=fx, fy=fy, cx=cx, cy=cy)

def _project_points(P_cam: np.ndarray, intr: Dict) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = P_cam[:, 0], P_cam[:, 1], np.maximum(P_cam[:, 2], 1e-6)
    u = intr["fx"] * (x / z) + intr["cx"]
    v = intr["fy"] * (y / z) + intr["cy"]
    valid = (z > 0) & (u >= 0) & (v >= 0) & (u < intr["width"]) & (v < intr["height"])
    return np.stack([u, v], axis=-1), valid

def _colorize_points_from_image(P_cam: np.ndarray, img_rgb: np.ndarray, intr: Dict) -> np.ndarray:
    uv, ok = _project_points(P_cam, intr)
    uv_i = uv[ok].astype(np.int32)
    colors = np.zeros((P_cam.shape[0], 3), dtype=np.uint8)
    if uv_i.size > 0:
        colors_ok = img_rgb[uv_i[:, 1], uv_i[:, 0]]
        colors[ok] = colors_ok
    else:
        colors[:] = 255
    return colors

def _iter_topic(bag, topic):
    for _topic, msg, t in bag.read_messages(topics=[topic]):
        yield _topic, msg, t

def merge_by_header_time(bag, topics):
    iters = [_iter_topic(bag, tp) for tp in topics]
    heap, counter = [], 0

    def key_from(topic, msg, t):
        h = getattr(getattr(msg, "header", None), "stamp", None)
        return (h.to_sec() if h else t.to_sec(), t.to_sec())

    # prime the heap
    for i, it in enumerate(iters):
        try:
            topic, msg, t = next(it)
            heapq.heappush(heap, (*key_from(topic, msg, t), counter, i, topic, msg, t))
            counter += 1
        except StopIteration:
            pass

    while heap:
        _, _, _, i, topic, msg, t = heapq.heappop(heap)
        yield topic, msg, t
        try:
            topic2, msg2, t2 = next(iters[i])
            heapq.heappush(heap, (*key_from(topic2, msg2, t2), counter, i, topic2, msg2, t2))
            counter += 1
        except StopIteration:
            pass

# ──────────────────────────────────────────────────────────────────────────────
class BagToRerunHydra:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # IO
        assert 'bag_path' in cfg, "You must provide path to rosbag"
        # ---- IO / paths (Hydra overrides) ----
        self.bag_path = pathlib.Path(cfg['bag_path'])
        bag_parent_dir = self.bag_path.parent.name
        assert self.bag_path and self.bag_path.exists(), f"Missing or invalid io.bag: {self.bag_path}"

        self.out_dir = pathlib.Path(cfg.get("out_dir", "offline_replay_outputs")) / bag_parent_dir
        if self.out_dir:
            self.out_dir.expanduser().mkdir(parents=True, exist_ok=True)

        self.out_rrd = self.out_dir / (self.bag_path.stem + ".rrd")
        self.write_blueprint = bool(cfg.get("write_blueprint", True))
        self.use_model = bool(cfg.get("use_model", False))

        # Robot config (SAME file you use for inference)
        # If you call hydra as: robot_config:=deployment/config/robot.yaml
        # we read it directly. Otherwise assume cfg already *is* that content.
        if "robot_config" in cfg:
            rcfg_path = pathlib.Path(cfg["robot_config"]).expanduser()
            import yaml
            with open(rcfg_path, "r") as f:
                robot_cfg = yaml.safe_load(f)
        else:
            # Allow the robot fields to live at top-level (as in your paste)
            robot_cfg = OmegaConf.to_container(cfg, resolve=True)

        robot_name = robot_cfg.get("robot_name", "robot")
        robot_cfg = _resolve_robot_placeholders(robot_cfg, robot_name)

        # Topics & frames from the SAME YAML
        self.topic_img = robot_cfg["camera_topic"]
        self.topic_caminfo = robot_cfg["camera_info_topic"]
        self.topic_odom = robot_cfg["odom_topic"]
        # Their tf is namespaced; derive tf_static sibling in same namespace
        self.topic_tf = robot_cfg.get("tf_topic", "/tf")
        if self.topic_tf.endswith("/tf"):
            self.topic_tf_static = self.topic_tf.replace("/tf", "/tf_static")
        else:
            self.topic_tf_static = self.topic_tf + "_static"

        self.topic_cloud = robot_cfg.get("cloud_topic", "/ferrite7/raw_velodyne_points")

        self.frame_world = robot_cfg["odom_frame"]
        self.frame_base = robot_cfg["base_link_frame"]
        self.frame_optical = robot_cfg["optical_frame"]

        # For optional model
        self.model = None
        self.preproc = None
        self.goal_cmd = DEFAULT_GOAL

        # TF buffer
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(60.0*60.0))  # 1h

        # cached intrinsics & latest RGB for point colorization
        self.intr: Optional[Dict] = None
        self.last_rgb: Optional[np.ndarray] = None
        self.last_img_stamp: Optional[rospy.Time] = None
        self.img_ts: list[float] = []         # sorted header times (seconds)
        self.img_rgbs: list[np.ndarray] = []  # RGB frames aligned with img_ts
        self.max_image_slop = float(cfg.get("max_image_slop", 0.25))  # seconds; set None to disable

        # Initialize start and end times if specified
        self.start_time = cfg.get("start_time", None)  # seconds from beginning of bag
        self.end_time = cfg.get("end_time", None)      # seconds from beginning of bag

        # Rerun init
        rerun_name = f"{robot_name}_rerun"
        rr.init(rerun_name, recording_id=None)

        rr.set_sinks(rr.FileSink(str(self.out_rrd)))
        # World as right-handed Z-up (typical robotics world)
        rr.log(ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(BASE, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Optional: write a default viewer layout if supported
        self.topic_to_entity = {
            self.topic_caminfo: f"{CAM}/info",
            self.topic_img: {
                "camera": CAM,   # now under /world/base_link
                "rgb":    IMG,
            },
            self.topic_cloud: {
                "points": f"{BASE}/cloud/points",
                "raw_points": f"{ROOT}/cloud_raw/points",
            },
            self.topic_odom:  f"{BASE}",
        }
        if self.write_blueprint:
            bp = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Robot (base_link)", origin="/world/base_link"),
                    rrb.Spatial2DView(name="Camera", origin="/world/base_link/camera/image"),
                )
            )
            bp.save(rerun_name, str(self.out_dir / (self.bag_path.stem + ".rbl")))

        # Optional: build model if requested and available
        if self.use_model and HAVE_MODEL and "model" in cfg:
            self._build_model_and_preproc(cfg)

    # ----------------- model bits (optional) -----------------
    def _build_model_and_preproc(self, cfg: DictConfig):
        mdl_cfg = cfg.model
        out_range = cfg.dataset["output_range"]
        self.preproc = ImagePreprocessor(
            height=int(self._img_h()), width=int(self._img_w()),
            out_min=out_range[0], out_max=out_range[1],
        )
        ckpt_path = mdl_cfg.get("weights_ckpt", None)
        model = build_model(
            mdl_cfg, ckpt_path, mdl_cfg.get("vision_weights_ckpt", None),
            seed=SEED, device=DEVICE,
        )
        self.model = model.to(DEVICE).eval()

    def _img_w(self) -> int:
        return int(self.cfg.get("image_width", 320))

    def _img_h(self) -> int:
        return int(self.cfg.get("image_height", 240))

    def _prepare_inputs(self, img_rgb: np.ndarray) -> Dict:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img_rgb)
        img_tensor = self.preproc(img_pil)
        inputs = {}
        for meta in self.cfg.model["dataloader_inputs"]:
            if meta["in_key"] == "front_rgb":
                inputs[meta["out_key"]] = img_tensor.to(DEVICE)
            elif meta["in_key"] == "goal_caption":
                inputs[meta["out_key"]] = [self.goal_cmd]
        return inputs

    # ----------------- TF helpers -----------------
    def _ingest_tf(self, msg: TFMessage, is_static: bool):
        for ts in msg.transforms:
            try:
                if is_static and hasattr(self.tf_buf, "set_transform_static"):
                    self.tf_buf.set_transform_static(ts, "bag")
                else:
                    self.tf_buf.set_transform(ts, "bag")
            except TypeError:
                self.tf_buf.set_transform(ts, "bag")

    def _lookup(self, target: str, source: str, stamp: rospy.Time):
        try:
            return self.tf_buf.lookup_transform(target, source, stamp, rospy.Duration(0.2))
        except Exception:
            return self.tf_buf.lookup_transform(target, source, rospy.Time(0))

    # ----------------- logging -----------------
    def _log_camera_img(self, img_msg: CompressedImage, caminfo: Optional[CameraInfo], topic_to_entity: Dict[str, str]):
        rgb = cv2.imdecode(np.frombuffer(img_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # If your intrinsics expect a resize/crop, apply exactly what your pipeline does:
        if self.intr is not None:
            # keep resolution consistent with intrinsics
            rgb = cv2.resize(rgb, (self.intr["width"], self.intr["height"]), interpolation=cv2.INTER_LINEAR)

        self._store_image(img_msg.header.stamp, rgb)
        self.last_rgb = rgb
        self.last_img_stamp = img_msg.header.stamp

        # time
        image_entity  = topic_to_entity[self.topic_img]["rgb"]      # "world/camera/image"
        camera_entity = topic_to_entity[self.topic_img]["camera"]   # "world/camera"    
        rr.set_time_seconds("t", img_msg.header.stamp.to_sec())
        
        rr.log(image_entity, rr.Image(rgb))

        if caminfo is not None:
            self.intr = _intrinsics_from_caminfo(caminfo)
            rr.log(
                camera_entity,
                rr.Pinhole(
                    width=int(self.intr["width"]),
                    height=int(self.intr["height"]),
                    focal_length=np.array([self.intr["fx"], self.intr["fy"]], dtype=np.float32),
                    principal_point=np.array([self.intr["cx"], self.intr["cy"]], dtype=np.float32),
                ),
                static=True,
            )

        # Pose of CAMERA in the BASE frame (base_link_T_camera):
        try:
            ts_cb = self._lookup(self.frame_base, self.frame_optical, img_msg.header.stamp)  # target=base, source=optical
            qc = ts_cb.transform.rotation
            tc = ts_cb.transform.translation
            rr.log(CAM, rr.Transform3D(
                translation=[tc.x, tc.y, tc.z],
                rotation=rr.components.RotationQuat(xyzw=[qc.x, qc.y, qc.z, qc.w]),
            ))
        except Exception:
            pass

        # Also (recommended) log BASE in WORLD so the subtree is placed in world:
        try:
            ts_wb = self._lookup(self.frame_world, self.frame_base, img_msg.header.stamp)   # world_T_base
            qb = ts_wb.transform.rotation
            tb = ts_wb.transform.translation
            rr.log(BASE, rr.Transform3D(
                translation=[tb.x, tb.y, tb.z],
                rotation=rr.components.RotationQuat(xyzw=[qb.x, qb.y, qb.z, qb.w]),
            ))
        except Exception:
            pass

    def _log_cloud(self, pc_msg: PointCloud2):
        """Colorize LiDAR with the latest image, then log in base_link frame."""
        # ---- 1) Read XYZ from the PointCloud2 ----
        fields = [f.name for f in pc_msg.fields]
        want = ("x", "y", "z")
        pts = []
        for tup in pc2.read_points(pc_msg, field_names=want, skip_nans=True):
            pts.append((tup[0], tup[1], tup[2]))
        if not pts:
            return

        P = np.asarray(pts, dtype=np.float32)  # (N,3)
        stamp = getattr(pc_msg.header, "stamp", rospy.Time(0))
        pc_frame = pc_msg.header.frame_id.lstrip("/")  # normalize just in case

        # ---- 2) Try to colorize using the latest image (image <= lidar time) ----
        colors = None
        img_for_cloud = self._nearest_image(float(stamp.to_sec()))
        if self.intr is not None and img_for_cloud is not None:
            try:
                # Transform LiDAR points into the camera optical frame at this time:
                ts_pc_to_cam = self._lookup(self.frame_optical, pc_frame, stamp)  # target=optical, source=pc
                q = ts_pc_to_cam.transform.rotation
                t = ts_pc_to_cam.transform.translation
                R = _quat_xyzw_to_rot([q.x, q.y, q.z, q.w])
                tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
                P_cam = (R @ P.T + tvec.reshape(3, 1)).T  # (N,3) in camera optical coords

                # Project & sample image colors
                colors = _colorize_points_from_image(P_cam, img_for_cloud, self.intr)
            except Exception:
                colors = None

        # ---- 3) Transform points to base_link for visualization ----
        cloud_entity = self.topic_to_entity[self.topic_cloud]["points"]        # "world/base_link/cloud/points"
        cloud_raw_entity   = self.topic_to_entity[self.topic_cloud]["raw_points"]    # "
        try:
            ts_pc_to_base = self._lookup(self.frame_base, pc_frame, stamp)  # target=base, source=pc
            qb = ts_pc_to_base.transform.rotation
            tb = ts_pc_to_base.transform.translation
            Rb = _quat_xyzw_to_rot([qb.x, qb.y, qb.z, qb.w])
            tb_vec = np.array([tb.x, tb.y, tb.z], dtype=np.float32)
            P_base = (Rb @ P.T + tb_vec.reshape(3, 1)).T  # (N,3) in base_link

            rr.set_time_seconds("t", stamp.to_sec())
            if colors is not None:
                rr.log(cloud_entity, rr.Points3D(positions=P_base, colors=colors))
            else:
                rr.log(cloud_entity, rr.Points3D(P_base))
        except Exception:
            # Fallback: log raw points (still useful to see something)
            rr.set_time_seconds("t", stamp.to_sec())
            if colors is not None:
                rr.log(cloud_raw_entity, rr.Points3D(P, colors=colors))
            else:
                rr.log(cloud_raw_entity, rr.Points3D(P))

    def _log_waypoints_world(self, waypoints_base: np.ndarray, stamp: rospy.Time):
        """waypoints_base: (T,3) in base_link → transform to world and log."""
        try:
            ts = self._lookup(self.frame_world, self.frame_base, stamp)
            q = ts.transform.rotation
            t = ts.transform.translation
            R = _quat_xyzw_to_rot([q.x, q.y, q.z, q.w])
            tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
            P_w = (R @ waypoints_base.T + tvec.reshape(3, 1)).T
            rr.set_time_seconds("t", stamp.to_sec())
            rr.log("/world/trajectory", rr.LineStrips3D([P_w.astype(np.float32)], radii=0.05, colors=[255, 0, 0]))
        except Exception:
            pass

    def _log_plan_overlay(self, base_rgb: np.ndarray, plan_mask: np.ndarray, name="/camera/plan_overlay"):
        """
        plan_mask: HxW float or bool — we alpha-blend on top of base_rgb for a quick visual.
        """
        if plan_mask.dtype != np.uint8:
            # normalize to 0..255 heat (simple)
            m = np.clip(plan_mask.astype(np.float32), 0, 1)
            heat = (m * 255).astype(np.uint8)
        else:
            heat = plan_mask
        # simple pseudo-color: put heat in red channel
        overlay = base_rgb.copy()
        r = overlay[:, :, 0].astype(np.int16)  # R
        r = np.clip(r * 0.6 + heat * 0.4, 0, 255).astype(np.uint8)
        overlay[:, :, 0] = r
        rr.log(name, rr.Image(overlay))

    def _store_image(self, stamp: rospy.Time, rgb: np.ndarray):
        """Insert an image into a sorted (by time) cache for nearest-time lookup."""
        ts = float(stamp.to_sec())
        # If images arrive in-order, a fast path append keeps O(1) amortized.
        if not self.img_ts or ts >= self.img_ts[-1]:
            self.img_ts.append(ts)
            self.img_rgbs.append(rgb)
            return
        # Otherwise keep list sorted (rare offline)
        i = bisect_left(self.img_ts, ts)
        self.img_ts.insert(i, ts)
        self.img_rgbs.insert(i, rgb)

    def _nearest_image(self, ts: float) -> Optional[np.ndarray]:
        """Return the RGB image whose timestamp is closest to ts (seconds)."""
        if not self.img_ts:
            return None
        i = bisect_left(self.img_ts, ts)
        cand = []
        if i < len(self.img_ts):
            cand.append((abs(self.img_ts[i] - ts), i))
        if i > 0:
            cand.append((abs(self.img_ts[i-1] - ts), i-1))
        if not cand:
            return None
        dt, idx = min(cand)
        if self.max_image_slop is not None and dt > self.max_image_slop:
            return None
        return self.img_rgbs[idx]

    # ----------------- main loop -----------------
    def run(self):
        rospy.init_node("bag_to_rerun_hydra", anonymous=True, disable_signals=True)

        # Optional model guard
        if self.use_model and not HAVE_MODEL:
            rospy.logwarn("io.use_model==True but model deps unavailable; continuing without model.")
            self.use_model = False

        with rosbag.Bag(str(self.bag_path), "r") as bag:
            # print bag topics and number of messages per topic
            print(f"[Rerun] Reading bag: {self.bag_path}")
            # Iterate though all topics and only rpint ones with point cluod type
            print("Bag topics:")
            for topic, info in bag.get_type_and_topic_info().topics.items():
                if info.msg_type == "sensor_msgs/PointCloud2":
                    print(f"  {topic}: {info.msg_type} [{info.message_count} msgs]")

            last_caminfo = None
            # Preload TF (namespaced)
            for topic, msg, _ in bag.read_messages(topics=[
                self.topic_tf_static, self.topic_tf, self.topic_caminfo
            ]):
                if topic == self.topic_caminfo:
                    last_caminfo = msg
                    if self.intr is None:
                        self.intr = _intrinsics_from_caminfo(last_caminfo)
                else:
                    self._ingest_tf(msg, is_static=(topic == self.topic_tf_static))

            assert last_caminfo is not None, f"No CameraInfo on topic: {self.topic_caminfo}"

            # ── PASS 1: ingest TF + CameraInfo, then cache & log ALL IMAGES ─────────────
            last_caminfo = None
            for topic, msg, _ in bag.read_messages(topics=[self.topic_tf_static, self.topic_tf, self.topic_caminfo]):
                if topic == self.topic_caminfo:
                    last_caminfo = msg
                    if self.intr is None:
                        self.intr = _intrinsics_from_caminfo(last_caminfo)
                else:
                    self._ingest_tf(msg, is_static=(topic == self.topic_tf_static))

            assert last_caminfo is not None, f"No CameraInfo on topic: {self.topic_caminfo}"
            
            start_time = self.start_time if self.start_time is not None else bag.get_start_time()
            end_time = self.end_time if self.end_time is not None else bag.get_end_time()
            print(f"[Rerun] Processing messages from {start_time:.2f}s to {end_time:.2f}s (bag time)")

            # Cache & log images first (keeps img cache complete and sorted)
            for _, img_msg, _ in bag.read_messages(topics=[self.topic_img]):
                if img_msg.header.stamp.to_sec() < start_time or img_msg.header.stamp.to_sec() > end_time:
                    continue    
                self._log_camera_img(img_msg, last_caminfo, self.topic_to_entity)

            # ── PASS 2: iterate clouds; for each, pick nearest image for colorizing ─────
            for _, cloud_msg, _ in bag.read_messages(topics=[self.topic_cloud]):
                if img_msg.header.stamp.to_sec() < start_time or img_msg.header.stamp.to_sec() > end_time:
                    continue
                self._log_cloud(cloud_msg)

        print(f"[Rerun] wrote: {self.out_rrd}")
        if self.write_blueprint:
            print(f"[Rerun] wrote: {self.out_dir / (self.bag_path.stem + '.rbl')}")

        try:
            rr.disconnect()   # <-- add this
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../config", config_name="policy")
def main(cfg: DictConfig):
    """
    Reuses the SAME Hydra policy config you use for inference.
    Add a small "io" group at runtime via overrides, e.g.:
      io.bag_path:=/path/to/bag.bag io.out_dir:=/tmp/rerun io.use_model:=true
    """
    runner = BagToRerunHydra(cfg)
    runner.run()

if __name__ == "__main__":
    main()
