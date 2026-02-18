#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpinFlow Planner (simplified & robust)
- Subscribes to camera image + camera_info + odometry
- Caches camera intrinsics; opportunistically fills extrinsics (TF base_link <-> optical)
- Ensures data readiness before inference
- Inference runs in a background thread; never blocks ROS spin
- Always processes the latest image (drops older frames)
- Publishes annotated image (CompressedImage) and trajectory (Path)
"""

import os
import pathlib
import threading
from collections import deque
from typing import Optional, Dict

import numpy as np
import torch
import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from PIL import Image as PILImage

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.inference.build_model import build_model
from deployment.src.constants import CAM_TO_BASE_OFFSET
from deployment.src.utils import (
    compressed_imgmsg_to_pil,
    rescale_intrinsics,
    pil_to_compressed_imgmsg,
    apply_intrinsics_to_image,
    waypoints_to_path_msg,
    camera_info_to_dict,
    tf_to_se3,
    pose_to_se3,
    ImagePreprocessor,
    viz_predictions,
    smooth_robot_actions,
    save_path_to_file
)
from deployment.src.convoi import ConvoiPlannerNode

# ──────────────────────────────────────────────────────────────────────────────
# Configurable constants
# ──────────────────────────────────────────────────────────────────────────────
RATE_HZ = 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR = torch.Generator(device=DEVICE).manual_seed(SEED)
NUM_WAYPOINTS = 50
# NUM_WAYPOINTS = 8
CFG_SCALE = 8.0
NUM_DENOISING_STEPS = 40
DEFAULT_GOAL = "Go to the next waypoint"

# ──────────────────────────────────────────────────────────────────────────────
# Node Implementation
# ──────────────────────────────────────────────────────────────────────────────
class SpinflowPlannerNode:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.robot_cfg = self._load_robot_cfg(cfg.get("robot_config", "deployment/config/robot.yaml"))
        self.image_size = (self.robot_cfg["image_width"], self.robot_cfg["image_height"])

        # Frames & TF
        self.odom_frame = self.robot_cfg["odom_frame"]
        self.base_link_frame = self.robot_cfg["base_link_frame"]
        self.optical_frame = self.robot_cfg["optical_frame"]
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State
        self.tf_infos: Dict = {}  # {"intrinsics": {...}, "T_optical_to_base": np.ndarray, "T_base_to_optical": np.ndarray}
        self.latest_img_msg: Optional[CompressedImage] = None
        self.latest_img_stamp: float = -1.0
        self._img_lock = threading.Lock()

        self.latest_odom_pose = None
        self.latest_odom_stamp = -1.0
        self._odom_lock = threading.Lock()

        self.goal_cmd = DEFAULT_GOAL
        self.infer_enabled = bool(rospy.get_param("~infer_enabled", True))

        # Path smoothing (optional)
        self.path_sm_cfg = self.robot_cfg.get("path_smoothing", {"enable": False, "window_size": 3})
        self.action_key = "smoothed_action_pred" if self.path_sm_cfg.get("enable") else "action_pred"
        self._smooth_buf = {
            "odom_ts": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "odom_poses": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "img_ts": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "action_pred": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
        }

        # Publishers / Subscribers
        self.path_pub = rospy.Publisher(self.robot_cfg["action_topic"], Path, queue_size=1)
        self.path_img_pub = rospy.Publisher(self.robot_cfg["image_plan_topic"], CompressedImage, queue_size=1)

        rospy.Subscriber(self.robot_cfg["camera_topic"], CompressedImage, self._camera_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        rospy.Subscriber(self.robot_cfg["camera_info_topic"], CameraInfo, self._camera_info_cb, queue_size=10)
        rospy.Subscriber(self.robot_cfg["odom_topic"], Odometry, self._odom_cb, queue_size=20)
        rospy.Subscriber("/spinflow/goal_cmd_override", String, self._goal_cb, queue_size=1)
        rospy.Subscriber("/spinflow/infer_enabled", Bool, self._infer_toggle_cb, queue_size=1)

        # Model
        self.model, self.infer_kwargs, self.preproc = self._build_model_and_preproc(cfg)
        self._inference_running = threading.Event()
        self._last_started_img_ts = -1.0

        # Clean logging
        rospy.loginfo("SpinflowPlannerNode initialized")

    # ─────────────── Callbacks ───────────────
    def _camera_cb(self, msg: CompressedImage):
        ts = msg.header.stamp.to_sec()
        with self._img_lock:
            # Keep only the latest image (drop older)
            if ts >= self.latest_img_stamp:
                self.latest_img_stamp = ts
                self.latest_img_msg = msg

    def _camera_info_cb(self, msg: CameraInfo):
        # Always update intrinsics
        self.tf_infos["intrinsics"] = rescale_intrinsics(
            camera_info_to_dict(msg),
            image_size=self.image_size
        )
        # Cache optical_frame the first time we see it
        if not self.optical_frame:
            self.optical_frame = msg.header.frame_id
            rospy.loginfo_once(f"[camera_info] optical_frame='{self.optical_frame}'")

        # Try to refresh extrinsics (non-blocking-ish lookup)
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.base_link_frame, self.optical_frame, rospy.Time(0), rospy.Duration(0.2)
            )
            T_optical_to_base = tf_to_se3(tf_stamped)
            self.tf_infos["T_optical_to_base"] = T_optical_to_base
            self.tf_infos["T_base_to_optical"] = np.linalg.inv(T_optical_to_base)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException):
            # Fine; we'll keep trying as messages arrive
            pass

    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self.latest_odom_stamp = msg.header.stamp.to_sec()
            self.latest_odom_pose = pose_to_se3(msg.pose.pose)

    def _goal_cb(self, msg: String):
        self.goal_cmd = msg.data.strip() or DEFAULT_GOAL

    def _infer_toggle_cb(self, msg: Bool):
        self.infer_enabled = bool(msg.data)
        rospy.logwarn(f"[spinflow] Inference {'ENABLED' if self.infer_enabled else 'PAUSED'}")

    # ─────────────── Main Loop ───────────────
    def spin(self):
        rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown():
            # Check readiness
            if not self.infer_enabled:
                rospy.loginfo("[spinflow] Inference paused by user.")
                rate.sleep(); continue

            if not self._ready_for_inference():
                rate.sleep(); continue

            # Always choose the latest image; ensure only one in-flight
            if not self._inference_running.is_set():
                with self._img_lock:
                    img_msg = self.latest_img_msg
                    img_ts = self.latest_img_stamp

                # Avoid launching duplicate work on the same frame
                if img_msg is not None and img_ts > self._last_started_img_ts:
                    self._last_started_img_ts = img_ts
                    self._inference_running.set()
                    threading.Thread(
                        target=self._inference_worker,
                        args=(img_msg, img_ts),
                        daemon=True
                    ).start()

            rate.sleep()

    # ─────────────── Inference Worker ───────────────
    def _inference_worker(self, img_msg: CompressedImage, img_ts: float):
        try:
            # Snapshot the required state up front
            with self._odom_lock:
                odom_pose = self.latest_odom_pose
                odom_ts = self.latest_odom_stamp

            goal = self.goal_cmd
            img_pil = compressed_imgmsg_to_pil(img_msg)
            img_pil = apply_intrinsics_to_image(
                img_pil, 
                self.tf_infos["intrinsics"]
            )
            # img_pil = img_pil.resize(self.image_size, PILImage.BILINEAR)
            mdl_inputs = self._prepare_inputs(img_pil, goal)
            with torch.no_grad():
                outputs = self.model.infer(mdl_inputs, **self.infer_kwargs)

            # Optional smoothing (requires action_pred present)
            if self.path_sm_cfg.get("enable"):
                self._smooth_buf["odom_ts"].append(odom_ts)
                self._smooth_buf["odom_poses"].append(odom_pose)
                self._smooth_buf["img_ts"].append(img_ts)
                self._smooth_buf["action_pred"].append(outputs["action_pred"][0].cpu().numpy())

                if len(self._smooth_buf["action_pred"]) >= self.path_sm_cfg.get("window_size", 3):
                    smoothed = torch.from_numpy(smooth_robot_actions(**self._smooth_buf)).to(DEVICE).unsqueeze(0)
                    outputs["smoothed_action_pred"] = smoothed
                else:
                    # Not enough history yet; skip publish to avoid jerky starts
                    return

            # If the model already produced an annotated image, use it; else render
            if "annotated_img" in outputs:
                img_vis = PILImage.fromarray(outputs["annotated_img"]).convert("RGB")
            else:
                img_vis = viz_predictions(
                    img_pil,
                    outputs,
                    obs_range=self.cfg.dataset["output_range"],
                    infos=self.tf_infos,
                    pred_key=self.action_key,
                    vis_path=self.robot_cfg.get("visualize_path", True)
                )

            # Offset x to account for camera-to-base
            if "action_pred" in outputs:
                outputs["action_pred"][:, :, :] += torch.tensor(CAM_TO_BASE_OFFSET).to(DEVICE)
                outputs["action_pred"][:, :, 2] = torch.mean(outputs["action_pred"][:, :, 2])  # Center z around 0
                # outputs["action_pred"][:, :, 2] = -0.3
                # -torch.mean(outputs["action_pred"][:, :, 2])

            # Convert waypoints -> Path in base_link, then (optionally) transform to odom
            waypoint_frame = self.base_link_frame
            path_msg = waypoints_to_path_msg(
                outputs,
                waypoint_frame,
                img_ts,
                pred_key=self.action_key,
                num_samples=NUM_WAYPOINTS
            )

            if self.odom_frame != waypoint_frame:
                try:
                    tf_time = rospy.Time.from_sec(img_ts)
                    tf_stamped = self.tf_buffer.lookup_transform(self.odom_frame, waypoint_frame, tf_time, rospy.Duration(0.5))
                    for ps in path_msg.poses:
                        ps.header.stamp = tf_time
                    path_msg_odom = Path()
                    path_msg_odom.header.frame_id = self.odom_frame
                    path_msg_odom.header.stamp = tf_time
                    path_msg_odom.poses = []
                    for idx, ps in enumerate(path_msg.poses):
                        path_msg_odom.poses.append(do_transform_pose(ps, tf_stamped))
                    # path_msg_odom.poses = [do_transform_pose(ps, tf_stamped) for ps in path_msg.poses]
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
                    rospy.logwarn(f"[spinflow] TF lookup (odom<-{waypoint_frame}) failed: {e}")
                    return
            else:
                path_msg_odom = path_msg

            # Publish annotated image
            img_msg_out = pil_to_compressed_imgmsg(img_vis, frame_id=self.base_link_frame, stamp=rospy.Time.from_sec(img_ts))
            self.path_img_pub.publish(img_msg_out)

            # Optionally save and publish path
            if self.robot_cfg.get("save_plan", False):
                save_path_to_file(path_msg_odom, "path_plan.bag", path_topic=self.robot_cfg["action_topic"])
            self.path_pub.publish(path_msg_odom)

        except Exception as e:
            rospy.logerr(f"[spinflow] Inference worker error: {e}")
        finally:
            self._inference_running.clear()

    # ─────────────── Helpers ───────────────
    def _ready_for_inference(self) -> bool:
        if "intrinsics" not in self.tf_infos:
            rospy.logwarn_throttle(2.0, "[spinflow] Waiting for camera intrinsics…")
            return False
        if "T_optical_to_base" not in self.tf_infos:
            rospy.logwarn_throttle(2.0, "[spinflow] Waiting for TF base_link ↔ optical…")
            return False
        with self._odom_lock:
            if self.latest_odom_pose is None:
                rospy.logwarn_throttle(2.0, "[spinflow] Waiting for odometry…")
                return False
        with self._img_lock:
            if self.latest_img_msg is None:
                rospy.logwarn_throttle(2.0, "[spinflow] Waiting for camera images…")
                return False
        return True

    def _build_model_and_preproc(self, cfg: DictConfig):
        mdl_cfg = cfg.model
        # Image preprocessor
        img_range = cfg.dataset["output_range"]
        preproc = ImagePreprocessor(
            height=self.robot_cfg["image_height"],
            width=self.robot_cfg["image_width"],
            out_min=img_range[0],
            out_max=img_range[1]
        )

        # Model
        if mdl_cfg['model_name'] in ["WaypointFlowPolicy", "LeLaN_clip"]:
            ckpt_path = mdl_cfg.get("weights_ckpt", None)
            assert ckpt_path, "Model checkpoint path must be provided for this model."
            model = build_model(
                mdl_cfg,
                ckpt_path,
                mdl_cfg.get("vision_weights_ckpt", None),
                seed=SEED,
                device=DEVICE
            )
            # Scheduler defaults (if present)
            infer_kwargs = {}
            if "validation" in mdl_cfg and "scheduler" in mdl_cfg["validation"]:
                infer_kwargs = dict(mdl_cfg["validation"]["scheduler"]["kwargs"])
                infer_kwargs["default_denoising_steps"] = NUM_DENOISING_STEPS
                infer_kwargs["cfg_scale"] = CFG_SCALE
                infer_kwargs["generator"] = GENERATOR
            model = model.to(DEVICE).eval()

        elif mdl_cfg['model_name'] == "ConvoiPlannerNode":
            model = ConvoiPlannerNode(mdl_cfg, self.robot_cfg)
            infer_kwargs = {}

        else:
            raise ValueError(f"Unknown model: {mdl_cfg['model_name']}")

        return model, infer_kwargs, preproc

    def _prepare_inputs(self, img_pil: PILImage.Image, goal_cmd: str):
        img_tensor = self.preproc(img_pil)
        mdl_cfg = self.cfg.model
        inputs = {}
        for meta in mdl_cfg["dataloader_inputs"]:
            if meta["in_key"] == "front_rgb":
                inputs[meta["out_key"]] = img_tensor.to(DEVICE)
            elif meta["in_key"] == "goal_caption":
                inputs[meta["out_key"]] = [goal_cmd]
            elif meta["in_key"] == "velodyne_points":
                # Not used here; add when needed
                raise RuntimeError("Model expects velodyne_points, but this script doesn't provide them.")
        # Provide TF infos if model needs them
        if mdl_cfg['model_name'] == "ConvoiPlannerNode":
            inputs.update(self.tf_infos)
        return inputs

    @staticmethod
    def _load_robot_cfg(path_like: str) -> DictConfig:
        path = pathlib.Path(path_like)
        assert path.exists(), f"Robot config not found: {path}"
        import yaml
        with open(path, "r") as f:
            robot_cfg = yaml.safe_load(f)
        # Expand robot name placeholders, if any
        robot_name = robot_cfg.get("robot_name", "robot")
        for k, v in list(robot_cfg.items()):
            if isinstance(v, str):
                robot_cfg[k] = v.replace("{ROBOT_NAME}", robot_name)
        return OmegaConf.create(robot_cfg)

# ──────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../config", config_name="policy")
def main(cfg: DictConfig) -> None:
    rospy.init_node("spinflow_planner", anonymous=False)
    rospy.loginfo(f"ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI')}")
    node = SpinflowPlannerNode(cfg)
    try:
        node.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        rospy.loginfo("Shutting down spinflow planner…")

if __name__ == "__main__":
    main()
