#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpinFlow Bag Inference
- Loads model via your existing build_model() and ImagePreprocessor
- Reads messages from a ROS1 bag using topics from your Hydra robot config
- Publishes annotated CompressedImage to <image_plan_topic> for live visualization
- Publishes Path to <action_topic> (same as before)
- Saves two MP4s: (1) annotated predictions, (2) original RGB
- Keeps TF lookups working by ingesting /tf and /tf_static from the bag
- Keeps your optional action smoothing path
- Keeps the same dataloader_inputs contract from cfg.model
- Hydra overrides to point at bag & output paths

CLI examples (any of these work):
  python run_bag_inference.py io.bag:=/path/to/data.bag io.out_dir:=/tmp/out
  python run_bag_inference.py io.bag:=/path/to/data.bag io.annotated_path:=/tmp/pred.mp4 io.rgb_path:=/tmp/rgb.mp4
  python run_bag_inference.py io.bag:=/path/to/data.bag io.fps:=20
"""

import os
import signal
import pathlib
from collections import deque
from typing import Optional, Dict, Iterable, Tuple

import hickle as hkl
import numpy as np
import torch
import cv2
import rospy
import rosbag
import tf2_ros

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String, Bool
from PIL import Image as PILImage

import hydra
from omegaconf import DictConfig, OmegaConf

# ── your existing utils / model builders ───────────────────────────────────────
from scripts.inference.build_model import build_model
from deployment.src.constants import CAM_TO_BASE_OFFSET
from deployment.src.utils import (
    compressed_imgmsg_to_pil,
    rescale_intrinsics,
    pil_to_compressed_imgmsg,
    intrinsics_to_camera_info_msg,
    apply_intrinsics_to_image,
    waypoints_to_path_msg,
    camera_info_to_dict,
    tf_to_se3,
    pose_to_se3,
    ImagePreprocessor,
    viz_predictions,
    smooth_robot_actions,
    save_path_to_file,
)

from deployment.src.convoi import ConvoiPlannerNode

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR = torch.Generator(device=DEVICE).manual_seed(SEED)
NUM_WAYPOINTS = 50
CFG_SCALE = 8.0
NUM_DENOISING_STEPS = 40
# DEFAULT_GOAL = "Go to the next waypoint"
# DEFAULT_GOAL = "Continue ahead on the sidewalk. Keep a safe distance from any loose clutter."
# DEFAULT_GOAL = "Stay on the sidewalk. Pass the parking sign from the left side."
# DEFAULT_GOAL = "Keep on the left side of the sidewalk and avoid obstacles." # Good for middle

# DEFAULT_GOAL = "Veer right onto the road and go between the construction barriers."
# START_FRAME=92

# DEFAULT_GOAL = "Go to the construction barriers while avoiding obstacles."
# START_FRAME=340

DEFAULT_GOAL = "Go to the next waypoint. Avoid obstacles from the right and stay on the sidewalk."
START_FRAME=300
DEFAULT_GOAL_FOREST = "Continue straight ahead on the paved path. Keep a safe distance from any vegetation."


# ──────────────────────────────────────────────────────────────────────────────
class BagPlanner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        assert 'bag_path' in cfg, "You must provide path to rosbag"
        # ---- IO / paths (Hydra overrides) ----
        self.bag_path = pathlib.Path(cfg['bag_path'])
        bag_parent_dir = self.bag_path.parent.name
        assert self.bag_path and self.bag_path.exists(), f"Missing or invalid io.bag: {self.bag_path}"

        self.out_dir = pathlib.Path(cfg.get("out_dir", "offline_replay_outputs")) / bag_parent_dir
        if self.out_dir:
            self.out_dir.expanduser().mkdir(parents=True, exist_ok=True)

        # optional explicit video paths
        # Extract parent directory of bad as prefix directory and add model name to video
        annotated_path = cfg.get("rgb_pred_path", "video_annotated.mp4")
        self.annotated_path = self.out_dir / str(annotated_path)

        raw_path = cfg.get("rgb_path", "video_rgb.mp4")
        self.rgb_path = self.out_dir / str(raw_path)

        self.model_pred_path = self.out_dir / annotated_path.replace(".mp4", ".hkl")

        # preferred FPS for writers (fallback if timestamps are irregular)
        self.out_fps = float(cfg.get("fps", 5.0))

        # ---- robot config (you had this already) ----
        self.robot_cfg = self._load_robot_cfg(cfg.get("robot_config", "deployment/config/robot.yaml"))
        self.image_size = (self.robot_cfg["image_width"], self.robot_cfg["image_height"])

        # frames
        self.odom_frame = self.robot_cfg["odom_frame"]
        self.base_link_frame = self.robot_cfg["base_link_frame"]
        self.optical_frame = self.robot_cfg["optical_frame"]

        # TF buffer; we will ingest /tf & /tf_static from the bag
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(60.0))

        # state
        self.tf_infos: Dict = {}  # intrinsics + (optional) cached SE(3) mats
        self.latest_odom_pose = None

        select_bags = ["ferrite7_2025-09-18-11-13-00_test", "ferrite7_2025-09-18-13-02-00_test"]
        # If select bags are in bag_path, use forest goal
        self.goal_cmd = cfg.get("goal_command", DEFAULT_GOAL_FOREST if any(x in str(self.bag_path) for x in select_bags) else DEFAULT_GOAL)
        
        # Path smoothing (optional)
        self.path_sm_cfg = self.robot_cfg.get("path_smoothing", {"enable": False, "window_size": 3})
        self.action_key = "smoothed_action_pred" if self.path_sm_cfg.get("enable") else "action_pred"
        self._smooth_buf = {
            "odom_ts": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "odom_poses": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "img_ts": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
            "action_pred": deque(maxlen=self.path_sm_cfg.get("window_size", 3)),
        }

        # publishers for live viz (optional, harmless if no roscore listeners)
        self.path_pub = rospy.Publisher(self.robot_cfg["action_topic"], Path, queue_size=1)
        self.path_img_pub = rospy.Publisher(self.robot_cfg["image_plan_topic"], CompressedImage, queue_size=1)

        # model & preproc
        self.model, self.infer_kwargs, self.preproc = self._build_model_and_preproc(cfg)

        # video writers (lazy-initialized on first frame once we know size)
        self._vw_annotated: Optional[cv2.VideoWriter] = None
        self._vw_rgb: Optional[cv2.VideoWriter] = None
        self._size_checked = False
        self._frames_started = 0
        self._frame_first = None      # int
        self._frame_last = None       # int
        self._frames_written = 0
        self._interrupted = False     # for clean shutdown on Ctrl-C

        # Register clean shutdown paths
        signal.signal(signal.SIGINT, self._sigint_handler)
        rospy.on_shutdown(self._on_shutdown)

    def _sigint_handler(self, signum, frame):
        rospy.logwarn("[BagPlanner] SIGINT received, requesting stop…")
        self._interrupted = True
        try:
            rospy.signal_shutdown("SIGINT")
        except Exception:
            pass  # okay outside a running rospy loop

    def _on_shutdown(self):
        # Always close writers; if interrupted and frames exist, rename with suffix
        self.finalize_videos(interrupted=self._interrupted)

    # ───────────────────────── helpers ─────────────────────────
    @staticmethod
    def _load_robot_cfg(path_like: str) -> DictConfig:
        path = pathlib.Path(path_like)
        assert path.exists(), f"Robot config not found: {path}"
        import yaml
        with open(path, "r") as f:
            robot_cfg = yaml.safe_load(f)
        robot_name = robot_cfg.get("robot_name", "robot")
        for k, v in list(robot_cfg.items()):
            if isinstance(v, str):
                robot_cfg[k] = v.replace("{ROBOT_NAME}", robot_name)
        return OmegaConf.create(robot_cfg)

    def _build_model_and_preproc(self, cfg: DictConfig):
        mdl_cfg = cfg.model
        img_range = cfg.dataset["output_range"]
        preproc = ImagePreprocessor(
            height=self.robot_cfg["image_height"],
            width=self.robot_cfg["image_width"],
            out_min=img_range[0],
            out_max=img_range[1],
        )

        if mdl_cfg["model_name"] in ["WaypointFlowPolicy", "LeLaN_clip"]:
            ckpt_path = mdl_cfg.get("weights_ckpt", None)
            assert ckpt_path, "Model checkpoint path must be provided."
            model = build_model(
                mdl_cfg,
                ckpt_path,
                mdl_cfg.get("vision_weights_ckpt", None),
                seed=SEED,
                device=DEVICE,
            )
            infer_kwargs = {}
            if "validation" in mdl_cfg and "scheduler" in mdl_cfg["validation"]:
                infer_kwargs = dict(mdl_cfg["validation"]["scheduler"]["kwargs"])
                infer_kwargs["default_denoising_steps"] = NUM_DENOISING_STEPS
                infer_kwargs["cfg_scale"] = CFG_SCALE
                infer_kwargs["generator"] = GENERATOR
            model = model.to(DEVICE).eval()

        elif mdl_cfg["model_name"] == "ConvoiPlannerNode":
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
                raise RuntimeError("Model expects velodyne_points, but this bag script doesn't provide them.")
        if mdl_cfg["model_name"] == "ConvoiPlannerNode":
            inputs.update(self.tf_infos)
        return inputs

    def _ensure_video_writers(self, w: int, h: int) -> None:
        if self._vw_annotated is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._vw_annotated = cv2.VideoWriter(self.annotated_path, fourcc, self.out_fps, (w, h))
        if self._vw_rgb is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._vw_rgb = cv2.VideoWriter(self.rgb_path, fourcc, self.out_fps, (w, h))

    def _close_video_writers(self) -> None:
        if self._vw_annotated is not None:
            self._vw_annotated.release()
        if self._vw_rgb is not None:
            self._vw_rgb.release()

    # ---- TF ingestion from bag (/tf, /tf_static) ----
    def _ingest_tf(self, tf_msg: TFMessage, is_static: bool) -> None:
        """Insert TFs from the bag into our Buffer, compatible with older tf2_py."""
        for ts in tf_msg.transforms:
            try:
                # Prefer explicit static API when present
                if is_static and hasattr(self.tf_buffer, "set_transform_static"):
                    # (transform_stamped, authority) — positional only on some builds
                    self.tf_buffer.set_transform_static(ts, "bag_playback")
                else:
                    # Fallback to dynamic insert (positional only)
                    self.tf_buffer.set_transform(ts, "bag_playback")
            except TypeError:
                # Very old API: only 2-arg positional signature; no static method.
                # Just insert as dynamic; still works for most offline lookups.
                self.tf_buffer.set_transform(ts, "bag_playback")

    def _preload_tf(self, bag: rosbag.Bag, tf_topics=("\/tf", "\/tf_static")) -> None:
        """Load all TF into the buffer before processing images to avoid future extrapolation."""
        rospy.loginfo("[BagPlanner] Preloading TF...")
        for topic, msg, t in bag.read_messages(topics=list(tf_topics)):
            self._ingest_tf(msg, is_static=(topic == "/tf_static"))
        rospy.loginfo("[BagPlanner] TF preload complete.")

    # ---- main pass over the bag ----
    def run(self) -> None:
        camera_topic = self.robot_cfg["camera_topic"]
        camera_info_topic = self.robot_cfg["camera_info_topic"]
        odom_topic = self.robot_cfg["odom_topic"]
        tf_topics = ["/tf", "/tf_static"]

        topics = [camera_topic, camera_info_topic, odom_topic] + tf_topics

        # Initialize republishers
        self.camera_topic_out = f'{camera_topic}'
        self.camera_info_topic_out = f'{camera_info_topic}'
        self.pub_cam_cal = rospy.Publisher(self.camera_topic_out, CompressedImage, queue_size=1)
        self.pub_caminfo_cal = rospy.Publisher(self.camera_info_topic_out, CameraInfo, queue_size=1)

        rospy.loginfo(f"[BagPlanner] Opening bag: {self.bag_path}")
        bag = rosbag.Bag(self.bag_path, "r")
        self._preload_tf(bag, tf_topics=("/tf", "/tf_static"))

        model_outputs_cache: Dict[float, Dict[str, np.ndarray]] = {}
        try:
            msg_iter = bag.read_messages(topics=topics)
            while not rospy.is_shutdown():
                if self._interrupted:
                    rospy.logwarn("[BagPlanner] Stopping bag replay due to SIGINT.")
                    break
                try:
                    topic, msg, t = next(msg_iter)
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    self._interrupted = True
                    rospy.logwarn("[BagPlanner] KeyboardInterrupt inside bag loop.")
                    break

                # 
                stamp = msg.header.stamp.to_sec() if hasattr(msg, "header") else t.to_sec()
                
                # --- TF messages ---
                if topic in tf_topics:
                    self._ingest_tf(msg, is_static=(topic == "/tf_static"))
                    self._maybe_update_extrinsics(stamp)
                    continue

                # --- CameraInfo (intrinsics) ---
                if topic == camera_info_topic:
                    self.tf_infos["intrinsics"] = rescale_intrinsics(
                        camera_info_to_dict(msg),
                        image_size=self.image_size,
                    )
                    if not self.optical_frame:
                        self.optical_frame = msg.header.frame_id
                        rospy.loginfo_once(f"[camera_info] optical_frame='{self.optical_frame}'")
                    self._maybe_update_extrinsics(stamp)

                    cal_info = intrinsics_to_camera_info_msg(
                        self.tf_infos["intrinsics"],
                        frame_id=self.optical_frame or msg.header.frame_id,
                        stamp=t,
                        height=self.image_size[1],
                        width=self.image_size[0],
                        base_msg=msg,
                    )
                    self.pub_caminfo_cal.publish(cal_info)
                    continue

                # --- Odometry ---
                if topic == odom_topic:
                    self.latest_odom_pose = pose_to_se3(msg.pose.pose)
                    continue
                
                # --- RGB image (CompressedImage) ---
                if topic == camera_topic:
                    if not self._ready_for_inference():
                        continue

                    self._frames_started += 1
                    if self._frames_started < START_FRAME:
                        continue
                    print("[BagPlanner] Processing image @ t={:.3f}".format(stamp))
                    # Get PIL and apply intrinsics crop/resize if you do that
                    img_pil = compressed_imgmsg_to_pil(msg)
                    img_pil = apply_intrinsics_to_image(img_pil, self.tf_infos["intrinsics"])
                    img_cal_msg = pil_to_compressed_imgmsg(
                        img_pil,
                        frame_id=self.optical_frame or msg.header.frame_id,
                        stamp=t,
                    )
                    self.pub_cam_cal.publish(img_cal_msg)

                    # Keep original-size RGB (for saving the raw RGB video)
                    rgb_np_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    # Prepare inputs & run model
                    mdl_inputs = self._prepare_inputs(img_pil, self.goal_cmd)
                    with torch.no_grad():
                        outputs = self.model.infer(mdl_inputs, **self.infer_kwargs)

                    wp_pre_norm = None
                    if "action_pred" in outputs and isinstance(outputs["action_pred"], torch.Tensor):
                        # keep exactly what the model produced (detach to CPU numpy)
                        wp_pre_norm = outputs["action_pred"].detach().cpu().numpy()

                    # Path mask (use your preferred key; fall back to common variants)
                    mask = None
                    for k in ("path_mask_pred", "plan_mask", "path_mask"):
                        if k in outputs:
                            v = outputs[k]
                            if isinstance(v, torch.Tensor):
                                mask = v.detach().cpu().numpy()
                            elif isinstance(v, np.ndarray):
                                mask = v
                            else:
                                # e.g., PIL or list -> convert to numpy
                                try:
                                    mask = np.asarray(v)
                                except Exception:
                                    mask = None
                            break

                    # Visualization image
                    if "annotated_img" in outputs:
                        img_vis = PILImage.fromarray(outputs["annotated_img"]).convert("RGB")
                    else:
                        img_vis = viz_predictions(
                            img_pil,
                            outputs,
                            obs_range=self.cfg.dataset["output_range"],
                            infos=self.tf_infos,
                            pred_key=self.action_key,
                            vis_path=self.robot_cfg.get("visualize_path", True),
                        )
                    # Store in cache keyed by the frame timestamp (seconds)
                    model_outputs_cache[stamp] = {
                        "annotated_image": np.array(img_vis),
                        "path_mask_prediction": mask,
                        "waypoint_actions_pre_norm": wp_pre_norm,
                    }

                    # Offset x to account for camera->base link offset (if present)
                    if "action_pred" in outputs:
                        outputs["action_pred"][:, :, :] += torch.tensor(CAM_TO_BASE_OFFSET).to(DEVICE)
                        outputs["action_pred"][:, :, 2] = torch.mean(outputs["action_pred"][:, :, 2])

                    # Build Path in base_link; optionally transform to odom
                    waypoint_frame = self.base_link_frame
                    path_msg = waypoints_to_path_msg(
                        outputs,
                        waypoint_frame,
                        stamp,
                        pred_key=self.action_key,
                        num_samples=NUM_WAYPOINTS,
                    )

                    path_msg_odom = self._maybe_transform_path(path_msg, stamp, waypoint_frame, self.odom_frame)

                    # Publish annotated image to topic (for live rviz)
                    img_msg_out = pil_to_compressed_imgmsg(
                        img_vis, frame_id=self.optical_frame, stamp=t
                    )
                    self.path_img_pub.publish(img_msg_out)

                    # Publish Path (optional save)
                    if path_msg_odom is not None:
                        if self.robot_cfg.get("save_plan", False):
                            # write single-message bag to out_dir if set
                            out_bag = (
                                pathlib.Path(self.out_dir) / "path_plan.bag"
                                if self.out_dir else "path_plan.bag"
                            )
                            save_path_to_file(path_msg_odom, str(out_bag), path_topic=self.robot_cfg["action_topic"])
                        self.path_pub.publish(path_msg_odom)

                    # Save videos (lazy init writers)
                    ann_np_bgr = cv2.cvtColor(np.array(img_vis), cv2.COLOR_RGB2BGR)
                    h, w = ann_np_bgr.shape[:2]
                    if not self._size_checked:
                        # enforce user-specified image_size if needed
                        if (w, h) != self.image_size:
                            ann_np_bgr = cv2.resize(ann_np_bgr, self.image_size, interpolation=cv2.INTER_LINEAR)
                            rgb_np_bgr = cv2.resize(rgb_np_bgr, self.image_size, interpolation=cv2.INTER_LINEAR)
                            h, w = self.image_size[1], self.image_size[0]
                        self._ensure_video_writers(w, h)
                        self._size_checked = True
                    else:
                        if (ann_np_bgr.shape[1], ann_np_bgr.shape[0]) != self.image_size:
                            ann_np_bgr = cv2.resize(ann_np_bgr, self.image_size, interpolation=cv2.INTER_LINEAR)
                        if (rgb_np_bgr.shape[1], rgb_np_bgr.shape[0]) != self.image_size:
                            rgb_np_bgr = cv2.resize(rgb_np_bgr, self.image_size, interpolation=cv2.INTER_LINEAR)

                    self._vw_annotated.write(ann_np_bgr)
                    self._vw_rgb.write(rgb_np_bgr)
                    
                    # --- update frame counters for suffix renaming ---
                    if self._frame_first is None:
                        self._frame_first = self._frames_started  # first written frame index
                    self._frames_written += 1
                    self._frame_last = self._frame_first + self._frames_written - 1

            rospy.loginfo("[BagPlanner] Finished bag.")
        finally:
            self._close_video_writers()
            # ---------- save the whole cache to disk ----------
            try:
                # compress to keep file sizes reasonable
                hkl.dump(model_outputs_cache, self.model_pred_path, mode='w', compression='gzip', compression_opts=4)
                rospy.loginfo(f"[BagPlanner] Saved predictions → {self.model_pred_path} "
                            f"({len(model_outputs_cache)} frames)")
            except Exception as e:
                rospy.logwarn(f"[BagPlanner] Failed to save predictions: {e}")
            bag.close()

    def _add_suffix_before_ext(self, path_str: str, suffix: str) -> str:
        p = pathlib.Path(path_str)
        return str(p.with_name(p.stem + suffix + p.suffix))

    def finalize_videos(self, interrupted: bool = False) -> None:
        """
        If interrupted and we wrote frames, rename the current video files to include
        the frame range suffix `_f<start>-<end>` so it's obvious they are partial.
        """
        try:
            # Make sure files are flushed/closed before renaming
            self._close_video_writers()
        except Exception:
            pass

        if not interrupted:
            return

        if (self._frame_first is None) or (self._frame_last is None) or (self._frames_written <= 0):
            rospy.logwarn("[BagPlanner] Ctrl-C caught, but no frames were written; skipping rename.")
            return

        suffix = f"_f{self._frame_first}-{self._frame_last}"

        # Annotated video
        try:
            new_ann = self._add_suffix_before_ext(self.annotated_path, suffix)
            if self.annotated_path != new_ann and pathlib.Path(self.annotated_path).exists():
                os.replace(self.annotated_path, new_ann)
                rospy.loginfo(f"[BagPlanner] Renamed annotated video → {new_ann}")
        except Exception as e:
            rospy.logwarn(f"[BagPlanner] Failed to rename annotated video: {e}")

        # RGB video
        try:
            new_rgb = self._add_suffix_before_ext(self.rgb_path, suffix)
            if self.rgb_path != new_rgb and pathlib.Path(self.rgb_path).exists():
                os.replace(self.rgb_path, new_rgb)
                rospy.loginfo(f"[BagPlanner] Renamed RGB video → {new_rgb}")
        except Exception as e:
            rospy.logwarn(f"[BagPlanner] Failed to rename RGB video: {e}")

    def _safe_lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        stamp_sec: float,
        timeout_sec: float = 0.5,
    ):
        """
        Try exact-time lookup; if it's in the future, fall back to the latest available transform.
        Returns (tf_stamped, used_time: rospy.Time).
        """
        req_time = rospy.Time.from_sec(stamp_sec)
        try:
            tf_st = self.tf_buffer.lookup_transform(target_frame, source_frame, req_time, rospy.Duration(timeout_sec))
            return tf_st, req_time
        except tf2_ros.ExtrapolationException as e:
            # If the request is in the future, use the latest available TF (Time(0))
            # This avoids dropping frames when TF lags image timestamps.
            try:
                tf_latest = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
                return tf_latest, tf_latest.header.stamp
            except Exception:
                # Re-raise the original if even latest isn't available
                raise e

    def _maybe_update_extrinsics(self, query_time_sec: float) -> None:
        """Populate base<->optical transforms if available in the buffer."""
        if "T_optical_to_base" in self.tf_infos:
            return
        if not self.optical_frame:
            return
        try:
            tf_stamped, _ = self._safe_lookup_transform(self.base_link_frame, self.optical_frame, query_time_sec, timeout_sec=0.2)
            T_optical_to_base = tf_to_se3(tf_stamped)
            self.tf_infos["T_optical_to_base"] = T_optical_to_base
            self.tf_infos["T_base_to_optical"] = np.linalg.inv(T_optical_to_base)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException):
            pass

    def _maybe_transform_path(self, path_msg: Path, stamp: float, from_frame: str, to_frame: str) -> Optional[Path]:
        if to_frame == from_frame:
            return path_msg
        try:
            tf_stamped, used_time = self._safe_lookup_transform(to_frame, from_frame, stamp, timeout_sec=0.5)
            out = Path()
            out.header.frame_id = to_frame
            out.header.stamp = used_time
            out.poses = []
            for ps in path_msg.poses:
                ps.header.stamp = used_time
                out.poses.append(do_transform_pose(ps, tf_stamped))
            return out
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
            rospy.logwarn(f"[BagPlanner] TF lookup ({to_frame}<-{from_frame}) failed: {e}")
            return None

    def _ready_for_inference(self) -> bool:
        if "intrinsics" not in self.tf_infos:
            rospy.logwarn_throttle(2.0, "[BagPlanner] Waiting for camera intrinsics (CameraInfo in bag)…")
            return False
        if "T_optical_to_base" not in self.tf_infos:
            rospy.logwarn_throttle(2.0, "[BagPlanner] Waiting for TF base_link ↔ optical (ensure /tf and /tf_static in bag)…")
            return False
        if self.latest_odom_pose is None:
            rospy.logwarn_throttle(2.0, "[BagPlanner] Waiting for Odometry…")
            return False
        return True

# ──────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base="1.3", config_path="../../config", config_name="policy")
def main(cfg: DictConfig) -> None:
    """
    Expected Hydra config fields used here:
      - robot_config: path to your YAML robot config (as before)
      - model: (unchanged) including model_name, dataloader_inputs, checkpoints, etc.
      - dataset.output_range: min/max for preprocessing

    New (but simple) group:
      io:
        bag: /abs/path/to/bag.bag           # REQUIRED (override via CLI)
        out_dir: /abs/path/to/out           # optional; if set, we create annotated.mp4 & rgb.mp4 here
        annotated_path: /abs/path/to/x.mp4  # optional; overrides out_dir default
        rgb_path: /abs/path/to/y.mp4        # optional; overrides out_dir default
        fps: 15.0                           # optional; default 15
    """
    rospy.init_node("spinflow_bag_inference", anonymous=False)
    rospy.loginfo(f"ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI')}")
    planner = BagPlanner(cfg)
    try:
        planner.run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        # Ensure we rename with frame-range suffix on Ctrl-C
        planner.finalize_videos(interrupted=True)
        return
    finally:
        rospy.loginfo("Shutting down bag inference…")

if __name__ == "__main__":
    main()
