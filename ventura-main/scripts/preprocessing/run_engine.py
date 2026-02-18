"""
This script mines for interesting dataset samples, removing difficult or uninteresting sequences

1. Mines for interesting dataset samples by computing distance from GPS location to
intersections from openstreetmap. Filtering out sequences that are too far away from intersections
2. Time filter - filters out entire sequences that are in the evening or night time
3. Brightness / color filter - filters out subsequences that are too homogeneous in color or brightness
4. Velocity Filter - Filters out subsequences with an average cmd velocity below a threshold
6. Filters out subsequences that are only turn in place (angular velocity but no linear)

"""
import os
for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'
import yaml
import argparse
from pathlib import Path
import random
import re
from tqdm import tqdm
import torch
import cv2
cv2.setNumThreads(0)        # or 1 – completely disables OpenCV’s TBB/OpenMP pool
cv2.ocl.setUseOpenCL(False) # avoid extra OpenCL threads on laptops / Macs

print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("torch.version.cuda:", torch.version.cuda)
print("nvcc path:", os.popen("which nvcc").read().strip())

import hickle as hkl
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import av # Efficient video writing library
import decord # Efficient video reading library

import spinflow.dataset.frodo_helpers as fh
from scripts.utils.time_utils import (apply_time_filter, interpolate_mask_to_timestamps, find_contiguous_true_intervals)
from scripts.utils.graph_utils import apply_graph_filter
from scripts.utils.log_utils import *
from scripts.utils.image_utils import (save_video_np)
from joblib import Parallel, delayed

from scripts.mapping.tracking_filters import apply_postfilter_single
from scripts.mapping.download_graphs import download_maps
try:
    from scripts.mapping.download_masks import compute_entity_masks
except Exception as e:
    logging.error(f"Failed to import compute_entity_masks: {e}")
    compute_entity_masks = None
from scripts.mapping.download_satellite import download_satellite_imagery
from scripts.mapping.download_routing import download_satellite_routes
from scripts.mapping.compute_infos import compute_ride_infos
from scripts.mapping.compute_odom import (
    odom_window_metrics,
    interpolate_se3,
    compute_odometry_goals
)
from scripts.utils.loader_utils import (
    load_timestamps, 
    load_controls, 
    load_gps, 
    load_graph_lut, 
    load_video, 
    load_odom,
    load_langlabel_info
)

from datetime import datetime, time
np.random.seed(42)
random.seed(42)

DEBUG_PIPELINE = False

def construct_filters(cfg):
    """Loops throgh and constructs filters"""
    filters = {}
    for filter_dict in cfg['filters']:
        name = filter_dict['name']
        filters[name] = {
            'type': filter_dict['type'],
            'params': filter_dict['params']
        }
    return filters

def save_subsequences_csv(
    ride_name: str,
    intervals: list[tuple[int, int]],
    out_path: str = None
) -> str:
    """
    Given:
        ride_name    : string identifier for the ride
        intervals  : list of (start_frame, end_frame) pairs, each ints
        out_path   : optional file path to write CSV. If None, only returns CSV string.

    Constructs a pandas DataFrame with columns:
        ["ride_name", "start_frame", "end_frame"]
    where each row is one interval. The ride‐specific columns are constant across rows.
    Returns:
        The CSV content as a string (comma‐separated, including header).
    If out_path is provided, also writes the CSV to that path.
    """
    # Number of intervals
    n = len(intervals)
    if n == 0:
        # Return just the header row if no intervals
        columns = ["ride_name", "start_frame", "end_frame"]
        header_only = ",".join(columns) + "\n"
        if out_path:
            with open(out_path, "w") as f:
                f.write(header_only)
        return header_only

    # Unzip intervals into two lists
    starts, ends = zip(*intervals)  # each of length n

    # Build a dict of columns, vectorized
    data = {
        "ride_name": [ride_name] * n,
        "start_frame": list(starts),
        "end_frame":   list(ends),
    }

    df = pd.DataFrame(data, columns=[
        "ride_name", "start_frame", "end_frame"
    ])

    # Optionally write to file
    if out_path:
        df.to_csv(df, out_path, index=False)

    return df

def apply_odometry_filter(data_dict, filter_dict):
    params = filter_dict["params"]
    H       = int(params["horizon_frames"])
    min_dist  = float(params["min_distance"])
    max_dyaw  = float(params["max_ang_disp"]) # deg
    max_omega = float(params["max_ang_vel"])  # deg/s

    if H < 2 or H % 2:
        raise ValueError("horizon_frames must be an even integer ≥ 2")

    odom = np.asarray(data_dict["odometry"], dtype=np.float64)
    # Interpolate odometry to match timestamps
    if odom.shape[1] < 8:
        raise ValueError("odometry array must have 8 columns")
    cam_ts = data_dict["timestamps"]
    odo_ts = odom[:, 0]  # (N,)
    odo_xyz = odom[:, 1:4]  # (N,3)
    odo_qwxyz = odom[:, 4:8]  # (N,4) qw qx qy qz
    odom_interp = interpolate_se3(cam_ts, odo_ts, odo_xyz, odo_qwxyz)
    N       = odom_interp.shape[0]
    half_H  = H // 2

    # ── vectorised metrics for *eligible* central indices
    odom_metrics = odom_window_metrics(odom_interp, H)   # each (N-H,)

    dist = odom_metrics["total_disp"]
    dyaw = odom_metrics["max_dyaw"]  # (N-H,)
    omega = odom_metrics["max_domega"]  # (N-H,)

    # central indices that own those windows
    centres = np.arange(half_H, N - half_H)

    # boolean test over all windows at once
    good = (dist >= min_dist) & (dyaw <= max_dyaw) & (omega <= max_omega)  # (N-H,)

    # build full-length mask
    mask = np.zeros(N, dtype=bool)
    mask[centres] = good

    return mask


def apply_velocity_filter(data_dict, filter_dict):
    """
    Build a boolean mask over N_frames (from data_dict["timestamps"]) filtering out:
      1) Any frame whose nearest‐neighbor‐averaged linear velocity is outside [lin_min, lin_max].
      2) Any frame whose nearest‐neighbor‐averaged angular velocity is outside [ang_min, ang_max].
      3) Any frame that lies within a "pure turn" window of length L, i.e. for L consecutive control
         samples, linear ≈ 0 AND angular ≠ 0.
      4) Any frame that lies within a "pure stop" window of length L, i.e. for L consecutive control
         samples, both linear ≈ 0 AND angular ≈ 0.

    `filter_dict["params"]` must contain:
        - low_pass_window: int (number of control samples to use in sliding window)
        - linear_velocity: [lin_min, lin_max]
        - angular_velocity: [ang_min, ang_max]

    Args:
        data_dict: {
            "controls": np.ndarray shape (N_controls, 3) columns = [lin_vel, ang_vel, ts],
            "timestamps": np.ndarray shape (N_frames,) of frame timestamps (UNIX seconds)
        }
        filter_dict: {
            "params": {
                "low_pass_window": int,
                "linear_velocity": [float, float],
                "angular_velocity": [float, float]
            }
        }

    Returns:
        mask: np.ndarray shape (N_frames,), dtype=bool
    """
    params = filter_dict.get("params", {})
    L = int(params["low_pass_window"])
    lin_min, lin_max = params["linear_velocity"]
    ang_min, ang_max = params["angular_velocity"]

    # 1) Extract controls
    controls = np.asarray(data_dict["controls"], dtype=float)
    if controls.ndim != 2 or controls.shape[1] < 3:
        raise ValueError("data_dict['controls'] must be shape (N_controls, 3).")
    lin = controls[:, 0]
    ang = controls[:, 1]
    ctrl_ts = controls[:, 2]
    N_ctrl = lin.shape[0]

    # 2) Compute moving‐average of linear and angular velocities (window length L)
    kernel = np.ones(L, dtype=float)
    mov_lin = np.convolve(lin, kernel, mode="same") / L
    mov_ang = np.convolve(ang, kernel, mode="same") / L

    # 3) Build masks for linear/ang bounds
    lin_ok = (mov_lin >= lin_min) & (mov_lin <= lin_max)
    ang_ok = (mov_ang >= ang_min) & (mov_ang <= ang_max)

    # 4) Detect "pure turn" windows: for L consecutive samples, linear ≈ 0 AND ang ≠ 0
    eps = 1e-2  # threshold for “≈ 0”
    pure_turn = np.zeros(N_ctrl, dtype=bool)
    if not params['allow_turns']:
        is_lin_zero = (np.abs(lin) < eps).astype(int)
        is_ang_nonzero = (np.abs(ang) >= eps).astype(int)
        sum_lin_zero = np.convolve(is_lin_zero, kernel, mode="same")
        sum_ang_nonzero = np.convolve(is_ang_nonzero, kernel, mode="same")
        pure_turn = (sum_lin_zero >= L) & (sum_ang_nonzero >= 1)

    # 5) Detect "pure stop" windows: for L consecutive samples, both |lin|<eps and |ang|<eps
    is_zero_total = ((np.abs(lin) < eps) & (np.abs(ang) < eps)).astype(int)
    sum_zero_total = np.convolve(is_zero_total, kernel, mode="same")
    pure_stop = sum_zero_total >= L

    # 6) Combine to get control‐level mask
    ctrl_mask = lin_ok & ang_ok & ~pure_turn & ~pure_stop

    return ctrl_mask # [N_ctrl,]

def apply_image_filter(data_dict, filter_dict):
    """
    Apply an image filter to the data_dict based on the filter_dict parameters.

    Inputs:
      data_dict: {
        "front_camera":  np.ndarray of shape (T, H, W, 3), dtype=uint8 or float in [0,255]
      }
      filter_dict: {
        "params": {
          "brightness_threshold": [low, high],   # fractions in [0,1], default [0.1, 0.9]
          "homogeneous_threshold": float         # minimum std‐dev in [0,1], default 0.05
        }
      }

    Returns:
      mask: np.ndarray of shape (T,), dtype=bool, True if frame is “good.”
    """
    params = filter_dict['params']
    # Get video first frame size
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(data_dict["front_camera"]))
    
    # Get first frame to get dimensions
    first_frame = vr[0]
    H, W, _ = first_frame.shape
    T = len(vr)  # Total number of frames
    acceptable_resolutions = [tuple(res) for res in params['resolutions'] ]
    if (H, W) not in acceptable_resolutions:
        logging.warning(f"Video resolution {(H, W)} not in acceptable resolutions {acceptable_resolutions}. Skipping image filter.")
        return np.zeros(T, dtype=bool)
    
    return np.ones(T, dtype=bool)  # Placeholder: return all frames as valid
    
def _download_one_satellite_image(ride_name, start_frame, end_frame, root_dir, out_dir):
    """
    Worker for a single subsequence. Returns
    (ride_name, start_frame, end_frame, success:bool).
    """
    logging.info(f"Downloading imagery for {ride_name} [{start_frame},{end_frame}]")
    ok = download_satellite_imagery(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )
    return ride_name, start_frame, end_frame, ok

def _download_one_satellite_route(ride_name, start_frame, end_frame, root_dir, out_dir):
    """
    Worker for a single subsequence. Returns
    (ride_name, start_frame, end_frame, success:bool).
    """
    logging.info(f"Downloading route for {ride_name} [{start_frame},{end_frame}]")
    ok = download_satellite_routes(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )
    return ride_name, start_frame, end_frame, ok

def _compute_one_odometry_goal(ride_name, start_frame, end_frame, root_dir, out_dir):
    """
    Worker for a single subsequence. Returns 
    (ride_name, start_frame, end_frame, success:bool).
    """
    logging.info(f"Computing odometry goal for {ride_name} [{start_frame},{end_frame}]")
    ok = compute_odometry_goals(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )
    return ride_name, start_frame, end_frame, ok

def visualize_subsequence(ride_name, start_frame, end_frame, root_dir, out_dir):
    """
    Visualizes a subsequence of a ride by loading the images and displaying them.
    This is a placeholder function that should be implemented to visualize the subsequence.
    
    Args:
        ride_name (str): Name of the ride.
        start_frame (int): Start frame index of the subsequence.
        end_frame (int): End frame index of the subsequence.
        out_dir (Path): Output directory where images are stored.
    """
    ride_dir = fh.set_frodo_dir(root_dir, *ride_name.split(' '))
    image_path = ride_dir / "front_camera.mp4"
    if not image_path.exists():
        logging.warning(f"Image path {image_path} does not exist. Cannot visualize subsequence.")
        return
    
    # Save video frames
    out_dir = out_dir / "front_camera" 
    out_dir.mkdir(parents=True, exist_ok=True)
    ride_name = ride_name.replace(' ', '_')  # Ensure valid filename
    out_path = out_dir / f"ride_{ride_name}_frames_{start_frame}_{end_frame}.mp4"
    logging.info(f"Visualizing subsequence {ride_name} from {start_frame} to {end_frame} to {out_path}.")

    # Load and save video from start/end frame 
    frames_np = load_video(image_path, start_frame=start_frame, end_frame=end_frame)
    if frames_np is None:
        logging.warning(f"No frames found in the specified range {start_frame}-{end_frame} for ride {ride_name}.")
        return
    save_video_np(frames_np, out_path, fps=20)

    
def process_ride(ride_dir, filters, modalities, ride_metadata=None):
    """
    Processes a single ride directory:
      1. Loads images, gps, cmd_vel, image timestamps 
      2. Applies filters to the data.
      3. Save a np array with valid subsequences that satisfy the filters (if any)
    """
    ride_id, driveid0, driveid1, date = fh.get_frodo_raw_id(ride_dir)

    #1 Load and verify data paths
    use_rgb = "front_camera" in modalities
    use_gps = "gps" in modalities
    use_ctrl = "control" in modalities
    use_odom = "odometry" in modalities

    data_dict = {}
    if use_rgb:
        video_path = ride_dir / "front_camera.mp4"
        timestamps_path = ride_dir / f"front_camera_timestamps_{driveid0}.csv"
        if not video_path.exists() or not timestamps_path.exists():
            logging.warning(f"Required files {video_path} or {timestamps_path} do not exist for ride {ride_id}. Skipping ride.")
            return None
        data_dict["timestamps"] = load_timestamps(timestamps_path)
        data_dict["front_camera"] = video_path  # Store path for later loading
        assert data_dict["timestamps"] is not None, f"Timestamps for {ride_id} are None. Check {timestamps_path}."
        assert data_dict["front_camera"] is not None, f"Video path for {ride_id} is None. Check {video_path}."

    if use_gps:
        gps_path = ride_dir / f"gps_data_{driveid0}.csv"
        if not gps_path.exists():
            logging.warning(f"Required file {gps_path} does not exist for ride {ride_id}. Skipping ride.")
            return None
        data_dict["gps"] = load_gps(gps_path)

    if use_ctrl:
        cmd_vel_path = ride_dir / f"control_data_{driveid0}.csv"
        if not cmd_vel_path.exists():
            logging.warning(f"Required file {cmd_vel_path} does not exist for ride {ride_id}. Skipping ride.")
            return None
        data_dict["controls"] = load_controls(cmd_vel_path)

    if use_odom:
        odom_path = ride_dir / f"odometry_data_{driveid0}.csv"
        if not odom_path.exists():
            logging.warning(f"Required file {odom_path} does not exist for ride {ride_id}. Skipping ride.")
            return None
        data_dict["odometry"] = load_odom(odom_path)

    N_frames = data_dict['timestamps'].shape[0]
    data_mask = np.ones(N_frames, dtype=bool)
    for filter_name, filter_dict in filters.items():
        if filter_dict['type'] == 'image':
            mask = apply_image_filter(data_dict, filter_dict)  # [N_frames,]
            if mask.shape[0] != N_frames:
                logging.warning(f"Number of images does not match timestamps, skipping sequence {ride_id}.")
                return None
            if mask is not None:
                data_mask = np.logical_and(data_mask, mask)
        elif filter_dict['type'] == "time":
            mask = apply_time_filter(data_dict, filter_dict)  # [N_frames,]
            data_mask = np.logical_and(data_mask, mask)
        elif filter_dict['type'] == 'gps':
            mask = apply_graph_filter(data_dict, filter_dict, ride_metadata) # [N_gps_frames,]
            if mask is not None:
                mask_aligned = interpolate_mask_to_timestamps(
                    data_dict['gps'][:, -1], mask, data_dict['timestamps'])
                data_mask = np.logical_and(data_mask, mask_aligned)
        elif filter_dict['type'] == 'controls':
            mask = apply_velocity_filter(data_dict, filter_dict)  # [N_frames,]
            if mask is not None:
                mask_aligned = interpolate_mask_to_timestamps(
                    data_dict['controls'][:, -1], mask, data_dict['timestamps'])
                data_mask = np.logical_and(data_mask, mask_aligned)
        elif filter_dict['type'] == 'odometry':
            mask = apply_odometry_filter(data_dict, filter_dict)
            assert mask.shape[0] == N_frames, f"Mask shape {mask.shape} does not match timestamps {data_dict['timestamps'].shape}"
            data_mask = np.logical_and(data_mask, mask)
        else:
            logging.warning(f"Unknown filter type '{filter_dict['type']}' for ride {ride_id}. Skipping filter.")
            continue

    #3 Find and save contiguous ranges of valid subsequences
    subsequences = find_contiguous_true_intervals(data_mask, window_len=cfg['window_length'])
    if len(subsequences) == 0:
        logging.info(f"No valid subsequences found for ride {ride_id}.")
        return None

    ride_name = fh.get_frodo_raw_id(ride_dir, format=True)
    df = save_subsequences_csv(
        ride_name=ride_name,
        intervals=subsequences
    )

    return df

def postfilter_parallel(tasks, filters, out_dir):
    """
    Applies post-filters in parallel to a list of tasks.
    Each task is a tuple (ride_name, start_frame, end_frame).
    Filters are applied based on the provided filter configuration.
    Returns a list of successful subsequences.
    """
    if not DEBUG_PIPELINE:
        logging.info(f"Running post-filtering on {len(tasks)} tasks in parallel...")
        # run in parallel
        results = Parallel(
            n_jobs=32,
            verbose=5,
        )(
            delayed(apply_postfilter_single)(
                ride_name=ride,
                start_frame=sf,
                end_frame=ef,
                out_dir=out_dir,
                filters=filters,
                debug_out_dir=Path("cotracker_postfilter_outputs")
            )
            for ride, sf, ef in tasks
        )

        # flatten successes/failures
        successes, failures = [], []
        for r in results:
            successes.extend(r.get('success', []))
            failures.extend(r.get('failure', []))

        total = len(successes) + len(failures)
        pct = (len(successes) / total * 100) if total > 0 else 0.0
        logging.info(f"Percentage of successful subsequences: {pct:.2f}%")
    else:
        # Seqeuntial processing
        successes = []
        failures = []
        for i, (ride, sf, ef) in enumerate(tasks):
            # if i <= 17:
            #     continue
            logging.info(f"Task {i+1}/{len(tasks)}: Applying post-filter for {ride} [{sf}, {ef}]")
            results = apply_postfilter_single(
                ride_name=ride,
                start_frame=sf,
                end_frame=ef,
                out_dir=out_dir,
                filters=filters,
                debug_out_dir=Path("cotracker_postfilter_outputs")
            )
            successes.extend(results['success'])
            failures.extend(results['failure'])

        num_successes, num_failures = len(successes), len(failures)
        logging.info(f"Percentage of successful subsequences: {num_successes / (num_successes + num_failures) * 100:.2f}%")

    return successes, failures

def build_mask_models(vlm_cfg, device):
    model_type = vlm_cfg["model_type"]
    caption_only = vlm_cfg['caption_only']

    if model_type == "openai":
        from deployment.src.gpt4v import BaseO4Mini
        multi_model = BaseO4Mini(
            api_key_env_var=vlm_cfg["api_key_env_var"],
            model_name=vlm_cfg["model_name"]
        )
        vlm_model_dict = {
            "model_type": model_type,
            "processor": None,  # OpenAI models do not use a processor
            "multi_model": multi_model,
        }
    elif model_type == "blip3o":
        from blip3o.utils import disable_torch_init
        from blip3o.model.builder import load_pretrained_model
        from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
        repo_id = "BLIP3o/BLIP3o-Model-8B"
        model_name = get_model_name_from_path(repo_id)
        diffusion_path = model_name + "/diffusion-decoder"

        #1 Load the BLIP3o processor and model
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        disable_torch_init()
        tokenizer, multi_model, context_len = load_pretrained_model(repo_id, device, model_name)

        vlm_model_dict = {
            "model_type": model_type,
            "processor": qwen_processor,
            "multi_model": multi_model,
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'openai' and 'blip3o'.")
    
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForZeroShotObjectDetection

    if not caption_only:
        #2 Load Grounding SAM2 model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_dir = "external/Grounded-SAM-2"
        sam2_checkpoint = f"{sam2_dir}/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_l.yaml"

        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)

        model_id = "IDEA-Research/grounding-dino-tiny"
        dino_processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    else:
        video_predictor = None
        image_predictor = None
        grounding_model = None
        dino_processor = None

    #3 Store models and processors
    sam2_model_dict = {
        "image_predictor": image_predictor,
        "video_predictor": video_predictor
    }
    dino_model_dict = {
        "processor": dino_processor,
        "grounding_model": grounding_model
    }
    return vlm_model_dict, sam2_model_dict, dino_model_dict

def _compute_entity_masks_single(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: Path,
    out_dir: Path,
    vlm_model_dict: dict,
    sam2_model_dict: dict,
    dino_model_dict: dict,
    caption_only: bool,
    cache_enabled: bool,
    device: str = "cuda"
):
    logging.info(f"Computing entity masks for {ride_name} [{start_frame},{end_frame}]")
    ok = compute_entity_masks(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir,
        vlm_model_dict=vlm_model_dict,
        dino_model_dict=dino_model_dict,
        sam2_model_dict=sam2_model_dict,
        save_visualizations=True,
        caption_only=caption_only,
        cache_enabled=cache_enabled,
        device=device
    )
    return ride_name, start_frame, end_frame, ok

def _compute_ride_info_single(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: Path,
    out_dir: Path,
): 
    logging.info(f"Computing ride info for {ride_name} [{start_frame},{end_frame}]")
    ok = compute_ride_infos(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )
    return ride_name, start_frame, end_frame, ok


# Now sort by ride_id (str) then drive0, drive1, timestamp (lexicographically)
def sort_key(item):
    ride_name, sf, ef = item
    # split on underscores (or spaces) to get the 4 components
    parts = re.split(r"[_\s]+", ride_name)
    ride_id = str(parts[0])
    rest    = tuple(parts[1:])  # e.g. (drive0, drive1, timestamp)
    return (ride_id,) + rest

def main(cfg, args):
    """
    Main entry point:
      1. Loads config.
      2. For each directory in 'rides_to_process', searches for subdirs 'ride_*_*'.
      3. Processes each ride into an HDF5 file with images, depth, segmentation, and BEV labels.
    """
    root_dir = Path(cfg['root_dir'])
    out_dir = Path(cfg['out_dir'])
    ride = args.ride
    split_path = Path(args.split)
    if not split_path.exists():
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.touch()

    assert root_dir.is_dir() and root_dir.exists(), f"Root directory {root_dir} does not exist."

    # Construct filters from config
    modalities = cfg['modalities']
    filters = construct_filters(cfg)

    # Precompute directory paths
    if len(ride) == 0:
        ride_list = sorted(root_dir.glob("output_rides_*"), key=lambda x: x.name.split('_')[-1])
    else:
        ride_list = [root_dir / Path(ride)]

        # Make sure to update both
        out_dir = out_dir / ride
        cfg['out_dir'] = str(out_dir)

    ride_dir_list = []
    for output_dir in ride_list:
        if not output_dir.is_dir() or not output_dir.exists():
            logging.warning(f"Output directory {output_dir} does not exist. Skipping.")
            continue        
        ride_dirs = sorted(output_dir.glob("ride_*_*"))
        ride_dir_list.extend(ride_dirs)
    logging.info(f"Found {len(ride_dir_list)} ride directories to process.")
    if len(ride_dir_list) == 0:
        logging.warning("No ride directories found. Exiting.")
        return

    # Run various pipeline stages:
    logging.info("Starting to process rides...")
    pipeline = cfg['pipeline']

    graph_df = None
    need_graphs = cfg.get('need_graphs', True)
    if pipeline['download_maps'] or pipeline["download_routing"]:
        logging.info("Downloading maps...")
        graph_df = download_maps(cfg, ride_dir_list)
    elif need_graphs:
        graph_lut_path = out_dir / cfg['download_maps']['graph_lut_path']
        if not graph_lut_path.exists():
            logging.warning(f"Graph LUT path {graph_lut_path} does not exist. Please download the maps first.")
            return
        # assert graph_lut_path.exists(), f"Graph LUT path {graph_lut_path} does not exist."
        graph_df = load_graph_lut(graph_lut_path)
        ride_dir_list = [ride_dir for ride_dir in ride_dir_list if fh.get_frodo_raw_id(ride_dir, format=True) in graph_df['ride'].values]

    if 'process_language_labels' in pipeline and pipeline['process_language_labels']:
        logging.info("Processing language labels...")
        langlabel_dir = root_dir / "language_labels"
        langlabel_paths = sorted(langlabel_dir.glob("*.json"))
        langlabel_paths = [p for p in langlabel_paths if str(ride) in str(p)]
        assert len(langlabel_paths) > 0, f"No language label files found in {langlabel_dir}."
        
        all_dfs = []
        try:
            for i, label_path in enumerate(langlabel_paths):
                logging.info(f"Processing language label file {i+1}/{len(langlabel_paths)}: {label_path}")
                if not label_path.exists():
                    logging.warning(f"Language label file {label_path} does not exist. Skipping.")
                    continue
                df = load_langlabel_info(label_path, root_dir)
                if df is not None:
                    all_dfs.append(df)
                
                if DEBUG_PIPELINE:
                    if i >= 5:
                        break
        except(KeyboardInterrupt, SystemExit):
            logging.info("Processing interrupted. Saving what we have so far.")
        combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        logging.info(f"Saving {len(combined_df)} language labels to {split_path}")
        combined_df.to_csv(split_path, index=False, header=True)

    if pipeline['apply_prefilters']:
        # Process each ride directory
        all_dfs = []
        try:  
            for i, ride_dir in enumerate(ride_dir_list):
                print(f"Processing ride directory: {ride_dir}")
                ride = fh.get_frodo_raw_id(ride_dir, format=True)

                ride_graph_metadata = None
                if need_graphs:
                    ride_graph_metadata = graph_df[graph_df['ride'] == ride].iloc[0]
                    assert not ride_graph_metadata.empty, f"Ride {ride} not found in graph metadata."
                
                df = process_ride(ride_dir, filters, modalities, ride_graph_metadata)
                if df is not None:
                    all_dfs.append(df)

                if DEBUG_PIPELINE:
                    if i >= 5:
                        break
        except(KeyboardInterrupt, SystemExit):
            logging.info("Processing interrupted. Saving what we have so far.")
        combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        # Write all subsequences to a .txt file
        logging.info(f"Saving {len(combined_df)} subsequences to {split_path}")
        combined_df.to_csv(split_path, index=False, header=True)
    
    if pipeline['visualize_samples']:
        # Load the split file and visualize samples
        assert split_path.exists(), f"Split file {split_path} does not exist."
        df = pd.read_csv(split_path)
        if df.empty:
            logging.warning(f"No subsequences found in {split_path}. Skipping visualization.")
            return

        num_viz_samples = cfg.get('num_viz_samples', 10)
        num_viz_samples = len(df) if num_viz_samples <= 0 or num_viz_samples >=len(df) else num_viz_samples
        logging.info(f"Visualizing {num_viz_samples} random subsequences from {split_path}")

        # Sample random subsequences
        sampled_indices = np.random.choice(df.index, size=num_viz_samples, replace=False)
        for idx in sampled_indices:
            subseq = df.iloc[idx]
            ride_name = subseq['ride_name']
            start_frame = subseq['start_frame']
            end_frame = subseq['end_frame']
            logging.info(f"Visualizing subsequence {idx}: {ride_name} [{start_frame}, {end_frame}]")

            visualize_subsequence(ride_name, start_frame, end_frame, root_dir, out_dir)
    
    if pipeline['compute_ride_infos']:
        logging.info("Computing ride information...")

        # Search through all .h5 files
        paths = fh.get_available_sequences(out_dir, name='path_tracker', ext='h5')
        ride_parts = [fh.get_frodo_raw_id(p, full=True) for p in paths]
        # Remove any paths that do not match the expected format
        ride_names = [" ".join(p[:4]) for p in ride_parts]
        # Create pandas df by joining the first four parts of the ride_infos
        df = pd.DataFrame({
            'ride_name': ride_names,
            'start_frame': [int(p[4]) for p in ride_parts],
            'end_frame': [int(p[5]) for p in ride_parts]
        })
        # Sort by ride_name, start_frame, end_frame
        df.sort_values(by=['ride_name', 'start_frame', 'end_frame'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        tasks = [
            (row.ride_name, row.start_frame, row.end_frame)
            for row in df.itertuples(index=False)
        ]

        if DEBUG_PIPELINE:
            # tasks = tasks[:5]
            results = []
            for i, (ride, sf, ef) in enumerate(tasks):
                logging.info(f"Task {i+1}/{len(tasks)}: Computing ride information for {ride} [{sf}, {ef}]")
                result = _compute_ride_info_single(
                    ride_name=ride,
                    start_frame=sf,
                    end_frame=ef,
                    root_dir=root_dir,
                    out_dir=out_dir
                )
                results.append(result)
        else:
            # Run in parallel (use n_jobs=-1 to use all CPUs, or set to some fixed number)
            results = Parallel(n_jobs=16, verbose=5)(
                delayed(_compute_ride_info_single)(ride, sf, ef, root_dir, out_dir)
                for ride, sf, ef in tasks
            )

        # Filter successes
        successes = [(ride, sf, ef) for ride, sf, ef, ok in results if ok]

        successes_sorted = sorted(successes, key=sort_key)
        # Save succcessful ride conversions to a csv
        success_df = pd.DataFrame(successes_sorted, columns=['ride_name', 'start_frame', 'end_frame'])
        success_path = out_dir / "full_rideinfos.txt"
        success_df.to_csv(success_path, index=False)
        logging.info(f"Saved successful ride infos to {success_path}")
    
    if pipeline['apply_postfilters']:
        logging.info("Applying post-filters...")

        assert split_path.exists(), f"Split file {split_path} does not exist."
        try:
            df = pd.read_csv(split_path)
        except EmptyDataError:
            logging.warning(f"Split file {split_path} is empty. Skipping post-filtering.")
            return
        
        tasks = [
            (row.ride_name, row.start_frame, row.end_frame)
            for row in df.itertuples(index=False)
        ]

        if DEBUG_PIPELINE:
            tasks = tasks[:100]

        filters = cfg['postfilters']
        successes, failures = postfilter_parallel(tasks, filters, out_dir)

        # Save successful subsequences to a csv in the same format as split path
        success_df = pd.DataFrame(successes, columns=['ride_name', 'start_frame', 'end_frame'])
        success_path = out_dir / "full_trackfiltered.txt"
        success_df.to_csv(success_path, index=False)
        logging.info(f"Saved successful subsequences to {success_path}")

        # Save failures and successes combines to a csv
        failure_df = pd.DataFrame(failures, columns=['ride_name', 'start_frame', 'end_frame'])
        failure_df["success"] = False
        success_df["success"] = True
        combined_df = pd.concat([failure_df, success_df], ignore_index=True)
        combined_df = combined_df.sort_values(by=['ride_name', 'start_frame', 'end_frame'])
        combined_df.reset_index(drop=True, inplace=True)

        combined_path = out_dir / "full_trackfiltered_combined.txt"
        combined_df.to_csv(combined_path, index=False)
        logging.info(f"Saved combined subsequences to {combined_path}")

    if pipeline['download_imagery']:
        logging.info("Downloading satellite imagery...")
        assert split_path.exists(), f"Split file {split_path} does not exist."
        try:
            df = pd.read_csv(split_path)
        except EmptyDataError:
            logging.warning(f"Split file {split_path} is empty. Skipping visualization.")
            return

        if df.empty:
            logging.warning(f"No subsequences found in {split_path}. Skipping visualization.")
            return

        tasks = [
            (row.ride_name, row.start_frame, row.end_frame)
            for row in df.itertuples(index=False)
        ]

        # Limit for debug
        if DEBUG_PIPELINE:
            tasks = tasks[:5]
            for i, (ride, sf, ef) in enumerate(tasks):
                logging.info(f"Task {i+1}/{len(tasks)}: Downloading imagery for {ride} [{sf}, {ef}]")
                _download_one_satellite_image(
                    ride_name=ride,
                    start_frame=sf,
                    end_frame=ef,
                    root_dir=root_dir,
                    out_dir=out_dir
                )
        else:
            # Run in parallel (use n_jobs=-1 to use all CPUs, or set to some fixed number)
            results = Parallel(n_jobs=16, verbose=5)(
                delayed(_download_one_satellite_image)(ride, sf, ef, root_dir, out_dir)
                for ride, sf, ef in tasks
            )

        # Filter successes
        successes = [(ride, sf, ef) for ride, sf, ef, ok in results if ok]

        successes_sorted = sorted(successes, key=sort_key)
        # Save succcessful downloads to a csv
        success_df = pd.DataFrame(successes_sorted, columns=['ride_name', 'start_frame', 'end_frame'])
        success_path = out_dir / "full_satellite.txt"
        success_df.to_csv(success_path, index=False)
        logging.info(f"Saved successful downloads to {success_path}")

    if pipeline['compute_routes']:
        logging.info("Computing global routes and turn-by-turn instructions...")
        assert split_path.exists(), f"Split file {split_path} does not exist."
        try:
            df = pd.read_csv(split_path)
        except EmptyDataError:
            logging.warning(f"Split file {split_path} is empty. Skipping route computation.")
            return
        if df.empty:
            logging.warning(f"No subsequences found in {split_path}. Skipping route computation.")
            return
        
        tasks = [
            (row.ride_name, row.start_frame, row.end_frame)
            for row in df.itertuples(index=False)
        ]

        results = []
        if DEBUG_PIPELINE:
            for i, (ride, sf, ef) in tqdm(enumerate(tasks), total=len(tasks), desc="Computing routes"):
                logging.info(f"Task {i+1}/{len(tasks)}: Computing routes for {ride} [{sf}, {ef}]")
                results.append(_download_one_satellite_route(
                    ride_name=ride,
                    start_frame=sf,
                    end_frame=ef,
                    root_dir=root_dir,
                    out_dir=out_dir
                ))
                # results.append((ride, sf, ef, ok))
        else:
            # Run in parallel (use n_jobs=-1 to use all CPUs, or set to some fixed number)
            results = Parallel(n_jobs=32, verbose=5)(
                delayed(_download_one_satellite_route)(ride, sf, ef, root_dir, out_dir)
                for ride, sf, ef in tasks
            )
        
        # Filter successes
        successes = [(ride, sf, ef) for ride, sf, ef, ok in results if ok]
        successes_sorted = sorted(successes, key=sort_key)
        # Save successful routes to a csv
        success_df = pd.DataFrame(successes_sorted, columns=['ride_name', 'start_frame', 'end_frame'])
        success_path = out_dir / "full_routes.txt"
        success_df.to_csv(success_path, index=False)
        logging.info(f"Saved successful routes to {success_path}")

    if pipeline['compute_entity_masks']:
        logging.info("Computing entity masks...")
        assert split_path.exists(), f"Split file {split_path} does not exist."
        try:
            df = pd.read_csv(split_path)
        except EmptyDataError:
            logging.warning(f"Split file {split_path} is empty. Skipping route computation.")
            return
        if df.empty:
            logging.warning(f"No subsequences found in {split_path}. Skipping route computation.")
            return

        tasks = [
            (row.ride_name, row.start_frame, row.end_frame)
            for row in df.itertuples(index=False)
        ]

        vlm_cfg = cfg['vlm_params']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        vlm_model_dict, sam2_model_dict, dino_model_dict = \
            build_mask_models(
                vlm_cfg,
                device
            )

        if DEBUG_PIPELINE:
            tasks = tasks[:20]
        results = []

        for i, (ride, sf, ef) in tqdm(enumerate(tasks), total=len(tasks), desc="Computing target masks"):
            logging.info(f"Task {i+1}/{len(tasks)}: Computing entity masks for {ride} [{sf}, {ef}]")
            results.append(_compute_entity_masks_single(
                ride_name=ride,
                start_frame=sf,
                end_frame=ef,
                root_dir=root_dir,
                out_dir=out_dir,
                vlm_model_dict=vlm_model_dict,
                sam2_model_dict=sam2_model_dict,
                dino_model_dict=dino_model_dict,
                caption_only=cfg.get('save_entity_caption_only', False),
                cache_enabled=cfg.get('cache_enabled', True),
                device=device
            ))
        
        # Filter successes
        successes = [(ride, sf, ef) for ride, sf, ef, ok in results if ok]
        successes_sorted = sorted(successes, key=sort_key)
        # Save successful routes to a csv
        success_df = pd.DataFrame(successes_sorted, columns=['ride_name', 'start_frame', 'end_frame'])
        success_path = out_dir / "full_entitymasks.txt"
        success_df.to_csv(success_path, index=False)
        logging.info(f"Saved successful entity masks to {success_path}")

    if pipeline['sample_splits']:
        logging.info("Sampling splits...")
        #1 Initialize models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to HDF5")
    parser.add_argument("--cfg_file", type=str, default="./scripts/preprocessing/config/frodo_mining.yaml", help="Path to config YAML")
    parser.add_argument("--split", type=str, default="frodo_mine.txt", help="Path to the split file")
    parser.add_argument("--ride", type=str, default="", help="List of rides to process")
    args = parser.parse_args()

    # Load config to dictionary
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg, args)
    
