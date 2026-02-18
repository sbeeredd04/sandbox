import math
import logging
from pathlib import Path

import numpy as np
import cv2
import h5py
import hickle as hkl

from scripts.utils.satellite_utils import (
    get_satellite_image,
    get_gmap_satellite_image,
    prepare_satellite_query,
    align_compass_to_ts,
    align_gps_heading_nearest,
    annotate_satellite_image,
    compute_visibility_and_distances
)
from scripts.utils.loader_utils import (
    load_timestamps,
    load_gps,
    load_inertial
)
from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)

def read_google_api_key(api_key_path=".gmap_api_key"):
    try:
        open(api_key_path, 'r').close()  # Check if file exists
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            logging.warning(f"Google API key file {api_key_path} is empty.")
            return None
        logging.info(f"Using Google API key from {api_key_path}")
        return api_key
    except FileNotFoundError:
        logging.warning(f"Google API key file {api_key_path} not found. Using Esri instead.")
        return None

def load_gps_aligned_data(ts_path, gps_path, imu_path):
    video_timestamps = load_timestamps(ts_path)         # [N,]
    gps_np = load_gps(gps_path, format='numpy')         # [Ngps×3] lat,lon,t
    if gps_np is None or gps_np.size == 0:
        logging.error(f"Failed to load GPS data from {gps_path}.")
        return False
    gps_np = gps_np[:, [2, 0, 1]]  # [Ngps×3] t, lat, lon
    gps_ts  = gps_np[:, 0]
    gps_pos = gps_np[:, 1:3]
    
    comp_np = load_inertial(imu_path, format='numpy')          # [N×2] ts, heading
    if comp_np is None or comp_np.size == 0:
        logging.error(f"Failed to load compass data from {imu_path}.")
        return False
    comp_ts = comp_np[:, 0].astype(np.float64)
    comp_hd = comp_np[:, 1].astype(np.float64)                  # degrees

    # Interpolate heading and GPS to match video timestamps
    comp_np = align_compass_to_ts(
        tgt_timestamps=gps_ts,
        compass_timestamps=comp_ts,
        compass_headings=comp_hd
    )  # [N]
    comp_np = np.concatenate((gps_ts[:, None], comp_np[:, None]), axis=1)

    gps_aligned_np, heading_aligned_np = align_gps_heading_nearest(
        gps_np, comp_np, video_timestamps
    )

    return {
        "gps_aligned": gps_aligned_np,
        "heading_aligned": heading_aligned_np,
        "video_timestamps": video_timestamps,
    }

def download_satellite_imagery(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str,
    zoom: int = 15,
    api_key: str | None = None,
    margin: float = 1.1,
):
    """
    For a given ride subsequence, load the GPS, Mag, and video timestamps,
    compute a bounding box around the future trajectory, download a satellite
    image centered on the robot and rotated so that its current heading points up,
    and save everything into an HDF5.

    Args:
        ride_name:    e.g. "1234_5678_91011_20250101T123000Z"
        start_frame:  index of the first video frame in this subsequence
        end_frame:    index of the last  video frame in this subsequence
        root_dir:     base path where your raw rides live
        out_dir:      base path where we should write output_ride_*/…
        zoom:         static maps zoom level
        api_key:      if provided, will use Google Static Maps; otherwise Esri
        margin:       expand bounding box by this factor
    """
    # ---- 1) Parse ride_name ----
    parts = ride_name.split(' ')
    ride_dir = set_frodo_dir(root_dir, *parts)  # e.g. "ride_1234_5678_91011_20250101T123000Z"
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    ride_id = parts[0]
    driveid0 = parts[1]
    driveid1 = parts[2]
    timestamp = "_".join(parts[3:])

    # ---- 2) load all the data ----
    # Paths may need tweaking to match your layout:
    ts_path   = ride_dir / f"front_camera_timestamps_{driveid0}.csv"
    gps_path  = ride_dir / f"gps_data_{driveid0}.csv"
    imu_path  = ride_dir / f"imu_data_{driveid0}.csv"
    gps_aligned, hdg_aligned, video_timestamps = load_gps_aligned_data(
        ts_path, gps_path, imu_path
    )

    # ---- 3) match video frames → GPS indices ----
    current_gps = gps_aligned[start_frame]  # [2]
    current_heading = hdg_aligned[start_frame]  # scalar
    future_gps      = gps_aligned[start_frame:]  # [M×2]
    future_heading  = hdg_aligned[start_frame:]  # [M,]
    future_ts       = video_timestamps[start_frame:]  # [M,] timestamps

    # ---- 3) compute map center ----
    lats = future_gps[:, 0]
    lons = future_gps[:, 1]
    # center_lat = float(lats.mean())
    # center_lon = float(lons.mean())
    center_lat = future_gps[0, 0]
    center_lon = future_gps[0, 1]

    # meters per degree at map center
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = (
        111132.954
      - 559.822 * math.cos(2 * lat_rad)
      +   1.175 * math.cos(4 * lat_rad)
      -   0.0023 * math.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat_rad)
      -   93.5   * math.cos(3 * lat_rad)
      +    0.118 * math.cos(5 * lat_rad)
    )

    # ---- 4) find max radial distance (meters) of any future point ----
    # 4a) get east/west & north/south extrema around the center
    dx = (lons - center_lon) * m_per_deg_lon   # east‐positive
    dy = (lats - center_lat) * m_per_deg_lat   # north‐positive (we’ll treat abs)

    # 4b) half‐extents in km *with* margin, separately
    half_w_km = abs(dx).max() / 1000.0 * margin
    half_h_km = abs(dy).max() / 1000.0 * margin

    # 4c) full ground‐box dimensions
    width_km  = 2 * half_w_km
    height_km = 2 * half_h_km
    # if width_km <= 50.0:
    #     logging.warning(f"Max distance {width_km:.2f} m is too small; skipping satellite image download.")
    #     return False  # no points far enough away to warrant a satellite image

    # half‐side in km, with margin
    # half_side_km = (max_dist_m / 1000.0) * margin
    # side_km = 2 * half_side_km

    DESIRED_PX = 1024
    DESIRED_PY = 576

    # current_heading is in degrees; convert to radians
    theta = np.deg2rad(current_heading)

    # compute exact bounding‐box of a DESIRED_PX×DESIRED_PY rectangle
    # when rotated by theta:
    cos_t = abs(np.cos(theta))
    sin_t = abs(np.sin(theta))
    # minimal enclosing box
    req_px = int(np.ceil(cos_t * DESIRED_PX + sin_t * DESIRED_PY))
    req_py = int(np.ceil(sin_t * DESIRED_PX + cos_t * DESIRED_PY))
    # but never smaller than our target crop window:
    req_px = max(req_px, DESIRED_PX)
    req_py = max(req_py, DESIRED_PY)
    # ---- fetch with the larger pixel size ----
    # choose a rectangular ground box so meters/pixel is uniform
    query = prepare_satellite_query(
        gps_np    = np.array([[center_lat, center_lon]]),
        width_km  = width_km,
        height_km = height_km,
        width_px  = req_px,
        height_px = req_py,
    )
    if query is None:
        logging.error("Failed to prepare satellite query; exiting.")
        return False

    # Check if Gmap API key is provided
    api_key = read_google_api_key(".gmap_api_key")
    # api_key = None
    # if api_key is None:
    #     logging.info("Using Esri satellite imagery (no API key provided).")
    # else:
    #     logging.info(f"Using Google Maps satellite imagery with API key from .gmap_api_key")
    
    if api_key:
        query['api_key'] = api_key
        img = get_gmap_satellite_image(query)
    else:
        img = get_satellite_image(query)
    
    if img is None:
        logging.error("Failed to download satellite image; exiting.")
        return False

    # ---- rotate about center ----
    h_px, w_px = query['height'], query['width']
    M = cv2.getRotationMatrix2D((w_px/2, h_px/2), current_heading, 1.0)
    img_rot = cv2.warpAffine(img, M, (w_px, h_px))
    # cv2.imwrite("test_rot.jpg", img_rot)
    # ---- now center‐crop to the final 640×640 ----
    img_crop = cv2.getRectSubPix(
        img_rot,
        patchSize=(DESIRED_PX, DESIRED_PY),
        center=(w_px/2.0, h_px/2.0),
    )
    # cv2.imwrite("test_crop.jpg", img_crop)

    # Compute the visibility and distances
    visible, distances = compute_visibility_and_distances(
        query,
        future_gps,
        current_heading=heading_aligned_np[start_frame],
        display_px=DESIRED_PX,
        display_py=DESIRED_PY,
    )

    false_idxs = np.where(~visible)[0]
    if false_idxs.size > 0:
        # points up to (but not including) the first False are visible
        last_visible = false_idxs[0] - 1
    else:
        # no False at all ⇒ everything is visible
        last_visible = visible.shape[0] - 1

    if last_visible >= 0:
        gps_to_plot      = future_gps[0 : last_visible + 1 : 10]  # every 10th
        heading_to_plot  = heading_aligned_np[start_frame : start_frame + last_visible + 1 : 10]
    else:
        gps_to_plot     = np.empty((0,2))
        heading_to_plot = np.empty((0,))

    # Grab every 10th gps and heading from np
    annotated_image = annotate_satellite_image(
        img_crop.copy(),
        gps_to_plot,
        heading_to_plot,
        query,
        heading_aligned=True
    )
    # cv2.imwrite("test_annotated.jpg", annotated_image)
    # import pdb; pdb.set_trace()
    # ---- 7) write out HDF5 ----
    seq_dir = (
        Path(out_dir)
        / f"ride_{driveid0}_{driveid1}_{timestamp}"
        / f"seq_{start_frame}"
    )
    seq_dir.mkdir(parents=True, exist_ok=True)
    h5_path = seq_dir / "satellite_info.h5"
    save_dict = {
        "satellite_image": img_crop,  # [640, 640, 3]
        "gt_route_image": annotated_image,  # [640, 640, 3]
        "current_gps": current_gps,  # [2]
        "current_heading": current_heading,  # scalar
        "future_ts": future_ts,  # [M,]
        "future_gps": future_gps,  # [M, 2]
        "future_heading": future_heading,  # [M,]
        "satellite_query": query,
        "future_visible": visible,  # [M,]
        "future_distances": distances,  # [M,]
    }
    try:
        hkl.dump(save_dict, h5_path, mode='w')
    except Exception as e:
        logging.error(f"Failed to write satellite info to {h5_path}: {e}")
        return False
    logging.info(f"Wrote satellite info (including query) to {h5_path}")
    return True