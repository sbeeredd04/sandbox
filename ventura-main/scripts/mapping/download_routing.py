# download_routing.py  (replace the old body with this)

from __future__ import annotations
from pathlib import Path
import os, json, logging, math, textwrap
import hickle as hkl
import cv2
import numpy as np
import requests
from typing import List, Tuple, Dict, Any, Sequence
import re, html

from scripts.mapping.download_satellite import (
    load_gps_aligned_data,
    read_google_api_key,
)
from scripts.utils.satellite_utils import (
    align_compass_to_ts, 
    align_gps_heading_nearest,
    annotate_satellite_image, 
    compute_visibility_and_distances,
    get_haversine_m,
    gps_to_local_xy,
)
from scripts.utils.loader_utils import (
    load_timestamps, 
    load_gps, 
    load_inertial
)
from spinflow.dataset.frodo_helpers import set_frodo_dir

# ───────────────────────── helpers ──────────────────────────────────────

def _sample_subgoals(
    future_gps: np.ndarray, N: int
) -> np.ndarray:
    """
    Return N sub-goals that split the remaining trajectory into
    N + 1 equal-length segments.

    Parameters
    ----------
    future_gps : (M, 2) array of [lat, lon] in degrees,  M ≥ 2
    N          : number of sub-goals to generate

    Notes
    -----
    • Uses the already-vectorised `get_haversine_m()`.  
    • Performs linear interpolation between bounding points instead of
      picking the nearest frame, so the spacing is truly uniform even
      when raw waypoint spacing is irregular.
    """
    if N <= 0 or len(future_gps) < 2:
        return np.empty((0, 2), dtype=future_gps.dtype)

    # ─── 1) cumulative path length ─────────────────────────────────────
    seg_d = get_haversine_m(
        future_gps[:-1, 0], future_gps[:-1, 1],
        future_gps[1:, 0],  future_gps[1:, 1]
    )                                                   # (M-1,)
    cum = np.insert(np.cumsum(seg_d), 0, 0.0)           # (M,)
    total = cum[-1]

    if total == 0.0:                                    # degenerate track
        return np.repeat(future_gps[:1], N, axis=0)

    # ─── 2) desired arc-length positions (skip 0 and total) ────────────
    targets = np.linspace(0, total, N + 2, endpoint=True)[1:-1]  # (N,)

    # ─── 3) locate each target within its segment ─────────────────────
    lo_idx = np.searchsorted(cum, targets, side="right") - 1      # (N,)
    hi_idx = lo_idx + 1

    # linear interpolation weight along the segment
    seg_len = cum[hi_idx] - cum[lo_idx]
    alpha   = (targets - cum[lo_idx]) / seg_len                   # (N,)

    # ─── 4) interpolate lat/lon ───────────────────────────────────────
    sub_goals_lat = (1.0 - alpha) * future_gps[lo_idx, 0] + alpha * future_gps[hi_idx, 0]
    sub_goals_lon = (1.0 - alpha) * future_gps[lo_idx, 1] + alpha * future_gps[hi_idx, 1]

    sub_goals = np.stack((sub_goals_lat, sub_goals_lon), axis=1).astype(future_gps.dtype)

    return sub_goals

def get_forward_gps_mask(
    gps: np.ndarray, hdg: np.ndarray
) -> np.ndarray:
    # Only keep waypoints that do not backtrack
    origin_xyz = np.hstack((gps[0][0], gps[0][1], hdg[0]))     # lat,lon,ψ₀
    loc_xy     = gps_to_local_xy(gps, origin_xyz)             # (M,2)  x,y
    forward  = loc_xy[:, 0]
    cum_max = np.maximum.accumulate(forward)
    keep_mask = np.concatenate(([True], forward[1:] > cum_max[:-1]))
    return keep_mask

def _decode_polyline(poly: str) -> list[tuple[float, float]]:
    coords, lat, lon = [], 0, 0
    i, length = 0, len(poly)

    while i < length:
        for coord in (lat, lon):
            shift, result = 0, 0
            while True:
                b = ord(poly[i]) - 63; i += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if result & 1 else (result >> 1)
            if coord is lat:
                lat += d
            else:
                lon += d
        coords.append((lat * 1e-5, lon * 1e-5))
    return coords

def _get_route_google(
    origin: Tuple[float, float],
    dest:   Tuple[float, float],
    api_key: str,
    spacing_m: float = 10.0,
) -> Dict[str, Any]:
    # --- helper: polyline → equally-spaced list via _sample_subgoals ----
    def _coords_from_poly(poly: str) -> list[tuple[float, float]]:
        pts = np.asarray(_decode_polyline(poly), dtype=np.float64)
        if pts.shape[0] < 2:
            return pts.tolist()

        total = get_haversine_m(pts[:-1, 0], pts[:-1, 1],
                                pts[1:, 0],  pts[1:, 1]).sum()
        if total == 0.0:
            return [tuple(pts[0])]

        N = max(1, int(total // spacing_m))        # interior sub-goals
        subs = _sample_subgoals(pts, N)            # (N,2)
        full = np.vstack([pts[0], subs, pts[-1]])
        return [tuple(p) for p in full]

    # ---------- preferred: Routes API -----------------------------------
    hdrs = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask":
            ("routes.polyline.encodedPolyline,"
             "routes.distanceMeters,"
             "routes.duration,"
             "routes.legs.steps.navigationInstruction")
    }
    body = {
        "origin":      {"location":{"latLng":{"latitude": origin[0],
                                             "longitude": origin[1]}}},
        "destination": {"location":{"latLng":{"latitude": dest[0],
                                             "longitude": dest[1]}}},
        "travelMode": "WALK",
        "languageCode": "en-US"
    }
    try:
        r = requests.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            headers=hdrs, json=body, timeout=5
        )
        if r.ok:
            data   = r.json()
            poly   = data["routes"][0]["polyline"]["encodedPolyline"]
            coords = _coords_from_poly(poly)
            return {"coords": coords, "scheme": "routes", "payload": data}
    except requests.RequestException as e:
        logging.warning(f"Routes API failed ({e}); falling back to legacy.")

    # ---------- fallback: legacy Directions API -------------------------
    url = ("https://maps.googleapis.com/maps/api/directions/json"
           f"?origin={origin[0]},{origin[1]}"
           f"&destination={dest[0]},{dest[1]}"
           f"&mode=walking"
           f"&language=en"
           f"&key={api_key}&alternatives=false")
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data  = r.json()
    poly  = data["routes"][0]["overview_polyline"]["points"]
    coords = _coords_from_poly(poly)
    return {"coords": coords, "scheme": "directions", "payload": data}

# def _get_turn_instruction(
#     route: Dict[str, Any],
#     robot_heading_deg: float,
#     heading_aligned: bool = False,
# ) -> str:
#     """
#     Return the first non-trivial instruction, replacing cardinal directions
#     with robot-relative words: forward / right / backward / left.

#     Parameters
#     ----------
#     route              : dict  – output of _get_route_google()
#     robot_heading_deg  : float – 0° = facing North, CW positive
#     heading_aligned    : bool  – if True, the map/instruction has already been
#                                  rotated so that robot-forward == North
#     """
#     # ── helper: cardinal → relative -----------------------------------
#     card2deg = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}

#     def rel_word(card: str) -> str:
#         hdg_robot = 0.0 if heading_aligned else robot_heading_deg % 360.0
#         diff = (card2deg[card] - hdg_robot) % 360.0
#         if diff < 45 or diff >= 315:
#             return "forward"
#         elif diff < 135:
#             return "right"
#         elif diff < 225:
#             return "backward"
#         else:
#             return "left"

#     def replace_cardinals(text: str) -> str:
#         for c in card2deg:
#             text = re.sub(rf"\b{c}\b",  rel_word(c), text, flags=re.I)
#         return text

#     # ── extract the first meaningful step -----------------------------
#     if route["scheme"] == "routes":
#         steps = route["payload"]["routes"][0]["legs"][0]["steps"]
#         extractor = lambda st: st["navigationInstruction"]["instructions"]
#     else:  # legacy Directions
#         steps = route["payload"]["routes"][0]["legs"][0]["steps"]
#         extractor = lambda st: re.sub("<[^>]+>", "", st["html_instructions"])
    
#     for st in steps:
#         txt = html.unescape(extractor(st)).replace("\u00A0", " ")
#         txt = re.sub(r"\bDestination\b.*", "", txt, flags=re.I).strip()  # ← NEW
#         low = txt.lower()
#         if not low.startswith(("arrive", "you have", "continue")):
#             print("txt:", txt, " cardinal ", replace_cardinals(txt))  # DEBUG
#             import pdb; pdb.set_trace()
#             return replace_cardinals(txt)
    
#     return "Continue forward"

def _get_turn_instruction(
    route: Dict[str, Any],
    robot_heading_deg: float,
    heading_aligned: bool = False,   # kept for call-site compatibility
    done_thresh_m: float = 10.0,     # “too close” = 10 m
) -> str:
    """
    Geometry-based instruction.

    • If the remaining route is ≤ done_thresh_m long, return "Destination".
    • Else compare the first segment bearing with robot_heading_deg and return
      "Continue forward" / "Turn right" / "Turn left".
    """
    coords = route.get("coords", [])
    if len(coords) < 2:
        return "Destination reached"

    # too-short 2-point route?
    if len(coords) == 2:
        d = get_haversine_m(coords[0][0], coords[0][1],
                            coords[1][0], coords[1][1])
        if d < done_thresh_m:
            return "Destination reached"

    # --- bearing P0 → P1 ------------------------------------------------
    lat1, lon1 = np.radians(coords[0])
    lat2, lon2 = np.radians(coords[1])

    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    # --- angle diff [-180°,180°] ----------------------------------------
    diff = ((bearing - robot_heading_deg + 540.0) % 360.0) - 180.0

    if abs(diff) < 30.0:
        return "Continue forward"
    elif diff > 0:
        return "Turn right"
    else:
        return "Turn left"


# ───────────────────────── main entry ───────────────────────────────────
def download_satellite_routes(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str,
    N: int = 20,
    draw_vis: bool = True
) -> bool:
    """
    • Sub-sample *N* way-points along future GPS.  
    • Query Google once per segment (origin→wpᵢ).  
    • Take the half-way waypoint, fetch one more route, pull its *first
      non-null* instruction.  
    • Save {'gps_subgoals', 'xy_subgoals', 'routes', 'turn_instruction'}
      to OUT_DIR/ride_name_routes.hkl.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key is None:
        api_key = read_google_api_key(".gmap_api_key")
    assert api_key is not None, \
        "Please set GOOGLE_MAPS_API_KEY in env or provide it in .gmap_api_key file."

    # ─── 1) path parsing ────────────────────────────────────────────────
    parts = ride_name.split()
    ride_dir = set_frodo_dir(root_dir, *parts)
    ride_id, driveid0 = parts[0], parts[1]

    # Load satellite_infos for plotting
    output_ride_dir = set_frodo_dir(out_dir.parent, *parts)
    try:
        sat_path = output_ride_dir / f"seq_{start_frame}" / "satellite_info.h5"
        sat_info = hkl.load(sat_path)

        path_tracker_path = output_ride_dir / f"seq_{start_frame}" / "path_tracker.h5"
        path_info = hkl.load(path_tracker_path)
    except FileNotFoundError:
        logging.error(f"Satellite or path info not found for {ride_name} at frame {start_frame}.")
        return False
    
    ts_path  = ride_dir / f"front_camera_timestamps_{driveid0}.csv"
    gps_path = ride_dir / f"gps_data_{driveid0}.csv"
    imu_path = ride_dir / f"imu_data_{driveid0}.csv"

    gps_aligned, hdg_aligned, video_timestamps = load_gps_aligned_data(
        ts_path, gps_path, imu_path
    ).values()
    if start_frame < 0 or end_frame >= len(video_timestamps):
        logging.error(f"Invalid frame range: {start_frame} to {end_frame} "
                      f"for {ride_name} ({len(video_timestamps)} frames total).")
        return False
    cur_gps = gps_aligned[start_frame]  # [2]
    cur_hdg = hdg_aligned[start_frame]  # scalar
    fut_gps = gps_aligned[start_frame:]  # [M×2]
    fut_hdg = hdg_aligned[start_frame:]  # [M,]
    fut_ts  = video_timestamps[start_frame:]  # [M,] timestamps

    # Filter future gps for forward points and in image bounds
    keep_mask = get_forward_gps_mask(fut_gps, fut_hdg)
    vis_mask, _ = compute_visibility_and_distances(
        sat_info["satellite_query"], fut_gps, fut_hdg[0]
    )
    keep_mask = keep_mask & vis_mask

    fut_gps = fut_gps[keep_mask]  # [M',2] where M' ≤ M
    fut_hdg = fut_hdg[keep_mask]  # [M',]
    fut_ts  = fut_ts[keep_mask]   # [M',] timestamps
  
    if len(fut_gps) < 2:
        logging.error("Too few future points"); return False

    try:
        # ─── 3) route & turn instruction ──────────────────────────────
        route = _get_route_google(cur_gps, fut_gps[-1], api_key)
        turn_text = _get_turn_instruction(route, cur_hdg, heading_aligned=True)
        print("Turn instruction:", turn_text)  # DEBUG
    except Exception as e:
        logging.error(f"Failed to fetch routes for {ride_name}: {e}")
        return False

    origin = (cur_gps[0], cur_gps[1], cur_hdg)  # [lat, lon, heading]
    snap_gps = np.array(route["coords"], dtype=np.float64)
    snap_xy  = gps_to_local_xy(snap_gps, origin)

    # ─── 3-bis) expected arrival-time at each sub-goal  ────────────
    avg_speed = 1.0

    # b) cumulative distance along the snapped poly-line
    seg  = get_haversine_m(snap_gps[:-1,0], snap_gps[:-1,1],
                            snap_gps[1:,0],  snap_gps[1:,1])               # (L-1,)
    cumd = np.insert(np.cumsum(seg), 0, 0.0)                      # (L,)

    # c) ETA for each sub-goal: nearest snapped point → cumd[idx]/v
    eta_secs = []
    for sg in snap_gps:
        idx = np.argmin(get_haversine_m(sg[0], sg[1],
                            snap_gps[:,0], snap_gps[:,1]))
        eta_secs.append(cumd[idx] / avg_speed)
    eta_secs = np.asarray(eta_secs, dtype=np.float64)              # (N,)
    
    # filter out gps along route that are not in the image bounds
    valid_mask, _ = compute_visibility_and_distances(
        sat_info["satellite_query"], snap_gps, cur_hdg
    ) 
    snap_gps = snap_gps[valid_mask]  # [K,2] where K ≤ N
    snap_xy  = snap_xy[valid_mask]   # [K,2] where
    eta_secs = eta_secs[valid_mask]  # [K,]

    if len(snap_gps) < 2:
        logging.error("No valid snapped GPS points found along the route.")
        return False

    # ─── 4) optional visualisation ──────────────────────────────────────
    if draw_vis:
        try:
            sub_hdg = np.zeros((len(snap_gps),), dtype=np.float64)
            sub_hdg[0] = cur_hdg

            snapped = np.array(route["coords"], dtype=np.float64)
            sat = annotate_satellite_image(           # your helper
                sat_info["satellite_image"],  # (H,W,3) RGB
                snap_gps,
                sub_hdg,
                sat_info["satellite_query"],
                heading_aligned=True,
                draw_cone=True,
            )

            vis_im = sat.copy()
            # draw top-left turn text
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            for i,line in enumerate(textwrap.wrap(turn_text, 40)):
                cv2.putText(vis_im, line, (10,30+25*i), font, fs,
                            (255,255,255), th, cv2.LINE_AA)
           
            vis_im = np.hstack([
                vis_im,
                cv2.cvtColor(path_info['front_rgb'], cv2.COLOR_RGB2BGR)
            ])
            test_dir = "output_route_vis"
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            vis_path = Path(test_dir) / f"ride_{ride_id}_{driveid0}_{parts[2]}_seq_{start_frame}.jpg"
            # cv2.imwrite(str(vis_path), vis_im)
            # cv2.imwrite("test.jpg", vis_im)  # for debugging
        except Exception as e:
            logging.warning(f"Vis failed: {e}")
            return False
    
    # ─── 5) save ────────────────────────────────────────────────────────
    out = {
        "snap_gps":   snap_gps.astype(np.float64),  # (K,2) degrees
        "snap_xy":    snap_xy.astype(np.float64),   # (K,2) meters
        "current_gps":    cur_gps.astype(np.float64),  # [2] degrees
        "current_heading": cur_hdg,                     # scalar degrees
        "route":         route,                          # raw JSON payloads
        "turn_instruction": turn_text,
        "eta_seconds": eta_secs.astype(np.float64),  # [K,] seconds
    }

    out_path = output_ride_dir / f"seq_{start_frame}" / "routing_info.h5"
    hkl.dump(out, out_path, mode="w")
    logging.info(f"Saved routing for {ride_name} to {out_path}")
    return True
