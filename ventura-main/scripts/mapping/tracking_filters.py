from scripts.utils.log_utils import logging
from pathlib import Path

import cv2
import numpy as np
import hickle as hkl  # or whatever loader you use
from shapely.geometry import MultiPoint
from scipy.spatial.distance import pdist
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter

from scripts.preprocessing.track_points_online_reinit import get_colormap
import spinflow.dataset.frodo_helpers as fh

DEBUG_TRACKS = False

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
from numpy.linalg import norm

def last_windows_ok(tracks, t_ids, sides,
                    *,
                    K=3,
                    n_min=2,
                    eps=60.0,
                    min_samples=2,
                    rms_max=4.0,
                    ang_max=120,
                    min_centroid_dist=12.0,
                    enable_midpoint_check=True,
                    **kwargs):
    """
    Returns
        True   → keep sequence
        False  → reject (bad tail)

    Extra rule:
        For each side, the centroids of the clusters at the last K timestamps
        must be separated from each other by ≥ min_centroid_dist pixels.
    """
    # ─── isolate points from last K unique timestamps ───────────────
    uniq_ts = np.unique(t_ids)
    if uniq_ts.size < K:
        logging.info("  DROPPED: not enough unique timestamps")
        return False

    tail_ts = uniq_ts[-K:]                            # ascending order
    sel     = np.isin(t_ids, tail_ts)
    P, T, S = tracks[sel], t_ids[sel], sides[sel]

    # will accumulate centroids per side, aligned with tail_ts
    centroids = { -1: [], +1: [] }

    # ─── per-timestamp & side checks ────────────────────────────────
    for t in tail_ts:
        for side in (-1, +1):
            pts = P[(T == t) & (S == side)]

            # (a) enough raw points
            if pts.shape[0] < n_min:
                logging.info(f"  DROPPED: {pts.shape[0]} pts on side {side} at t={t}")
                return False

            # (b) DBSCAN ⇒ exactly one cluster
            if pts.shape[0] < min_samples:
                logging.info(f"  DROPPED: < min_samples pts on side {side} at t={t}")
                return False
            lbls     = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
            clusters = {l for l in lbls if l >= 0}
            if len(clusters) != 1:
                logging.info(f"  DROPPED: {len(clusters)} clusters on side {side} at t={t}")
                return False

            # store centroid for later spacing test
            c = pts[lbls == next(iter(clusters))].mean(0)
            centroids[side].append(c)

    # ─── NEW: centroid separation check ─────────────────────────────
    for side in (-1, +1):
        cs = np.vstack(centroids[side])      # shape (K,2)
        # pairwise distances between every pair of timesteps
        d2 = np.linalg.norm(cs[None,:,:] - cs[:,None,:], axis=-1)
        # ignore diagonal zeros by adding large value on diag then take min
        np.fill_diagonal(d2, np.inf)
        if d2.min() < min_centroid_dist:
            logging.info(f"  DROPPED: centroids on side {side} "
                         f"closer than {min_centroid_dist}px (min={d2.min():.1f})")
            return False

    if enable_midpoint_check:
        # ─── midpoint spline / heading tests (unchanged) ───────────────
        mids = []
        for t in tail_ts:
            l = P[(T == t) & (S == -1)].mean(0)
            r = P[(T == t) & (S == +1)].mean(0)
            mids.append((l + r) * 0.5)
        mids = np.asarray(mids)

        if mids.shape[0] < 3:
            logging.info("  DROPPED: not enough midpoints for spline check")
            return False

        s_x = UnivariateSpline(np.arange(mids.shape[0]), mids[:,0], k=2, s=0)
        s_y = UnivariateSpline(np.arange(mids.shape[0]), mids[:,1], k=2, s=0)
        fit = np.column_stack([s_x(range(mids.shape[0])),
                            s_y(range(mids.shape[0]))])
        rms = np.sqrt(((mids - fit)**2).sum(1).mean())
        if rms > rms_max:
            logging.info(f"  DROPPED: RMS spline residual {rms:.2f} > {rms_max:.2f}")
            return False

        v1, v2 = mids[-2] - mids[-3], mids[-1] - mids[-2]
        if norm(v1) < 1e-3 or norm(v2) < 1e-3:
            logging.info("  DROPPED: zero vector in last 2 midpoints")
            return False
        dtheta = np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                                            (norm(v1)*norm(v2)), -1, 1)))
        if dtheta > ang_max:
            logging.info(f"  DROPPED: heading change {dtheta:.2f}° > {ang_max:.2f}°")
            return False
    
    return True

def detect_and_visualize_protrusions(
    path_mask: np.ndarray,
    image_bgr: np.ndarray,
    open_radius: int = 10,
    min_prop_length: int = 25,
    min_aspect: float = 4.0,
    # min_area: float = 200.0
) -> tuple[int, np.ndarray]:
    """
    Detects thin protrusions in path_mask using oriented bounding boxes
    and draws them in red on image_bgr.

    Returns:
      count        : number of protruding components
      vis_image_bgr: copy of image_bgr with protrusions highlighted
    """
    # 1) Remove main body by opening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*open_radius+1, 2*open_radius+1)
    )
    mask_u8 = (path_mask.astype(np.uint8) * 255)
    opened  = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    residual= cv2.subtract(mask_u8, opened)
    residual_bool = residual.astype(bool)

    # prepare visualization
    vis = image_bgr.copy()

    # 2) Label and examine each residual component
    num_labels, labels = cv2.connectedComponents(residual.astype(np.uint8))
    count = 0

    for lbl in range(1, num_labels):
        comp = (labels == lbl)
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue

        # build point list for minAreaRect
        pts = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(pts)      # ((cx,cy),(w,h),angle)
        (w_rect, h_rect) = rect[1]
        long_side  = max(w_rect, h_rect)
        short_side = min(w_rect, h_rect)
        area = long_side * short_side

        if DEBUG_TRACKS:
            print(f"Aspect ratio: {long_side / (short_side+1e-6):.2f} ")

        # decide if this is a protrusion
        if long_side >= min_prop_length and short_side > 0 \
           and (long_side / short_side) >= min_aspect:
            count += 1

            # draw oriented bounding box filled in red
            box = cv2.boxPoints(rect).astype(np.int32)  # shape (4,2)
            overlay = vis.copy()
            cv2.drawContours(overlay, [box], 0, (0,0,255), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0, vis)

    if DEBUG_TRACKS:
        # save debugging images
        cv2.imwrite("residual_protrusions.png", residual)
        cv2.imwrite("protrusions_vis.png", vis)
        # import pdb; pdb.set_trace()
    return count, vis

def cluster_and_midpoints(
    tracks: np.ndarray,        # shape (N,2)
    crumb_times: np.ndarray,   # shape (N,)
    sides: np.ndarray,         # shape (N,), values in {-1,1}
    eps: float,
    min_samples: int
) -> dict:
    """
    For each integer time t in crumb_times:
      - DBSCAN‐cluster the left‐side pts (side=-1),
      - DBSCAN‐cluster the right‐side pts (side=+1),
      - Count #clusters on each side per timestamp,
      - If both sides have ≥1 cluster, pick the largest cluster’s centroid,
        record that centroid + the midpoint between them.

    Also computes the overall DBSCAN cluster count across all left/right points.

    Returns:
      {
        'times': np.ndarray (M,),
        'left_centroids': np.ndarray (M,2),
        'right_centroids': np.ndarray (M,2),
        'midpoints': np.ndarray (M,2),
        'cluster_counts_left': list[int] length = #timestamps,
        'cluster_counts_right': list[int],
        'global_cluster_count_left': int,
        'global_cluster_count_right': int
      }
    """
    # --- global cluster counts over all points on each side ---
    ptsL_all = tracks[sides == -1]
    ptsL_all = ptsL_all[np.all(np.isfinite(ptsL_all), axis=1)]
    if ptsL_all.shape[0] >= min_samples:
        lblsL_all = DBSCAN(eps=eps, min_samples=min_samples).fit(ptsL_all).labels_
        global_clusters_L = len({l for l in lblsL_all if l >= 0})
    else:
        global_clusters_L = 0

    ptsR_all = tracks[sides == 1]
    ptsR_all = ptsR_all[np.all(np.isfinite(ptsR_all), axis=1)]
    if ptsR_all.shape[0] >= min_samples:
        lblsR_all = DBSCAN(eps=eps, min_samples=min_samples).fit(ptsR_all).labels_
        global_clusters_R = len({l for l in lblsR_all if l >= 0})
    else:
        global_clusters_R = 0

    # --- per‐time clustering & centroiding ---
    times_all = sorted(np.unique(crumb_times.astype(int)))
    left_cs, right_cs, mids, kept_times = [], [], [], []
    counts_L, counts_R = [], []

    for t in times_all:
        mask_t = (crumb_times.astype(int) == t)
        ptsL = tracks[mask_t & (sides == -1)]
        ptsR = tracks[mask_t & (sides ==  1)]
        ptsL = ptsL[np.all(np.isfinite(ptsL), axis=1)]
        ptsR = ptsR[np.all(np.isfinite(ptsR), axis=1)]

        # left side
        if ptsL.shape[0] >= min_samples:
            cL = ptsL.mean(axis=0) if ptsL.size > 0 else None
        else:
            counts_L.append(0)
            cL = None

        # right side
        if ptsR.shape[0] >= min_samples:
            cR = ptsR.mean(axis=0) if ptsR.size > 0 else None
        else:
            counts_R.append(0)
            cR = None

        # keep only if both sides valid
        if cL is not None and cR is not None:
            left_cs .append(cL)
            right_cs.append(cR)
            mids    .append((cL + cR) * 0.5)
            kept_times.append(t)

    return {
        'times': np.array(kept_times, dtype=int),
        'left_centroids': np.vstack(left_cs)   if left_cs else np.zeros((0,2)),
        'right_centroids': np.vstack(right_cs) if right_cs else np.zeros((0,2)),
        'midpoints': np.vstack(mids)           if mids else np.zeros((0,2)),
        'cluster_counts_left':  counts_L,
        'cluster_counts_right': counts_R,
        'global_cluster_count_left':  global_clusters_L,
        'global_cluster_count_right': global_clusters_R,
    }


def bottom_mask_width_exceeds(mask: np.ndarray, min_width: int, max_width: int) -> bool:
    """
    Checks the bottom row of a binary mask for the leftmost and rightmost 'on' pixels.

    Args:
        mask      : H×W boolean or uint8 mask.
        max_width : maximum allowed width in pixels.

    Returns:
        True  if either no pixel exists on the bottom row, or the width between
              the leftmost and rightmost pixels exceeds max_width.
        False otherwise.
    """
    # ensure boolean
    bottom = mask[-1].astype(bool)  # last row

    xs = np.where(bottom)[0]
    # no pixels at bottom → fail
    if xs.size == 0:
        return True

    left, right = xs.min(), xs.max()
    width = right - left

    # too wide → fail
    return width > max_width or width < min_width

def mask_clear_of_vertical_edges(
    mask: np.ndarray,
    *,
    ignore_bottom: bool = True
) -> bool:
    """
    Fast check that the path mask does NOT touch the left, top or right
    image edge (the bottom edge is usually allowed, so it is skipped by
    default).

    Args
    ----
    mask : H×W bool / uint8 mask.
    ignore_bottom : if False the bottom row will also be tested.

    Returns
    -------
    True  – mask pixels are clear of the checked edges  → keep sample  
    False – at least one pixel hits a checked edge      → drop sample
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D (H×W)")

    # coerce to boolean just once (no copy thanks to NumPy views)
    m = mask.astype(bool, copy=False)

    touches = m[:, 0].any()          # left edge
    touches |= m[0, :].any()         # top edge
    touches |= m[:, -1].any()        # right edge
    if not ignore_bottom:
        touches |= m[-1, :].any()    # bottom edge (optional)

    return not touches               # True → pass, False → fail

def mask_starts_from_sides(mask: np.ndarray, max_offset: int) -> bool:
    """
    Checks whether the bottom-most part of a binary mask touches either
    the left or right edge of the image.

    Args:
        mask       : H×W boolean or uint8 array.
        max_offset : Right-hand x-threshold in pixels (e.g. int(0.8*W)).
                     The corresponding left-hand threshold is W-max_offset.

    Returns
        True  → mask starts too close to a side   (caller should DROP)
        False → mask starts within the central 60 % band
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")

    H, W = mask.shape
    # locate the lowest row that contains at least one foreground pixel
    rows_with_fg = np.where(mask.any(axis=1))[0]
    if rows_with_fg.size == 0:            # empty mask → fail
        return True

    y_bot = rows_with_fg.max()            # deepest row containing fg
    xs    = np.where(mask[y_bot])[0]
    if xs.size == 0:                      # shouldn’t happen, but be safe
        return True

    left_x  = xs.min()
    right_x = xs.max()

    right_thresh = max_offset            # ≈ 0.8 W
    left_thresh  = W - max_offset        # ≈ 0.2 W

    # fail if either edge pixel lies outside the central band
    return (left_x < left_thresh) or (right_x > right_thresh)

def apply_postfilter_single(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    out_dir: Path,
    filters: dict,
    debug_out_dir: Path = None
):
    """
    Applies post-filters to a single subsequence of a ride.
    filters should contain:
      - min_area                     : float (pixels^2)
      - max_corners                  : int
      - min_corner_angle_deg         : float (degrees)
      - max_end_missing_frames       : int
      - max_end_frame                : int
      - min_length_width_ratio       : float
      - max_length_width_ratio       : float (optional)
    """
    logging.info(f"Applying post-filters for {ride_name} [{start_frame}, {end_frame}]")

    # unpack filters
    min_area                   = float(filters['min_area'])
    min_mask_area               = float(filters['min_mask_area'])
    protrusion_kwargs = filters['protrusion_kwargs']
    cluster_kwargs = filters['cluster_kwargs']
    window_kwargs = filters['window_kwargs']

    sequence_status = { "success": [], "failure": [] }
    ride_dir = fh.set_frodo_dir(out_dir.parent, *ride_name.split(' '))
    # seq_dirs = sorted(
    #     [d for d in ride_dir.glob("seq_*")],
    #     key=lambda x: int(x.name.split('_')[-1])
    # )
    # import pdb; pdb.set_trace()
    # seq_dirs = [d for d in seq_dirs
    #             if start_frame <= int(d.name.split('_')[-1]) <= end_frame]
    # import pdb; pdb.set_trace()
    seq_dir = ride_dir / f"seq_{start_frame}"
    if not seq_dir.exists():
        logging.warning(f"Sequence directory {seq_dir} does not exist.")
        return sequence_status
    seq_dirs = [seq_dir]

    did_fail = False
    num_frames = (end_frame - start_frame + 1)
    for idx, seq_dir in enumerate(seq_dirs):
        start_frame = int(seq_dir.name.split('_')[-1])
        if did_fail:
            if DEBUG_TRACKS:
                import pdb; pdb.set_trace()
            sequence_status["failure"].append((ride_name, start_frame, start_frame + num_frames))
        
        logging.info(f" Processing {seq_dir.name}")
        track_path = seq_dir / "path_tracker.h5"
        entity_caption_path = seq_dir / "entity_caption.h5"
        if not track_path.exists(): # or not entity_caption_path.exists():
            logging.warning("  no path_tracker.h5 → skipping")
            continue

        data        = hkl.load(track_path)
        visibility  = data['visibility'][0]  # (T, H,W)
        tracks      = data['tracks'][0]    # (T,N,2)
        last_tracks = tracks[-1][visibility[-1]]  # (N,2) for last frame
        crumbs      = data['crumbs'][0][visibility[-1]]    # (N,3) [t,x,y]
        sides       = data['sides'][0][visibility[-1]]     # (N,)
        path_mask   = data['path_mask']    # (H,W)
        front_rgb   = data['front_rgb']    # (H,W,3)
        num_frames  = tracks.shape[0] 

        front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
        # --- visualization (unchanged) ---
        overlay   = front_bgr.copy()
        overlay[path_mask==1] = (255,255,51)
        front_vis_nocrumbs = cv2.addWeighted(front_bgr, 0.5, overlay, 0.5, 0)
        times     = crumbs[:,0]
        colormap  = get_colormap(times)
        last_xy   = last_tracks             # (N,2)
        front_vis = front_vis_nocrumbs.copy()
        for i,(x,y) in enumerate(last_xy):
            if not np.isnan(x):
                cv2.circle(front_vis, (int(x),int(y)), 3, colormap[i].tolist(), -1)
        H,W,_ = front_vis.shape
        if DEBUG_TRACKS:
            cv2.imwrite("path_mask_nocrumbs.png", front_vis_nocrumbs)
            cv2.imwrite("path_mask.png", cv2.resize(front_vis, (W//2, H//2)))

        # 1) Tiny‐spatial‐spread filter
        pts = crumbs[:,1:3]
        area = MultiPoint(pts).convex_hull.area
        mask_area = path_mask.sum()
        # print("Mask area:", mask_area)
        if area < min_area:
            logging.info(f"  DROPPED: hull area {area:.1f} < {min_area:.1f}")
            did_fail = True
            continue
        if mask_area < min_mask_area:
            logging.info(f"  DROPPED: mask area {mask_area:.1f} < {min_mask_area:.1f}")
            did_fail = True
            continue

        # 1.5) Check if two left and right sides at bottom of mask are too wide
        if bottom_mask_width_exceeds(path_mask, min_width=50, max_width=300):
            logging.info("  DROPPED: bottom mask width exceeds 300 pixels")
            did_fail = True
            continue
        
        if mask_starts_from_sides(path_mask, max_offset=int(0.9*W)):
            logging.info("  DROPPED: mask starts from sides")
            did_fail = True
            continue

        if not mask_clear_of_vertical_edges(path_mask):
            logging.info("  DROPPED: mask touches left / top / right image border")
            did_fail = True
            continue

        # 2) DBSCAN cluster‐count filter per‐side  ‹— MODIFIED
        if filters['cluster_kwargs']['enabled']:
            out = cluster_and_midpoints(
                tracks      = last_tracks,
                crumb_times = crumbs[:,0],
                sides       = sides,
                eps         = cluster_kwargs['eps'],
                min_samples = cluster_kwargs['min_samples']
            )

            # if fewer than 3 valid times, fail
            times = out['times']  # 1D array of kept integer timestamps
            if times.shape[0] < 3:
                logging.info(
                    f"  DROPPED: only {times.shape[0]} timestamps with both clusters (<3)"
                )
                did_fail = True
                continue

            if out['global_cluster_count_left'] < 3 or \
                out['global_cluster_count_right'] < 3:
                logging.info(
                    f"  DROPPED: global cluster counts too low: "
                    f"L={out['global_cluster_count_left']}, "
                    f"R={out['global_cluster_count_right']}"
                )
                did_fail = True
                continue
            
        if window_kwargs['enabled'] and not last_windows_ok(
                tracks      = last_tracks,
                t_ids       = crumbs[:,0],
                sides       = sides,
                **window_kwargs
                ):
                # K           = 3,
                # n_min       = 3,
                # eps         = 20.0,
                # min_samples = 3,
                # rms_max     = 4.0,
                # ang_max     = 20):
                logging.info("  DROPPED: poor tracking in last N windows")
                did_fail = True
                continue

        # # compute midpoint arc-length exactly as before
        # mids = out['midpoints']  # shape (M,2)

        # # 4) Length‐width ratio filter
        # ys, xs = np.nonzero(path_mask)
        # if ys.size < 2:
        #     logging.info("  DROPPED: empty mask")
        #     continue

        # h_bbox = float(ys.max() - ys.min() + 1)
        # w_bbox = float(xs.max() - xs.min() + 1)
        # wh_ratio = w_bbox / (h_bbox + 1e-6)

        # # mids: list of [x,y] centroids you built earlier
        # M = np.array(mids, dtype=float)
        # # plot the midpoints
        # if DEBUG_TRACKS:
        #     for x,y in M:
        #         cv2.circle(front_vis_nocrumbs, (int(x),int(y)), 3, (0,255,0), -1)
        #     cv2.imwrite("path_mask_nocrumbs.png", front_vis_nocrumbs)
        # if M.shape[0] >= 2:
        #     L = float(np.linalg.norm(np.diff(M, axis=0), axis=1).sum())
        # else:
        #     L = 0.0

        # min_length = filters.get('min_length', 190.0)
        # if L < min_length:
        #     logging.info(f"  DROPPED: length {L:.1f} < {min_length:.1f}")
        #     did_fail = True
        #     # import pdb; pdb.set_trace()
        #     continue

        # max_wh = float(filters.get('max_wh_ratio', 2.0))
        # h_bbox = float(ys.max() - ys.min() + 1)
        # w_bbox = float(xs.max() - xs.min() + 1)
        # wh_ratio = w_bbox / (h_bbox + 1e-6)
        # if wh_ratio > max_wh:
        #     logging.info(f"  DROPPED: width/height ratio {wh_ratio:.2f} > {max_wh:.2f}")
        #     did_fail = True
        #     continue

        # 5) Protrusion filter
        if protrusion_kwargs['enabled']:
            count, vis = detect_and_visualize_protrusions(
                path_mask=path_mask,
                image_bgr=front_bgr,
                **protrusion_kwargs
            )
            if count > 0:
                logging.info(f"  DROPPED: {count} thin protrusions detected")
                did_fail = True
                continue

        # 6) Lost‐at‐end filter (unchanged)
        # pct_tracks_alive = visibility[-1].sum() / visibility.shape[1]
        # if pct_tracks_alive < 0.75:
        #     logging.info(
        #         f"  DROPPED: {pct_tracks_alive:.2%} tracks alive in last frame < 75%"
        #     )
        #     did_fail = True
        #     continue

        # if ride_name == "10 41994 f6a757 20240508131902" and start_frame == 4621:
        #     import pdb; pdb.set_trace()

        # if start_frame == 24714:
        #     import pdb; pdb.set_trace() 

        did_fail = False
        logging.info("  PASSED all filters, proceeding with mask build.")
        sequence_status["success"].append((ride_name, start_frame, start_frame + num_frames))
        
        if debug_out_dir is not None:
            # save debug images
            debug_out_dir.mkdir(parents=True, exist_ok=True)
            ride_id, did0, did1, ts = ride_name.split(' ')
            debug_out_path = debug_out_dir / f"ride_{ride_id}_{did0}_{did1}_{ts}_{start_frame}_{end_frame}.jpg"
            cv2.imwrite(debug_out_path, front_vis_nocrumbs)
        if DEBUG_TRACKS:
            logging.info("  DEBUG: saving debug images")
            cv2.imwrite("front_vis_nocrumbs.png", front_vis_nocrumbs)
            cv2.imwrite("front_vis.png", front_vis)
            # import time
            # time.sleep(1.5)
            import pdb; pdb.set_trace()
    
    # Handle edge case last failure
    if did_fail:
        sequence_status["failure"].append((ride_name, start_frame, start_frame + num_frames))

    return sequence_status