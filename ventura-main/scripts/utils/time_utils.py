from datetime import datetime, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from timezonefinder import TimezoneFinder  # pip install timezonefinder
import pytz

def interpolate_mask_to_timestamps(src_timestamps: np.ndarray,
                                      src_mask: np.ndarray,
                                      target_timestamps: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor “interpolation” of a boolean mask from GPS times to target times.

    Args:
        src_timestamps    (shape=(N_gps,))         : sorted array of timestamps
        src_mask          (shape=(N_gps,))         : boolean array at each timestamp
        target_timestamps (shape=(N_timestamps,))  : sorted array of target timestamps

    Returns:
        np.ndarray of shape (N_timestamps,) of booleans,
        where each entry is gps_mask at the GPS timestamp nearest to that target timestamp.
    """
    src_ts = np.asarray(src_timestamps).flatten()
    src_b  = np.asarray(src_mask).flatten()
    tgt_ts = np.asarray(target_timestamps).flatten()

    # Find insertion indices: idx[i] is first j such that gps_ts[j] >= tgt_ts[i]
    idx = np.searchsorted(src_ts, tgt_ts, side="left")

    # Compute candidate indices to pick from gps_b
    ng = len(src_ts)
    left = np.clip(idx - 1, 0, ng - 1)
    right = np.clip(idx,     0, ng - 1)

    # Distances to left and right neighbors
    dist_left  = tgt_ts - src_ts[left]    # could be negative if idx=0, left=0
    dist_right = src_ts[right] - tgt_ts   # could be negative if idx>=ng, right=ng-1

    # By default, pick left; override when right is strictly closer
    use_right = dist_right < dist_left
    chosen_idx = np.where(use_right, right, left)

    return src_b[chosen_idx]

# def find_true_windows_swview(mask: np.ndarray, window_len: int):
#     """
#     Same as above, but uses `sliding_window_view` and `.all(axis=1)`.

#     Returns a list of (start, end) index pairs for every length‐`window_len` subarray
#     that is all True.
#     """
#     N = mask.shape[0]
#     if window_len < 1 or window_len > N:
#         return []

#     # 1) Create a view of all length‐window_len chunks
#     #    Resulting shape is (N-window_len+1, window_len)
#     windows = sliding_window_view(mask, window_len)

#     # 2) Check which windows are all True
#     all_true = windows.all(axis=1)  # boolean array of length (N-window_len+1)

#     # 3) Gather the starting indices
#     starts = np.nonzero(all_true)[0]

#     # 4) Build (start, end) pairs
#     return [(int(i), int(i + window_len - 1)) for i in starts]

def find_contiguous_true_intervals(mask: np.ndarray, window_len: int):
    """
    Given a 1D boolean array `mask` of length N and an integer `window_len`,
    return a list of (start, end) index pairs for each contiguous run of True
    whose length is >= window_len. The (start, end) indices are inclusive.

    Example:
        mask = np.array([False, True, True, True, False, True, True, True, True])
        window_len = 3
        # There are two runs: indices [1..3] length=3, and [5..8] length=4.
        # Both are >= 3, so return [(1, 3), (5, 8)].
    """
    mask = np.asarray(mask, dtype=bool).flatten()
    N = mask.size
    if window_len < 1 or window_len > N:
        return []

    # Pad mask with False at both ends to detect edges
    padded = np.concatenate(([False], mask, [False]))
    # Compute differences: +1 at run start, -1 at run end
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)     # indices in `mask` where a True-run starts
    ends   = np.flatnonzero(diff == -1)    # indices in `mask` where a True-run ends

    intervals = []
    for s, e in zip(starts, ends):
        run_len = e - s
        if run_len >= window_len:
            intervals.append((int(s), int(e - 1)))  # end is exclusive in diff, so use e-1

    return intervals

def apply_time_filter(data_dict, filter_dict):
    """
    Create a boolean mask of length N_frames indicating which frames satisfy:
      1) Their timestamp is within `max_desync` seconds of the nearest GPS fix.
      2) Their timestamp’s time‐of‐day (in the GPS location’s local timezone)
         falls between `time_range[0]` and `time_range[1]`.
      3) The total number of True frames is at least `min_length`; otherwise all False.

    Args:
        data_dict: {
            "gps": np.ndarray of shape (N_gps, 3) → [latitude, longitude, gps_ts],
            "timestamps": np.ndarray of shape (N_frames,) → frame timestamps (UNIX seconds),
            "ride": str (optional, unused here)
        }
        filter_dict: {
            "params": {
                "max_desync": float,            # seconds
                "time_range": [str, str],       # ["HH:MM:SS", "HH:MM:SS"]
                "min_length": int               # minimum number of valid frames
            }
        }

    Returns:
        np.ndarray of shape (N_frames,), dtype=bool.
    """
    params = filter_dict.get("params", {})
    max_desync = float(params.get("max_desync", 2.0))
    time_range = params.get("time_range", ["00:00:00", "23:59:59"])
    min_length = int(params.get("min_length", 0))

    # 1) Extract GPS array and frame timestamps
    gps_arr = np.asarray(data_dict["gps"], dtype=float)
    if gps_arr.ndim != 2 or gps_arr.shape[1] < 3:
        raise ValueError("data_dict['gps'] must be shape (N_gps, 3) with [lat, lon, ts].")
    # Use GPS timestamps for desync
    gps_ts = gps_arr[:, 2]
    frame_ts = np.asarray(data_dict["timestamps"], dtype=float)

    N_frames = len(frame_ts)
    if N_frames == 0:
        return np.zeros(0, dtype=bool)

    # 2) Sort GPS timestamps (and keep lat/lon for timezone)
    sort_idx = np.argsort(gps_ts)
    gps_ts = gps_ts[sort_idx]
    gps_lats = gps_arr[sort_idx, 0]
    gps_lons = gps_arr[sort_idx, 1]

    # 3) Determine timezone from first GPS location
    tzf = TimezoneFinder()
    lat0, lon0 = float(gps_lats[0]), float(gps_lons[0])
    tz_name = tzf.timezone_at(lat=lat0, lng=lon0)
    if tz_name is None:
        # Fallback to UTC
        tz = pytz.UTC
    else:
        tz = pytz.timezone(tz_name)

    # 4) Build mask for desync (nearest‐GPS) condition
    idx = np.searchsorted(gps_ts, frame_ts, side="left")
    ng = len(gps_ts)
    left = np.clip(idx - 1, 0, ng - 1)
    right = np.clip(idx,     0, ng - 1)

    dist_left = np.abs(frame_ts - gps_ts[left])
    dist_right = np.abs(gps_ts[right] - frame_ts)
    choose_right = dist_right < dist_left
    nearest_idx = np.where(choose_right, right, left)
    desync = np.abs(frame_ts - gps_ts[nearest_idx])
    mask = desync <= max_desync

    # 5) Build mask for time‐of‐day using local timezone
    t_start = time.fromisoformat(time_range[0])
    t_end   = time.fromisoformat(time_range[1])

    # Convert frame_ts (UNIX sec) → localized datetime → .time()
    # Use tz.localize or datetime.fromtimestamp with tz
    # datetime.fromtimestamp with tz parameter yields localized time directly
    times_of_day = np.array([
        datetime.fromtimestamp(ts, tz).time()
        for ts in frame_ts
    ], dtype=object)

    within_time = np.logical_and(times_of_day >= t_start, times_of_day <= t_end)
    mask = np.logical_and(mask, within_time)

    # 6) Enforce minimum length
    if mask.sum() < min_length:
        return np.zeros(N_frames, dtype=bool)

    return mask