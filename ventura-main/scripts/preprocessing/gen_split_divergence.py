#!/usr/bin/env python3
"""
Sample Splitter
===============
Utility to build balanced train/val/test splits for a trajectory‑based
robotics dataset.

Overview
--------
1. Loads *candidate* samples (one row = a clip) from a user‑supplied
   function.
2. Computes per‑sample statistics needed to decide whether a clip is
   "interesting" (currently: heading disagreement & topological degree).
3. Writes a *stats.csv* containing **all** clips plus these metrics so it
   can be inspected or re‑used later.
4. Produces three more CSV files – *train.csv*, *val.csv*, *test.csv* –
   each listing the subset of rows assigned to that split.

"Interesting" definition (default)
----------------------------------
*  **Heading filter:** absolute cosine distance between goal‑arrow
   heading and ground‑truth path heading is > *sigma* · σ
   (σ = std.‑dev. of the dataset).
*  **Degree filter:** topological degree at sample location > 2.

Both criteria must hold for a clip to be flagged *interesting*.

The code is deliberately modular: add custom filters by appending to
`ACTIVE_FILTERS`.

Note: This script only *loads* the per‑ride GraphML once per unique path
(it is cached in memory) then re‑uses it for every row that belongs to
that ride.

Usage
-----
```
python sample_splitter.py \
    --root_dir /data/frodobots8k \
    --graph_lut /data/maps/ride_to_graph.csv \
    --output_dir /data/splits \
    --sigma 1.0 \
    --val_ratio 0.1 --test_ratio 0.1
```

Dependencies
------------
```
pandas, numpy, scikit‑learn, networkx, osmnx
```

You will also need to provide an implementation of
`get_valid_samples(root_dir)` suited to your repository.  A stub is
included below.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, List
import h5py
import hickle as hkl
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_cdt
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import networkx as nx  # noqa: F401
    import osmnx as ox      # noqa: F401
    from shapely.geometry import Point
except ImportError as e:
    sys.exit(
        "[FATAL] networkx and osmnx are required for degree computation.\n"
        "         Install with: pip install networkx osmnx"
    )

from scripts.utils.loader_utils import (
    combine_ride_splits,
    combine_ride_graphs,
    load_gps,
    load_inertial
)
from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)
from scripts.utils.satellite_utils import (
    get_haversine_m,
    gps_to_local_xy
)
from scripts.utils.polyline_utils_smoothed import (
    get_center_spline
)
from scripts.mapping.compute_curvatures import (
    compute_curvature_single,
    plot_curvature_histogram
)
from scripts.mapping.project_odom import (
    project_odom_single
)
from scripts.mapping.odometry_filters import (
    filter_by_odometry
)

DEBUG_MODE = False
MAX_DEBUG_SAMPLES = 100
COMPUTE_NODE_DEGREE = True
HEADING_SIGMA_THRESHOLD = 1.0 # multiplier for std.‑dev.
INTERSECTION_THRESHOLD = 6.4  # metres, used for goal selection
STRAIGHT_PATH_DEGREE = 2  # degree threshold for "straight" path
FOV_DEG = 120.0                     # camera field-of-view (total)
HALF_FOV = FOV_DEG / 2.0            # ±60 °
# Don't worry about entity info, we'll replace with spatial lang later
_REQ_FILES = ("path_tracker.h5", "ride_info.h5", "odometry_info.h5" ) #, "entity_caption.h5")

LOG_STATISTICS = {                     # ←-- edit here only
    "deg_mean":  dict(op="mean", column="degree"),
    "deg_std":   dict(op="std",  column="degree"),
    "div_mean":  dict(op="mean", column="cos_dist"),
    "depth_mean":dict(op="mean", column="depth_px"),
    "depth_std": dict(op="std",  column="depth_px"),
    "high_deg":  dict(op="sum",  column="high_degree"),
    "hdg_out":   dict(op="sum",  column="heading_outlier"),
}
_OP_FUN = {                            # tiny helper map
    "mean": pd.Series.mean,
    "std":  pd.Series.std,
    "sum":  pd.Series.sum,
}

# ---------------------------------------------------------------------------
# 2.  Helper maths
# ---------------------------------------------------------------------------

_FOUR = ndi.generate_binary_structure(2, 1)

def path_mask_depth(mask: np.ndarray) -> int:
    """
    Parameters
    ----------
    mask : (H, W) uint8 / bool
        Binary path mask (True/1 = traversable).

    Returns
    -------
    depth_px : int
        Maximum 4-connected distance, in pixels, that you can travel
        *inside the mask* starting from any mask-pixel on the **bottom
        image row**.
    """
    m = mask.astype(bool)
    seeds = np.zeros_like(m, bool)
    seeds[-1] = m[-1]                 # bottom-row seeds

    if not seeds.any():               # no contact with bottom
        return 0

    # ------------------------------------------------------------------
    # 1.  Grow the connected component that touches the bottom row
    #     (avoids wasting time inside other, disjoint blobs).
    # ------------------------------------------------------------------
    comp, _ = ndi.label(m, structure=_FOUR)
    comp_id = comp[-1][m[-1]][0]      # id of the touching component
    comp_mask = comp == comp_id       # restrict search to that blob

    # ------------------------------------------------------------------
    # 2.  Geodesic distance via iterative binary-dilation BFS
    #     Each iteration is a fully-vectorised C-routine; loop runs
    #     at most depth_px times (≤ image height).
    # ------------------------------------------------------------------
    frontier = seeds.copy()
    visited  = seeds.copy()
    depth    = 0

    while True:
        frontier = ndi.binary_dilation(frontier, _FOUR)
        frontier &= comp_mask & ~visited   # stay inside component
        if not frontier.any():
            return depth
        visited |= frontier
        depth   += 1


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest signed difference *a - b* in degrees, in (-180, 180]."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff

def cosine_distance(heading_goal: float, heading_path: float) -> float:
    """Return 1 -cos(Δθ) where Δθ is the angular difference in degrees."""
    delta_rad = math.radians(angular_difference_deg(heading_goal, heading_path))
    return 1.0 - math.cos(delta_rad)

def _wrap180(angle_deg: float) -> float:
    """Wrap any angle to (-180, 180] degrees."""
    return (angle_deg + 180.0) % 360.0 - 180.0

def sort_depth_deg_div(df: pd.DataFrame) -> pd.DataFrame:
    """depth_px ↓, degree ↓, cos_dist ↓, then ride_name ↑ for determinism."""
    return df.sort_values(
        by=["depth_px", "degree", "cos_dist", "ride_name"],
        ascending=[False,     False,    False,      True]
    )

# ---------------------------------------------------------------------------
# 1.  User‑defined hooks (replace with real loaders for your dataset)
# ---------------------------------------------------------------------------

def _row_has_all_h5(root: Path, ride_name: str, start_frame: int) -> bool:
    """
    Return True iff *all* required .h5 files exist for this row.
    """
    base = set_frodo_dir(root, *ride_name.split(" ")) / f"seq_{start_frame}"
    for fname in _REQ_FILES:
        fp = base / fname
        if not h5py.is_hdf5(fp):          # also False if fp missing
            return False
    return True

def filter_missing_h5(df: pd.DataFrame,
                      root: Path,
                      n_jobs: int = 32) -> pd.DataFrame:
    """
    Drop rows whose required .h5 files are missing *or* corrupted.

    The I/O is purely metadata reads and therefore CPU-light; we parallelise
    with joblib-threads to saturate the filesystem without Python-GIL issues.
    """
    root = Path(root).expanduser().resolve()

    keep_mask = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_row_has_all_h5)(root, r.ride_name, r.start_frame)
        for r in df.itertuples()
    )
    keep_mask = np.fromiter(keep_mask, dtype=bool, count=len(df))

    n_drop = len(df) - int(keep_mask.sum())
    if n_drop:
        print(f"[INFO] Filtering out {n_drop:,} rows (missing/corrupt .h5)")
    return df[keep_mask].reset_index(drop=True)

def choose_goal_index(
    gps_np: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_dist_m: float,
    max_dist_m: float,
) -> int:
    """Return the first *future* index within [min,max] metres, else *end_idx*."""
    lat0, lon0 = gps_np[start_idx, :2]
    for k in range(start_idx + 1, min(end_idx + 1, len(gps_np))):
        if min_dist_m <= get_haversine_m(lat0, lon0, *gps_np[k, :2]) <= max_dist_m:
            return k
    return end_idx

def compute_goal_heading(
    root_dir: str,
    df: pd.DataFrame,
    ride_to_graph: Dict[str, str] | None = None,
    *,
    min_dist: float = 20.0,
    max_dist: float = 100.0,
    n_jobs: int = 48,  # use all cores by default
) -> pd.DataFrame:
    """Append goal_heading_deg, path_heading_deg, cos_dist to *df* (parallel).

    The core per‑row logic is preserved verbatim from the original single‑thread
    implementation; we simply run it concurrently with joblib.
    """

    root = Path(root_dir).expanduser().resolve()

    def _worker(ride: str, s_f: int, e_f: int) -> Tuple[float, float]:
        parts = ride.split(" ")
        rdir = set_frodo_dir(root, *parts)

        track_path, satellite_path, route_path, entity_path = None, None, None, None
        sat_data, path_data, route_data, entity_data = None, None, None, None
        try:
            for fname in _REQ_FILES:
                current_path = rdir / f"seq_{s_f}" / fname
                if fname == "path_tracker.h5":
                    track_path = current_path
                    path_data = hkl.load(track_path)
                elif fname == "satellite_info.h5":
                    satellite_path = current_path
                    sat_data = hkl.load(satellite_path)
                elif fname == "routing_info.h5":
                    route_path = current_path
                    route_data = hkl.load(route_path)
                elif fname == "entity_info.h5":
                    entity_path = current_path
                    entity_data = hkl.load(entity_path)
                else:
                    raise ValueError(f"Unknown file name: {fname}")
        except Exception as e:
            # logging.error(
            #     f"[ERROR] Failed to load data for {ride} at {s_f}-{e_f}: {e}")
            return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

        gps_np = sat_data["future_gps"]
        heading_np = sat_data["future_heading"]
        num_gps = gps_np.shape[0]
        goal_idx = choose_goal_index(gps_np, 0, num_gps, min_dist, max_dist)
        goal_idx = min(goal_idx, num_gps - 1)

        origin = np.array([*gps_np[0, :2], heading_np[0]])
        goal_xy = gps_to_local_xy(gps_np[goal_idx:goal_idx + 1, :2], origin)
        theta = math.degrees(math.atan2(goal_xy[0][1], goal_xy[0][0]))
        goal_h = float(np.clip(theta, -90.0, 90.0))

        tracks = path_data["tracks"][0][-1]
        crumb_times = path_data["crumbs"][0][:, 0]
        sides = path_data["sides"][0]
        spline_out = get_center_spline(tracks, crumb_times, sides, 150.0)
        cx, cy = spline_out["cx"], spline_out["cy"]
        mask = path_data["path_mask"].astype(np.uint8)    # (H,W) 0/1
        depth_px = path_mask_depth(mask)

        H, W = path_data["front_rgb"].shape[:2]
        bx, by = W // 2, H - 1
        bcy = -cx + by
        bcx = -cy + bx
        path_h = float(np.clip(math.degrees(math.atan2(bcy[0], bcx[0])), -90.0, 90.0))

        return goal_h, path_h, gps_np[0][0], gps_np[0][1], heading_np[0], depth_px

    # materialise rows list for stable iteration ordering & progress bar
    rows = list(zip(df["ride_name"], df["start_frame"], df["end_frame"]))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker)(ride, s_f, e_f)
        for ride, s_f, e_f in tqdm(rows, desc="Goal headings", leave=False)
    )

    goal_deg, path_deg, lat, lon, hdg, depth_px = map(list, zip(*results))  # unzip list of tuples
    out = df.copy()
    # Set the corresponding columns
    out["goal_heading_deg"] = goal_deg
    out["path_heading_deg"] = path_deg
    out["latitude"] = lat
    out["longitude"] = lon
    out["heading_deg"] = hdg
    out["cos_dist"] = out.apply(
        lambda r: cosine_distance(r["goal_heading_deg"], r["path_heading_deg"]), axis=1
    )
    out["depth_px"] = depth_px
    return out
# ---------------------------------------------------------------------------
# 3.  Degree from GraphML
# ---------------------------------------------------------------------------

class GraphCache:
    """Lightweight cache to avoid re‑loading the same GraphML many times."""

    def __init__(self):
        self._store: Dict[Path, 'nx.MultiDiGraph'] = {}

    def get(self, path: Path):
        if path not in self._store:
            self._store[path] = ox.load_graphml(path)
        return self._store[path]

graph_cache = GraphCache()

def nearest_degree(
    graph_path: Path,
    lat: float,
    lon: float,
    hdg: float,                       # ENU: 0 ° = East, +CW
    max_search_dist_m: float = 12.8,
) -> int:
    """
    Return the degree of the closest node that is
    (a) within *max_search_dist_m* **and**
    (b) lies inside a ±60 ° field-of-view about *hdg* (ENU CW).
    """
    G = graph_cache.get(graph_path)

    try:
        G_proj = ox.project_graph(G)

        # current position → projected CRS
        pt_proj, _ = ox.projection.project_geometry(
            Point(lon, lat), crs="EPSG:4326", to_crs=G_proj.graph["crs"]
        )
        x0, y0 = pt_proj.x, pt_proj.y

        # nearest node
        node, dist = ox.distance.nearest_nodes(G_proj, x0, y0, return_dist=True)
        if dist > max_search_dist_m:
            return 2

        x1, y1 = G_proj.nodes[node]["x"], G_proj.nodes[node]["y"]

        # ------------------------------------------------------------------
        # CCW → CW conversion
        # ------------------------------------------------------------------
        # theta_math: 0 ° = East, +CCW (standard atan2)
        theta_math = np.rad2deg(np.arctan2(y1 - y0, x1 - x0))

        # # Convert to ENU heading: 0 ° = East, +CW
        theta_cw   = _wrap180(-theta_math)

        # angular difference to current heading
        diff = _wrap180(theta_math - hdg)
        if abs(diff) > HALF_FOV:
            return 2                    # node outside FOV

        return int(G_proj.degree[node])

    except Exception:
        return 2

# ---------------------------------------------------------------------------
# 4.  Filters
# ---------------------------------------------------------------------------

def heading_outlier_filter(df: pd.DataFrame, sigma: float) -> pd.Series:
    """Boolean mask – heading cosine distance > sigma·σ."""
    std = df["cos_dist"].std(ddof=0)
    return np.abs(df["cos_dist"]) > sigma * std

def degree_filter(df: pd.DataFrame) -> pd.Series:
    """Boolean mask – degree > 2."""
    return df["degree"] > STRAIGHT_PATH_DEGREE

# List of (name, callable) used to mark *interesting* samples
ACTIVE_FILTERS: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("heading_outlier", lambda d: heading_outlier_filter(d, sigma=HEADING_SIGMA_THRESHOLD)),
    ("high_degree", degree_filter),
]

# ---------------------------------------------------------------------------
# 5.  Split logic
# ---------------------------------------------------------------------------

def _split_group(df: pd.DataFrame, val_ratio: float, test_ratio: float,
                 rng: int | None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, val, test DataFrames for *df* (no shuffling outside)."""
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()
    # test first
    train_val, test = train_test_split(
        df, test_size=test_ratio, shuffle=True, random_state=rng, stratify=None
    )
    # val from the remainder
    val_fraction_of_train_val = val_ratio / max(1e-9, (1.0 - test_ratio))
    train, val = train_test_split(
        train_val, test_size=val_fraction_of_train_val, shuffle=True,
        random_state=rng, stratify=None
    )
    return train, val, test

def balanced_split(
    df: pd.DataFrame,
    val_ratio: float,           # e.g. 0.15  → ≤15 % of all rows in val
    test_ratio: float,          # e.g. 0.20  → ≤20 % of all rows in test
    rng: int | None = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/val/test with two extra rules:

    • |val|  ≤ val_ratio · |df|
      |test| ≤ test_ratio · |df|

    • Inside those splits, a **fixed fraction** of the rows are forced to be
      “interesting” (flag column `interesting == True`):

          VAL_INT_PCT   = 0.40    (40 % of the val set)
          TEST_INT_PCT  = 0.50    (50 % of the test set)

      The remainder are “boring”.  If there aren’t enough interesting or boring
      rows to satisfy the target, the code backs-off gracefully.

    The train split receives everything left over.
    """

    # ───────────────────────────── parameters you can tweak ──────────────
    VAL_INT_PCT  = 0.20          # hard-coded percentage of interesting rows
    TEST_INT_PCT = 0.30          # in val / test respectively
    # ──────────────────────────────────────────────────────────────────────

    assert 0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0
    assert val_ratio + test_ratio < 1.0, "val_ratio + test_ratio must be < 1"

    rng = np.random.default_rng(rng)
    df  = df.sample(frac=1.0, random_state=rng.bit_generator)      # full shuffle

    # split interesting vs boring
    int_df = df[df["interesting"]].sample(frac=1.0, random_state=rng.bit_generator)
    bor_df = df[~df["interesting"]].sample(frac=1.0, random_state=rng.bit_generator)

    n_total   = len(df)
    n_val     = int(round(n_total * val_ratio))
    n_test    = int(round(n_total * test_ratio))

    # ---- pick interesting rows for val / test ---------------------------------
    n_val_int  = min(len(int_df), int(round(n_val  * VAL_INT_PCT)))
    n_test_int = min(len(int_df) - n_val_int, int(round(n_test * TEST_INT_PCT)))

    val_int   = int_df.iloc[:n_val_int]
    test_int  = int_df.iloc[n_val_int : n_val_int + n_test_int]
    int_rest  = int_df.iloc[n_val_int + n_test_int :]           # for train later

    # ---- pick boring rows for val / test --------------------------------------
    n_val_bor  = n_val  - len(val_int)
    n_test_bor = n_test - len(test_int)

    bor_val   = bor_df.iloc[:n_val_bor]
    bor_test  = bor_df.iloc[n_val_bor : n_val_bor + n_test_bor]
    bor_rest  = bor_df.iloc[n_val_bor + n_test_bor :]           # for train later

    # ---- assemble splits ------------------------------------------------------
    val_df   = pd.concat([val_int,  bor_val ]).sample(frac=1.0, random_state=rng.bit_generator)
    test_df  = pd.concat([test_int, bor_test]).sample(frac=1.0, random_state=rng.bit_generator)
    train_df = pd.concat([int_rest, bor_rest]).sample(frac=1.0, random_state=rng.bit_generator)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def _weighted_sample(df: pd.DataFrame, n: int, rng):
    """
    Sample *n* rows without replacement.

    Priority order
    --------------
    1. rows whose depth_px ≥ μ + 2·σ
    2. (fallback) rows whose depth_px ≥ μ + 1·σ
    3. (fallback) all remaining rows

    Within the chosen subset the probability is
        depth_px · |degree| · |cos_dist|
    (clipped to ≥1 to avoid zeros).  Uniform if all weights are zero.
    """
    mu, std = df["depth_px"].mean(), df["depth_px"].std()

    thresholds = [2, 1.75, 1.5, 1.25, 1.0, 0.5, 0.25, 0.0]

    pool = None
    for t in thresholds:
        tier = df[df["depth_px"] >= mu + t * std]
        print("[INFO] Found", len(tier), "rows with depth_px ≥", mu + t * std)
        if len(tier) >= n:
            pool = tier
            break
    assert pool is not None, "No rows meet the depth threshold criteria"

    w = (pool["depth_px"].clip(lower=1) *
         pool["degree"].abs().clip(lower=1) *
         pool["cos_dist"].abs().clip(lower=1)).astype(float).fillna(0.0)

    if (w > 0).any():
        return pool.sample(n=n, weights=w, random_state=rng, replace=False)

    # fallback – uniform
    return pool.sample(n=n, random_state=rng, replace=False)

def sample_splits_default(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    rng: int | None,
    max_samples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Original 'interesting' logic moved here. Keeps the weighting rules and calls balanced_split().
    """
    rng_np = np.random.default_rng(rng)
    df = df.sample(frac=1.0, random_state=rng_np.bit_generator)

    # Ensure presence of 'interesting' column
    if "interesting" not in df.columns:
        df["interesting"] = True

    interesting_df = df[df["interesting"]]
    boring_df      = df[~df["interesting"]]

    if max_samples is not None and len(df) > max_samples:
        remaining = max_samples - len(interesting_df)
        if remaining > 0 and len(boring_df) > 0:
            sampled_boring = _weighted_sample(
                boring_df, n=min(remaining, len(boring_df)), rng=rng
            )
            df = pd.concat([interesting_df, sampled_boring], ignore_index=True)
        else:
            df = _weighted_sample(interesting_df, n=max_samples, rng=rng)
        df = df.reset_index(drop=True)

    assert not df.duplicated(subset=["ride_name", "start_frame", "end_frame"]).any()
    return balanced_split(df, val_ratio, test_ratio, rng)

def sample_splits_curve(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    rng: int | None,
    max_samples: int | None,
    n_bins: int = 11,
    binning: str = "quantile",       # kept for API compatibility; edges come from histogram
    oversample_train: bool = True,
    plot_dir: Path | None = None,
    hist_kwargs: dict | None = None,
    max_copies_per_sample: int = 3,  # cap total copies (incl. original) in TRAIN
    tau: float = 0.5,                # kept for API compatibility (unused in this simplified version)
    rho_max: float = 0.30,           # max fraction of TRAIN that can be duplicates
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simplified pipeline:

    1) Compute global curvature bins (from full df) and plot the ORIGINAL histogram.
    2) If max_samples is set and < len(df), subsample *N = max_samples* items
       WITHOUT replacement with per-sample weights proportional to inverse
       BIN frequency (rarer bins get sampled more).
       Otherwise, keep the full df.
    3) Split the (subsampled) set into disjoint train / val / test by unique keys.
    4) (Optional) Oversample TRAIN *with replacement* to make bin counts more even,
       while respecting per‑sample duplication cap and a global duplication budget.
    5) Plot final TRAIN histogram using the ORIGINAL bin edges (if plot_dir set).
    """
    # -------------------- setup & checks --------------------
    rng_np = np.random.default_rng(rng)
    df = df.copy()

    if "curvature" not in df.columns:
        raise ValueError("curve sampling requires a 'curvature' column")
    df = df[np.isfinite(df["curvature"])].reset_index(drop=True)
    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 1) ORIGINAL histogram --------------------
    if hist_kwargs is None:
        hist_kwargs = dict(n_bins=n_bins, range_method="quantile",
                           q_low=0.05, q_high=0.95, symmetric=True)

    # This should exist in your codebase; returns (edges, counts)
    pre_edges, pre_counts = plot_curvature_histogram(
        df, save_path=(plot_dir / "curvature_hist_full_pre.png") if plot_dir else None,
        **hist_kwargs
    )

    # Bin the entire dataframe once, using the ORIGINAL edges
    curv_full = df["curvature"].to_numpy()
    df["curve_bin"] = np.digitize(curv_full, pre_edges[1:-1], right=False).astype(int)

    # -------------------- 2) Inverse-frequency SUBSAMPLING to N --------------------
    if max_samples is not None and max_samples < len(df):
        # counts per bin in the FULL df
        bin_counts = df["curve_bin"].value_counts().to_dict()
        # per-row weights = 1 / count(bin)
        w = np.array([1.0 / max(1, bin_counts[b]) for b in df["curve_bin"]], dtype=float)
        w /= w.sum()
        N = int(max_samples)

        idx = rng_np.choice(len(df), size=N, replace=False, p=w)
        df_sub = df.iloc[idx].reset_index(drop=True)
    else:
        df_sub = df.reset_index(drop=True)

    # -------------------- helper: unique key for disjoint splits --------------------
    def _key_series(d: pd.DataFrame) -> pd.Series:
        if all(c in d.columns for c in ("ride_name", "start_frame", "end_frame")):
            return (d["ride_name"].astype(str) + "|" +
                    d["start_frame"].astype(str) + "|" +
                    d["end_frame"].astype(str))
        # last resort: unique per row (won't de-duplicate)
        return pd.Series(np.arange(len(d)), index=d.index, dtype=object)

    # -------------------- 3) Random KEY split: train / val / test --------------------
    keys = _key_series(df_sub)
    uniq_keys = keys.drop_duplicates().to_numpy()
    rng_np.shuffle(uniq_keys)

    n_total = len(uniq_keys)
    n_test = int(round(test_ratio * n_total))
    n_val  = int(round(val_ratio  * n_total))
    test_keys = set(uniq_keys[:n_test])
    val_keys  = set(uniq_keys[n_test:n_test + n_val])
    train_keys= set(uniq_keys[n_test + n_val:])

    is_test  = keys.isin(test_keys)
    is_val   = keys.isin(val_keys)
    is_train = keys.isin(train_keys)

    test_df  = df_sub[is_test].drop_duplicates(subset=["curve_bin"] if "ride_name" not in df_sub.columns else ["ride_name","start_frame","end_frame"]).reset_index(drop=True)
    val_df   = df_sub[is_val ].drop_duplicates(subset=["curve_bin"] if "ride_name" not in df_sub.columns else ["ride_name","start_frame","end_frame"]).reset_index(drop=True)
    train_df = df_sub[is_train].reset_index(drop=True)  # keep all rows for train (duplicates OK)

    # -------------------- 4) Optional TRAIN oversampling to even bins --------------------
    if oversample_train and len(train_df) > 0 and max_copies_per_sample > 1:
        counts = train_df["curve_bin"].value_counts().sort_index()
        if len(counts) > 0:
            M = int(counts.sum())
            target = int(counts.max())  # aim to bring all bins up to the current max
            need_per_bin = (target - counts).clip(lower=0).astype(int)

            # global duplication budget
            max_added = int(np.floor(rho_max * M))
            desired_added = int(need_per_bin.sum())
            if desired_added > max_added and desired_added > 0:
                scale = max_added / desired_added
                need_per_bin = np.floor(need_per_bin * scale).astype(int)

            # per-key duplication cap
            key_ser = _key_series(train_df)
            cur_counts = key_ser.value_counts()  # how many copies per key currently
            remain_allowed = (max_copies_per_sample - cur_counts).clip(lower=0).to_dict()

            by_bin = dict(tuple(train_df.groupby("curve_bin", sort=True)))
            upsampled = []
            for b, g in by_bin.items():
                need = int(need_per_bin.get(b, 0))
                if need <= 0 or len(g) == 0:
                    upsampled.append(g)
                    continue

                g_keys = _key_series(g).to_numpy()
                g_idx  = g.index.to_numpy()

                # how many more times can each row's key appear?
                per_key_remain = np.array([remain_allowed.get(k, 0) for k in g_keys], dtype=int)
                slot_pool = np.repeat(g_idx, np.clip(per_key_remain, 0, None))

                if slot_pool.size == 0:
                    upsampled.append(g)
                    continue

                take = min(need, slot_pool.size)
                chosen = rng_np.choice(slot_pool, size=take, replace=False)
                extra = train_df.loc[chosen].copy()

                # update global remaining allowance per key
                for k in _key_series(extra):
                    remain_allowed[k] = max(remain_allowed.get(k, 0) - 1, 0)
                    cur_counts[k] = cur_counts.get(k, 1) + 1

                upsampled.append(pd.concat([g, extra], ignore_index=True))

            train_df = pd.concat(upsampled, ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    # -------------------- 5) Final histogram on TRAIN (original edges) --------------------
    if plot_dir is not None and len(train_df) > 0:
        # use same edges as original for apples-to-apples comparison
        train_curv = train_df["curvature"].to_numpy()
        _ = plot_curvature_histogram(
            train_df,
            save_path=plot_dir / "curvature_hist_train_post.png",
            **hist_kwargs
        )
        post_counts, _ = np.histogram(train_curv, bins=pre_edges)
        np.savetxt(plot_dir / "curvature_hist_full_pre_counts.txt", pre_counts, fmt="%d")
        np.savetxt(plot_dir / "curvature_hist_train_post_counts.txt", post_counts, fmt="%d")
        np.savetxt(plot_dir / "curvature_hist_edges_used.txt", pre_edges)

    # -------------------- return (sorted for stability) --------------------
    sort_cols = ["ride_name"] if "ride_name" in df_sub.columns else None
    if sort_cols:
        train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
        val_df   = val_df.sort_values(sort_cols).reset_index(drop=True)
        test_df  = test_df.sort_values(sort_cols).reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 6.  Main pipeline
# ---------------------------------------------------------------------------

def process_split(root_dir: Path, out_dir: Path, split_file: Path, 
            filter: bool = False, use_cache: bool = False,
            cache_dir: Path | None = None,
            sigma: float = 1.0, val_ratio: float = 0.1, test_ratio: float = 0.05,
            rng: int | None = 42, max_samples: int | None = None, frame_horizon: int = 60,
            alignment_pct: float = 0.5,):

    out_dir.mkdir(parents=True, exist_ok=True)

    if use_cache:
        print("[INFO] Loading candidate samples from cache …")
        stats_path = cache_dir / "combined_split_divergence.csv"
        assert stats_path.exists(), \
            f"Cache file {stats_path} does not exist. " \
            "Run without --use_cache to generate it first."
    else:
        print("[INFO] Generating candidate samples …")
        stats_path = out_dir / "combined_split_divergence.csv"

    if not use_cache or not stats_path.exists():
        df = combine_ride_splits(root_dir, split_file)
        if DEBUG_MODE:
            max_debug_samples = min(MAX_DEBUG_SAMPLES, len(df))
            df = df.sample(n=max_debug_samples, random_state=rng)
            df = df.reset_index(drop=True)
        
        if filter == "graph":
            ride_to_graph = combine_ride_graphs(root_dir, "ride_to_graph.csv")

            # Filter the rides that have valid infos processed
            print(f"[INFO] Loaded {len(df):,} valid samples from {root_dir}.")

            df = compute_goal_heading(root_dir, df, ride_to_graph)
            df = df.merge(
                ride_to_graph[["ride", "graph_path"]]
                            .rename(columns={"ride": "ride_name"}),   # align column names
                on="ride_name",
                how="left",
                validate="many_to_one",
            )
            # Drop rows with nans
            df = df.dropna(subset=["graph_path", "goal_heading_deg", "path_heading_deg",
                                "latitude", "longitude", "heading_deg", "depth_px"])

            # ------------------------------------------------------------------
            # Compute cosine distance
            # ------------------------------------------------------------------
            print("[INFO] Computing node degrees (may take a while) …")
            def _deg(row):
                if pd.isna(row["graph_path"]):
                    return STRAIGHT_PATH_DEGREE
                return nearest_degree(
                    Path(row["graph_path"]),
                    row["latitude"],
                    row["longitude"],
                    row["heading_deg"],
                    max_search_dist_m=INTERSECTION_THRESHOLD,
                )

            df["degree"] = df.apply(_deg, axis=1)

            # ------------------------------------------------------------------
            # Determine *interesting*
            # ------------------------------------------------------------------
            filter_results: Dict[str, pd.Series] = {}
            for fname, f in ACTIVE_FILTERS:
                if fname == "heading_outlier":
                    mask = heading_outlier_filter(df, sigma)
                else:
                    mask = f(df)
                filter_results[fname] = mask
                df[fname] = mask
            
            # interesting == logical AND across all active filters
            combined_mask = np.logical_and.reduce(list(filter_results.values()))
            df["interesting"] = combined_mask
            df = df.sort_values("ride_name").reset_index(drop=True)
        elif filter == "curve":
            # Compute the curvature of the rides
            print("[INFO] Computing curvature …")

            if DEBUG_MODE:
                # Don't parallelise in debug mode
                n_jobs = 1
            else:
                n_jobs = 96

            # Filter missing df rows
            print(f"[INFO] Filtering missing .h5 files in {root_dir} …")
            df = filter_missing_h5(df, root_dir, n_jobs=48)
            print(f"[INFO] Loaded {len(df):,} valid samples from {root_dir}.")

            # Randomly sample 10 rides for debugging
            if DEBUG_MODE:
                max_debug_samples = min(MAX_DEBUG_SAMPLES, len(df))
                df = df.sample(n=max_debug_samples, random_state=rng)
                df = df.reset_index(drop=True)

            # Job lib parallelisation call of compute_curvature_single
            inputs = list(zip(df["ride_name"], df["start_frame"], df["end_frame"]))
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(compute_curvature_single)(root_dir, ride_name, start_frame, end_frame)
                for ride_name, start_frame, end_frame in tqdm(inputs, desc="Computing curvature", leave=False)
            )
            df["curvature"] = np.array(results)[:, 0]  # Extract curvature values
            print(f"[INFO] Computing odometry filter for {len(df):,} samples …")

            # Plot histogram of curvatures
            # edges, counts = plot_curvature_histogram(
            #     df, n_bins=21, range_method="quantile",
            #     q_low=0.05, q_high=0.95, symmetric=False,
            #     save_path = out_dir / "curvature_histogram.png"
            # )
            # df["curve_bin"] = np.digitize(df["curvature"].to_numpy(), edges[1:-1])

            # Also postfilter sequences based on visual-action alignment
            if alignment_pct > 0.0:
                # reinit inputs because already used in the previous parallel call
                alignment_results = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(project_odom_single)(root_dir, ride_name, start_frame, end_frame)
                    for ride_name, start_frame, end_frame in tqdm(inputs, desc="Computing curvature", leave=False)
                )
                df["pct_aligned"] = np.array(alignment_results)
                print(f"[INFO] Computed visual-action alignment for {len(df):,} samples.")
            else:
                df["pct_aligned"] = 1.0

            # Postfilter seqeuences by odometry
            odom_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(filter_by_odometry)(root_dir, ride_name, start_frame, end_frame, frame_horizon=frame_horizon)
                for ride_name, start_frame, end_frame in tqdm(inputs, desc="Filtering by odometry", leave=False)
            )
            df["odometry_valid"] = np.array(odom_results)
            df_len = len(df)
            df = df[df["odometry_valid"]].reset_index(drop=True)
            print(f"[INFO] Filtered {df_len - len(df):,} samples by odometry.")
            df_len = len(df)
            df = df[df["pct_aligned"] >= alignment_pct]
            print(f"[INFO] Filtered {df_len - len(df):,} samples by visual-action alignment.")
            print(f"[INFO] {len(df):,} valid samples after filtering by odometry and alignment.")
        else:
            # Default to all samples being "interesting"
            df["interesting"] = True
        df["ride_name"] = df["ride_name"].astype(str)
        df["start_frame"] = df["start_frame"].astype(int)
        df["end_frame"] = df["end_frame"].astype(int)

        # ------------------------------------------------------------------
        # Save full statistics CSV before splitting
        # ------------------------------------------------------------------
        with open(stats_path, "w", encoding="utf-8") as f:
            # prepend metadata as comment lines
            f.write(f"# heading sigma threshold , {HEADING_SIGMA_THRESHOLD}\n")
            f.write(f"# intersection threshold  , {INTERSECTION_THRESHOLD}\n")
            df.to_csv(f, index=False)
        print(f"[INFO] Wrote sample statistics to {stats_path}")
    else:
        filtered_stats_path = out_dir / "filtered_samples.csv"
        if not filtered_stats_path.exists():
            df = pd.read_csv(stats_path, header=2)
            df = filter_missing_h5(df, root_dir, n_jobs=48)
            df.to_csv(filtered_stats_path, index=False)
            print(f"[INFO] Filtered samples saved to {filtered_stats_path}")
        df = pd.read_csv(filtered_stats_path)

        print(f"[INFO] Loaded {len(df):,} samples from {filtered_stats_path}")

    df = df.drop_duplicates(subset=["ride_name", "start_frame", "end_frame"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Balanced train/val/test split
    # ------------------------------------------------------------------
    if not cache_dir:
        print("[INFO] Creating balanced splits …")

        # Ensure an 'interesting' flag exists for default flow
        if "interesting" not in df.columns:
            df["interesting"] = True

        # Choose sampler
        if filter == "curve":
            train_df, val_df, test_df = sample_splits_curve(
                df, val_ratio, test_ratio, rng, max_samples,
                n_bins=11, binning="quantile", oversample_train=True,
                plot_dir=out_dir,
                hist_kwargs=dict(n_bins=11, range_method="quantile",
                                q_low=0.05, q_high=0.95, symmetric=False)
            )
        else:
            train_df, val_df, test_df = sample_splits_default(
                df, val_ratio, test_ratio, rng, max_samples
            )
    else:
        print("[INFO] Using precomputed splits from the dataset directory …")
        train_df, val_df, test_df = (
            pd.read_csv(cache_dir / f"{name}.txt")
            .merge(df[["ride_name", "start_frame", "end_frame"]],
                   on=["ride_name", "start_frame", "end_frame"],
                   how="inner", validate="many_to_one")
            for name in ("train", "val", "test")
        )

    # ------------------------------------------------------------------
    # Sort by ride_name for reproducibility
    # ------------------------------------------------------------------
    train_df = train_df.sort_values("ride_name").reset_index(drop=True)
    val_df   = val_df.sort_values("ride_name").reset_index(drop=True)
    test_df  = test_df.sort_values("ride_name").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save splits
    # ------------------------------------------------------------------
    for split_name, split_df in zip(
        ["train", "val", "test"], [train_df, val_df, test_df]
    ):
        txt_path = out_dir / f"{split_name}.txt"
        split_df.to_csv(txt_path, index=False)
        print(f"    - {split_name:5s}: {len(split_df):,} rows → {txt_path}")

    # Save full txt that contains all samples
    full_txt_path = out_dir / "full.txt"
    df.to_csv(full_txt_path, index=False)
    print(f"    - full: {len(df):,} rows → {full_txt_path}")

    # ------------------------------------------------------------------
    # Save *interesting* subset of the test split
    # ------------------------------------------------------------------
    test_div_path = out_dir / "test_divergence.txt"
    test_df[test_df["interesting"]].to_csv(test_div_path, index=False)
    print(f"    - test_divergence: {test_df['interesting'].sum():,} rows "
          f"→ {test_div_path}")

    print("[DONE] Splitting complete.")

    # ------------------------------------------------------------------
    # Print/save dataset statistics per split
    # ------------------------------------------------------------------
    stats_txt = out_dir / "dataset_stats.txt"
    with stats_txt.open("w", encoding="utf-8") as f:

        # ----------- 1. build header dynamically -----------------------
        hdr_cols = ["split", "count"]
        for name, cfg in LOG_STATISTICS.items():
            if cfg["column"] in df.columns:          # keep only valid ones
                hdr_cols.append(name)
        hdr_cols.append("interesting")

        header = " | ".join(col.ljust(15) for col in hdr_cols)
        print("\n" + header)
        f.write(header + "\n")

        # ----------- 2. per-split lines --------------------------------
        for split_name, split_df in zip(
            ["train", "val", "test"], [train_df, val_df, test_df]
        ):
            stats = {
                "split": split_name,
                "count": len(split_df),
                "interesting": int(split_df.get("interesting", []).sum()),
            }

            for name, cfg in LOG_STATISTICS.items():
                col = cfg["column"]
                if col not in split_df.columns:
                    continue                         # column absent → skip
                op  = _OP_FUN[cfg["op"]]
                stats[name] = op(split_df[col])

            # --- pretty-print with the same order as header ------------
            line = " | ".join(
                str(stats.get(c, "")).ljust(15) if c != "count"
                else f"{stats['count']:<15d}"        # keep count integer aligned
                for c in hdr_cols
            )
            print(line)
            f.write(line + "\n")

    print(f"\n[INFO] Dataset statistics written to {stats_txt}")


# ---------------------------------------------------------------------------
# 7.  CLI entry‑point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build balanced dataset splits.")
    p.add_argument("--root_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--filter", choices=["graph", "curve", "none"], default="none",
                   help="Use graph degree for divergence split (default: none)")
    p.add_argument("--use_cache", action="store_true",
                   help="Use cached divergence split csv files if they exist")
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="Directory with precomputed cache files")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Std-dev multiplier for heading outlier filter")
    p.add_argument("--val_ratio", type=float, default=0.02, help="Validation set ratio [default 0.02]")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio [default 0.1]")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of total samples (all interesting + random boring)")
    p.add_argument("--split_file", type=str, default="full_trackfiltered.txt",
                   help="Name of the split file to read from each ride directory [full_trackfiltered.txt, full_entitymasks.txt]")
    p.add_argument("--frame_horizon", type=int, default=60,
                     help="Minimum number of future frames requires for each sample (default: 60)") 
    p.add_argument("--min_alignment_pct", type=float, default=0.0,
                     help="Minimum percentage of frames that must be aligned with the visual action (default: 0.0)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_split(
        root_dir=args.root_dir,
        out_dir=args.output_dir,
        split_file=args.split_file,
        filter=args.filter,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        sigma=args.sigma,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        rng=args.seed,
        max_samples=args.max_samples,
        frame_horizon=args.frame_horizon,
        alignment_pct=args.min_alignment_pct,
    )
