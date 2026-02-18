import numpy as np
from pathlib import Path
import hickle as hkl
import matplotlib.pyplot as plt
from skimage.measure import CircleModel, LineModelND, ransac

from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)

FRAME_HORIZON = 60 # ~ 6 meters at 0.1m intervals


def plot_curvature_histogram(
    df,
    col: str = "curvature",
    *,
    n_bins: int = 21,
    range_method: str = "quantile",   # "quantile" | "std" | "iqr" | "data"
    q_low: float = 0.01,
    q_high: float = 0.99,
    std_k: float = 3.0,
    symmetric: bool = True,           # make [lo, hi] symmetric about 0
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot a histogram of curvature values with min/max derived from statistics.
    Returns (edges, counts). Edges can be reused for binning/stratification.
    """
    x = np.asarray(df[col].to_numpy(), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite curvature values to plot.")

    # ---- choose range based on statistics ----
    if range_method == "quantile":
        lo, hi = np.quantile(x, [q_low, q_high])
    elif range_method == "std":
        m, s = float(np.mean(x)), float(np.std(x))
        lo, hi = m - std_k * s, m + std_k * s
    elif range_method == "iqr":
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    elif range_method == "data":
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        raise ValueError(f"Unknown range_method={range_method!r}")

    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        eps = 1e-9
        lo, hi = float(np.min(x) - eps), float(np.max(x) + eps)

    # ---- compute histogram with fixed edges (returned for reuse) ----
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(x, bins=edges)

    # ---- plot ----
    plt.figure()
    plt.hist(x, bins=edges)                 # matplotlib only; no custom colors
    plt.axvline(0.0, linestyle=":", linewidth=1)      # reference at κ=0
    plt.axvline(np.mean(x), linestyle="--", linewidth=1)  # mean
    plt.title(f"Curvature histogram (n={x.size})")
    plt.xlabel("Curvature κ [1/m]")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return edges, counts

def compute_curvature_single(
    root_dir,
    ride_name: str,
    start_frame: int,
    end_frame: int,
    *,
    plot: bool = False,
    pca_thresh: float = 0.01,          # s2/s1 below this → straight
    turn_deg_thresh: float = 10.0,     # total turning (deg) below this → straight
    kappa_min: float = 1e-4,           # |κ| below this → straight
    radius_over_length: float = 8.0,   # R > α·path_length → straight
    residual_scale: float = 2.0,
    ransac_max_trials: int = 1000,
):
    """
    Returns (curvature, rmse). Curvature signed CCW-positive; 0.0 if straight.
    """
    xy = _load_xy_from_odom(root_dir, ride_name, start_frame, end_frame)
    if xy is False:
        return False
    if xy.shape[0] < 2:
        return 0.0, 0.0

    # --- Precompute geometry ---
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if xy.shape[0] >= 2 else np.array([])
    total_len = float(steps.sum())
    sigma = float(np.median(steps)) if steps.size else 1.0
    residual_threshold = max(residual_scale * sigma, 1e-3)

    # PCA straightness (thinness ratio s2/s1)
    Xc = xy - xy.mean(axis=0, keepdims=True)
    if Xc.shape[0] >= 2:
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        thinness = (S[1] / S[0]) if len(S) >= 2 and S[0] > 0 else 0.0
    else:
        thinness = 0.0

    # Total turning (degrees)
    v = np.diff(xy, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dtheta = np.unwrap(np.diff(ang))
    total_turn_deg = np.degrees(np.sum(np.abs(dtheta))) if dtheta.size else 0.0
    signed_turn = float(np.sum(dtheta)) if dtheta.size else 0.0  # for curvature sign

    # Always compute line fit and its RMSE (used for straight and fallback)
    line = LineModelND()
    ok_line = line.estimate(xy)
    line_rmse = float(np.sqrt(np.mean(line.residuals(xy)**2))) if ok_line else 0.0
    line_info = {"kind": "line", "origin": (xy.mean(axis=0) if not ok_line else line.params[0]),
                 "direction": (np.array([1.0, 0.0]) if not ok_line else line.params[1]),
                 "rmse": line_rmse}

    # --- Straightness gate: skip circle if clearly straight ---
    clearly_straight = (thinness < pca_thresh) or (total_turn_deg < turn_deg_thresh)
    if clearly_straight or xy.shape[0] < 3:
        if plot:
            _plot_xy_and_model(xy, line_info,
                f"Straight (gate) — thin={thinness:.3g}, turn={total_turn_deg:.1f}°, κ=0, RMSE={line_rmse:.3f}")
        return 0.0, line_rmse

    # --- Try circle via RANSAC only if not clearly straight ---
    model_c, inliers = ransac(
        xy, CircleModel, min_samples=3,
        residual_threshold=residual_threshold, max_trials=ransac_max_trials
    )

    if model_c is None or not np.any(inliers):
        if plot:
            _plot_xy_and_model(xy, line_info, "Circle RANSAC failed — line fallback (κ=0)")
        return 0.0, line_rmse

    xc, yc, r = map(float, model_c.params)
    in_xy = xy[inliers]
    resid = model_c.residuals(in_xy)
    rmse_circle = float(np.sqrt(np.mean(resid**2)))

    # Curvature sign from overall heading change
    sign = 1.0 if signed_turn > 0 else (-1.0 if signed_turn < 0 else 1.0)
    kappa = sign / r

    # Nearly-straight final clamp
    straight_by_kappa = abs(kappa) < kappa_min
    straight_by_radius = (total_len > 0.0) and (r > radius_over_length * total_len)
    if straight_by_kappa or straight_by_radius:
        if plot:
            _plot_xy_and_model(
                xy, line_info,
                f"Clamped straight — |κ|<{kappa_min} or R>{radius_over_length}·L, κ=0, RMSE={line_rmse:.3f}"
            )
        return 0.0, line_rmse

    circle_info = {"kind": "circle", "center": (xc, yc), "radius": r,
                   "rmse": rmse_circle, "kappa": float(kappa)}
    if plot:
        _plot_xy_and_model(xy, circle_info,
            f"Circle fit — κ={kappa:.5g}, R={r:.3f}, RMSE={rmse_circle:.3f}")

    return float(kappa), rmse_circle


# ------------------------- helpers -------------------------

def _load_xy_from_odom(root_dir, ride_name, start_frame, end_frame):
    parts = ride_name.split(' ')
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    seq_dir = set_frodo_dir(root_dir, *parts) / f"seq_{start_frame}"
    odom_path = seq_dir / "odometry_info.h5"
    if not odom_path.exists():
        print(f"Missing odometry file for ride {ride_name}.")
        return False

    # Only load FRAME_HORIZON frames
    try:
        odom = hkl.load(odom_path)['smoothed_poses'][:FRAME_HORIZON]
    except KeyError:
        print(f"Key 'smoothed_poses' not found in {odom_path}.")
        import pdb; pdb.set_trace()
        return False
    N = odom.shape[0]
    xy = odom[:, 1:3].astype(np.float64)

    # Drop duplicate consecutive points
    # if xy.shape[0] >= 2:
    #     step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    #     keep = np.hstack([True, step > 1e-9])
    #     xy = xy[keep]

    return xy

def _plot_xy_and_model(xy: np.ndarray, model_info: dict, title: str | None = None):
    """Single plotter for both circle and line models, with circle clipped to x-range of points."""
    plt.figure()
    # Odometry points connected by lines
    plt.plot(xy[:, 0], xy[:, 1], marker='o')

    subtitle = "default"
    if model_info["kind"] == "circle":
        xc, yc = model_info["center"]
        r = model_info["radius"]

        # Clip the circle to the x-range of the data
        xmin = float(np.min(xy[:, 0]))
        xmax = float(np.max(xy[:, 0]))
        ymin = float(np.min(xy[:, 1]))
        ymax = float(np.max(xy[:, 1]))

        # Dense parametric circle
        th = np.linspace(0.0, 2.0 * np.pi, 2048)
        cx = xc + r * np.cos(th)
        cy = yc + r * np.sin(th)

        # Keep only parts with x within [xmin, xmax]
        eps = 1e-9
        mask = (cx >= xmin - eps) & (cx <= xmax + eps) & (cy >= ymin - eps) & (cy <= ymax + eps)

        if np.any(mask):
            idx = np.where(mask)[0]
            # split into contiguous segments so we don't draw across gaps
            split_points = np.where(np.diff(idx) > 1)[0] + 1
            segments = np.split(idx, split_points)
            for seg in segments:
                if seg.size > 1:
                    plt.plot(cx[seg], cy[seg], linewidth=2)
        # (optional) else: nothing to draw if no arc lies within x-limits

        # Show center
        # plt.scatter([xc], [yc], s=25)

        subtitle = f"RMSE={model_info['rmse']:.3f}, R={r:.3f}, κ={model_info.get('kappa', 0):.5g}"

    else:
        o = np.asarray(model_info["origin"])
        d = np.asarray(model_info["direction"])
        # Span the best-fit line across the data via projection
        d2 = float(np.dot(d, d)) if float(np.dot(d, d)) > 0 else 1.0
        t = ((xy - o) @ d) / d2
        p0, p1 = o + t.min() * d, o + t.max() * d
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2)
        subtitle = f"RMSE={model_info['rmse']:.3f}, κ=0"

    plt.gca().set_aspect('equal', adjustable='box')
    if title:
        plt.title(f"{title}\n{subtitle}")
    else:
        plt.title(subtitle)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("test_curve.jpg", dpi=150)
    plt.close()