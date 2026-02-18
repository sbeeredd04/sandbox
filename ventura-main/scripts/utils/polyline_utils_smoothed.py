import cv2, numpy as np
from functools import partial
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate   import UnivariateSpline
from scipy.spatial       import cKDTree
from scipy.ndimage       import gaussian_filter1d

DEBUG_MASK = False

def robust_spline(t, y, smooth, k=3, max_iter=10, tol=1e-3):
    """
    Iteratively-re-weighted LSQ spline     (Tukey bi-square ρ).
    Returns a callable spline exactly like UnivariateSpline.
    """
    # internal: prepare uniform knots (open uniform B-spline)
    if t.size < k + 2:
        raise ValueError("not enough points for spline")
    # one knot every ~ (len/4) points  --> very similar to s parameter
    step      = max(int(t.size // 4), k + 1)
    knot_idx  = np.arange(step, t.size - step, step)
    tk        = t[knot_idx]

    # start with all weights = 1
    w = np.ones_like(t)
    for _ in range(max_iter):
        spl = LSQUnivariateSpline(t, y, tk, w=w, k=k)
        resid = y - spl(t)
        median_abs = np.median(np.abs(resid)) + 1e-9   # avoid /0
        u = resid / (6.0 * median_abs)                 # Tukey constant = 6
        w_new = (1 - u**2) ** 2
        w_new[np.abs(u) >= 1] = 0

        if np.mean(np.abs(w - w_new)) < tol:
            break
        w = w_new

    return spl

def build_mask_from_crumbs(
    img_bgr: np.ndarray,
    tracks: np.ndarray,          # (K,2)
    crumb_times: np.ndarray,     # (K,)
    sides: np.ndarray,           # (K,)
    *,
    # spline_smooth: float = 150.0,
    # resolution: int = 24,        # arc-len px per dense point
    # blur_sigma: float = 5.0,     # σ for Gaussian smoothing of width curve
    # min_width_frac: float = 0.2, # never collapse below this × global median
    # spoke_length: int = 20,
    # Tight turns sparse
    spline_smooth: float = 50.0,
    resolution: int = 20,        # arc-len px per dense point
    blur_sigma: float = 5.0,     # σ for Gaussian smoothing of width curve
    min_width_frac: float = 0.45, # never collapse below this × global median
    spoke_length: int = 15,
    viz_path: str = "centerline_viz.png",
) -> np.ndarray:
    """Return bool mask (H,W) covering the swept path.  Saves debug PNG."""
    H, W = img_bgr.shape[:2]
    default_output = {"mask": None, "success": False}
    # ───────────────────────────── 1. filter NaNs
    ok = (np.isfinite(tracks).all(1) &
          np.isfinite(crumb_times)    &
          np.isfinite(sides))
    if not np.any(ok):
        if DEBUG_MASK:
            print("build_mask_from_crumbs: no valid crumbs found")
        return default_output
    tracks, crumb_times, sides = tracks[ok], crumb_times[ok], sides[ok]

    # ───────────────────────────── 2-3. left/right centroids per int timestamp
    degree = 3
    out = get_center_spline(
        tracks, crumb_times, sides, degree=degree, spline_smooth=spline_smooth
    )
    if out is None:
        if DEBUG_MASK:
            print("build_mask_from_crumbs: failed to compute center spline")
        return default_output

    Lc, Rc, Mc, ts, cx, cy, px, py = (
        out["Lc"], out["Rc"], out["Mc"],
        out["ts"], out["cx"], out["cy"],
        out["px"], out["py"]
    )

    # ───────────────────────────── 4. debug overlay (unchanged)
    viz = img_bgr.copy()
    for p in Lc.astype(int): cv2.circle(viz, tuple(p), 3, (255, 0, 0), -1)
    for p in Rc.astype(int): cv2.circle(viz, tuple(p), 3, (  0, 0,255), -1)
    for p in Mc.astype(int): cv2.circle(viz, tuple(p), 3, (  0,255, 0), -1)
    cv2.polylines(viz, [np.round(np.c_[cx, cy]).astype(int)], False, (255,255,0), 2)
    for (mx, my), (nx, ny) in zip(Mc.astype(int), zip(px, py)):
        p1 = (mx + int(nx * spoke_length), my + int(ny * spoke_length))
        p2 = (mx - int(nx * spoke_length), my - int(ny * spoke_length))
        cv2.line(viz, (mx, my), p1, (0,255,255), 1)
        cv2.line(viz, (mx, my), p2, (0,255,255), 1)

    # ───────────────────────────── 5. densify along centreline (just to pick a density)
    seg     = np.hypot(np.diff(cx), np.diff(cy))
    s       = np.concatenate(([0.], np.cumsum(seg)))
    total   = s[-1]
    n_dense = max(int(total / resolution) + 1, 3 * len(s))

    # ───────────────────────────── 6. fit two splines to Lc and Rc
    ts_min, ts_max = ts[0], ts[-1]
    ts_d = np.linspace(ts_min, ts_max, n_dense)
    try:
        # side splines (reuse same smoothing knob)
        # sxL = UnivariateSpline(ts, Lc[:,0], k=degree, s=spline_smooth)
        # syL = UnivariateSpline(ts, Lc[:,1], k=degree, s=spline_smooth)
        # sxR = UnivariateSpline(ts, Rc[:,0], k=degree, s=spline_smooth)
        # syR = UnivariateSpline(ts, Rc[:,1], k=degree, s=spline_smooth)
        sxL = robust_spline(ts, Lc[:, 0], smooth=spline_smooth)
        syL = robust_spline(ts, Lc[:, 1], smooth=spline_smooth)
        sxR = robust_spline(ts, Rc[:, 0], smooth=spline_smooth)
        syR = robust_spline(ts, Rc[:, 1], smooth=spline_smooth)
    except Exception as e:
        print("build_mask_from_crumbs: spline fit failed:", e)
        return default_output

    # evaluate dense left/right
    left_xy  = np.vstack([sxL(ts_d), syL(ts_d)]).T
    right_xy = np.vstack([sxR(ts_d), syR(ts_d)]).T
    
    # ───────────────────────────── 7. smooth each rail
    left_xy[:,0]  = gaussian_filter1d(left_xy[:,0],  blur_sigma, mode="nearest")
    left_xy[:,1]  = gaussian_filter1d(left_xy[:,1],  blur_sigma, mode="nearest")
    right_xy[:,0] = gaussian_filter1d(right_xy[:,0], blur_sigma, mode="nearest")
    right_xy[:,1] = gaussian_filter1d(right_xy[:,1], blur_sigma, mode="nearest")

    def extrapolate_to_bottom(xy):
        p0, p1 = xy[-2], xy[-1]
        dx, dy = p1 - p0
        if dy == 0:
            new = np.array([p1[0], H-1])
        else:
            t   = (H-1 - p1[1]) / dy
            new = p1 + t*np.array([dx, dy])
        return np.vstack([xy, new])

    left_xy  = extrapolate_to_bottom(left_xy)
    right_xy = extrapolate_to_bottom(right_xy)

    # ───────────────────────────── 7. debug‐draw the two rails
    cv2.polylines(viz, [np.round(left_xy ).astype(int)], False, (0,255,0), 2)
    cv2.polylines(viz, [np.round(right_xy).astype(int)], False, (0,0,255), 2)
    cv2.imwrite(viz_path, viz)

    # ───────────────────────────── 8. fill between them
    poly = np.vstack([left_xy, right_xy[::-1]])
    mask = np.zeros((H, W), np.uint8)
    if poly.shape[0] >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)

    # ───────────────────────────── 8b. health‐check & fallback
    # a) fraction of crumbs covered
    pts = np.round(tracks).astype(int)
    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)
    inside = mask[pts[:,1], pts[:,0]]
    frac_in = float(inside.sum()) / len(inside)

    # b) convex‐hull of crumbs (fallback geometry)
    hull_pts = cv2.convexHull(tracks.astype(np.int32))
    hull_mask = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(hull_mask, hull_pts, 255)

    # d) carve out small holes 
    open_radius = 5  # tweak as needed
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*open_radius+1, 2*open_radius+1)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ───────────────────────────── 8e. keep only the largest connected component
    cc_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if cc_count > 1:                                    # 0 = background
        areas          = stats[1:, cv2.CC_STAT_AREA]    # skip background
        largest_label  = 1 + np.argmax(areas)           # restore offset
        mask           = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # ───────────────────────────── 8f. re-evaluate health metrics
    # a) fraction of crumbs covered
    inside   = mask[pts[:, 1], pts[:, 0]]
    frac_in  = float(inside.sum()) / len(inside)

    # b) convex hull of crumbs (same as before)
    mask_area = float(mask.sum())
    hull_area = float(hull_mask.sum()) + 1e-6
    solidity  = mask_area / hull_area

    # decide failure
    COVERAGE_THRESH = 0.7    # e.g. at least 70% of crumbs must lie inside
    SOLIDITY_THRESH = 0.4   # ribbon must fill ≥40% of its convex hull
    if frac_in < COVERAGE_THRESH or solidity < SOLIDITY_THRESH:
        # fallback to simple hull
        if DEBUG_MASK:
            print(f"Mask failure: frac_in={frac_in:.2f}, solidity={solidity:.2f}")
        return { "mask": (hull_mask > 0), "success": False }

    # ───────────────────────────── 9. all good
    return { "mask": mask.astype(bool), "success": True }

def get_center_spline(
    tracks: np.ndarray,
    crumb_times: np.ndarray,
    sides: np.ndarray,
    degree: int,
    spline_smooth: float,
):
    """
    Return (Lc, Rc, Mc, ts, cx, cy, px, py) or None on failure.
    - Lc/Rc: (N,2) left / right centroids per integer timestamp
    - Mc   : centre pts ½(Lc+Rc)
    - ts   : timestamps used for spline fit  (float64)
    - cx,cy: centre spline evaluated at ts
    - px,py: unit normals at ts  (for debug spokes)
    """
    try:
        ok = (np.isfinite(tracks).all(1) &
            np.isfinite(crumb_times)    &
            np.isfinite(sides))
        if not np.any(ok):
            return None
        tracks, crumb_times, sides = tracks[ok], crumb_times[ok], sides[ok]

        t0, t1 = int(np.ceil(crumb_times.min())), int(np.floor(crumb_times.max()))
        Lc, Rc, Mc, ts_mid = [], [], [], []
        for t in range(t0, t1 + 1):
            iL = np.where((crumb_times.astype(int) == t) & (sides == -1))[0]
            iR = np.where((crumb_times.astype(int) == t) & (sides == +1))[0]
            if iL.size and iR.size:
                cL, cR = tracks[iL].mean(0), tracks[iR].mean(0)
                Lc.append(cL); Rc.append(cR); Mc.append((cL + cR) * 0.5); ts_mid.append(t)
        if len(Mc) < 2:
            return None

        Lc, Rc, Mc = map(np.vstack, (Lc, Rc, Mc))
        ts = np.asarray(ts_mid, float)

        # smooth centre spline --------------------------------------------------
        # sx = UnivariateSpline(ts, Mc[:, 0], k=degree, s=spline_smooth)
        # sy = UnivariateSpline(ts, Mc[:, 1], k=degree, s=spline_smooth)
        sx = robust_spline(ts, Mc[:, 0], smooth=spline_smooth)
        sy = robust_spline(ts, Mc[:, 1], smooth=spline_smooth)
        cx, cy = sx(ts), sy(ts)

        # unit normal at sparse points
        dx, dy = sx.derivative()(ts), sy.derivative()(ts)
        nrm    = np.hypot(dx, dy); nrm[nrm == 0] = 1.0
        px, py = -dy / nrm, dx / nrm
    except Exception as e:
        if DEBUG_MASK:
            print("get_center_spline: error:", e)
        return None

    return {
        "Lc": Lc, "Rc": Rc, "Mc": Mc,
        "ts": ts, "cx": cx, "cy": cy,
        "px": px, "py": py
    }

# Old mostly good but uses time
# def get_center_spline(
#     tracks: np.ndarray,
#     crumb_times: np.ndarray,
#     sides: np.ndarray,
#     spline_smooth: float,
# ):
#     """
#     Return (Lc, Rc, Mc, ts, cx, cy, px, py) or None on failure.
#     - Lc/Rc: (N,2) left / right centroids per integer timestamp
#     - Mc   : centre pts ½(Lc+Rc)
#     - ts   : timestamps used for spline fit  (float64)
#     - cx,cy: centre spline evaluated at ts
#     - px,py: unit normals at ts  (for debug spokes)
#     """
#     try:
#         ok = (np.isfinite(tracks).all(1) &
#             np.isfinite(crumb_times)    &
#             np.isfinite(sides))
#         if not np.any(ok):
#             return None
#         tracks, crumb_times, sides = tracks[ok], crumb_times[ok], sides[ok]

#         t0, t1 = int(np.ceil(crumb_times.min())), int(np.floor(crumb_times.max()))
#         Lc, Rc, Mc, ts_mid = [], [], [], []
#         for t in range(t0, t1 + 1):
#             iL = np.where((crumb_times.astype(int) == t) & (sides == -1))[0]
#             iR = np.where((crumb_times.astype(int) == t) & (sides == +1))[0]
#             if iL.size and iR.size:
#                 cL, cR = tracks[iL].mean(0), tracks[iR].mean(0)
#                 Lc.append(cL); Rc.append(cR); Mc.append((cL + cR) * 0.5); ts_mid.append(t)
#         if len(Mc) < 2:
#             return None

#         Lc, Rc, Mc = map(np.vstack, (Lc, Rc, Mc))
#         ts = np.asarray(ts_mid, float)

#         # smooth centre spline --------------------------------------------------
#         sx = UnivariateSpline(ts, Mc[:, 0], k=3, s=spline_smooth)
#         sy = UnivariateSpline(ts, Mc[:, 1], k=3, s=spline_smooth)
#         cx, cy = sx(ts), sy(ts)

#         # unit normal at sparse points
#         dx, dy = sx.derivative()(ts), sy.derivative()(ts)
#         nrm    = np.hypot(dx, dy); nrm[nrm == 0] = 1.0
#         px, py = -dy / nrm, dx / nrm
#     except Exception as e:
#         if DEBUG_MASK:
#             print("get_center_spline: error:", e)
#         return None

#     return {
#         "Lc": Lc, "Rc": Rc, "Mc": Mc,
#         "ts": ts, "cx": cx, "cy": cy,
#         "px": px, "py": py
#     }

def sweep_path_mask(
    center_pts: np.ndarray,
    left_pts:   np.ndarray,
    right_pts:  np.ndarray,
    image_shape: tuple[int,int],
    cap_style: str = 'round',
    resolution: int = 16
) -> np.ndarray:
    """
    Build a 2D mask by buffering the center‐line to approximate the swept volume.
    Handles both Polygon and MultiPolygon cases.
    """
    # 1) Compute median corridor half‐width
    widths = np.linalg.norm(right_pts - left_pts, axis=1)
    if widths.size == 0:
        return np.zeros(image_shape, dtype=np.uint8)
    radius = float(np.median(widths)) / 2.0

    # 2) Buffer the LineString
    ls = LineString(center_pts.tolist())
    cap_map = {'round':1, 'flat':2, 'square':3}
    cap = cap_map.get(cap_style, 1)
    poly = ls.buffer(radius,
                     resolution=resolution,
                     cap_style=cap,
                     join_style=1)

    # 3) Ensure we have a list of Polygons
    if isinstance(poly, Polygon):
        polys = [poly]
    elif isinstance(poly, MultiPolygon):
        # unify overlapping pieces just in case
        polys = list(poly)
    else:
        # unexpected geometry
        return np.zeros(image_shape, dtype=np.uint8)

    # 4) Rasterize each exterior ring
    mask = np.zeros(image_shape, dtype=np.uint8)
    ext_pts = []
    for p in polys:
        coords = np.array(p.exterior.coords, dtype=np.int32)
        # make sure it's (N,2)
        if coords.ndim == 2 and coords.shape[1] == 2:
            ext_pts.append(coords)
    if ext_pts:
        cv2.fillPoly(mask, ext_pts, color=255)

    return mask

def build_mask_from_polylines(
    left_poly: np.ndarray,
    right_poly: np.ndarray,
    image_shape: tuple[int, int],
    buffer_radius: int = 2
) -> np.ndarray:
    """
    Join two polylines by pairing their bottommost (highest y) points and connecting
    them down to the bottom edge of the image, fill the interior polygon, then apply a dilation buffer.

    Args:
        left_poly    : np.ndarray of shape (N_L, 3) with columns [t, x, y]
        right_poly   : np.ndarray of shape (N_R, 3) with columns [t, x, y]
        image_shape  : (H, W)
        buffer_radius: how many pixels to dilate the filled polygon

    Returns:
        mask: np.ndarray of shape (H, W), dtype=uint8, 255 inside the buffered region, 0 outside.
    """
    H, W = image_shape

    # Extract and round to ints
    left_pts = np.round(left_poly[:, 1:3]).astype(np.int32)
    right_pts = np.round(right_poly[:, 1:3]).astype(np.int32)

    # Clip to image bounds
    left_pts[:, 0] = np.clip(left_pts[:, 0], 0, W - 1)
    left_pts[:, 1] = np.clip(left_pts[:, 1], 0, H - 1)
    right_pts[:, 0] = np.clip(right_pts[:, 0], 0, W - 1)
    right_pts[:, 1] = np.clip(right_pts[:, 1], 0, H - 1)

    if len(left_pts) < 1 or len(right_pts) < 1:
        return np.zeros((H, W), dtype=np.uint8)

    # Find bottommost points (max y)
    iL = np.argmax(left_pts[:, 1])
    iR = np.argmax(right_pts[:, 1])
    BL_x, BL_y = left_pts[iL]
    BR_x, BR_y = right_pts[iR]

    # Build sequences starting from those bottommost indices
    left_rot = np.vstack((left_pts[iL:], left_pts[:iL]))
    right_rot = np.vstack((right_pts[iR:], right_pts[:iR]))

    # Sort those rotated arrays by decreasing y (trace upward)
    left_sorted = left_rot[np.argsort(-left_rot[:, 1])]
    right_sorted = right_rot[np.argsort(-right_rot[:, 1])]

    # Points on bottom edge directly below bottommost points
    BLeft = np.array([BL_x, H - 1], dtype=np.int32)
    BRight = np.array([BR_x, H - 1], dtype=np.int32)

    # Build polygon:
    # 1) Start at left_sorted[0] (bottommost left), go down to BLeft,
    # 2) then across bottom edge to BRight, then up to right_sorted[0] (bottommost right),
    # 3) then follow right_sorted upward, then follow left_sorted downward back to start.
    polygon_pts = []
    polygon_pts.append(left_sorted[0])     # bottommost left
    polygon_pts.append(BLeft)              # connect down to bottom edge
    polygon_pts.append(BRight)             # connect across bottom
    polygon_pts.append(right_sorted[0])    # bottommost right

    # 4) Up the right side
    for i in range(1, len(right_sorted)):
        polygon_pts.append(right_sorted[i])

    # 5) Down the left side (skip the first element, which is bottommost)
    for i in range(len(left_sorted) - 1, 0, -1):
        polygon_pts.append(left_sorted[i])

    polygon_pts = np.array(polygon_pts, dtype=np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], color=255)

    if buffer_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * buffer_radius + 1, 2 * buffer_radius + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return mask