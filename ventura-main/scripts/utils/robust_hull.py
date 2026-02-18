import cv2, numpy as np
from sklearn.covariance import EllipticEnvelope        # ①
from shapely.geometry import MultiPoint                # ②
import alphashape                                      # ② (optional)
from skimage.draw import polygon                       # ③

# -------------------------------------------------- #
def swept_mask(points_xy,  img_shape, *,               # points = (N,2) ndarray
               use_concave=False, filter_threshold=0.0,  alpha=1.5):
    """
    Returns an 8-bit (0/255) mask whose white region tightly covers the inlier
    points.  Everything heavy is done by external libs.

    Parameters
    ----------
    points_xy   : Nx2 float  … [x,y] pixel coords  (column order!)
    img_shape   : (H,W)      … final mask size
    use_concave : bool       … switch convex/concave hull (default convex)
    filter_threshold : float … threshold for outlier removal (default 0.0)
    alpha       : float      … alphashape α when concave
    """
    # ---------- 1) remove gross outliers in one call -----------------
    # Set to 0 to not filter outliers at all.
    if filter_threshold > 0.0:
        inlier_mask = EllipticEnvelope(contamination=filter_threshold).fit_predict(points_xy) >= 0
    else:
        inlier_mask = np.ones(points_xy.shape[0], dtype=bool)
    inliers     = points_xy[inlier_mask]  # (M,2) ndarray

    if inliers.shape[0] < 3:                    # keep calm & carry on
        raise ValueError("Too few inliers for a hull – check the data.")

    # ---------- 2) hull from Shapely (convex) or alphashape (concave)-
    if use_concave:
        # alpha = 0.95 * alphashape.optimizealpha(inliers)
        # poly = alphashape.alphashape(inliers, alpha)
        # import pdb; pdb.set_trace()  # for debugging, remove later
        # convert inliers to list of tuples for alphashape
        pts = [tuple(pt.tolist()) for pt in inliers]  # (x,y) tuples
        # alpha = alphashape.optimizealpha(pts)
        poly = alphashape.alphashape(pts, alpha=alpha)
        # import pdb; pdb.set_trace()  # for debugging, remove later
        # hull_pts = hull.exterior.coords.xy
        # poly = alphashape.alphashape(inliers, alpha)
    else:                                       # convex by default
        poly = MultiPoint(inliers).convex_hull

    # ---------- 3) rasterise polygon to a mask -----------------------
    H, W = img_shape
    mask = np.zeros((H, W), np.uint8)

    # poly.exterior.coords → [(x0,y0), (x1,y1) …]  convert to rows/cols
    if poly.geom_type == "Polygon":
        xs, ys = np.array(poly.exterior.coords.xy, dtype=np.float32)
        rr, cc = polygon(ys, xs, shape=mask.shape)    # sk-image wants r,c
        mask[rr, cc] = 255
    else:                                             # MultiPolygon ⇢ union
        for subpoly in poly.geoms:
            xs, ys = np.array(subpoly.exterior.coords.xy, dtype=np.float32)
            rr, cc = polygon(ys, xs, shape=mask.shape)
            mask[rr, cc] = 255

    return mask, poly, inliers                      # (for debugging / viz)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # example usage --------------------------------------------------
    pts = np.loadtxt("track_pts.txt")               # your blue dots
    frame = np.zeros((288, 512, 3), np.uint8)  # dummy frame for size
    H, W = frame.shape[:2]

    mask, poly, inliers = swept_mask(pts, (H, W), use_concave=True, filter_threshold=0.0,  alpha=0.5)

    # overlay for sanity-check
    vis = frame.copy()
    vis[mask == 255] = (0, 255, 0)                  # green swept area
    for x, y in inliers.astype(int):                # red remaining dots
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

    cv2.imwrite("swept_mask.png", vis)  # save the overlay