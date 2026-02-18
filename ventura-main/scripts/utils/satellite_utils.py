import math
import requests
import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg

import logging
from scripts.utils.log_utils import *

import hashlib
_NO_IMG_HASH = "f5e4ef6082f67a91c5cd77f9c0d6b762"  # example MD5
_EARTH_R = 6_371_000.0  # m

def get_haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth (all args in degrees).
    Returns metres.  Works with numpy arrays.
    """
    dlat  = np.radians(lat2 - lat1)
    dlon  = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * _EARTH_R * np.arcsin(np.sqrt(a))

def gps_to_sat_pixels(
    gps_np: np.ndarray,              # (N,2) lat,lon  (deg)
    sat_query: dict,                 # same keys as prepare_satellite_query
    current_gps: Tuple[float, float],
    current_heading_deg: float | None = None,
    heading_aligned: bool = False,
    display_px: int | None = None,   # final cropped size (width)
    display_py: int | None = None,   # final cropped size (height)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GPS coords to pixel coords on the **cropped** satellite image.

    If `heading_aligned=True`, first rotate the whole set so the *current*
    heading faces up-centre (same convention as annotate_satellite_image).

    Returns
    -------
    xs, ys : (N,) float arrays  — pixel coords in the cropped frame
    """
    def _latlon_to_pixel(lat: np.ndarray,
                     lon: np.ndarray,
                     zoom: int,
                     scale: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised conversion to *world* pixel coords used by Google Maps.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        sin_lat = np.sin(np.radians(lat))
        n       = 256.0 * scale * (2 ** zoom)                # world size in px
        x = (lon + 180.0) / 360.0 * n
        y = (0.5 - np.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * np.pi)) * n
        return x, y

    # 1) basic parameters
    lat_c   = sat_query["center_lat"]
    lon_c   = sat_query["center_lon"]
    z       = int(sat_query["zoom"])
    s       = int(sat_query.get("scale", 1))
    W_full  = int(sat_query["width"])  * s
    H_full  = int(sat_query["height"]) * s

    # 2) world-pixel coords
    cx, cy          = _latlon_to_pixel(lat_c, lon_c, z, s)
    px, py          = _latlon_to_pixel(gps_np[:, 0], gps_np[:, 1], z, s)
    xs_full         = px - cx + W_full / 2.0
    ys_full         = py - cy + H_full / 2.0

    # 3) crop offset (query vs. displayed image)
    if display_px is None:  display_px = W_full
    if display_py is None:  display_py = H_full
    dx = (W_full  - display_px) / 2.0
    dy = (H_full -  display_py) / 2.0
    xs = xs_full - dx
    ys = ys_full - dy

    # 4) optional heading-align
    if heading_aligned and current_heading_deg is not None:
        lat0, lon0 = current_gps
        # rotate around display centre so heading points up
        theta = -np.deg2rad(current_heading_deg)
        c, s  = np.cos(theta), np.sin(theta)
        cx0, cy0 = display_px / 2.0, display_py / 2.0
        vx, vy   = xs - cx0, ys - cy0
        xs =  vx * c - vy * s + cx0
        ys =  vx * s + vy * c + cy0

    return xs, ys

def filter_gps_in_satellite(
    gps_np: np.ndarray,
    sat_query: Dict[str, float | int | str],
    radius_m: float | None = None,
    return_pixels: bool = False,
):
    """
    Keep only GPS poses that

    1. lie within `radius_m` (metres) of the satellite‐map centre **AND**
    2. project to pixels inside the satellite image bounds.

    If `radius_m is None`, only the pixel‐bounds test is applied.

    Parameters
    ----------
    gps_np : (N,2) ndarray
        [[lat, lon], ...]  (degrees)
    sat_query : dict
        center_lat, center_lon, zoom, scale, width, height  (Static-Maps style)
    radius_m : float | None
        Maximum great-circle distance.  None → no distance filter.
    return_pixels : bool
        Whether to also return (x,y) pixels of retained points.

    Returns
    -------
    gps_kept : (M,2) ndarray
    pix_xy   : (M,2) int ndarray       (only if `return_pixels` is True)
    """
    # -------- map parameters -------------------------------------------------
    z   = int(sat_query["zoom"])
    s   = int(sat_query.get("scale", 1))
    W   = int(sat_query["width"])  * s
    H   = int(sat_query["height"]) * s
    lat_c = float(sat_query["center_lat"])
    lon_c = float(sat_query["center_lon"])

    # -------- pixel coords for all points -----------------------------------
    cx, cy      = latlon_to_pixel(lat_c, lon_c, z, s)
    px, py      = latlon_to_pixel(gps_np[:, 0], gps_np[:, 1], z, s)
    u           = px - cx + W / 2.0
    v           = py - cy + H / 2.0
    in_bounds   = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # -------- optional distance filter --------------------------------------
    if radius_m is not None:
        dist_m   = get_haversine_m(lat_c, lon_c, gps_np[:, 0], gps_np[:, 1])
        in_radius = dist_m <= radius_m
        keep      = in_bounds & in_radius
    else:
        keep      = in_bounds

    gps_kept = gps_np[keep]

    if return_pixels:
        return gps_kept, np.stack([u[keep], v[keep]], axis=1).astype(int)

    return gps_kept

def is_no_imagery_tile(arr: np.ndarray) -> bool:
    """
    Return True iff `arr` is the standard Google 'no imagery' tile.
    Fast version: compare MD5 of the compressed PNG bytes.
    Fallback (if hash changes): variance test.
    """
    # -- exact hash (fast) ------------------------------------------------
    md5 = hashlib.md5(cv2.imencode(".png", arr)[1]).hexdigest()
    if md5 == _NO_IMG_HASH:
        return True

    # -- variance test (robust) ------------------------------------------
    # real satellite tiles have large colour variance; placeholder is flat
    if arr.var() < 150:             # tweak threshold if needed
        # additionally, check that most pixels are near the grey bg colour
        mean_col = arr.reshape(-1, 3).mean(axis=0)
        if np.allclose(mean_col, (223, 221, 216), atol=15):  # BGR
            return True
    return False

def prepare_satellite_query(
    gps_np: np.ndarray,
    width_km: float,
    height_km: float,
    width_px: int,
    height_px: int,
    maptype: str = "satellite",
    scales: list[int] = (1, 2),
    min_zoom: int = 0,
    max_zoom: int = 21
) -> dict:
    """
    Compute a static‐map query centered on gps_np[0], 
    covering at least width_km×height_km, and returning exactly 
    width_px×height_px pixels by choosing zoom & scale automatically.

    Args:
        gps_np:      (N,2) array of [lat, lon] (we use only [0]).
        width_km:    east–west ground coverage in km.
        height_km:   north–south ground coverage in km.
        width_px:    desired output width in pixels.
        height_px:   desired output height in pixels.
        maptype:     e.g. "satellite".
        scales:      list of allowed scale factors (1 or 2).
        min_zoom:    min allowed zoom level.
        max_zoom:    max allowed zoom level.

    Returns:
        dict with keys:
          center_lat, center_lon,
          zoom, scale,
          width, height (map‐pixels = size param),
          size (="{width}x{height}"),
          maptype
    """
    # 1) center point
    center_lat = float(gps_np[0,0])
    center_lon = float(gps_np[0,1])

    # 2) degrees per km at center_lat
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = (
        111132.954
      - 559.822 * math.cos(2*lat_rad)
      +   1.175 * math.cos(4*lat_rad)
      -   0.0023 * math.cos(6*lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat_rad)
      -   93.5   * math.cos(3*lat_rad)
      +    0.118 * math.cos(5*lat_rad)
    )
    km_per_deg_lat = m_per_deg_lat / 1e3
    km_per_deg_lon = m_per_deg_lon / 1e3

    # 3) required degree spans
    lat_span_deg = height_km / km_per_deg_lat
    lon_span_deg = width_km  / km_per_deg_lon

    if lat_span_deg <= 1e-6 or lon_span_deg <= 1e-6:
        logging.warning("Height or width too small, cannot process")
        return None

    best = None  # (zoom, scale)
    for s in scales:
        # solve width:  width_px/s = lon_span_deg / (360/(256*2^zoom))
        # ⇒ 2^zoom = (360 * (width_px/s)) / (256 * lon_span_deg)
        # ⇒ zoom = log2(...)
        # Default tile size 256px,each zoom level doubles tile size
        zoom_w = math.log2((360.0 * (width_px / s)) / (256.0 * lon_span_deg))
        zoom_h = math.log2((360.0 * (height_px / s)) / (256.0 * lat_span_deg))
        zoom_real = min(zoom_w, zoom_h)
        zoom_int  = int(math.floor(zoom_real))
        zoom_int  = max(min_zoom, min(max_zoom, zoom_int))

        # record highesgfiltert zoom
        if best is None or zoom_int > best[0]:
            best = (zoom_int, s)

    if best is None:
        logging.warning("No valid zoom/scale found for the given parameters")
        return None

    zoom, scale = best

    # map‐pixels size = display_px / scale
    map_w = int(math.ceil(width_px  / scale))
    map_h = int(math.ceil(height_px / scale))

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom":       zoom,
        "scale":      scale,
        "width":      map_w,
        "height":     map_h,
        "size":       f"{map_w}x{map_h}",
        "maptype":    maptype
    }

def get_satellite_image(args: dict) -> np.ndarray:
    """
    Drop-in replacement for Google Static Maps using Esri World Imagery tiles,
    with corrected tile coverage to avoid off-by-one cropping errors.
    """
    center_lat = args['center_lat']
    center_lon = args['center_lon']
    width_px   = args['width']
    height_px  = args['height']
    zoom       = args['zoom']

    TILE_SIZE = 256
    num_tiles = 2**zoom
    map_size  = TILE_SIZE * num_tiles

    # convert lat/lon -> global Web Mercator pixel coords
    def latlon_to_pixel(lat: float, lon: float):
        lat_rad = math.radians(lat)
        x = (lon + 180.0) / 360.0 * map_size
        y = (1.0 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2.0 * map_size
        return x, y

    cx, cy     = latlon_to_pixel(center_lat, center_lon)
    half_w, half_h = width_px / 2.0, height_px / 2.0

    # bounding box in pixel space
    x0, y0 = cx - half_w, cy - half_h

    # tile indices covering that box
    tx0 = int(math.floor(x0 / TILE_SIZE))
    ty0 = int(math.floor(y0 / TILE_SIZE))
    tx1 = int(math.ceil((x0 + width_px ) / TILE_SIZE)) - 1
    ty1 = int(math.ceil((y0 + height_px) / TILE_SIZE)) - 1

    # fetch & stitch tiles
    rows = []
    for ty in range(ty0, ty1 + 1):
        row_tiles = []
        ty_clamped = min(max(ty, 0), num_tiles - 1)
        for tx in range(tx0, tx1 + 1):
            tx_wrapped = tx % num_tiles
            url = (
                f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{zoom}/{ty_clamped}/{tx_wrapped}"
            )
            resp = requests.get(url)
            resp.raise_for_status()
            tile = cv2.imdecode(
                np.frombuffer(resp.content, np.uint8),
                cv2.IMREAD_COLOR
            )
            if is_no_imagery_tile(tile):
                print("Invalid tile received, returning None")
                return None
            row_tiles.append(tile)
        rows.append(np.hstack(row_tiles))
    big_img = np.vstack(rows)

    # compute pixel offset of our box within the stitched image
    x_off = int(math.floor(x0)) - tx0 * TILE_SIZE
    y_off = int(math.floor(y0)) - ty0 * TILE_SIZE

    # now crop exactly width_px×height_px
    cropped = big_img[
        y_off : y_off + height_px,
        x_off : x_off + width_px
    ]

    # sanity check
    if cropped.shape[0] != height_px or cropped.shape[1] != width_px:
        raise RuntimeError(
            f"Esri stitch/crop failed: got {cropped.shape[:2]} "
            f"instead of ({height_px}, {width_px})"
        )
    # import pdb; pdb.set_trace()
    # cv2.imwrite("testsatelliteesri.png", cropped)
    return cropped

def get_gmap_satellite_image(args: dict) -> np.ndarray:
    """
    Fetch & stitch Google Maps satellite tiles to produce an arbitrary-size image
    covering the exact geographic box defined by args.

    args must contain:
      - center_lat (float)
      - center_lon (float)
      - width      (int): desired output width in pixels
      - height     (int): desired output height in pixels
      - zoom       (int): zoom level (0–21)
      - api_key    (str)
    Optional:
      - scale   (int): 1 or 2 (retina); default 1
      - servers (list[str]): tile subdomains; default ['mt0','mt1','mt2','mt3']
      - layer   (str): 's' for satellite; default 's'
    """
    center_lat = args['center_lat']
    center_lon = args['center_lon']
    width_px   = args['width']
    height_px  = args['height']
    zoom       = args['zoom']
    api_key    = args['api_key']
    scale      = args.get('scale', 1)
    servers    = args.get('servers', ['mt0','mt1','mt2','mt3'])
    layer      = args.get('layer', 's')  # 's'=satellite, 'r'=roadmap, etc.

    TILE_SIZE  = 256
    num_tiles  = 2 ** zoom
    map_size   = TILE_SIZE * num_tiles

    def latlon_to_pixel(lat: float, lon: float):
        """Convert lat/lon to global WebMercator pixel coords at this zoom."""
        lat_rad = math.radians(lat)
        x = (lon + 180.0) / 360.0 * map_size
        y = (1.0 - math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi) / 2.0 * map_size
        return x, y

    # center pixel coords
    cx, cy = latlon_to_pixel(center_lat, center_lon)
    half_w, half_h = width_px / 2.0, height_px / 2.0
    x0, y0 = cx - half_w, cy - half_h

    # which tile indices cover that box?
    tx0 = int(math.floor(x0 / TILE_SIZE))
    ty0 = int(math.floor(y0 / TILE_SIZE))
    tx1 = int(math.ceil ((x0 + width_px ) / TILE_SIZE)) - 1
    ty1 = int(math.ceil ((y0 + height_px) / TILE_SIZE)) - 1

    # fetch & stitch
    rows = []
    for ty in range(ty0, ty1 + 1):
        tiles_row = []
        ty_clamped = min(max(ty, 0), num_tiles - 1)
        for tx in range(tx0, tx1 + 1):
            tx_wrapped = tx % num_tiles
            # pick a server in round-robin fashion
            server = servers[(tx_wrapped + ty_clamped) % len(servers)]
            url = (
                f"https://{server}.google.com/vt/lyrs={layer}"
                f"&x={tx_wrapped}&y={ty_clamped}&z={zoom}"
                f"&scale={scale}&key={api_key}"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            tile = cv2.imdecode(
                np.frombuffer(resp.content, np.uint8),
                cv2.IMREAD_COLOR
            )
            if is_no_imagery_tile(tile):
                print("Invalid tile received, returning None")
                return None
            tiles_row.append(tile)
        # horizontally concatenate this row of tiles
        rows.append(np.hstack(tiles_row))

    big_img = np.vstack(rows)

    # pixel‐offset of our desired box within the stitched image
    x_off = int(math.floor(x0)) - tx0 * TILE_SIZE
    y_off = int(math.floor(y0)) - ty0 * TILE_SIZE

    # crop exactly width_px×height_px
    cropped = big_img[
        y_off : y_off + height_px,
        x_off : x_off + width_px
    ]

    if cropped.shape[0] != height_px or cropped.shape[1] != width_px:
        print(f"Got {cropped.shape[:2]} instead of ({height_px}, {width_px})")
        return None

    return cropped

def align_gps_heading_nearest(
    gps_data: np.ndarray,
    heading_data: np.ndarray,
    target_timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each t in target_timestamps, find the GPS & heading sample
    with the closest timestamp.

    Args:
        gps_data:          [N×3] array of [timestamp, latitude, longitude]
        heading_data:      [N×2] array of [timestamp, heading_degrees]
        target_timestamps: [M]   array of times to align to

    Returns:
        positions: [M×2] array of [latitude, longitude]
        headings:  [M]   array of heading_degrees
    """
    # unpack
    ts   = gps_data[:, 0]
    lats = gps_data[:, 1]
    lons = gps_data[:, 2]
    h_ts = heading_data[:, 0]
    hdgs = heading_data[:, 1]

    if not np.allclose(ts, h_ts):
        raise ValueError("GPS and heading timestamps must match")

    # assume ts is sorted ascending
    # for each target t, find insertion point in ts
    idxs = np.searchsorted(ts, target_timestamps, side='left')

    # clamp both neighbors to [0,N-1]
    idx0 = np.clip(idxs - 1, 0, len(ts)-1)
    idx1 = np.clip(idxs,     0, len(ts)-1)

    # choose whichever neighbor is closer
    dist0 = np.abs(target_timestamps - ts[idx0])
    dist1 = np.abs(target_timestamps - ts[idx1])
    use1  = dist1 < dist0
    idxs_nearest = np.where(use1, idx1, idx0)

    # gather results
    positions = np.stack([lats[idxs_nearest], lons[idxs_nearest]], axis=1)
    headings  = hdgs[idxs_nearest]

    return positions, headings

def align_compass_to_ts(
    tgt_timestamps: np.ndarray,
    compass_timestamps: np.ndarray,
    compass_headings: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """
    Smooth compass_headings in a sliding window, then linearly interpolate
    so that you get one heading per gps_timestamp.

    Args:
        tgt_timestamps:      shape (N,) array of GPS times (float seconds)
        compass_timestamps:  shape (M,) array of IMU/compass times
        compass_headings:    shape (M,) array of headings in degrees
        window_size:         odd int, size of the smoothing window (in samples)

    Returns:
        shape (N,) array of smoothed & interpolated headings at each gps_timestamp.
    """
    # 1) cast to float and sort compass data by time
    tgt_ts  = tgt_timestamps.astype(np.float64)
    comp_ts = compass_timestamps.astype(np.float64)
    comp_hdgs = compass_headings.astype(np.float64)
    sort_idx = np.argsort(comp_ts)
    comp_ts = comp_ts[sort_idx]
    comp_hdgs = comp_hdgs[sort_idx]

    # 2) circular smoothing: average unit‐vectors over a sliding window
    #    convert to radians and to unit‐vectors
    thetas = np.deg2rad(comp_hdgs)
    cosines = np.cos(thetas)
    sines   = np.sin(thetas)

    # build uniform window
    w = np.ones(window_size) / window_size
    cos_smooth = np.convolve(cosines, w, mode='same')
    sin_smooth = np.convolve(sines,   w, mode='same')

    # recompose into angles, back to degrees
    smoothed_thetas = np.arctan2(sin_smooth, cos_smooth)
    comp_hdgs_smooth = np.rad2deg(smoothed_thetas) % 360.0

    # 3) interpolate (np.interp clamps by default at ends)
    aligned_headings = np.interp(tgt_ts, comp_ts, comp_hdgs_smooth)

    return aligned_headings

def _blend_gradient_path_mpl(bgr_img: np.ndarray,
                             xs: np.ndarray,
                             ys: np.ndarray,
                             color: None,
                             lw: float = 2.0) -> np.ndarray:
    """
    Blend a Turbo-gradient poly-line onto `bgr_img` using Matplotlib/Agg.

    • figure/axes are fully transparent  → no white background
    • RGBA buffer is converted to BGR before alpha blending
    """
    h, w = bgr_img.shape[:2]

    # --- transparent figure -------------------------------------------
    dpi  = 100.0
    fig  = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=(0, 0, 0, 0))
    ax   = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor("none")

    # --- build gradient line ------------------------------------------
    pts      = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

    # Default to turbo, otherwise solid color   
    if color is None:
        # Turbo gradient
        lc = LineCollection(segments,
                            cmap="turbo",
                            linewidths=lw,
                            antialiased=True)
        lc.set_array(np.linspace(0.0, 1.0, len(xs)))   # colour along path
    else:
        # Solid RGB / BGR colour (accept either)
        rgb = np.asarray(color, np.uint8)
        if rgb.shape[0] == 3:          # assume BGR as from OpenCV → RGB
            rgb = rgb[[2, 1, 0]]
        lc = LineCollection(segments,
                            colors=[rgb / 255.0],      # 0-1 floats
                            linewidths=lw,
                            antialiased=True)
    ax.add_collection(lc)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)                                # invert y to match img

    # --- render to RGBA and blend -------------------------------------
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)

    overlay_bgr = rgba[..., [2, 1, 0]]               # RGBA → BGR
    alpha       = rgba[..., 3:] / 255.0
    out = (overlay_bgr * alpha + bgr_img * (1 - alpha)).astype(np.uint8)
    return out

def annotate_satellite_image(
    img: np.ndarray,
    gps_np: np.ndarray,
    heading_np: np.ndarray,
    satellite_query: dict,
    heading_aligned: bool = True,
    fov_angle_deg: float = 110.0,
    fov_length_frac: float = 0.1,
    point_size: int = 3,
    point_color: tuple[int,int,int] = None,
    draw_cone: bool = True,
    cone_fill_color: tuple[int,int,int]  = (255, 225, 186),
    alpha: float = 0.7
) -> np.ndarray:
    """
    Annotate a (possibly cropped) satellite image with GPS trajectory and an FOV cone,
    taking into account any center-cropping compared to the original query size.

    Args:
        img:               H_img×W_img×3 BGR cropped satellite image
        gps_np:            (N×2) array of [lat, lon]
        heading_np:        (N,)   array of headings in degrees (0=north, CW+)
        satellite_query:   dict with 'center_lat','center_lon',
                           'width','height','zoom' of the original fetched image
        heading_aligned:   if True, rotate points so first heading is up-center
        fov_angle_deg:     total width of the FOV cone in degrees
        fov_length_frac:   fraction of min(W_img,H_img) to use as cone radius
        point_size:        radius of GPS dots
        point_color:       BGR for GPS dots
        cone_fill_color:   BGR for the semi-transparent FOV
        alpha:             transparency for cone fill (0..1)

    Returns:
        Annotated copy of `img`.
    """
    annotated = img.copy()
    H_img, W_img = img.shape[:2]
    W_full = satellite_query['width']
    H_full = satellite_query['height']
    center_lat = satellite_query['center_lat']
    center_lon = satellite_query['center_lon']
    zoom       = satellite_query['zoom']

    # compute cropping offset (how many px were cut off on each side)
    dx = (W_full - W_img) / 2.0
    dy = (H_full - H_img) / 2.0

    # degrees-per-pixel based on the full image
    deg_per_px = 360.0 / (256 * 2**zoom)
    lon_span = W_full * deg_per_px
    lat_span = H_full * deg_per_px

    # project lat/lon -> full-image pixel coords, then shift by crop offset
    xs, ys = gps_to_sat_pixels(
        gps_np,
        satellite_query,
        current_gps=(gps_np[0,0], gps_np[0,1]),
        current_heading_deg=heading_np[0],
        heading_aligned=heading_aligned,
        display_px=W_img,
        display_py=H_img,
    )

    # image-center in cropped coords
    px0 = W_img / 2.0
    py0 = H_img / 2.0

    if draw_cone:
        # draw FOV cone at the first point
        x0, y0 = xs[0], ys[0]
        half_fov = fov_angle_deg / 2.0

        # --- CHOOSE CONE ORIENTATION ----------------------------------
        base_hdg = 0.0 if heading_aligned else heading_np[0]   # ← NEW
        angles   = np.linspace(-half_fov, half_fov, num=30) + base_hdg
        # --------------------------------------------------------------

        radius = int(min(W_img, H_img) * fov_length_frac)
        pts = []
        for a in angles:
            r = np.deg2rad(a)
            xi = int(round(x0 + radius * np.sin(r)))
            yi = int(round(y0 - radius * np.cos(r)))
            pts.append((xi, yi))

        poly = np.array(
            [(int(round(x0)), int(round(y0)))] + pts + [(int(round(x0)), int(round(y0)))],
            dtype=np.int32
        )

        overlay = annotated.copy()
        cv2.fillPoly(overlay, [poly], cone_fill_color)
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
    
    # ------------------------------------------------------------------
    # draw Turbo-coloured polyline + dots
    # ------------------------------------------------------------------
    n_pts = len(xs)
    valid_mask = np.ones(n_pts, dtype=bool)
    if n_pts:
        if point_color is None:
            # Turbo colours in BGR (OpenCV’s default order)
            vals   = np.linspace(0, 255, n_pts, dtype=np.uint8).reshape(-1, 1)
            colors = cv2.applyColorMap(vals, cv2.COLORMAP_TURBO)[:, 0, :]  # (N,3)
        else:
            colors = np.full((n_pts, 3), point_color, dtype=np.uint8)

        # ── 1) polyline with segment-wise colour (smooth enough for N≫1) ──
        if len(xs) >= 2:
            annotated = _blend_gradient_path_mpl(annotated, xs, ys, color=point_color, lw=10.0)
            annotated = np.ascontiguousarray(annotated)

        # ── 2) overlay dots so way-points remain visible ─────────────────
        for i, (x, y) in enumerate(zip(xs, ys)):
            color = tuple(int(v) for v in colors[i])
            cv2.circle(annotated, (int(round(x)), int(round(y))),
                    point_size, color, -1)

            # (optional) bounds check
            if x < 0 or x >= W_img or y < 0 or y >= H_img:
                valid_mask[i] = False
                logging.warning(f"Point {i} ({x}, {y}) outside image bounds")
    return annotated

def gps_to_local_xy(
    goals: np.ndarray,
    origin: np.ndarray,
    earth_radius: float = 6_371_000.0
) -> np.ndarray:
    """
    Transform GPS goals (lat/lon) into a local XY frame.

    Args:
        goals:   (N,2) array of [lat, lon] in degrees.
        origin:  (3,) array = [lat0, lon0, heading_deg],
                 heading_deg = clockwise from North.
        earth_radius: Earth radius in meters.

    Returns:
        (N,2) array of [X, Y] in meters, where
          X = rightward (relative to heading),
          Y = forward  (along heading).
    """
    lat0, lon0, heading_deg = origin
    lat  = goals[:, 0]
    lon  = goals[:, 1]

    # to radians
    d2r = np.pi / 180.0
    lat0r = lat0 * d2r
    latr  = lat  * d2r
    lonr  = lon  * d2r
    lon0r = lon0 * d2r

    # delta
    dlat = latr - lat0r
    dlon = lonr - lon0r

    # ENU linearization
    east  = earth_radius * dlon * np.cos(lat0r)
    north = earth_radius * dlat

    # heading
    psi = heading_deg * d2r
    c, s = np.cos(psi), np.sin(psi)

    # right‐vector = [ cos, sin ]
    # forward-vector = [ -sin,  cos ]
    X =  east * c - north * s   # positive to your right
    Y =  east * s + north * c   # positive forward

    # Convert ENU to local XY
    Xlocal = Y
    Ylocal = -X

    return np.stack((Xlocal, Ylocal), axis=1)

def compute_visibility_and_distances(
    query: dict,
    future_gps: np.ndarray,
    current_heading: float,
    display_px: int = 1024,
    display_py: int = 576
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
      query: {
        'center_lat': float,
        'center_lon': float,
        'width': int,     # map‐pixels (size/scale)
        'height': int,
        'zoom': int,
        'scale': int      # 1 or 2
      }
      future_gps: (M,2) array of [lat, lon]
      current_heading: rotation applied (degrees, clockwise)
      display_px: final crop size (px), e.g. 640

    Returns:
      visible:   (M,) bool array – True if that point lands inside the rotated & cropped image
      distances: (M,) float array – ground‐distance in meters from the current GPS
    """
    # unpack
    center_lat = query['center_lat']
    center_lon = query['center_lon']
    map_w      = query['width']
    map_h      = query['height']
    zoom       = query['zoom']
    scale      = query.get('scale', 1)

    # 1) compute meters‐per‐degree at center
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = (
        111132.954
      - 559.822 * math.cos(2*lat_rad)
      +   1.175 * math.cos(4*lat_rad)
      -   0.0023 * math.cos(6*lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat_rad)
      -   93.5   * math.cos(3*lat_rad)
      +    0.118 * math.cos(5*lat_rad)
    )

    # 2) compute raw distances (meters)
    lats = future_gps[:,0]
    lons = future_gps[:,1]
    dx_m = (lons - center_lon) * m_per_deg_lon
    dy_m = (lats - center_lat) * m_per_deg_lat
    distances = np.hypot(dx_m, dy_m)

    # 3) figure out the pixel‐scale of the raw (pre‐crop) image
    xs, ys = gps_to_sat_pixels(
        future_gps,
        query,
        current_gps=(future_gps[0,0], future_gps[0,1]),
        current_heading_deg=current_heading,
        heading_aligned=True,
        display_px=display_px,
        display_py=display_py,
    )
    visible = ((xs >= 0) & (xs < display_px) &
               (ys >= 0) & (ys < display_py))

    return visible, distances
