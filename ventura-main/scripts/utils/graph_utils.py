# Third-party libraries
import osmnx as ox                       # pip install osmnx
from scipy.spatial import cKDTree        # pip install scipy
from pyproj import Transformer           # pip install pyproj

from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Module‐level cache for KD‐trees (one per graphml path)
# ----------------------------------------------------------------------
_kdtree_cache = {}


_kdtree_cache = {}

def _get_intersection_kdtree(graph_path):
    """
    Load a GraphML file (pedestrian routing graph), identify intersection nodes
    (nodes with degree ≥ 3), project their coordinates to EPSG:3857,
    and build a cKDTree for fast nearest‐intersection queries.

    Returns:
        (tree, intersections_proj): 
            tree            = cKDTree built on (x, y) of intersection nodes in meters
            intersections_proj = GeoDataFrame of intersection nodes, in EPSG:3857
    Caches the result so repeated calls with the same graph_path reuse the same tree.
    """
    graph_path = Path(graph_path)
    if graph_path in _kdtree_cache:
        return _kdtree_cache[graph_path]

    # 1) Load the graph
    G = ox.load_graphml(str(graph_path))

    # 2) Extract node GeoDataFrame (lat/lon)
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)

    # 3) Compute node degrees and filter to intersections (degree >= 3)
    degrees = dict(G.degree())  # {node_id: degree}
    nodes_gdf["degree"] = nodes_gdf.index.map(degrees)
    intersections = nodes_gdf[nodes_gdf["degree"] >= 3].copy()

    if intersections.empty:
        # No intersections in this graph
        _kdtree_cache[graph_path] = (None, None)
        return None, None

    # 4) Project intersection geometries to WebMercator (EPSG:3857)
    intersections_proj = intersections.to_crs(epsg=3857)
    coords = np.vstack((intersections_proj.geometry.x.values,
                        intersections_proj.geometry.y.values)).T

    # 5) Build KD‐tree on (x, y) in meters
    tree = cKDTree(coords)

    # Cache and return
    _kdtree_cache[graph_path] = (tree, intersections_proj)
    return tree, intersections_proj


def apply_graph_filter(data_dict, filter_dict, ride_metadata):
    """
    Return a boolean mask (length = number of GPS fixes) indicating which fixes
    are within `distance` meters of ANY intersection (graph node) for this ride.

    Assumptions:
      - data_dict["gps"] is an array‐like of shape (N, 3) or (N, 2), where
        column 0 = latitude, column 1 = longitude. (Timestamp column is ignored.)
      - data_dict["ride"] is a string identifier matching the 'ride' column in the
        ride->graph CSV (filter_dict["params"]["ride_to_graph_csv"]).
      - filter_dict["params"] must contain:
           * "ride_to_graph_csv": path to a CSV with columns ["ride", "graph_path"].
           * "distance": (optional) radius in meters for filtering. Defaults to 30.0.
      - ride_metadata pandas DataFrame with graph ml file path
    Returns:
      np.ndarray of shape (N,), dtype=bool, True if that GPS fix is within
      `distance` meters of ANY intersection node in the ride’s graph.
    """
    params = filter_dict['params']
    distance = float(params['distance'])
    graph_path = Path(ride_metadata['graph_path'])

    # 1) Extract GPS array and ride name
    gps_arr = np.asarray(data_dict["gps"])
    if gps_arr.ndim == 2 and gps_arr.shape[1] >= 2:
        lats = gps_arr[:, 0].astype(float)
        lons = gps_arr[:, 1].astype(float)
    else:
        raise ValueError("data_dict['gps'] must have shape (N,2) or (N,3) with lat/lon in cols 0,1.")

    ride = ride_metadata['ride']
    # 2) Load ride->graph lookup and find this ride’s GraphML path
    if not graph_path.is_file():
        raise FileNotFoundError(f"GraphML for ride '{ride}' not found at {graph_path}.")
    
    # 3) Get or build KD‐tree for this graph
    tree, intersections_proj = _get_intersection_kdtree(graph_path)

    if tree is None:
        return np.zeros(len(lats), dtype=bool)
    
    # 4) Project GPS fixes (lat/lon) to EPSG:3857 (meters)
    tf = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_m, y_m = tf.transform(lons, lats)
    
    points_xy = np.column_stack((x_m, y_m))

    assert np.isfinite(points_xy).all(), "GPS coordinates must be finite values."

    # 5) Query the KD‐tree for nearest neighbor distance to any node
    dists, _ = tree.query(points_xy, k=1)

    return dists <= distance
