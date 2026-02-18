#!/usr/bin/env python3
"""
Module: download.py

Provides a single entry point, `download_maps(cfg, ride_dirs)`, which:
  1. Loads and subsamples GPS points from each ride directory.
  2. Clusters all GPS points (ensuring each ride stays in one cluster).
  3. Computes bounding boxes (with padding) and centers for each cluster.
  4. Plots all GPS points colored by cluster and draws circles around cluster centers.
  5. Builds a ride→cluster lookup table and saves it.
  6. Downloads cropped .osm.pbf extracts (pedestrian routing data) around each cluster in parallel.
  7. Post‐processes each bounding box by building a pedestrian routing graph using OSMnx,
     consolidating intersections by a specified tolerance (subsample), and saving the graph.

Usage (from another script):
    from scripts.mapping.download import download_maps
    download_maps(cfg, ride_dir_list)

Dependencies:
    - Python 3.7+
    - pandas
    - numpy
    - scikit-learn
    - PyYAML
    - geopandas
    - shapely
    - contextily
    - matplotlib
    - requests
    - osmnx
"""

import os
import math
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import requests
import osmnx as ox

# Earth radius in meters (for Haversine conversions)
EARTH_RADIUS = 6_371_000.0

from scripts.utils.loader_utils import (load_timestamps, load_controls, load_gps)
from spinflow.dataset.frodo_helpers import get_frodo_raw_id

def subsample_gps(ride_dirs, max_per_ride):
    """
    For each ride_dir in ride_dirs, load gps_data.csv (expects 'latitude','longitude').
    Randomly subsample up to max_per_ride points per ride.
    Returns one DataFrame with columns: ['ride', 'latitude', 'longitude'].
    """
    all_dfs = []
    for ride_path in ride_dirs:
        ride_name = Path(ride_path).name
        df = load_gps(ride_path, format='pandas')

        if df is None or df.empty:
            logging.warning(f"[WARN] No valid GPS data in {ride_path}, skipping.")
            continue
        
        # Limit to max_per_ride points, sampling uniformly
        ids = np.linspace(0, len(df) - 1, num=min(len(df), max_per_ride), dtype=int)
        df = df.iloc[ids][["latitude", "longitude", "timestamp"]].reset_index(drop=True)
        if len(df) > max_per_ride:
            logging.info(f"[INFO] Ride {ride_name} has {len(df)} points, sampling down to {max_per_ride}.")

        df["ride"] = get_frodo_raw_id(ride_path, format=True)  # e.g. "9 27689 20240405015101"
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No valid GPS data loaded from any ride.")
    combined = pd.concat(all_dfs, ignore_index=True)

    return combined

def haversine_eps(meters):
    return meters / EARTH_RADIUS

def find_best_clustering(df, eps_vals, min_samples):
    """
    Try each eps (in meters) → convert to radians → DBSCAN(haversine).
    Pick the first eps so that each ride's points share a single cluster.
    If none work, fallback to largest eps.
    Returns (df_with_cluster_labels, used_eps_meters).
    """
    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    for eps_m in eps_vals:
        eps_rad = haversine_eps(eps_m)
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
        labels = db.fit_predict(coords_rad)
        df["cluster"] = labels
        ride_nclust = df.groupby("ride")["cluster"].nunique()
        if (ride_nclust == 1).all():
            logging.info(f"[INFO] Selected eps={eps_m} m for clustering.")
            return df.copy(), eps_m
    # fallback
    fallback = eps_vals[-1]
    logging.warning(f"[WARN] No eps satisfied coherence. Using eps={fallback} m anyway.")
    df["cluster"] = DBSCAN(
        eps=haversine_eps(fallback),
        min_samples=min_samples,
        metric="haversine"
    ).fit_predict(coords_rad)
    return df.copy(), fallback

def get_cluster_uuid(cid, bounds):
    """Generates a unique identifier for a cluster based on its ID and bounding box."""
    s, w, n, e = bounds["south"], bounds["west"], bounds["north"], bounds["east"]
    return f"cluster_{cid}_{s:.6f}_{w:.6f}_{n:.6f}_{e:.6f}".replace(".", "_").replace("-", "m")

def compute_cluster_bounds(df, padding_m):
    """
    For each non‐noise cluster in df (has 'cluster' column):
        - Compute min/max lat/lon
        - Compute center (mean lat, mean lon)
        - Expand by padding_m (in meters → degrees approx).
    Returns: {cluster_id: {"bounds": {south, west, north, east}, "center": (lat, lon)}}
    """
    clusters = {}
    for cid, sub in df[df["cluster"] >= 0].groupby("cluster"):
        lat_min, lat_max = sub["latitude"].min(), sub["latitude"].max()
        lon_min, lon_max = sub["longitude"].min(), sub["longitude"].max()
        lat_c = sub["latitude"].mean()
        lon_c = sub["longitude"].mean()

        # Approx. conversion: 1° lat ~ 111.32 km. Lon scaling by cos(lat).
        lat_pad_deg = padding_m / 111320.0
        lon_pad_deg = padding_m / (111320.0 * math.cos(math.radians(lat_c)))

        bounds = {
            "south": lat_min - lat_pad_deg,
            "north": lat_max + lat_pad_deg,
            "west": lon_min - lon_pad_deg,
            "east": lon_max + lon_pad_deg,
        }
        clusters[cid] = {"bounds": bounds, "center": (lat_c, lon_c)}
    return clusters

def plot_clusters(df, clusters, out_dir):
    """
    1) Draws one large figure showing all GPS points colored by cluster, with
       circles around each cluster center (using OpenStreetMap background).
    2) Draws a second figure with a grid of subplots—one subplot per non‐noise cluster—
       each zoomed in on its padded bounding box and overlaid on a satellite basemap
       (Esri.WorldImagery). The GPS points for that cluster are plotted on top.

    Args:
        df (pd.DataFrame): must contain 'latitude','longitude','cluster' columns.
        clusters (dict): mapping cluster_id → { "bounds": {south, west, north, east}, 
                                               "center": (lat_c, lon_c) }.
        out_dir (Path or str): path (including filename prefix) where figures will be saved.
                                Two files will be written:
                                  - `{out_dir}/overview.png`
                                  - `{out_dir}/per_cluster.png`
    """
    # -----------------------------------------------------------------------
    # Part 1: Overall overview map (colored by cluster, circle outlines)
    # -----------------------------------------------------------------------
    # 1a) Build a GeoDataFrame in WebMercator (EPSG:3857)
    gdf_all = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    unique_c = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab20", len(unique_c))

    for idx, cid in enumerate(unique_c):
        sub_all = gdf_all[gdf_all["cluster"] == cid]
        color = "lightgray" if cid == -1 else cmap(idx)
        ax1.scatter(
            sub_all.geometry.x,
            sub_all.geometry.y,
            s=5,
            color=color,
            label=(f"cluster {cid}" if cid >= 0 else "noise"),
        )

    # Haversine helper to compute radius (m)
    def haversine(a_lat, a_lon, b_lat, b_lon):
        φ1, φ2 = math.radians(a_lat), math.radians(b_lat)
        dφ = math.radians(b_lat - a_lat)
        dλ = math.radians(b_lon - a_lon)
        a = (math.sin(dφ / 2) ** 2 +
             math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2)
        return 2 * EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(max(0, 1 - a)))

    # Draw circles around each cluster center
    for cid, info in clusters.items():
        lat_c, lon_c = info["center"]
        # Project the center point to EPSG:3857
        center_merc = (
            gpd.GeoSeries([Point(lon_c, lat_c)], crs="EPSG:4326")
            .to_crs(epsg=3857)
        )
        x_c, y_c = center_merc.geometry.iloc[0].x, center_merc.geometry.iloc[0].y

        # Compute max distance from center to cluster bounds (in meters)
        sub_df = df[df["cluster"] == cid]
        lat_min, lat_max = sub_df["latitude"].min(), sub_df["latitude"].max()
        lon_min, lon_max = sub_df["longitude"].min(), sub_df["longitude"].max()
        span_m = max(
            haversine(lat_c, lon_c, lat_min, lon_c),
            haversine(lat_c, lon_c, lat_max, lon_c),
            haversine(lat_c, lon_c, lat_c, lon_min),
            haversine(lat_c, lon_c, lat_c, lon_max),
        )

        circle = plt.Circle(
            (x_c, y_c),
            span_m,
            edgecolor="black",
            facecolor="none",
            linewidth=1.2
        )
        ax1.add_patch(circle)

    # Add a simple basemap (OpenStreetMap) underneath
    try:
        ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik)
    except (AttributeError, KeyError):
        # In case your contextily version has a different naming for OSM:
        ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron)

    ax1.set_axis_off()
    ax1.legend(markerscale=3, fontsize="small", loc="lower left")
    plt.tight_layout()
    overview_path = f"{str(out_dir)}/overview.png"
    fig1.savefig(overview_path, dpi=200)
    plt.close(fig1)
    logging.info(f"[INFO] Overview cluster map saved to: {overview_path}")

    # -----------------------------------------------------------------------
    # Part 2: One subplot per cluster on a high‐res satellite basemap
    # -----------------------------------------------------------------------
    # Filter out the “noise” cluster (-1) if present
    cluster_ids = sorted(cid for cid in clusters.keys() if cid >= 0)
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        logging.warning("[WARN] No non‐noise clusters to plot in detail.")
        return

    # Determine subplot grid size (square-ish)
    n_cols = int(math.ceil(math.sqrt(n_clusters)))
    n_rows = int(math.ceil(n_clusters / n_cols))

    fig2, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )

    # Pre‐compute the full GeoDataFrame in WebMercator once
    # (we already have gdf_all)
    for idx, cid in enumerate(cluster_ids):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx][col_idx]

        # Extract this cluster's GeoDataFrame slice (in WebMercator)
        sub_merc = gdf_all[gdf_all["cluster"] == cid]

        # If no points, skip
        if sub_merc.empty:
            ax.set_title(f"cluster {cid} (no points)", fontsize=10)
            ax.set_axis_off()
            continue

        # Get this cluster's padded bounds in lat/lon
        b = clusters[cid]["bounds"]
        south, north = b["south"], b["north"]
        west, east = b["west"], b["east"]

        # Convert that lat/lon bbox to WebMercator
        # Create two corner points and project to 3857:
        sw_merc = gpd.GeoSeries([Point(west, south)], crs="EPSG:4326").to_crs(epsg=3857)
        ne_merc = gpd.GeoSeries([Point(east, north)], crs="EPSG:4326").to_crs(epsg=3857)
        x_min, y_min = sw_merc.geometry.iloc[0].x, sw_merc.geometry.iloc[0].y
        x_max, y_max = ne_merc.geometry.iloc[0].x, ne_merc.geometry.iloc[0].y

        # Plot the GPS points for this cluster
        ax.scatter(
            sub_merc.geometry.x,
            sub_merc.geometry.y,
            s=10,
            color="red",
            label=f"cluster {cid}"
        )

        # Set the visible extent / limits, adding a tiny margin if you like
        margin_x = 0.02 * (x_max - x_min)
        margin_y = 0.02 * (y_max - y_min)
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

        # Add high‐res satellite basemap (Esri World Imagery)
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.Esri.WorldImagery,
                crs="EPSG:3857"
            )
        except (AttributeError, KeyError):
            # Fallback: if Esri isn’t available, use another
            ctx.add_basemap(
                ax,
                source=ctx.providers.Stamen_Terrain.terrain_background,
                crs="EPSG:3857"
            )

        ax.set_title(f"cluster {cid}", fontsize=10)
        ax.set_axis_off()

    # Turn off any empty subplots if n_rows*n_cols > n_clusters
    total_plots = n_rows * n_cols
    if total_plots > n_clusters:
        for empty_idx in range(n_clusters, total_plots):
            r = empty_idx // n_cols
            c = empty_idx % n_cols
            axes[r][c].set_axis_off()

    plt.tight_layout()
    per_cluster_path = f"{str(out_dir)}/per_cluster.png"
    fig2.savefig(per_cluster_path, dpi=200)
    plt.close(fig2)
    logging.info(f"[INFO] Detailed per‐cluster map saved to: {per_cluster_path}")

def build_overpass_query(bounds, tags):
    """
    Given bounding box dict: {south, west, north, east}, and a list of Overpass tag filters
    (e.g. ['["highway"~"footway|path"]']), produce a properly‐formatted Overpass QL string.

    The resulting string looks like:

        [bbox:south,west,north,east]
        [out:json]
        [timeout:25];
        (
          way[tag_filter](south,west,north,east);
        );
        out geom;

    If you want PBF instead of JSON, change `[out:json]` → `[out:pbf]` and `out geom;` → `out;`.
    """
    s = bounds["south"]
    w = bounds["west"]
    n = bounds["north"]
    e = bounds["east"]

    # Combine all tag filters into a single string, e.g. '["highway"~"footway|path"]'
    tag_filter = "".join(tags)

    # Build the multiline Overpass QL query
    q = (
        f"[bbox:{s:.6f},{w:.6f},{n:.6f},{e:.6f}]\n"
        f"[out:json]\n"
        f"[timeout:25];\n"
        "(\n"
        f"  way{tag_filter}({s:.6f},{w:.6f},{n:.6f},{e:.6f});\n"
        ");\n"
        "out geom;\n"
    )
    return q


def download_pbf_for_cluster(cid, info, out_path, api_url, tags):
    """
    Executes a POST to Overpass API with the bounding box query (as JSON output), 
    and streams the response to out_dir/cluster_{cid}_SWNE.json (or .osm.pbf if you switch to PBF).

    Note: We send the query in the form field named "data", exactly as Overpass expects.
    """
    bounds = info["bounds"]
    query = build_overpass_query(bounds, tags)

    if out_path.exists():
        logging.info(f"[INFO] Cluster {cid} already downloaded: {out_path.name}")
        return out_path

    try:
        # Overpass expects the query in a form‐field called "data".
        resp = requests.post(
            api_url,
            data={"data": query},
            stream=True,
            timeout=300
        )
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        logging.info(f"[INFO] Downloaded cluster {cid} to {out_path.name}")
        return out_path

    except Exception as e:
        logging.error(f"[ERROR] Failed to download cluster {cid}: {e}")
        return None

def build_and_save_graph(cid, info, graph_dir, tolerance):
    """
    Build a pedestrian‐routing graph from the locally downloaded .osm.pbf file
    (instead of re‐calling Overpass). Then consolidate intersections and save as GraphML.

    Args:
        cid (int): cluster ID
        info (dict): {"bounds": {...}, "center": (...)}  # we no longer actually need bounds here
        out_dir (Path): directory where both PBF and resulting GraphML live
        tolerance (float): in meters, for node‐merging
    """
    tile_id = get_cluster_uuid(cid, info["bounds"])
    b = info["bounds"]
    west, north, east, south = b["west"], b["north"], b["east"], b["south"]

    graph_path = graph_dir / f"{tile_id}.graphml"
    if graph_path.exists():
        logging.info(f"[INFO] Graph for cluster {cid} already exists: {graph_path.name}")
        return graph_path

    try:
        # 2) Load the raw PBF directly into an OSMnx graph (network_type="walk")
        G = ox.graph_from_bbox([west, north, east, south], network_type="walk")

        # 3) Consolidate intersections in UTM (so distance tolerance is in meters)
        G_proj = ox.project_graph(G)  
        G_simpl = ox.consolidate_intersections(
            G_proj,
            tolerance=tolerance,
            rebuild_graph=True,
            dead_ends=False
        )
        # 4) Reproject back to WGS84 so that edges are in lat/lng
        G_final = ox.project_graph(G_simpl, to_crs="EPSG:4326")

        # 5) Save out a GraphML so you can load this graph quickly later
        ox.save_graphml(G_final, filepath=graph_path)
        logging.info(f"[INFO] Saved walking graph for cluster {cid} to '{graph_path.name}'")
        return graph_path
    except Exception as e:
        logging.error(f"[ERROR] Failed to build/save graph for cluster {cid}: {e}")
        return None

def download_maps(cfg: dict, ride_dirs: list):
    """
    Orchestrates the entire map‐downloading and postprocessing pipeline.

    Args:
        cfg (dict): Configuration dictionary (parsed from YAML).
                    Expects all map/routing‐related settings under cfg["download_maps"].
        ride_dirs (list of pathlib.Path): List of ride directory paths, each containing a gps_data.csv.

    The `cfg["download_maps"]` section expects:
        subdir: (str) subdirectory under out_dir where maps/graphs will be saved
        padding: (float) padding in meters to expand each cluster's bounding box
        subsample: (float) tolerance in meters to consolidate intersections in the graph
        max_points_per_ride: (int) max GPS points to sample per ride
        eps_values: (list of float) DBSCAN epsilons (in meters) to try, in ascending order
        min_samples: (int) min_samples for DBSCAN
        osm_pbf:
            api_url: (str) Overpass API endpoint
            tags: (list of str) Overpass tag filters, e.g. ['["highway"~"footway|path"]']
    """
    # --------------------------------------------------
    # 1) Unpack and validate download_maps‐specific configs
    # --------------------------------------------------
    dm_cfg = cfg.get("download_maps", {})
    meta_subdir = dm_cfg.get("metadata_subdir", "maps/metadata")
    tile_subdir = dm_cfg.get("tile_subdir", "maps/tiles")
    graph_subdir = dm_cfg.get("graph_subdir", "maps/graphs")
    graph_lut_path = dm_cfg.get("graph_lut_path", "maps/metadata/ride_to_graph.csv")
    padding_m = dm_cfg.get("padding", 50.0)
    use_cache = dm_cfg.get("use_cache", False)
    max_per_ride = int(dm_cfg.get("max_points_per_ride", 500))
    min_samples_per_cluster = int(dm_cfg.get("min_samples_per_cluster", 100))
    eps_list = dm_cfg.get("eps_values", [100.0, 200.0, 500.0])
    viz_clusters = dm_cfg.get("viz_clusters", True)

    osm_cfg = dm_cfg.get("osm_pbf", {})
    api_url = osm_cfg.get("api_url", "https://overpass-api.de/api/interpreter")
    tags = osm_cfg.get("tags", ['["highway"~"footway|path|pedestrian"]'])
    subsample_tol = dm_cfg.get("subsample", 10.0)  # meters

    assert cfg['pipeline']["download_routing"], \
        "[ERROR] download_maps requires download_routing to be enabled in the pipeline."

    out_dir = cfg['out_dir']
    meta_dir = Path(out_dir) / meta_subdir
    tile_dir = Path(out_dir) / tile_subdir
    graph_dir = Path(out_dir) / graph_subdir
    graph_lut_path = Path(out_dir) / graph_lut_path
    for subdir in [meta_dir, tile_dir, graph_dir, graph_lut_path.parent]:
        Path(subdir).mkdir(parents=True, exist_ok=True)
        
    # --------------------------------------------------
    # 2) Load & subsample GPS points
    # --------------------------------------------------

    logging.info(f"[INFO] Subsampling GPS points from {len(ride_dirs)} rides, max {max_per_ride} per ride.")
    df_gps = subsample_gps(ride_dirs, max_per_ride)

    # --------------------------------------------------
    # 3) Perform DBSCAN clustering (haversine) at multiple epsilons
    # --------------------------------------------------
    
    if not use_cache or not (meta_dir / "clusters.csv").exists():
        logging.info(f"[INFO] Clustering {len(df_gps)} GPS points from {len(df_gps['ride'].unique())} rides.")
        df_clustered, chosen_eps = find_best_clustering(df_gps, eps_list, min_samples_per_cluster)
        df_clustered["eps_meters"] = chosen_eps
        df_clustered.to_csv(meta_dir / "clusters.csv", index=False)
        logging.info(f"[INFO] Clustering complete. Saved to {meta_dir / 'clusters.csv'} with eps={chosen_eps} m.")
    else:
        logging.info(f"[INFO] Loading cached clusters from {meta_dir / 'clusters.csv'}")
        df_clustered = pd.read_csv(meta_dir / "clusters.csv")
        chosen_eps = float(df_clustered["eps_meters"].iloc[0])
        logging.info(f"[INFO] Using cached eps={chosen_eps} m for clustering.")

    # --------------------------------------------------
    # 4) Compute padded bounding boxes & centers per cluster
    # --------------------------------------------------

    clusters_info = compute_cluster_bounds(df_clustered, padding_m)

    # --------------------------------------------------
    # 5) Plot clusters + cluster circles, save PNG
    # --------------------------------------------------

    if viz_clusters:
        logging.info("[INFO] Plotting clusters and saving overview map.")
        plot_clusters(df_clustered, clusters_info, meta_subdir)

    # --------------------------------------------------
    # 6) Build ride→cluster lookup CSV
    # --------------------------------------------------
    ride_to_cluster = (
        df_clustered.groupby("ride")["cluster"]
        .first()
        .reset_index()
        .rename(columns={"cluster": "cluster_id"})
    )

    # Drop unassigned clusters
    ride_to_cluster = ride_to_cluster[ride_to_cluster["cluster_id"] >= 0]
    ride_to_cluster = ride_to_cluster.reset_index(drop=True)

    # Generate ride UUIDs based on cluster and bounds
    ride_to_cluster["cluster_uuid"] = ride_to_cluster["cluster_id"].map(
        lambda cid: get_cluster_uuid(
            cid,
            clusters_info[cid]["bounds"]
        )
    )
    ride_to_cluster.to_csv(graph_lut_path, index=False)
    logging.info(f"[INFO] Ride→cluster lookup saved to {graph_lut_path}")
    
    # --------------------------------------------------
    # 7) Download .osm.pbf extracts for each cluster (parallel)
    # --------------------------------------------------

    if cfg["pipeline"]["download_maps"]:
        logging.info("[INFO] Downloading .osm.pbf extracts for each cluster.")
        pbf_paths = {}
        workers = min(8, max(1, len(clusters_info)))
        # for cid, info in clusters_info.items():
        #     maps_path = tile_dir / get_cluster_uuid(cid, info["bounds"])
        #     pbf_paths[cid] = download_pbf_for_cluster(
        #         cid, info, maps_path, api_url, tags
        #     )

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {}
            for cid, info in clusters_info.items():
                tile_id = ride_to_cluster.loc[
                    ride_to_cluster["cluster_id"] == cid, "cluster_uuid"
                ].values[0]
                tile_path = tile_dir / f"{tile_id}.osm.pbf"
                futures[cid] = exe.submit(
                    download_pbf_for_cluster,
                    cid, info, tile_path, api_url, tags
                )
            for cid, future in futures.items():
                pbf_paths[cid] = future.result()
    else:
        logging.info("[INFO] download_map disabled; skipping .osm.pbf downloads.")

    # --------------------------------------------------
    # 8) Build & save walk graphs (GraphML) for each cluster
    # --------------------------------------------------

    valid_cluster_graphs = {}  # cid -> Path to saved GraphML
    if cfg.get("pipeline", {}).get("download_routing", True):
        with ThreadPoolExecutor(max_workers=min(4, len(clusters_info))) as exe:
            futures = {}
            for cid, info in clusters_info.items():
                futures[cid] = exe.submit(
                    build_and_save_graph,
                    cid, info, graph_dir, subsample_tol
                )
            for cid, future in futures.items():
                graph_path = future.result()
                if graph_path is not None:
                    valid_cluster_graphs[cid] = graph_path
    else:
        logging.info("[INFO] Skipping graph postprocessing as download_routing is disabled.")
    
    # --------------------------------------------------
    # 9) Build ride→graph mapping based on valid clusters
    # --------------------------------------------------
    ride_to_graph_records = []
    ride_dir_to_graph = {}

    # Create a lookup: ride_name -> ride_dir Path
    ride_name_to_dir = {
        get_frodo_raw_id(Path(rd), format=True): Path(rd) 
        for rd in ride_dirs
    }
    
    for _, row in ride_to_cluster.iterrows():
        ride_name = row["ride"]
        cid = int(row["cluster_id"])
        uuid = row["cluster_uuid"]

        # Only keep rides whose cluster had a successfully built graph
        if cid in valid_cluster_graphs:
            graph_path = valid_cluster_graphs[cid]
            ride_dir_path = str(ride_name_to_dir.get(ride_name))
            if ride_dir_path is None:
                # logging.warning(f"[WARN] Ride '{ride_name}' not found in the original ride_dirs list.")
                continue

            # Convert all paths to absolute paths
            ride_to_graph_records.append({
                "ride": ride_name,
                "ride_dir": Path(ride_dir_path).resolve(),
                "cluster_id": cid,
                "cluster_uuid": uuid,
                "graph_path": Path(graph_path).resolve()
            })
            ride_dir_to_graph[ride_name] = graph_path

    # Save the ride→graph CSV
    ride_graph_df = pd.DataFrame.from_records(ride_to_graph_records)
    ride_graph_df.to_csv(graph_lut_path, index=False)
    logging.info(f"[INFO] Valid ride→graph lookup saved to {graph_lut_path}")
    logging.info(f"[INFO] Found {len(ride_dir_to_graph)} rides with valid graphs out of {len(ride_to_cluster)} total rides.")   

    # --------------------------------------------------
    # 10) Return the ride_dir→graph_path dictionary
    # --------------------------------------------------
    logging.info("[INFO] Map download and processing complete.")
    return ride_graph_df