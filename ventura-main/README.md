# Ventura: Adapting Image Diffusion Models for Unified Task Conditioned Navigation

## 🛠️ Setup

Installing the environment
```bash
conda create -n spinflow python=3.10
conda activate spinflow
pip install -e .
pip install -r requirements.txt
```

This has been tested on:
- Ubuntu 22.04
- Python 3.10
- CUDA 12.8
- pip 24.3.1

If you make changes to the environment, you can update the environment with:
```bash
./scripts/aws/update_env.sh
```

## 📚 Dataset

### Sync Datasets Across Machines
To sync the dataset across machines, you can use the `sync_dataset.sh` script. This script sync 200 randomly selected the `.h5` files from the source directory to the destination directory. The source directory is expected to be on the local machine, and the destination directory is expected to be on a remote machine.

```bash
python scripts/aws/sync_frodo.py --src /robodata/public_datasets/frodobots8k/spinflow_processed --dst ec2-user@52.13.84.237:/data/public_datasets/frodo8k --split ./data/frodo8k/splits/spinflow_full/full.txt --num 200
```

## 🔨 Building the Dataset

To build the dataset, first you will need download the frodobots dataset. Then, you should use the `run_engine.py` script to filter the dataset to only include interesting rides and download additional imagery not included in the original dataset (e.g. satellite imagery) and routing directions.

```bash
./scripts/preprocessing/run_engine_sequential.sh ./data/frodobots8k
```

Next, you will need to generate path trajectory masks for supervising the model. We provide an example below for generating the masks for all of the rides with ride id 6 in the command below.

```bash
python scripts/preprocessing/track_points_online_iterative.py \
    --data_path ./data/frodobots8k \
    --split_path ./data/frodobots8k_processed/output_rides_6/full_raw.txt \
    --out_dir ./data/frodobots8k_processed \
    --dataset split
```

To parallelize this processing, you can run the script below which will loop through all the output rides and process them on multiple GPUs in parallel. The script below will start from gpu 0, using the next 4 gpus, and process 4 rides per gpu at a time.
```bash
./scripts/preprocessing/track_points_sequential.sh ./data/frodobots8k ./data/frodobots8k_processed 0 4 4
```


## 📖 Dataset Description

The main inputs to the model are the `front_camera.mp4` file and `path_tracker.h5` files. The remaining information (satellite image pairs, GPS goals, routing directions, etc.) still need to added.

## Dataset format

```
frodobots8k_processed/
├── output_rides_6
│   ├── full_raw.txt
│   ├── ride_{ride_id}_{did0}_{did1}_{rs}/
│   │   ├── seq_{start_ts}/
│   │   │   ├── front_camera.mp4
│   │   │   ├── path_tracker.h5
│   │   │   ├── satellite_info.h5
│   │   ...
│   │   maps/
│   │   ├── graphs/
│   │   │   ├── cluster_{cid}_{S}_m_{W}_{N}_m_{E}.graphml
│   │   │   ├── ...
│   │   ├── metadata/
│   │   |   ├── clusters.csv
│   │   |   ├── ride_to_graph.csv
│   │   ├── tiles/
```
`front_camera.mp4` contains the next `N` frames ending at `end_ts` in the ride. The number of frames `N` may vary across videos and may not match `start_ts`. We guarantee that frame for which the mask is generated is `start_ts`. This is because we only track crumbs until a window exists where we have not tracked any crumbs The `path_tracker.h5` contains the path trajectory masks for supervising the model and following the format below:
```json
{
    "tracks": "[1, T, N, 2] np array float32",
    "visibility": "[1, T, N] np array bool",
    "crumbs": "[1, N, 3] np array float32 [t x y] pixels",
    "sides": "[1, N] np array of int8",
    "path_mask": "[H, W] np array bool True means pixel is part of the path"
}
```

The `satellite_infos.h5` contains all information regarding satellite imagery, global gps paths, and transformation between GPS to pixels.

```json
{
    "satellite_image": "[640, 640, 3] np array haligned sat. rgb",
    "gt_route_image": "[640, 640, 3] np array ann. haligned sat. rgb",
    "current_gps": "[2,] np array cur lat lon",
    "current_heading": "[1, np array heading in deg]",
    "future_gps": "[M, 2] np array curr + future lat lon",
    "satellite_query": "dict for transforming gps->pixel"
}
```

`ride_to_graph.csv` format. It enables mapping from the ride id to the corresponding graph (contains network node info for planning).
```
ride,ride_dir,cluster_id,cluster_uuid,graph_path
6 29923 ca3e01 20240411205030,$CLUSTER_DIR,$GRAPH_DIR
...
```
