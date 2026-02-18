# Dataset Preprocessing

Edit the `scripts/preprocessing/config/frodo_mining.yaml` config file to set what engine preprocessing steps you want to run. Below we will provide the exact settings we used to preprocess our dataset.

**Final Dataset Structure**
```markdown
output_directory
|_ouput_rides_0
    |_ ride_x_y_z
        |_ seq_{s}
            |_ entity_info.h5
            |_ front_camera.mp4
            |_ path_tracker.h5
            |_ satellite_info.h5
    full_entitymasks.txt
    full_raw.txt
    full_satellite.txt
    full_trackfiltered.txt
```

**Downloads network graphs and performs initial sequence prefiltering**
All pipeline steps set to false except `download_routing`, `apply_prefilters`. Saves sequences to `full_raw.txt`.
```bash
./scripts/run_engine.parallel.sh ./data/frodobots8k_processed 0 40 full_raw.txt
```

```markdown
output_directory
|_ouput_rides_0
    |_ maps
        |_ graphs
        |_ metadata
    full_raw.txt
```
 

**Extract path trajectory masks using CoTracker**

The following script will extract path trajectory masks with CoTracker and save the successfully processed sequences to `full_raw.txt`
```bash
./scripts/track_points_parallel.sh ./data/frodobots8k ./data/frodobots8k_processed --gpus 0,1,2,3,4,5,6,7 --jobs 4
```

```markdown
output_directory
|_ouput_rides_0
    |_ ride_x_y_z
        |_ seq_{s}
            |_ front_camera.mp4
            |_ path_tracker.h5
```


**Postfilter path trajectory masks and download satellite imagery**
Set all pipeline steps to false except `apply_postfilters`. This will filter only the valid path trajectory masks from amongst the set of processed masks.
```bash
./scripts/run_engine.parallel.sh ./data/frodobots8k_processed 0 40 full_raw.txt
```

Set all pipeline steps to false except `download_imagery`. This will use a `.gmap_api_key` if you have a google static maps api key and use esri imagery otherwise.
```bash
./scripts/run_engine.parallel.sh ./data/frodobots8k_processed 0 40 full_trackfiltered.txt
```

```markdown
output_directory
|_ouput_rides_0
    |_ ride_x_y_z
        |_ seq_{s}
            |_ satellite_info.h5
    full_satellite
    full_trackfiltered.txt
```


**Compute target entity masks**
Set all pipeline steps to false except `compute_entity_masks`. This will run `BLIP3o` + `Grounding-SAM2` to caption and track masks corresponding to interesting targets of interest.
```bash
./scripts/run_engine.parallel.sh ./data/frodobots8k_processed 0 40 full_trackfiltered.txt
```
Outputs:
```markdown
output_directory
|_ouput_rides_0
    |_ ride_x_y_z
        |_ seq_{s}
            |_ entity_info.h5
    full_entitymasks.txt
```

