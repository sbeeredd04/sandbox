# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import cv2
from pathlib import Path
import h5py

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def choose_h5_file(data_dir: Path, chosen: str | None) -> Path:
    if chosen is None:
        files = sorted(data_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No .h5 files in {data_dir}")
        return random.choice(files)
    p = data_dir / chosen
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_video(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        vid = f["images"]["front_camera"][()]
    if vid.dtype != np.uint8:
        vid = (vid.clip(0, 1) * 255).astype(np.uint8)
    if vid.shape[-1] != 3:
        raise ValueError("Expect RGB video")
    return vid

def reverse_video(video: np.ndarray, save_video=False) -> np.ndarray:
    """Reverse the video frames to match the original order."""
    reversed_video =  video[::-1].copy()

    # save video
    if save_video:
        output_path = "reversed_video.mp4"
        # Save video using cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = reversed_video[0].shape
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
        for frame in reversed_video:
            out.write(frame)
        out.release()

        print(f"Reversed video saved to {output_path}")
    return reversed_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_path",
        default="./data/frodo8k/h5files_10hz/ride_12_46796_25901d_20240518084145.h5",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    if not Path(args.h5_path).exists():
        raise ValueError("Video file does not exist")
    
    video = load_video(Path(args.h5_path))
    # Reverse video frames to match the original order
    video = reverse_video(video, save_video=False)

    # Only extract middle 100 frames for testing
    if video.shape[0] > 100:
        start_frame = video.shape[0] // 2 - 50
        end_frame = video.shape[0] // 2 + 50
        video = video[start_frame:end_frame]
    print(f"Loaded video with {video.shape[0]} frames, each of shape {video.shape[1:]}")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, queries, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    H, W = None, None
    for i, frame in enumerate(video):
        H, W, _ = frame.shape
        # Sample square grid in bottom center of frame
        queries_x = np.linspace(
            frame.shape[1] // 2 - 8*args.grid_size,
            frame.shape[1] // 2 + 8*args.grid_size,
            args.grid_size,
        ).astype(int).clip(0, frame.shape[1] - 1)
        queries_y = np.linspace(
            frame.shape[0] - 8*args.grid_size,
            frame.shape[0],
            args.grid_size,
        ).astype(int).clip(0, frame.shape[0] - 1)
        queries = np.stack(np.meshgrid(queries_x, queries_y), axis=-1).reshape(-1, 2)

        # Add time of query frame to queries
        query_time = np.ones((queries.shape[0], 1)) * i
        queries_th = np.concatenate([query_time, queries], axis=-1).astype(int)
        queries_th = torch.from_numpy(queries_th).to(DEFAULT_DEVICE).float()[None, :, :]

        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries_th,
                # grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries=queries_th,
        grid_query_frame=args.grid_query_frame,
    )

    print("Tracks are computed")
    # Query mask for grid queries [1, 1, H, W]
    query_mask = torch.zeros(
        (1, 1, frame.shape[0], frame.shape[1]), device=DEFAULT_DEVICE
    )
    query_mask[0, 0, queries_y[0] : queries_y[-1] + 1, queries_x[0] : queries_x[-1] + 1] = 1
    # Make segm_mask [1, T, H, W] for time dimension
    query_mask = query_mask.repeat(1, pred_tracks.shape[1], 1, 1)

    # save a video with predicted tracks
    seq_name = args.h5_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(
        save_dir="./saved_videos", 
        pad_value=120, 
        linewidth=3,
        mode="rainbow",
        tracks_leave_trace=-1,
    )
    import pdb; pdb.set_trace()
    vis.visualize(
        video, 
        pred_tracks.cpu(), 
        pred_visibility, 
        query_frame=args.grid_query_frame,
        segm_mask=query_mask.cpu(),
        compensate_for_camera_motion=True
    )