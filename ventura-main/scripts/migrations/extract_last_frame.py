#!/usr/bin/env python3
"""
Extract the last frame from every
…/output_rides_{ride_id}/ride_{did0}_{did1}_{ts}/seq_{start}/front_camera.mp4
and save it as …/front_camera.jpg.

Usage
-----
python extract_last_frame.py /path/to/root_dir
# (optionally) overwrite existing .jpg files
python extract_last_frame.py /path/to/root_dir --overwrite
"""
import argparse, sys
from pathlib import Path

# --------------------------------------------------------------------------------------
def extract_last_frame(video_path: Path, out_path: Path) -> bool:
    """
    Decode the final frame (BGR) of an MP4 and write it as JPEG.
    Uses decord if available, otherwise falls back to OpenCV.
    Returns True on success.
    """
    try:
        import decord, cv2, numpy as np
        decord.bridge.set_bridge("numpy")
        vr = decord.VideoReader(str(video_path), num_threads=2)
        frame = vr[len(vr) - 1]                    # ndarray HxWxC, uint8
    except Exception:
        # --> fallback to opencv
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERR] cannot open {video_path}", file=sys.stderr)
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"[ERR] cannot read last frame {video_path}", file=sys.stderr)
            return False

    import cv2
    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        print(f"[ERR] cannot write {out_path}", file=sys.stderr)
    return ok

# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Extract final frame of each front_camera.mp4")
    p.add_argument("root", type=Path, help="Dataset root directory")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing front_camera.jpg")
    args = p.parse_args()

    mp4_files = list(args.root.rglob("front_camera.mp4"))
    if not mp4_files:
        print("No front_camera.mp4 found under that root.", file=sys.stderr)
        sys.exit(1)

    for mp4 in mp4_files:
        jpg = mp4.with_name("front_camera.jpg")
        if jpg.exists() and not args.overwrite:
            print(f"[SKIP] {jpg} already exists — use --overwrite to replace.")
            continue
        if extract_last_frame(mp4, jpg):
            print(f"[OK]   {jpg}")
        else:
            print(f"[FAIL] {mp4}", file=sys.stderr)

if __name__ == "__main__":
    main()
