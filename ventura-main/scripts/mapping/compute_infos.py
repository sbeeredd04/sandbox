

from pathlib import Path
import numpy as np
import hickle as hkl

from typing import Dict, List

from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)
from scripts.mapping.compute_odom import (
    compute_odometry_goals
)
from scripts.utils.loader_utils import (
    load_intrinsics,
    load_extrinsics
)

def compute_frame_transform(
    from_frame: str,
    to_frame: str,
    extrinsics: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Computes 4x4 transformations from T^{to_frame}_{from_frame}.
    If the transformation is not available, raises KeyError.
    """
    # identity if same frame
    if from_frame == to_frame:
        return np.eye(4)

    # Reverse this order because extrinsics are stored in the form T^{from_frame}_{to_frame}
    # and we need T^{to_frame}_{from_frame} for the transformation.
    key = f"{to_frame} {from_frame}"
    rev_key = f"{from_frame} {to_frame}"

    if key in extrinsics:
        return extrinsics[key]
    elif rev_key in extrinsics:
        # invert the available transform
        return np.linalg.inv(extrinsics[rev_key])
    else:
        raise KeyError(f"No transform available between '{from_frame}' and '{to_frame}'")


def compute_transforms(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str
):
    """This function reads the transforms and saves the necessary 
    information for project from cam to base link and camera infos
    """
    parts = ride_name.split(' ')
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    ride_id = parts[0]
    driveid0 = parts[1]
    driveid1 = parts[2]
    timestamp = "_".join(parts[3:])

    root_ride_dir = set_frodo_dir(root_dir, *parts)

    # Load the transforms
    extr_path = root_ride_dir / f"tf_static_{driveid0}.yaml"
    intr_path = root_ride_dir / f"front_camera_info_{driveid0}.yaml"

    intr = load_intrinsics(intr_path)
    extr = load_extrinsics(extr_path)
    if intr is None or extr is None:
        print(f"Failed to load intrinsics or extrinsics for ride {ride_name}.")
        return False

    try:
        # Precompute transform from camera to base link frames
        optical_frame = intr['frame_id']
        base_link_frame = f'{driveid0}/base_link'
        T_optical_to_base = compute_frame_transform(optical_frame, base_link_frame, extr)
    except KeyError as e:
        print(f"Missing transform for ride {ride_name}: {e}")
        return False

    # Aggregate transforms into single ride infos dictionary
    ride_infos = {
        'ride_id': ride_id,
        'driveid0': driveid0,
        'driveid1': driveid1,
        'timestamp': timestamp,
        'start_frame': start_frame,
        'end_frame': end_frame,
    }
    ride_infos.update({
        'intrinsics': intr,
        'extrinsics': extr,
        'T_optical_to_base': T_optical_to_base
    })
    ride_dir = set_frodo_dir(out_dir.parent, *parts) / f"seq_{start_frame}"
    ride_dir.mkdir(parents=True, exist_ok=True)

    out_path = ride_dir / "ride_info.h5"
    hkl.dump(ride_infos, out_path)

    return True

def compute_ride_infos(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str
) -> bool:
    """
    Computes necessary information for training from the ride data.
    This includes extracting metadata, computing odometry, etc.
    """
    # ---- 1) Load all the data
    transforms_passed = compute_transforms(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )

    # Compute odometry goals
    odom_passed = compute_odometry_goals(
        ride_name=ride_name,
        start_frame=start_frame,
        end_frame=end_frame,
        root_dir=root_dir,
        out_dir=out_dir
    )
    ride_passed = transforms_passed and odom_passed
    return ride_passed