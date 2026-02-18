"""
To be used for all camera related utilities in SpinFlow.
"""
import torch
import cv2
import numpy as np

def get_pixel2pts_transform(calib_dict):
    """
    Returns a transformation matrix that converts image pixels to 3D points in LiDAR frame

    Inputs:
        calib_dict: [dict] calibration dictionary
    Outputs:
        pix2pts: [4 x 4] transformation matrix
    """
    # Assume finite camera eqn
    T_cam_world = calib_dict['cam_to_world_matrix']

    T_canon = np.eye(4)
    T_canon[:3, :3] = calib_dict['rectification_matrix'].T  # 4x4

    M = calib_dict['new_camera_matrix'][:3, :3]
    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = np.linalg.inv(M)  # 4x4

    T_rect_to_world = T_cam_world @ T_canon @ P_pix_cam

    return T_rect_to_world

def get_pts2pixel_transform(calib_dict):
    """
    Returns a transformation matrix that converts 3D points in LiDAR frame to image pixel coordinates
    Boilerplate function to get the projection matrix from the calibration dictionary

    P =  Pcam @ Eye(Re | 0) @ T_lidar_cam

    Inputs:
        calib_dict: [dict] calibration dictionary
    Outputs:
        pts2pix: [4 x 4] transformation matrix
    """
    T_cam_world = calib_dict['cam_to_world_matrix']
    T_world_cam = np.linalg.inv(T_cam_world)

    T_canon = np.eye(4)
    T_canon[:3, :3] = calib_dict['rectification_matrix']

    M = calib_dict['new_camera_matrix'][:3, :3]
    P_pix_cam = np.eye(4)
    P_pix_cam[:3, :3] = M

    T_world_to_rect = P_pix_cam @ T_canon @ T_world_cam

    return T_world_to_rect

def project_xyz_to_pixel(xyz, intrinsics, T_cam_to_world):
    """
    Projects Nx3 xyz points to image camera intrinsics and extrinsics.
    """
    assert xyz.ndim == 2 and xyz.shape[-1] == 3, "XYZ should be of shape [N, 3]."
    assert 'K' in intrinsics and 'R' in intrinsics, "Intrinsics should contain 'K' and 'R'."
    assert intrinsics['K'].shape[-2:] == (3, 3), "K should be of shape [3, 3]."
    assert intrinsics['R'].shape[-2:] == (3, 3), "R should be of shape [3, 3]."
    assert T_cam_to_world.shape[-2:] == (4, 4), "T_cam_to_world should be of shape [4, 4]."

    # Move inputs to numpy if necessary
    K, R = intrinsics['K'], intrinsics['R']
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(T_cam_to_world, torch.Tensor):
        T_cam_to_world = T_cam_to_world.cpu().numpy()

    # Ensure that K and R are [1, 3, 3]
    if K.ndim == 2:
        K = K[None, :, :]
    if R.ndim == 2:
        R = R[None, :, :]
    if T_cam_to_world.ndim == 2:
        T_cam_to_world = T_cam_to_world[None, :, :]

    # TODO: implement the projection logic
    calib_dict = {
        'cam_to_world_matrix': T_cam_to_world,
        'new_camera_matrix': K,
        'rectification_matrix': R,
    }
    T_world_to_rect = get_pts2pixel_transform(calib_dict)[0]
    pts_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])  # [N, 4]
    uv_homogeneous = (T_world_to_rect @ pts_homogeneous.T).T  # [N, 4]
    uv = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:3]  # Normalize by z to get pixel coordinates

    # Clip invalid points to be within image bounds and in front of the camera
    image_H, image_W = int(intrinsics['image_height']), int(intrinsics['image_width'])
    valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_W) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < image_H) & \
                 (uv_homogeneous[:, 2] > 0)  # Ensure points are in front of the camera

    if not valid_mask.any():
        return None
    
    uv = uv[valid_mask]  # Keep only valid points
    return uv

def project_odom_to_pixel(odom, T_world_optical, calib_dict):
    """Converts SE2 odometry to pixel coordinates using camera intrinsics and extrinsics.
    Args:
        odom (torch.Tensor): [B, T, 3, 3]
        T_world_optical (torch.Tensor): [B, 4, 4]
        calib_dict (dict): Dictionary containing camera intrinsics and extrinsics.
            Should contain 'new_camera_matrix' for intrinsic matrix 
            and 'cam_to_world_matrix' for extrinsic matrix.
    Returns:
        torch.Tensor: Pixel coordinates of shape (N, 2).
    """
    assert odom.shape[-2:] == (3, 3), "Odom should be of shape [B, T, 3, 3]."
    assert T_world_optical.shape[-2:] == (4, 4), "T_world_optical should be of shape [B, 1, 4, 4]."

    T_cam_world = calib_dict['cam_to_world_matrix']
    
    # convert SE2 to xyz
    x = odom[:, :, 0, -1]
    y = odom[:, :, 1, -1]
    # z = -0.5 * torch.ones_like(x)  # Assuming z is constant, can be adjusted based on calibration
    z = -T_cam_world[:, 2, 3] * torch.ones_like(x)  # Assuming z is constant, can be adjusted based on calibration
    pts = torch.stack([x, y, z], dim=-1)  # Shape: (1, N, 3)
    pts = torch.cat([pts, torch.ones_like(pts[:, :, :1])], dim=-1)  # Add homogeneous coordinate, shape: (B, T, 4)
    
    uvd = torch.matmul(T_world_optical, pts.unsqueeze(-1)).squeeze(-1)  # Shape: (B, T, 4)

    # Clip invalid points to first valid point
    image_H, image_W = calib_dict['image_height'].item(), calib_dict['image_width'].item()
    uv = uvd[:, :, :2] / (uvd[:, :, 2:3])  # Normalize by z to get pixel coordinates
    valid_mask = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < image_W) & \
                (uv[:, :, 1] >= 0) & (uv[:, :, 1] < image_H)
    
    # Return all zero if no valid points
    if not valid_mask.any():
        return torch.zeros_like(uv[:, :, :2])  # Return zero tensor if no valid points
    
    # Set invalid points to the first valid point
    uv[~valid_mask] = uv[valid_mask].unsqueeze(1).expand(-1, uv.shape[1], -1)[0, 0, :]

    return uv  # Shape: (B, T, 2)