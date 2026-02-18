import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Import your utility functions here; adjust the module path as needed.
from spinflow.util.math_utils import (
    se3_matrix,
    odom_to_local_pose,
)
from scripts.mapping.compute_odom import (
    se3_inverse_batch,
    se3_log_batch,
    se3_exp_batch,
    quat_to_se3,
    interpolate_se3,
)

def random_se3_matrix():
    # Generate random SE3 via random rotation and translation
    rot = R.random().as_quat()  # returns [x, y, z, w]
    q_wxyz = np.array([[rot[3], rot[0], rot[1], rot[2]]])
    t = np.random.rand(1, 3)
    return quat_to_se3(q_wxyz, t)[0]

def test_se3_inverse_batch_identity():
    I = np.eye(4)[None, ...]
    inv = se3_inverse_batch(I)
    assert np.allclose(inv, I)

def test_se3_inverse_batch_random_roundtrip():
    Ts = np.stack([random_se3_matrix(), random_se3_matrix()], axis=0)
    inv_inv = se3_inverse_batch(se3_inverse_batch(Ts))
    assert np.allclose(inv_inv, Ts)

def test_se3_log_exp_roundtrip():
    Ts = np.stack([random_se3_matrix(), random_se3_matrix()], axis=0)
    xi = se3_log_batch(Ts)
    Ts_recon = se3_exp_batch(xi)
    assert np.allclose(Ts_recon, Ts, atol=1e-6)

def test_quat_to_se3_and_se3_matrix_equivalence():
    q = np.array([[1.0, 0.0, 0.0, 0.0]])  # identity quaternion wxyz
    xyz = np.array([[0.5, -0.3, 0.8]])
    T1 = quat_to_se3(q, xyz)
    T2 = se3_matrix(xyz[:,0], xyz[:,1], xyz[:,2], q)
    assert np.allclose(T1, T2)

def test_odom_to_local_pose_se3():
    odom = np.array([
        [0,0,0, 1,0,0,0],
        [1,0,0, 1,0,0,0],
    ])
    local = odom_to_local_pose(odom, mode='se3')
    # First pose at origin
    assert np.allclose(local[0], [0,0,0,1,0,0,0])
    # Second pose relative: translation (1,0,0), no rotation
    assert np.allclose(local[1], [1,0,0,1,0,0,0])

def test_odom_to_local_pose_se2():
    odom = np.array([
        [0,0,0, 1,0,0,0],
        [0,1,0, 1,0,0,0],
    ])
    local = odom_to_local_pose(odom, mode='se2')
    assert local.shape == (2,3)
    assert np.allclose(local[0], [0,0,0])
    assert np.allclose(local[1], [0,1,0])

def test_interpolate_se3_linear_translation():
    # Simple 1D motion along x, identity rotation
    odo_ts = np.array([0.0, 1.0])
    odo_xyz = np.array([[0.0,0.0,0.0], [1.0,0.0,0.0]])
    odo_q = np.array([[1,0,0,0], [1,0,0,0]])
    cam_ts = np.array([0.0, 0.5, 1.0])
    res = interpolate_se3(cam_ts, odo_ts, odo_xyz, odo_q)

    # result shape: (F, 8) = ts + x,y,z + qw,qx,qy,qz
    assert res.shape == (3, 8)
    # timestamps preserved
    assert np.allclose(res[:,0], cam_ts)
    # x equals t, y,z zero
    assert np.allclose(res[:,1], cam_ts)
    assert np.allclose(res[:,2], 0)
    assert np.allclose(res[:,3], 0)
    # quaternion identity
    assert np.allclose(res[:,4:], np.array([1,0,0,0]))

def test_invalid_odom_to_local_pose_args():
    with pytest.raises(ValueError):
        odom_to_local_pose(np.zeros((5,6)), mode='se3')
    with pytest.raises(ValueError):
        odom_to_local_pose(np.zeros((5,7)), mode='invalid')

