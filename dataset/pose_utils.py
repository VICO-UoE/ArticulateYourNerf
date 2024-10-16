import numpy as np
import math
import random
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

def look_at(eye, target=np.array([0., 0., 0.]), up=np.array([0., 0., -1.])):
    # Calculate the forward, right, and up vectors
    forward = (target - eye)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    # Construct the rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = new_up
    rotation_matrix[:3, 2] = forward  # Negate forward to make it look towards the target
    rotation_matrix[:3, 3] = eye
    # print(forward)
    # Homogeneous coordinates
    rotation_matrix[3, 3] = 1.0

    return rotation_matrix

def generate_extrinsic_params(point):
   """
   Generate OpenGL camera extrinsic parameters given a point.
   The camera is assumed to be positioned at the given point and facing the origin (0, 0, 0).

   Parameters:
   point (tuple or list): The coordinates of the point where the camera is positioned.

   Returns:
   np.array: The 4x4 transformation matrix representing the camera extrinsic parameters.
   """
   # Convert the point to a numpy array
   point = np.array(point)

   # Define the camera orientation
   # The camera is facing the negative z-axis, and the up direction is the positive y-axis
  
   # rotation_matrix = calculate_rotation_matrix(point)
   rotation_matrix = look_at(point)[:3, :3]

   # Create the transformation matrix
   T = np.eye(4)
   T[:3, :3] = rotation_matrix
   T[:3, 3] = point

   return T

def get_sapien_pose(cam_pos):
    '''
    NGP coordinate: 

    right, down, front
    '''
   # Compute the camera pose by specifying forward(x), left(y) and up(z)
    # cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44

def pose2view(pos):
  column_swap = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0]
    ]) 
  pos_R = pos[:3, :3]
  r = np.matmul(pos_R, column_swap)
  pos[:3, :3] = r
  return pos

def random_point_in_sphere(radius, theta_range=[0, 2*math.pi], phi_range=[0, math.pi]):
    '''
    return np.array()
    '''
    # Generate random spherical coordinates
    theta_low, theta_high = theta_range
    phi_low, phi_high = phi_range
    
    theta = random.uniform(theta_low, theta_high)       # Azimuthal angle
    phi = random.uniform(phi_low, phi_high)            # Polar angle
    r = random.uniform(radius-0.5, radius+0.5)      # Radial distance
    
    # Convert spherical coordinates to Cartesian coordinates
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return np.array([x, y, z])

def radius_to_pose(radius):
    point = random_point_in_sphere(radius=radius)
    ngp_pose = generate_extrinsic_params(point)
    return ngp_pose

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    axis = quaternions[..., 1:] / sin_half_angles_over_angles
    axis = F.normalize(axis, p=2., dim=0)
    deg = angles * 180 / torch.pi
    return axis, angles, deg



def normalize(v):
    return v / np.sqrt(np.sum(v**2))

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    k = normalize(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def get_quaternion_axis_angle(k, theta):
    rot_m = get_rotation_axis_angle(k, theta)
    rot = R.from_matrix(rot_m)
    q = rot.as_quat()
    # change to w, x, y, z
    q_ret = np.zeros_like(q)
    q_ret[0] = q[-1]
    q_ret[1:] = q[:3]
    return q_ret