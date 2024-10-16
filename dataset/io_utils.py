import torch
import numpy as np
import open3d as o3d
import json
from .pose_utils import get_rotation_axis_angle

def load_ply_to_tensor(fname, color=False, device='cpu'):
    pcd = o3d.io.read_point_cloud(fname)
    pts = np.asarray(pcd.points)
    
    return torch.Tensor(pts).to(device)

def load_gt_from_json_revolute(meta, state):
    
    axis_o = np.array(meta['trans_info']['axis']['o'])
    axis_d = np.array(meta['trans_info']['axis']['d'])
    rotate_l = meta['trans_info']['rotate']['l']
    rotate_r = meta['trans_info']['rotate']['r']
    if state == 'start':
        angle_radian = (rotate_r - rotate_l)/180 * np.pi
    else:
        angle_radian = (rotate_l - rotate_r)/180 * np.pi
    transform_T = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]) # (x, y, z) -> (-y, x, z)
    axis_d_ngp = np.dot(transform_T, axis_d)
    axis_o_ngp = np.dot(transform_T, axis_o)

    gt_info = {
        "axis_o": torch.Tensor(axis_o_ngp),
        "axis_d": torch.Tensor(axis_d_ngp),
        "R": torch.Tensor(get_rotation_axis_angle(axis_d_ngp, angle_radian)),
        "theta": angle_radian,
        "dist": torch.Tensor([0]) 
    }
    return gt_info

def load_gt_from_json(fname, state, motion_type):
    meta = json.load(open(fname))
    if motion_type == 'r':
        return load_gt_from_json_revolute(meta, state)
    else:
        return load_gt_from_json_prismatic(meta, state)

def load_gt_from_json_prismatic(meta, state):
    # meta = json.load(open(fname))
    axis_o = np.array(meta['trans_info']['axis']['o'])
    axis_d = np.array(meta['trans_info']['axis']['d'])
    trans_l = meta['trans_info']['translate']['l']
    trans_r = meta['trans_info']['translate']['r']
    if state == 'start':
        scale = trans_r - trans_l
    else:
        scale = trans_l - trans_r
    transform_T = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]) # (x, y, z) -> (-y, x, z)
    axis_d_ngp = np.dot(transform_T, axis_d)
    axis_o_ngp = np.dot(transform_T, axis_o)
    gt_info = {
        "dist": torch.Tensor([scale]).view(1),
        "axis_o": torch.Tensor(axis_o_ngp),
        "axis_d": torch.Tensor(axis_d_ngp),
        "R": torch.eye(3),
        "theta": torch.Tensor([0])
    }
    return gt_info

def load_multipart_gt(fname, state, motion_type):
    with open(fname, 'r') as f:
        motion_meta = json.load(f)
    info_list = []
    for i in motion_meta:
        if motion_type == 'r':
            info  = load_gt_from_json_revolute(i, state=state)
        else:
            info = load_gt_from_json_prismatic(i, state=state)
        info_list += [info]
    return info_list