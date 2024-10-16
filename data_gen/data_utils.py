import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
import math
import random
import matplotlib.pyplot as plt
from pathlib import Path as P
import json
from tqdm import tqdm
# camera position coordinate: https://sapien.ucsd.edu/docs/2.2/tutorial/basic/hello_world.html, check the viewer section

# openGL coordinate definition: https://medium.com/@christophkrautz/what-are-the-coordinates-225f1ec0dd78

# from camera position coordinate to openGL: x = -y, y = z, z = -x

conversion_matrix = np.array([
    [0, -1, 0], 
    [0, 0, 1], 
    [-1, 0, 0]
])

def min_max_depth(depth):
    max_depth = depth.max()
    min_depth = depth[depth>0].min()
    return min_depth, max_depth

def model_rot_cvt_trans(camera):
    model_mat = camera.get_model_matrix()
    model_trans = model_mat[:3, -1:]

    # from forward(x), left(y) and up(z), to right(x), up(y), backwards(z)
    cvt_matrix_3x3 = np.array([
        [0, -1, 0],  # left(y) -> right(x)
        [0, 0, 1], # up(z) -> up(y)
        [-1, 0, 0] # forward(x) -> backward(z)
    ])
    new_trans = np.dot(cvt_matrix_3x3, model_trans)
    model_mat[:3, -1:] = new_trans
    return model_mat

def calculate_pose_openGL(translation):
    """
    recalculate the rotation matrix for camera extrinsic, camera is facing the origin
    input
        @param translation: object position given in viwer coordinate, row vector
        
    """
    trans_gl = np.dot(conversion_matrix, translation.T) # permute
    forward = -trans_gl / np.linalg.norm(trans_gl)
    right = np.cross([0, 1, 0], forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([right, up, forward], axis=1)
    mat44[:3, 3] = trans_gl
    return mat44

# def calculate_pose_openGL_reverse(mat44):
#     """
#     recalculate the rotation matrix for camera extrinsic, camera is facing the origin
#     input
#         @param translation: object position given in viwer coordinate, row vector
        
#     """
#     trans_gl = np.dot(conversion_matrix, translation.T) # permute
#     forward = -trans_gl / np.linalg.norm(trans_gl)
#     right = np.cross([0, 1, 0], forward)
#     right = right / np.linalg.norm(right)
#     up = np.cross(forward, right)
#     mat44 = np.eye(4)
#     mat44[:3, :3] = np.stack([right, up, forward], axis=1)
#     mat44[:3, 3] = trans_gl
#     return mat44

def custom_openGL(camera):
    model_mat = camera.pose.to_transformation_matrix()
    model_trans = model_mat[:3, -1:]
    return calculate_pose_openGL(model_trans.reshape(-1))

def random_point_in_sphere(radius, theta_range=[0, 2*math.pi], phi_range=[0, math.pi]):
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
    
    return x, y, z

def point_in_sphere(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return x, y, z

def get_depth(camera):
    position = camera.get_float_texture('Position')  # [H, W, 4]
    # Depth
    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    return depth_pil

def get_joint_type(asset) -> list:
    joints = asset.get_joints()
    j_type = []
    for joint in joints:
        if joint.get_dof() != 0:
            j_type += [joint.type[0]] * joint.get_dof()
    return j_type

def get_joint_limit(asset):
    joints = asset.get_joints()
    j_limits = []
    for joint in joints:
        if joint.get_dof() != 0:
            j_limits += [joint.get_limits()] 
    return np.array(j_limits) # should be Nx2

def calculate_cam_ext(point):
    cam_pos = np.array(point)
    # def update_cam_pose(cam_pos):
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44

def render_img(point, save_path, camera_mount_actor, scene, camera, asset, q_pos=None, pose_fn=None, save=True):
    mat44 = calculate_cam_ext(point)
    if camera_mount_actor is None:
        camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    else:
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    if q_pos is not None:
        asset.set_qpos(q_pos)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    rgba = camera.get_float_texture('Color')  # [H, W, 4]
        
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    

    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    
    rgba_pil = Image.fromarray(rgba_img, 'RGBA')
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    label0_pil = Image.fromarray(color_palette[label0_image])
    label1_pil = Image.fromarray(color_palette[label1_image])
    label2_pil = Image.fromarray(label1_image)
    camera_pose = camera.get_pose()
    qpos = asset.get_qpos()
    if pose_fn is not None:
        save_pose = pose_fn(camera).tolist()
    else:
        save_pose = camera.get_extrinsic_matrix().tolist()
    model_mat = camera.get_model_matrix()
    cv_ext = camera.get_extrinsic_matrix()
    meta_dict={
        "pose": save_pose,
        "ext_pose": camera.get_extrinsic_matrix().tolist(),
        "model_mat": model_mat.tolist(),
        "qpos": qpos.tolist(),
        "joint_type": get_joint_type(asset),
        "cam_param": camera.get_intrinsic_matrix().tolist()}
        
    depth_pil = get_depth(camera)
    min_d, max_d = min_max_depth(np.array(depth_pil))
    if save:
        depth_pil.save(str(save_path/'depth.png'))
        label0_pil.save(str(save_path/'label0.png'))
        label1_pil.save(str(save_path/'label1.png'))
        label2_pil.save(str(save_path/'label_actor.png'))
        rgba_pil.save(str(save_path/'color.png'))
        json_fname = str(save_path/'meta.json')
        with open(json_fname, 'w') as f:
            json.dump(meta_dict, f)
    ret_dict = {
        'rgba': rgba_pil,
        'depth': depth_pil,
        'label_0': label0_pil,
        'label_1': label1_pil,
        'label_actor': label2_pil,
        'meta': meta_dict,
        'min_d': min_d,
        'max_d': max_d,
        'mat44': mat44
    }
    return ret_dict

def gen_articulated_object_nerf_s1(num_pos_img, radius_, split, camera, asset, scene, object_path, camera_mount_actor=None, theta_range = [0*math.pi, 2*math.pi], phi_range = [0*math.pi, 1*math.pi], render_pose_file_dir = None, with_seg=False):
    if split is not None:
        save_base_path = object_path / split
    else:
        save_base_path = object_path
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    if with_seg:
        save_seg_path = save_base_path / 'seg'
        save_seg_path.mkdir(exist_ok=True)
    else:
        save_seg_path = None
    render_pose_dict = {}
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy,
        "fy": camera.fy,
        "fx": camera.fx
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf
    for i in tqdm(range(num_pos_img)):
        instance_save_path = None
        point = random_point_in_sphere(radius=radius_, theta_range=theta_range, phi_range=phi_range)
        # point = points[i]
        ret_dict = render_img(point, instance_save_path, camera_mount_actor, scene, camera, asset, pose_fn=custom_openGL, save=False)
        frame_id = 'r_'+str(i)
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        
        render_pose = ret_dict['mat44']
        render_pose_dict[frame_id] = render_pose.tolist()
        
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname))   
        
        depth_fname = save_depth_path / ('depth' + str(i) + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))

        if with_seg:
            fname = frame_id + '.png'
            ret_dict['label_actor'].save(str(save_seg_path / fname))
        
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']  
    print('min_d = ', min_d)
    print('max_d = ', max_d)

    transform_json['frames'] = frame_dict
    transform_fname = str(save_base_path / 'transforms.json')
    if render_pose_file_dir is not None:
        P(render_pose_file_dir).mkdir(parents=True, exist_ok=True)
        render_pose_fname = P(render_pose_file_dir) / (split + '.json')
        with open(render_pose_fname, 'w') as f:
            json.dump(render_pose_dict, f)
            
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    pass

def gen_articulated_object_nerf_s2(num_pos_img, radius_, split, camera, asset, scene, object_path, \
                                   camera_mount_actor=None, theta_range = [0*math.pi, 2*math.pi], \
                                    phi_range = [0*math.pi, 1*math.pi], render_pose_file_dir = None, q_pos=None):
    
    with_seg=True
    dof = asset.dof
    if q_pos is not None:
        asset.set_qpos(q_pos)
    elif dof > 1:
        q_pos = [-np.inf] * dof
        q_pos[0] = np.inf
        asset.set_qpos(q_pos)
    else:
        asset.set_qpos([np.inf] * dof)
    if split is not None:
        save_base_path = object_path / split
    else:
        save_base_path = object_path
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    
    if with_seg:
        save_seg_path = save_base_path / 'seg'
        save_seg_path.mkdir(exist_ok=True)
    else:
        save_seg_path = None
    render_pose_dict = {}
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf
    for i in tqdm(range(num_pos_img)):
        instance_save_path = None
        point = random_point_in_sphere(radius=radius_, theta_range=theta_range, phi_range=phi_range)
        # point = points[i]
        ret_dict = render_img(point, instance_save_path, camera_mount_actor, scene, camera, asset, pose_fn=custom_openGL, save=False)
        frame_id = 'r_'+str(i)
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        
        render_pose = ret_dict['mat44']
        render_pose_dict[frame_id] = render_pose.tolist()
        
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname))   
        
        depth_fname = save_depth_path / ('depth' + str(i) + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))

        if with_seg:
            fname = frame_id + '.png'
            ret_dict['label_actor'].save(str(save_seg_path / fname))
        
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']
    print('min_d = ', min_d)
    print('max_d = ', max_d)

    transform_json['frames'] = frame_dict

    joint_types = [joint.type for joint in asset.get_joints()]
    transform_json['joint_types'] = joint_types
    transform_json['qpos'] = asset.get_qpos().tolist()

    transform_fname = str(save_base_path / 'transforms.json')
    if render_pose_file_dir is not None:
        P(render_pose_file_dir).mkdir(parents=True, exist_ok=True)
        render_pose_fname = P(render_pose_file_dir) / (split + '.json')
        with open(render_pose_fname, 'w') as f:
            json.dump(render_pose_dict, f)
            
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    pass

def generate_articulation_test(num_pos_img, radius_, split, camera, asset, scene, object_path, \
                                   camera_mount_actor=None, theta_range = [0*math.pi, 2*math.pi], \
                                    phi_range = [0*math.pi, 1*math.pi], render_pose_file_dir = None):
    with_seg=True
    if split is not None:
        save_base_path = object_path / split
    else:
        save_base_path = object_path
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    
    if with_seg:
        save_seg_path = save_base_path / 'seg'
        save_seg_path.mkdir(exist_ok=True)
    else:
        save_seg_path = None
    render_pose_dict = {}
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf

    dof = asset.dof
    qlimits = asset.get_qlimits()
    qrange = qlimits[:, 1] - qlimits[:, 0]
    qpos_list = np.random.randn(num_pos_img, dof)
    qpos_list = qpos_list * qrange - qlimits[:,0]


    for i in tqdm(range(num_pos_img)):
        instance_save_path = None
        point = random_point_in_sphere(radius=radius_, theta_range=theta_range, phi_range=phi_range)
        asset.set_qpos(qpos_list[i])
        # point = points[i]
        ret_dict = render_img(point, instance_save_path, camera_mount_actor, scene, camera, asset, pose_fn=custom_openGL, save=False)
        frame_id = 'r_'+str(i)
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        
        render_pose = ret_dict['mat44']
        render_pose_dict[frame_id] = render_pose.tolist()
        
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname))   
        
        depth_fname = save_depth_path / ('depth' + str(i) + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))

        if with_seg:
            fname = frame_id + '.png'
            ret_dict['label_actor'].save(str(save_seg_path / fname))
        
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']
    print('min_d = ', min_d)
    print('max_d = ', max_d)

    transform_json['frames'] = frame_dict
    
    joint_types = [joint.type for joint in asset.get_joints()]
    transform_json['joint_types'] = joint_types
    transform_json['qpos'] = qpos_list.tolist()
    transform_fname = str(save_base_path / 'transforms.json')
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    pass


def generate_img_with_pose(pose_dir, split, camera, asset, scene, object_path, camera_mount_actor=None, qpos=0):
    
    if split is not None:
        save_base_path = object_path / split
    else:
        save_base_path = object_path
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf
    # load camera pose
    pose_fname = P(pose_dir) / (split + '.json')
    asset.set_qpos(np.array(qpos))
    print('generating images from saved pose file: ', pose_fname)
    render_pose = json.load(open(str(pose_fname)))
    for frame_id in tqdm(render_pose.keys()):
        img_pose = np.array(render_pose[frame_id])
        
        ret_dict = render_img_with_pose(img_pose, None, camera_mount_actor, scene, camera, asset, save=False)
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname)) 
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        depth_fname = save_depth_path / ('depth' + frame_id[2:] + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']  
    print('min_d = ', min_d)
    print('max_d = ', max_d)
    transform_json['frames'] = frame_dict
    transform_fname = str(save_base_path / 'transforms.json')
            
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    
    pass

def render_img_with_pose(pose, save_path, camera_mount_actor, scene, camera, asset, q_pos=None, pose_fn=None, save=True):
    mat44 = pose
    if camera_mount_actor is None:
        camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    else:
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    if q_pos is not None:
        asset.set_qpos(q_pos)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    rgba = camera.get_float_texture('Color')  # [H, W, 4]
        
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    

    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    
    rgba_pil = Image.fromarray(rgba_img, 'RGBA')
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    label0_pil = Image.fromarray(color_palette[label0_image])
    label1_pil = Image.fromarray(color_palette[label1_image])
    label2_pil = Image.fromarray(label1_image)
    camera_pose = camera.get_pose()
    qpos = asset.get_qpos()
    if pose_fn is not None:
        save_pose = pose_fn(camera).tolist()
    else:
        save_pose = camera.get_extrinsic_matrix().tolist()
    model_mat = camera.get_model_matrix()
    cv_ext = camera.get_extrinsic_matrix()
    meta_dict={
        "pose": save_pose,
        "ext_pose": camera.get_extrinsic_matrix().tolist(),
        "model_mat": model_mat.tolist(),
        "qpos": qpos.tolist(),
        "joint_type": get_joint_type(asset),
        "cam_param": camera.get_intrinsic_matrix().tolist()}
        
    depth_pil = get_depth(camera)
    min_d, max_d = min_max_depth(np.array(depth_pil))
    if save:
        depth_pil.save(str(save_path/'depth.png'))
        label0_pil.save(str(save_path/'label0.png'))
        label1_pil.save(str(save_path/'label1.png'))
        label2_pil.save(str(save_path/'label_actor.png'))
        rgba_pil.save(str(save_path/'color.png'))
        json_fname = str(save_path/'meta.json')
        with open(json_fname, 'w') as f:
            json.dump(meta_dict, f)
    ret_dict = {
        'rgba': rgba_pil,
        'depth': depth_pil,
        'label_0': label0_pil,
        'label_1': label1_pil,
        'label_actor': label2_pil,
        'meta': meta_dict,
        'min_d': min_d,
        'max_d': max_d,
        'mat44': mat44
    }
    return ret_dict


def generate_art_imgs(output_dir, split, index_list, scene_dict, num_imgs, radius=5, pose_dict=None, reuse_pose=False):
    """
    Generate and save files into the specified directory structure for a single split.

    Args:
        output_dir (str): The root directory where the structure will be created.
        split (str): The name of the split (e.g., 'train', 'val', 'test').
        index_list (list): List of folder names for each index.
        scene_dict (dict): ditc{
            "scene": scene,
            "camera": camera,
            "asset": asset,
            "pose": pose,
            "q_pos": qpos,
            "save_path": save_path,
            "save": save,
            "pose_fn": pose_fn,
            "camera_mount_actor": None
        }
        num_imgs (int): number of images to save for each articulation index (configuration)
        pose_dict (dict): default None
        reuse_pose (bool): if true, return pose used here.
    Returns:
        None or pose_dict
    """
    output_path = P(output_dir)

    # Create the root output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the directory for the specified split
    split_dir = output_path / split
    split_dir.mkdir(exist_ok=True)
    
    pose_dict = {}
    
    for i in range(num_imgs):
        frame_id = 'r_' + str(i)
        point = random_point_in_sphere(radius=radius)
        mat44 = calculate_cam_ext(point)
        pose_dict[frame_id] = mat44

    # Iterate through each index folder (e.g., 'idx0_folder', 'idx1_folder', ...)
    for index in tqdm(index_list):
        if index == 0:
            # skip the 0 articulation
            continue
        index_folder = f'idx_{index}'
        index_folder_dir = split_dir / index_folder
        index_folder_dir.mkdir(exist_ok=True)
        q_limits = get_joint_limit(scene_dict['asset']).reshape([-1, 2])
        if split == 'train':
            q_ratio = index/max(index_list)
        else:
            q_ratio = (index-0.5)/max(index_list) 
        q_ratio = index/max(index_list)
        cur_qpos = q_limits[:, 0] + q_ratio * (q_limits[:, 1] - q_limits[:, 0])
        
        scene_dict['q_pos'] = cur_qpos
        transforms_data = {
        }
        # qpos_dict[frame_id] = cur_qpos
        # Create subdirectories for 'rgb', 'depth', 'seg'
        for subfolder in ['rgb', 'depth', 'seg']:
            subfolder_dir = index_folder_dir / subfolder
            subfolder_dir.mkdir(exist_ok=True)

        for img_idx in range(num_imgs):
            frame_id = 'r_' + str(img_idx)
            fname = frame_id + '.png'
            if pose_dict is None:
                if not reuse_pose:
                    point = random_point_in_sphere(radius=radius)
                    mat44 = calculate_cam_ext(point)
                    scene_dict['pose'] = mat44
                else:
                    scene_dict['pose'] = pose_dict[frame_id]
            else:
                scene_dict['pose'] = np.array(pose_dict[frame_id])
            ret_dict = render_img_with_pose(**scene_dict)
            ret_dict['rgba'].save(str(index_folder_dir / 'rgb' / fname))
            ret_dict['depth'].save(str(index_folder_dir / 'depth' / fname))
            ret_dict['label_actor'].save(str(index_folder_dir / 'seg' / fname))
            transforms_data[frame_id] = scene_dict['camera'].get_model_matrix().tolist()
        # Create and save 'transforms.json' file
        transforms_file = index_folder_dir / 'transforms.json'
        focal = (camera.fx + camera.fy) / 2
        save_dict = {}
        save_dict['frame'] = transforms_data
        save_dict['focal'] = focal
        joints_type = get_joint_type(asset)
        save_dict['j_type'] = joints_type
        save_dict['qpos'] = cur_qpos.tolist()
        with transforms_file.open('w') as json_file:
            json.dump(save_dict, json_file)
    return pose_dict

def scene_setup(urdf_file, h=480, w=640, n=0.1, f=100):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    asset = loader.load_kinematic(str(urdf_file))
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = n, f
    width, height = w, h
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    return camera, asset, scene
    
    
def gen_articulated_object_paris(num_pos_img, radius_, split, camera, asset, scene, object_path, \
                                   camera_mount_actor=None, theta_range = [0*math.pi, 2*math.pi], \
                                    phi_range = [0*math.pi, 1*math.pi], state='start'):
    
    with_seg=True
    
    state_path = object_path / state
    state_path.mkdir(exist_ok=True)
    
    save_base_path = state_path / split
        
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path 
    
    
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    
    if with_seg:
        save_seg_path = save_base_path / 'seg'
        save_seg_path.mkdir(exist_ok=True)
    else:
        save_seg_path = None
    render_pose_dict = {}
    
    camera_K = camera.get_intrinsic_matrix()
    
    frame_dict = {
        'K': camera_K.tolist()
    }
    max_d = 0
    min_d = np.inf
    for i in tqdm(range(num_pos_img)):
        instance_save_path = None
        point = random_point_in_sphere(radius=radius_, theta_range=theta_range, phi_range=phi_range)
        # point = points[i]
        ret_dict = render_img(point, instance_save_path, camera_mount_actor, scene, camera, asset, pose_fn=custom_openGL, save=False)
        frame_id = str(i).zfill(4)
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        
        render_pose = ret_dict['mat44']
        render_pose_dict[frame_id] = render_pose.tolist()
        
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname))   
        
        depth_fname = save_depth_path / ('depth' + str(i) + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))

        if with_seg:
            fname = frame_id + '.png'
            ret_dict['label_actor'].save(str(save_seg_path / fname))
        
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']
    print('min_d = ', min_d)
    print('max_d = ', max_d)

    transform_json = frame_dict

    transform_fname = str(state_path / ('camera_' + split + '.json'))
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    pass

def gen_paris_art(object_name, object_id, op_path, motion, camera, scene, asset, radius, json_fname, joint_idx, motion_idx, motion_type='r', only_motion_json=False):
    img_output_path = P(op_path) / 'load' / 'sapien' / object_name / object_id
    img_output_path.mkdir(exist_ok=True, parents=True)
    
    
    train_args_dict = {
        "num_pos_img": 100,
        "radius_": radius,
        "split": "train",
        "camera": camera,
        "asset": asset,
        "scene": scene,
        "object_path": img_output_path,
        "phi_range": [0.1*math.pi, 0.55*math.pi]
    }
    val_args_dict = {
        "num_pos_img": 20,
        "radius_": radius,
        "split": "val",
        "camera": camera,
        "asset": asset,
        "scene": scene,
        "object_path": img_output_path,
        "phi_range": [0.1*math.pi, 0.55*math.pi]
    }
    test_args_dict = {
        "num_pos_img": 50,
        "radius_": radius,
        "split": "test",
        "camera": camera,
        "asset": asset,
        "scene": scene,
        "object_path": img_output_path,
        "phi_range": [0.1*math.pi, 0.55*math.pi]
    }
    if not only_motion_json:
        if motion_type == 'r':
            asset.set_qpos(motion[0]/180*np.pi)
        else:
            asset.set_qpos(motion[0])
        gen_articulated_object_paris(**train_args_dict, state='start')
        gen_articulated_object_paris(**test_args_dict, state='start')
        gen_articulated_object_paris(**val_args_dict, state='start')
        start_qpos = asset.get_qpos()
        print(start_qpos)
        if motion_type == 'r':
            
            asset.set_qpos(motion[1]/180*np.pi)
        else:
            asset.set_qpos(motion[1])
        gen_articulated_object_paris(**train_args_dict, state='end')
        gen_articulated_object_paris(**test_args_dict, state='end')
        gen_articulated_object_paris(**val_args_dict, state='end')
        end_qpos = asset.get_qpos()
        print(end_qpos)
    with open(str(json_fname), 'r') as f:
        obj_meta = json.load(f)
        
    joint_info_list = []
    for idx, m_idx in enumerate(joint_idx):
        m_dict = obj_meta[m_idx]
        if "axis" in m_dict['jointData'].keys():
            cur_dict = {}
            if m_dict['joint'] == 'hinge':
                motion_type = 'revolute'
            else:
                motion_type = 'prismatic'
            
            R_coord = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            json_o = np.array(m_dict['jointData']['axis']['origin'])
            json_d = np.array(m_dict['jointData']['axis']['direction'])
            axis_o = np.matmul(R_coord, json_o).tolist()
            # if motion_type == 'revolute':
            axis_d = np.matmul(R_coord, json_d).tolist()
            # else:
            #     axis_d = json_d.tolist()
            cur_dict['trans_info'] = {
                "axis":{
                    "o": axis_o,
                    "d": axis_d
                },
                "limit":{
                    "a": m_dict['jointData']['limit']['a'],
                    "b": m_dict['jointData']['limit']['b']
                },
                "rotate":{
                    "l": float(motion[0][motion_idx[idx]]),
                    "r": float(motion[1][motion_idx[idx]])
                },
                "translate":{
                    "l": float(motion[0][motion_idx[idx]]),
                    "r": float(motion[1][motion_idx[idx]])
                },
                "motion_type": motion_type
            }
            joint_info_list += [cur_dict]
    
    
    # data_paris_path = P("data/load/glasses/102612/textured_objs")
    data_paris_path = P(op_path) / 'data' / 'sapien' / object_name / object_id / 'textured_objs'
    data_paris_path.mkdir(exist_ok=True, parents=True)
    save_json_name = data_paris_path / 'trans.json'
    with open(str(save_json_name), 'w') as f:
        json.dump(joint_info_list, f)
    pass
    
if __name__ == '__main__':
    # test articulation image generation
    base_path = "/home/s2262444/codes/articulate_object/articulated-object-nerf"
    urdf_file = base_path + "/data_base/laptop/10211/mobility.urdf"
    output_dir = base_path + "/data/laptop_art_same_pose"
    camera, asset, scene = scene_setup(urdf_file)
    qpos_list = np.arange(10)
    scene_dict = {
            "scene": scene,
            "camera": camera,
            "asset": asset,
            "pose": None,
            "q_pos": None,
            "save_path": None,
            "save": None,
            "pose_fn": None,
            "camera_mount_actor": None
        }
    
    splits = ('train', 'val')
    generate_art_imgs(output_dir, 'train', qpos_list, scene_dict, 10, reuse_pose=True)
    generate_art_imgs(output_dir, 'val', qpos_list, scene_dict, 10)