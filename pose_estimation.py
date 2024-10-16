import numpy as np
import sys
import torch
import json
from dataset.io_utils import load_ply_to_tensor
from dataset.ray_utils import transfrom_to_NGP
import cv2
from pathlib import Path as P
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvf
import torch.nn as nn
import torch.nn.functional as F
import random
from pytorch3d.loss import chamfer_distance
import open3d as o3d
from config import get_opts
from tqdm import tqdm
from dataset.ray_utils import camera_matrix_scaling
import torchvision
import torchvision.transforms as tvT
from pytorch3d.ops import knn_points

def find_unique_closest_indices(data_tensor, threshold_tensor):
    # Prepare a tensor to store unique closest indices
    closest_indices_unique = torch.zeros(threshold_tensor.shape[0], dtype=torch.long)

    for threshold_index, threshold_value in enumerate(threshold_tensor):
        difference = torch.abs(data_tensor - threshold_value)

        for _ in range(data_tensor.shape[0]):
            # Find the index with the smallest difference
            _, smallest_diff_index = torch.min(difference, dim=0)
            # Check if this index already exists in closest_indices_unique
            if smallest_diff_index.item() in closest_indices_unique:
                # If it already exists, set the difference to a high value and redo the iteration
                difference[smallest_diff_index] = float('inf')
            else:
                # Store unique closest index
                closest_indices_unique[threshold_index] = smallest_diff_index
                # Break inner loop and continue with the next value in the threshold_tensor
                break
                
    return closest_indices_unique

class PoseModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        init_Q = torch.Tensor([1, 0, 0, 0])
        axis_origin = torch.Tensor([ 0, 0, 0 ])
        self.Q = nn.Parameter(init_Q, requires_grad=True)
        self.axis_origin = nn.Parameter(axis_origin, requires_grad=True)
        # gt stapler_103111
        # origin: [0, 0.752066445189761, 0.10500870623151695]
        # Q: [ 0.93937271, -0.34289781,  0.        ,  0.        ]
        # gt laptop_10211
        # origin:
        # Q: 

    @torch.no_grad()
    def norm_Q(self):
        Q_norm = F.normalize(self.Q, p=2., dim=0)
        self.Q = nn.Parameter(Q_norm, requires_grad=True)
        
    def init_param(self):
        init_Q = torch.Tensor([1, 0, 0, 0])
        axis_origin = torch.Tensor([ 0, 0, 0 ])
        self.Q = nn.Parameter(init_Q, requires_grad=True)
        self.axis_origin = nn.Parameter(axis_origin, requires_grad=True)
    # def forward(self)
    
    def R_from_quaternions(self):
        '''
        w, x, y, z
        '''
        quaternions = F.normalize(self.Q, p=2., dim=0)

        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)

    def get_transform(self):
        R = self.R_from_quaternions()
        tr = torch.eye(4).to(self.Q)
        tr[:3, :3] = R
        tr[:3, -1] = self.axis_origin
        return tr

    def construct_transform(self):
        tr = self.get_transform()
        R = tr[:3, :3]
        t = tr[:3, -1]
        # Create 4x4 identity matrices for base transformations
        T_to_origin = torch.eye(4, device=t.device)
        T_back_to_t = torch.eye(4, device=t.device)

        # Construct the transformation matrices for translation
        T_to_origin[:3, 3] = -t.squeeze()   # Translation by -t
        T_back_to_t[:3, 3] = t.squeeze()  # Translation by t

        # Construct the 4x4 rotation matrix R
        R_4x4 = torch.eye(4, device=R.device)  
        R_4x4[:3, :3] = R  # Add 3x3 R to top-left of 4x4 matrix

        # Combine the transformations
        T = T_back_to_t @ R_4x4 @ T_to_origin

        return T
    
    def transform_c2w(self, c2w, use_inverse=False):
        T = self.construct_transform()
        if use_inverse:
            T = torch.linalg.inv(T)
        if len(c2w.shape) == 3:
            # batched
            T = T.unsqueeze(0).repeat([c2w.shape[0], 1, 1])
            return torch.bmm(T, c2w)
        else:
            
            return T @ c2w

class PoseModule_prismatic(PoseModule):
    
    def __init__(self) -> None:
        init_dir = torch.Tensor([1, 0, 0])
        init_scale = torch.Tensor([0])
        self.dir = nn.Parameter(init_dir, requires_grad=True)
        self.scale = nn.Parameter(init_scale, requires_grad=True)
        
    def construct_transform(self):
        norm_dir = torch.nn.functional.normalize(self.dir)
        translation = self.scale * norm_dir
        transform =torch.eye(4).to(translation)
        transform[:3, -1] = translation.view(3)
        return transform


class PoseModule_se3(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        init_Q = torch.Tensor([1, 0, 0, 0])
        axis_origin = torch.Tensor([ 0, 0, 0 ])
        self.Q = nn.Parameter(init_Q, requires_grad=True)
        self.axis_origin = nn.Parameter(axis_origin, requires_grad=True)
        init_dir = torch.Tensor([1, 0, 0])
        init_scale = torch.Tensor([0])
        self.dir = nn.Parameter(init_dir, requires_grad=True)
        self.scale = nn.Parameter(init_scale, requires_grad=True)
        # init_trans = torch.Tensor([0, 0, 0])
        # self.trans = nn.Parameter(init_trans, requires_grad=True)
        
        
        # gt stapler_103111
        # origin: [0, 0.752066445189761, 0.10500870623151695]
        # Q: [ 0.93937271, -0.34289781,  0.        ,  0.        ]
        # gt laptop_10211
        # origin:
        # Q: 

    @torch.no_grad()
    def norm_Q(self):
        Q_norm = F.normalize(self.Q, p=2., dim=0)
        self.Q = nn.Parameter(Q_norm, requires_grad=True)
        
    def norm_dir(self):
        dir_norm = F.normalize(self.dir.view(1, -1), dim=1)
        self.dir = nn.Parameter(dir_norm.view(-1), requires_grad=True)
        
    def init_param(self):
        device = self.Q.device
        init_Q = torch.Tensor([1, 0, 0, 0]).to(device)
        axis_origin = torch.Tensor([ 0, 0, 0 ]).to(device)
        self.Q = nn.Parameter(init_Q, requires_grad=True)
        self.axis_origin = nn.Parameter(axis_origin, requires_grad=True)
        init_dir = torch.Tensor([1, 0, 0]).to(device)
        init_scale = torch.Tensor([0]).to(device)
        self.dir = nn.Parameter(init_dir, requires_grad=True)
        self.scale = nn.Parameter(init_scale, requires_grad=True)
    
        # init_trans = torch.Tensor([0, 0, 0])
        # self.trans = nn.Parameter(init_trans, requires_grad=True)
    
    def R_from_quaternions(self):
        '''
        w, x, y, z
        '''
        quaternions = F.normalize(self.Q, p=2., dim=0)

        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)

    def get_transform(self):
        R = self.R_from_quaternions()
        tr = torch.eye(4).to(self.Q)
        tr[:3, :3] = R
        tr[:3, -1] = self.axis_origin
        return tr

    def construct_transform_revolute(self):
        tr = self.get_transform()
        R = tr[:3, :3]
        t = tr[:3, -1]
        # Create 4x4 identity matrices for base transformations
        T_to_origin = torch.eye(4, device=t.device)
        T_back_to_t = torch.eye(4, device=t.device)

        # Construct the transformation matrices for translation
        T_to_origin[:3, 3] = -t.squeeze()   # Translation by -t
        T_back_to_t[:3, 3] = t.squeeze()  # Translation by t

        # Construct the 4x4 rotation matrix R
        R_4x4 = torch.eye(4, device=R.device)  
        R_4x4[:3, :3] = R  # Add 3x3 R to top-left of 4x4 matrix

        # Combine the transformations
        T = T_back_to_t @ R_4x4 @ T_to_origin

        return T
    
    def construct_transform_prismatic(self):
        norm_dir = torch.nn.functional.normalize(self.dir.view(1, -1))
        translation = self.scale * norm_dir
        transform =torch.eye(4).to(translation)
        transform[:3, -1] = translation.view(3)
        return transform
    
    def transform_c2w(self, c2w, use_inverse=False):
        T_revolute = self.construct_transform_revolute()
        T_primatic = self.construct_transform_prismatic()
        if use_inverse:
            T_revolute = torch.linalg.inv(T_revolute)
            T_primatic[:3, -1] *= -1
        if len(c2w.shape) == 3:
            # batched
            T_revolute = T_revolute.unsqueeze(0).repeat([c2w.shape[0], 1, 1])
            T_primatic = T_primatic.unsqueeze(0).repeat([c2w.shape[0], 1, 1])
            return torch.bmm(T_primatic, torch.bmm(T_revolute, c2w))
        else:
            
            return T_primatic @ T_revolute @ c2w

class CoarsePoseEstimator():
    def __init__(self, ref_pts_fname, renderer, dataset, root_dir, vsize=3/128, use_frame_num=100) -> None:
        # self.ref_3d = ref_3d_pts
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.K = renderer.dataset.K
        self.vsize = vsize
        # self.frames = frames
        self.renderer = renderer
        if type(root_dir) == str:
            self.root_dir = P(root_dir)
        self.use_frame_num = use_frame_num
        self.load_meta()
        self.dataset = dataset
        self.src_ref_3d_torch = self.load_src_ref_pts(ref_pts_fname).to(self.device)
        self.construct_learnable_pose()
        self.collect_ref_and_tgt_pts()
        # self.mark_part_in_src_ref()
        self.configure_optimizer()
        pass
    
    def load_meta(self):
        meta_fname = self.root_dir/'transforms.json'
        with open(str(meta_fname)) as f:
            self.meta = json.load(f)
        self.frames = self.meta['frames']
    
    def load_src_ref_pts(self, fname):
        pts_tensor = load_ply_to_tensor(fname)
        return pts_tensor
    
    def construct_learnable_pose(self):
        self.pose_param = PoseModule().cuda()
        
    def get_transform(self):
        R = self.R_from_quaternions()
        tr = torch.eye(4).to(self.pose_param.Q)
        tr[:3, :3] = R
        tr[:3, -1] = self.pose_param.axis_origin
        return tr
    
    def configure_optimizer(self):
        # param_dict = {
        #     "Q": self.pose_param.Q,
        #     "origin": self.pose_param.axis_origin
        # }
        self.optimizer = torch.optim.Adam(
            self.pose_param.parameters(),
            lr=1e-1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
    
    def collect_ref_and_tgt_pts(self):
        part_pts = []
        ref_pts = {}
        poses = []
        co_mask = []
        frame_cnt = 0
        for k, v in self.frames.items():
            test_pose = transfrom_to_NGP(np.asarray(v))
            render_batch = self.renderer.gen_img_given_pose(test_pose)
            
            pred_mask = render_batch['acc'] > (1 - 1e-2)
            pred_mask = pred_mask.squeeze(-1)
            name = k + '.png'
            fname = str(self.root_dir / 'rgb' / name)
            ref_img = tvf.pil_to_tensor(Image.open(fname))
            ref_mask = ref_img[-1].to(pred_mask)
            part_img = (pred_mask.float() - ref_mask.float()) * pred_mask.float()
            cur_part_pts = render_batch['raw_pts'][part_img == 1]
            part_pts += [cur_part_pts]
            cur_ref_pts = self.get_pts_from_mask(ref_mask).to(self.device) #[K, 2]
            ref_pts[k] = cur_ref_pts
            poses += [torch.Tensor(test_pose).to(self.device)]
            co_mask += [ref_mask.float() * pred_mask.float()]
            
            # choose how many frames to use
            frame_cnt += 1
            if frame_cnt >= self.use_frame_num:
                break
            
        self.part_pts = torch.cat(part_pts, dim=0).to(self.device)
        self.ref_pts = ref_pts
        self.poses = poses
        self.co_mask = co_mask
        
    @staticmethod
    def get_pts_from_mask(mask):
        y_coords, x_coords = torch.where(mask == 1)
        pts = torch.stack([x_coords, y_coords]).T
        return pts
    
    def mark_part_in_src_ref(self):
        geo_dist = torch.cdist(self.src_ref_3d_torch, self.part_pts)
        min_geo_dist, _ = geo_dist.min(dim=-1, keepdim=True)
        part_ref_pts_mask = min_geo_dist < (self.vsize/2)
        part_ref_pts = self.src_ref_3d_torch[part_ref_pts_mask.view(-1)]
        
        # filter point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_ref_pts.cpu().float().numpy())
        o3d.io.write_point_cloud('part_stapler_paris.ply', pcd)
        cl = pcd.remove_radius_outlier(nb_points=3, radius=1/32)
        cl = cl[0].remove_radius_outlier(nb_points=2, radius=1/32)
        cl = cl[0].remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        
        # generate final part point mask
        final_part_ref_pts = torch.Tensor(np.asarray(cl[0].points)).to(self.src_ref_3d_torch)
        final_geo_dist = torch.cdist(self.src_ref_3d_torch, final_part_ref_pts)
        final_geo_dist_v, _ = final_geo_dist.min(dim=-1)
        final_part_mask = final_geo_dist_v < self.vsize
        self.part_mask = final_part_mask
        
    
    def proj_3d_to_2d(self, idx):
        cur_pose = self.poses[idx]
        tgt_key = 'r_' + str(idx)
        tgt_pts = self.ref_pts[tgt_key]
        # step 1 proj dynamic part
        tr = self.get_transform()
        R = tr[:3, :3]
        t = tr[:3, -1]
        T = self.construct_transform(R, t)
        dy_c2w = T @ cur_pose
        dy_pts = self.src_ref_3d_torch[self.part_mask]
        dy_2d_pts = self.project_point_cloud(dy_pts, dy_c2w, self.K.to(dy_pts))
        # step 2 proj static part
        # st_mask = 1 - self.part_mask.float()
        # st_pts = self.src_ref_3d_torch[st_mask == 1]
        # st_2d_pts = self.project_point_cloud(st_pts, cur_pose, self.K.to(st_pts))
        # pts_2d = torch.cat([dy_2d_pts, st_2d_pts], dim=-1).T
        co_mask = self.co_mask[idx]
        
        co_2d_pts = self.get_pts_from_mask(co_mask).to(dy_2d_pts)
        final_2d_pts = torch.cat([co_2d_pts, dy_2d_pts.T], dim=0)
        
        return final_2d_pts, tgt_pts.to(dy_2d_pts), dy_2d_pts.T
    
    def construct_transform(self, R, t):
        # Create 4x4 identity matrices for base transformations
        T_to_origin = torch.eye(4, device=t.device)
        T_back_to_t = torch.eye(4, device=t.device)

        # Construct the transformation matrices for translation
        T_to_origin[:3, 3] = -t.squeeze()   # Translation by -t
        T_back_to_t[:3, 3] = t.squeeze()  # Translation by t

        # Construct the 4x4 rotation matrix R
        R_4x4 = torch.eye(4, device=R.device)  
        R_4x4[:3, :3] = R  # Add 3x3 R to top-left of 4x4 matrix

        # Combine the transformations
        T = T_back_to_t @ R_4x4 @ T_to_origin

        return T
    
    def estimate_pose(self, step, img_ind=3):
        idx_float = random.uniform(0, 1)
        idx = int(idx_float * len(self.poses))
        train_bar = tqdm(range(step + 1))
        
        for s in train_bar:
            pred_pts, tgt_pts, _ = self.proj_3d_to_2d(img_ind)
            chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction='mean', batch_reduction='mean')
            self.optimizer.zero_grad()
            chamfer_loss.backward()
            self.optimizer.step()
            train_bar.set_description(f'trainig at step: {step}, current loss: {chamfer_loss:.2f}')
            # self.pose_param.norm_Q()
        pass
    
    def estimate_pose_accum_grad(self, step, accum_iter=4):
        idx_float = random.uniform(0, 1)
        idx = int(idx_float * len(self.poses))
        train_bar = tqdm(range(int(step * accum_iter + 1)))
        loss = 0
        s_cnt = 0
        # self.optimizer.zero_grad()
        for s in train_bar:
            with torch.set_grad_enabled(True):
                idx_float = random.uniform(0, 1)
                idx = int(idx_float * len(self.poses))
                pred_pts, tgt_pts, _ = self.proj_3d_to_2d(idx)
                
                chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction='mean', batch_reduction='mean')
                
                loss += chamfer_loss
                # final_loss = chamfer_loss / 
                if ((s + 1) % accum_iter == 0) or (s == step):
                    loss = loss / accum_iter
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_bar.set_description(f'trainig at step: {s}, current loss: {loss:.2f}')
                    loss = 0
                    # self.pose_param.norm_Q()
            with torch.no_grad():
                if s_cnt % 100 == 0:
                    self.pose_param.norm_Q()
                    print('norm Q')
                    self.pose_param.norm_dir()
            s_cnt += 1
        pass
    
    
    def project_point_cloud(self, pts, c2w, K):
        '''
        return [2, N]
        '''
        # Add a dimension of ones to the point cloud to make it homogeneous
        ones = torch.ones((pts.shape[0], 1), device=pts.device)
        homogeneous_point_cloud = torch.cat((pts, ones), dim=1)
        
        # Transform the point cloud from world coordinates to camera coordinates
        points_in_camera_coordinates = torch.inverse(c2w) @ homogeneous_point_cloud.t()
        
        # Normalize the coordinates
        points_in_camera_coordinates /= points_in_camera_coordinates[3, :].clone()
        
        # Project the points onto the 2D plane using the intrinsic matrix
        projected_points = K @ points_in_camera_coordinates[:3, :]
        
        # Normalize the projected points
        projected_points /= projected_points[2, :].clone()
        
        return projected_points[:2, :]

    def project_point_cloud_to_image(self, c2w, K):
        # Project 3D points to 2D
        projected_points = self.project_point_cloud(self.ref_3d, c2w, K)

        # Convert these coordinates into integer indices
        x_coords, y_coords = torch.round(projected_points).long()

        # Estimate image size from intrinsic matrix K
        image_width = 2 * K[0, 2].int()
        image_height = 2 * K[1, 2].int()

        # Keep points within image boundaries
        x_coords = torch.clamp(x_coords, 0, image_width - 1)
        y_coords = torch.clamp(y_coords, 0, image_height - 1)

        # Create an image initially with all zeros (black)
        image = torch.zeros((image_height, image_width), dtype=torch.uint8)
        
        # Paint our points on the blank canvas as white pixels
        image[y_coords, x_coords] = 255

        return image
    
    def pts_2d_to_img(self, pts_2d):
        '''
        pts_2d in shape [N, 2]
        '''
        coord_2d = torch.round(pts_2d).long().T
        x_coords, y_coords = coord_2d
        # Estimate image size from intrinsic matrix K
        image_width = 2 * self.K[0, 2].int()
        image_height = 2 * self.K[1, 2].int()

        # Keep points within image boundaries
        x_coords = torch.clamp(x_coords, 0, image_width - 1)
        y_coords = torch.clamp(y_coords, 0, image_height - 1)

        # Create an image initially with all zeros (black)
        image = torch.zeros((image_height, image_width), dtype=torch.uint8)
        
        # Paint our points on the blank canvas as white pixels
        image[y_coords, x_coords] = 255

        return image
    
    def R_from_quaternions(self):
        '''
        w, x, y, z
        '''
        quaternions = F.normalize(self.pose_param.Q, p=2., dim=0)

        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)
    
class PoseEstimator():
    def __init__(self, renderer, dataset, output_dir, use_num_frames=100, device='cpu', scaling=0.5, use_se3=False, motion_type='r', idx_list=None, eps=0.5, select_frame=4, N=128, min_points=100) -> None:
        self.use_se3 = use_se3
        self.renderer = renderer
        self.dataset = dataset
        self.motion_type = motion_type
        self.output_path = P(output_dir)
        self.w, self.h = self.dataset.img_wh
        self.N = N
        self.device = device
        self.use_num_frames = use_num_frames
        total_frame = self.dataset.poses.shape[0]
        if idx_list is None:
            if self.use_num_frames >= total_frame:
                idx_list = np.arange(total_frame)
            else:
                idx_list = np.random.choice(total_frame, self.use_num_frames, replace=False)
        
        self.eps = eps
        self.idx_list = idx_list
        self.availiable_idx_list = idx_list
        self.K = camera_matrix_scaling(dataset.K, scaling)
        self.op_w, self.op_h = int(scaling * self.w), int(scaling * self.h)
        self.resize = tvT.Resize([self.op_w, self.op_h], interpolation=tvT.InterpolationMode.NEAREST_EXACT)
        self.min_points = min_points
        self.preprocess(select_frame)
        pass
    
    def preprocess(self, select_frame):
        print('------------------------------------------')
        print('collecting points for parts')
        print('------------------------------------------')
        self.collect_part()
        self.frame_selection(num_frames=select_frame)
        
        print('------------------------------------------')
        print('collecting points for parts again after frame selection')
        print('------------------------------------------')
        
        self.scan_nerf()
        self.nerf_pts_check()
        self.collect_part()
        self.process_part_pts()
        self.construct_learnable_param()
        self.configure_optimizer()
    
    
    def frame_selection(self, num_frames=4):
        # start with max pixel occupacity
        # max different pixels
        # recover point position of c2w, pick frames with equal azimuth distance, max azimuth is 90 deg
        
        # azimuth_gap = 360 / num_frames
        # if azimuth_gap > 90:
        #     azimuth_gap = 90
        
        # pose_list = self.dataset.poses
        # valid_score = self.valid_cnt / self.valid_cnt.max()
        diff_score = self.diff_cnt / self.diff_cnt.max()
        # sum_score = valid_score + diff_score
        # max_diff = torch.argmax(diff_score).view(1)
        # max_valid = torch.argmax(valid_score).view(1)
        # max_sum = torch.argmax(sum_score).view(1)
        # unique_id = torch.stack([max_diff, max_valid, max_sum]).unique()
        # _, diff_sort_idx = torch.sort(self.diff_cnt, descending=True)
        # key_frame = torch.argmax(valid_score)
        # pts_list = []
        # for p_idx in self.idx_list:
        #     pts = pose_list[p_idx].view(4, 4)[:3, 3].view(1, -1)
        #     pts_list += [pts]
        # pts_batch = torch.cat(pts_list).unsqueeze(0) # [1, N, 3]
        # def points_azimuth(points):
        #     # points tensor has shape [B, N, 2] where last dimension represents x, y respectively
        #     x, y = points[..., 0], points[..., 1]

        #     # compute azimuth (in radians) using torch.atan2
        #     azimuth = torch.atan2(y, x)

        #     # as atan2 returns values in range [-pi, pi], shift all negative radians by adding 2pi
        #     azimuth[azimuth < 0] += 2 * torch.pi

        #     # convert radians into degrees
        #     azimuth = torch.rad2deg(azimuth)

        #     return azimuth  # shape [B, N]
        # pts_batch_azimuth = points_azimuth(pts_batch).unsqueeze(-1)
        # key_pts = pts_batch[:, key_frame, :].view(1, 1, 3)
        # final_pts_batch = torch.cat([key_pts, pts_batch], dim=1)[:, :, :2]
        # key_idx = self.idx_list[key_frame.cpu().item()]
        # self.idx_list = [key_idx, *self.idx_list]
        # from pytorch3d.ops import sample_farthest_points
        # _, fps_idx = sample_farthest_points(final_pts_batch, K=num_frames, random_start_point=True)
        # fps_idx = fps_idx.view(-1)
        # selected_frames = []
        
        # # ================= use top k
        v, i = torch.topk(diff_score, k=num_frames)
        max_diff_idx = i.view(-1)
        # _, topk_sum_idx = torch.topk(sum_score, k=num_frames)
        # _, topk_valid_idx = torch.topk(valid_score, k=num_frames)
        
        # def get_half_score(score, num_frames):
        #     if num_frames % 2 == 0:
        #         half_num = int(num_frames/2)
        #     else:
        #         raise RuntimeError('currently only support num_frames%2 == 0')
        #     _, topk_idx = torch.topk(score, k=half_num)
        #     _, lowk_idx = torch.topk(score, largest=False, k=half_num)
        #     final_idx = torch.cat([topk_idx.view(-1), lowk_idx.view(-1)])
        #     return final_idx
        # # max_v = diff_score.max()
        # # min_v = diff_score.min()
        # # step_v = (max_v - min_v) / (num_frames - 1)
        # # threshold_list = torch.arange(start=min_v, end=max_v, step=step_v)
        # # f_idx = find_unique_closest_indices(diff_score, threshold_list)
        # # random_idx = 
        
        # def get_fidx(score: torch.Tensor):
        #     max_v = score.max()
        #     min_v = score.min()
        #     med_v = score.median()
        #     step_v = (max_v - med_v) / num_frames 
        #     threshold_list = torch.arange(start=med_v, end=max_v, step=step_v)
        #     f_idx = find_unique_closest_indices(score, threshold_list)
        #     return f_idx
        
        # # f_idx = get_fidx(diff_score)
        # f_idx = get_fidx(valid_score)
        # # f_idx = get_fidx(sum_score)
        # azimuth_step = 15
        
        # azimuth_thres_list = torch.arange(start=-30, end=30, step=15).to(max_valid)
        # cur_azimuth = pts_batch_azimuth.view(-1)[max_valid.cpu().item()]
        # batch_azimuth = pts_batch_azimuth.view(-1) - cur_azimuth
        # azimuth_idx = find_unique_closest_indices(batch_azimuth.cpu(), azimuth_thres_list.cpu())
        
        
        
        # while True:
        #     rand_idx = np.random.choice(len(self.idx_list), num_frames - 1)
        #     if max_valid not in rand_idx:
        #         break
        # f_idx = torch.cat([torch.Tensor(rand_idx).view(-1).to(max_valid), max_valid.view(-1)])
        # f_idx = f_idx[1:]
      
        f_idx = max_diff_idx
        # f_idx = get_half_score(valid_score, num_frames=num_frames)
        # print(f'training with frames: {f_idx}')
        selected_frames = []
        for idx in f_idx:
            selected_frames += [self.idx_list[idx]]
        # self.idx_list = np.array(selected_frames)
        # self.idx_list = np.array([44, 12, 28, 35, 10, 63, 51, 13, 20, 97,  1, 73, 56, 30, 95])
        # self.idx_list = np.array(selected_frames)
        gt_path = self.output_path / 'train_set'
        gt_path.mkdir(exist_ok=True)
        
        w, h = self.dataset.img_wh
        for frame_idx in self.idx_list:
            rgb_gt = self.dataset.rgb[frame_idx].view(w, h, 3).permute(2, 0, 1)
            rgb_pil = tvf.to_pil_image(rgb_gt)
            fname = gt_path / f'{frame_idx:4d}.png'
            rgb_pil.save(str(fname))
        
        
    def scan_nerf(self):
        N = self.N
        dist = 3/N
                
        # Create i, j, k tensors
        i = torch.arange(N).view(-1,1,1,1).expand(N, N, N, 1)
        j = torch.arange(N).view(1,-1,1,1).expand(N, N, N, 1)
        k = torch.arange(N).view(1,1,-1,1).expand(N, N, N, 1)

        # Concatenate i, j, k to a 128x128x128x3 tensor
        pos = torch.cat((i, j, k), dim=3)
        pos = pos * dist + torch.Tensor([-1.5, -1.5, -1.5])
        pos = pos.to(self.device)
        query_pos = pos.view(-1, 3)
        _, sigma = self.renderer.model.ngp(query_pos, query_pos)
        sigma = sigma.view(N, N, N, 1)
        opacity = 1 - torch.exp(-sigma * dist)
        valid_grid = opacity > 0.1
        pts = pos[valid_grid.view(N, N, N)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        # pcd_final, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        fname = self.output_path / 'nerf_scan.ply'
        # down_pcd = pcd.voxel_down_sample(3/128)
        # down_pts = torch.Tensor(np.asarray(down_pcd.points)).to(pts)
        
        o3d.io.write_point_cloud(str(fname), pcd)
        self.src_pts = pts
        
    def collect_part(self):
        N=self.N
        part_pts_list = []
        co_mask_list = []
        pred_mask_list = []
        # gt_mask_list = []
        occupy_mask_list = []
        diff_cnt = []
        valid_cnt = []
        diff_mask = []
        cache_idxs = []
        for i in tqdm(self.idx_list):
            render_batch = self.renderer.gen_img_given_pose(self.dataset.poses[i].view(4,4))
            pred_mask = render_batch['acc'] > 0.5
            gt_mask = self.dataset.mask[i].to(pred_mask).view(self.w, self.h, 1)
            pred_points = render_batch['raw_pts']
            co_mask = torch.logical_and(pred_mask, gt_mask)
            diff = (pred_mask.float() - gt_mask.float()) > 0
            part_pts = pred_points[diff.squeeze(-1)]    
            part_pts_list += [part_pts]
            co_mask_list += [co_mask.view(self.w, self.h)]
            pred_mask_list += [pred_mask.view(self.w, self.h)]
            
            occupy_mask = torch.logical_or(pred_mask, gt_mask)
            occupy_mask_list += occupy_mask
            
            valid_cnt += [pred_mask.sum()]
            diff_cnt += [diff.sum()]
            diff_mask += [diff]
            cache_idx = self.collect_important_pixels(
                render_batch['rgb'].view(-1, 3), self.dataset.rgb[i].view(-1, 3), i)
            cache_idxs += [cache_idx]
            
        self.valid_cnt = torch.stack(valid_cnt) # N,
        self.diff_cnt = torch.stack(diff_cnt) # N,
        self.part_pts_list = part_pts_list
        self.co_mask_list = co_mask_list
        self.pred_mask_list = pred_mask_list
        self.occupy_mask_list = occupy_mask_list
        self.cache_idxs = torch.cat(cache_idxs, dim=0)
        # part_pts_whole = torch.cat(part_pts_list, dim=0)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(part_pts_whole.cpu().numpy())
        # voxel_part = pcd.voxel_down_sample(voxel_size=3/N)
        # voxel_part_pts = torch.Tensor(np.asarray(voxel_part.points)).to(self.src_pts)
        # dist = torch.cdist(self.src_pts.unsqueeze(0), voxel_part_pts.unsqueeze(0))
        # dist = dist.squeeze(0)
        # dist_min, _ = dist.min(dim=-1)
        # dist_mask = dist_min < (3/N)
        # final_part_pts = self.src_pts[dist_mask]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(final_part_pts.cpu().numpy())
        # inliner, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        # fname = self.output_path / 'final_part.ply'
        # o3d.io.write_point_cloud(str(fname), inliner)
        # del dist, dist_min, dist_mask
        # torch.cuda.empty_cache()
        
        # self.nerf_part_pts = torch.Tensor(np.asarray(inliner.points)).to(self.src_pts)
        # self.co_mask_list = co_mask_list
        # self.pred_mask_list = pred_mask_list
        
        #  # save co_mask
        # occu_mask_fname = self.output_path / 'co_mask.pth'
        # occu_mask_tensor = torch.cat(occupy_mask_list, dim=0)
        # torch.save(occu_mask_tensor, occu_mask_fname)
        # self.occu_mask_fname = occu_mask_fname
    
    def collect_important_pixels(self, rgb_pred:torch.Tensor, rgb_gt:torch.Tensor, frame_idx):
        '''
        rgb_pred in shape [-1, 3]
        '''
        rgb_diff = (rgb_pred - rgb_gt.to(rgb_pred)).abs().sum(dim=-1)
        rgb_diff_valid = rgb_diff[rgb_diff>0]
        threshold = rgb_diff_valid.max() * 0.5
        mask = rgb_diff > threshold
        save_path = self.output_path / 'debug_important_pixel'
        save_path.mkdir(exist_ok=True)
        fname = str(frame_idx) + '.png'
        mask_pil = tvf.to_pil_image(mask.view(1, 800, 800).to(rgb_gt))
        mask_pil.save(str(save_path/fname))
        pix_idx = torch.arange(rgb_diff.shape[0]).to(rgb_pred)
        valid_pix_idx = pix_idx[rgb_diff > threshold]
        img_idx = torch.ones_like(valid_pix_idx) * frame_idx
        cache_idx = torch.cat([img_idx.view(-1, 1), valid_pix_idx.view(-1, 1)], dim=1)
        return cache_idx
    
    
    def process_part_pts(self):
        N = self.N
        part_pts_whole = torch.cat(self.part_pts_list, dim=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_pts_whole.cpu().numpy())
        voxel_part = pcd.voxel_down_sample(voxel_size=3/N)
        voxel_part_pts = torch.Tensor(np.asarray(voxel_part.points)).to(self.src_pts)
        dist = torch.cdist(self.src_pts.unsqueeze(0), voxel_part_pts.unsqueeze(0))
        dist = dist.squeeze(0)
        dist_min, _ = dist.min(dim=-1)
        dist_mask = dist_min < (3/N)
        final_part_pts = self.src_pts[dist_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_part_pts.cpu().numpy())
        inliner, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        fname = self.output_path / 'final_part.ply'
        o3d.io.write_point_cloud(str(fname), inliner)
        del dist, dist_min, dist_mask
        torch.cuda.empty_cache()
        
        self.nerf_part_pts = torch.Tensor(np.asarray(inliner.points)).to(self.src_pts)
        
         # save co_mask
        occu_mask_fname = self.output_path / 'co_mask.pth'
        occu_mask_tensor = torch.cat(self.occupy_mask_list, dim=0)
        torch.save(occu_mask_tensor, occu_mask_fname)
        self.occu_mask_fname = occu_mask_fname
        
    def construct_learnable_param(self):
        if self.use_se3:
            self.pose_param = PoseModule_se3().cuda()
        else:
            self.pose_param = PoseModule().cuda()
    
    def get_transform(self):
        R = self.R_from_quaternions()
        tr = torch.eye(4).to(self.pose_param.Q)
        tr[:3, :3] = R
        tr[:3, -1] = self.pose_param.axis_origin
        return tr
    
    def configure_optimizer(self, lr_Q=1e-1, lr_T=1e-1, lr_scale=1e-2, lr_dir=1e-1, lr=1e-1, scheduler_step=150, gamma=0.1):
        # param_dict = {
        #     "Q": self.pose_param.Q,
        #     "origin": self.pose_param.axis_origin
        # }
        if self.motion_type == 'r':
            q_dict = {
                "params": self.pose_param.Q,
                "lr": lr_Q,
                "name": "Q"
            }
            origin_dict = {
                "params": self.pose_param.axis_origin,
                "lr": lr_T,
                "name": "axis_origin"
            }
            param_list = [q_dict, origin_dict]
        else:
            dir_dict = {
                "params": self.pose_param.dir,
                "lr": lr_dir,
                "name": "pose_dir"
            }
            scale_dict = {
                "params": self.pose_param.scale,
                "lr": lr_scale,
                "name": "pose_scale"
            }
            param_list = [dir_dict, scale_dict]
        
        self.optimizer = torch.optim.Adam(
            param_list,
            lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
    
    def construct_transform(self, R, t):
        # Create 4x4 identity matrices for base transformations
        T_to_origin = torch.eye(4, device=t.device)
        T_back_to_t = torch.eye(4, device=t.device)

        # Construct the transformation matrices for translation
        T_to_origin[:3, 3] = -t.squeeze()   # Translation by -t
        T_back_to_t[:3, 3] = t.squeeze()  # Translation by t

        # Construct the 4x4 rotation matrix R
        R_4x4 = torch.eye(4, device=R.device)  
        R_4x4[:3, :3] = R  # Add 3x3 R to top-left of 4x4 matrix

        # Combine the transformations
        T = T_back_to_t @ R_4x4 @ T_to_origin

        return T

    def R_from_quaternions(self):
        '''
        w, x, y, z
        '''
        quaternions = F.normalize(self.pose_param.Q, p=2., dim=0)

        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)
    
    def estimate_pose(self, step, img_ind=3):
        idx_float = random.uniform(0, 1)
        # idx = int(idx_float * len(self.poses))
        train_bar = tqdm(range(step + 1))
        
        for s in train_bar:
            pred_pts, tgt_pts, _ = self.proj_3d_to_2d(img_ind)
            chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction='mean', batch_reduction='mean')
            self.optimizer.zero_grad()
            chamfer_loss.backward()
            self.optimizer.step()
            train_bar.set_description(f'trainig at step: {step}, current loss: {chamfer_loss:.2f}')
            # self.pose_param.norm_Q()
        pass
    
    def estimate_pose_accum_grad(self, step, accum_iter=4, norm_Q=True):
        # idx_float = random.uniform(0, 1)
        # idx = int(idx_float * len(self.idx_list))
        train_bar = tqdm(range(int(step * accum_iter + 1)))
        loss = 0
        total_train_num = len(self.idx_list)
        cur_step = 0
        
        diff_score = self.diff_cnt / self.diff_cnt.sum()
        diff_score = diff_score.cpu().numpy()
        # self.optimizer.zero_grad()
        for s in train_bar:
            with torch.set_grad_enabled(True):
                # idx_float = random.uniform(0, 1)
                # idx = np.random.choice(len(self.idx_list), 1, p=diff_score)[0]
                # idx = int(idx_float * len(self.idx_list))
                idx = cur_step % total_train_num
                pred_pts, tgt_pts, _ = self.proj_3d_to_2d(idx)
                
                chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction='mean', batch_reduction='mean')
                
                # add outlier rejection to CD
                # chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction=None, batch_reduction=None)
                
                loss += chamfer_loss
                if chamfer_loss > 300:
                    print(f'current idx: {idx}, current loss: {chamfer_loss}')
                # final_loss = chamfer_loss / 
                if ((s + 1) % accum_iter == 0) or (s == step):
                    loss = loss / accum_iter
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_bar.set_description(f'trainig at step: {step}, current loss: {loss:.2f}')
                    cur_loss = loss.detach()
                    loss = 0
                    
                    # self.pose_param.norm_Q()
            # if s % 1000 == 0:
            #     self.pose_param.norm_Q()
            cur_step += 1
        if norm_Q:
            self.norm_Q()
            self.norm_dir()
        return cur_loss
    
    def norm_Q(self):
        self.pose_param.norm_Q()
        
    def norm_dir(self):
        self.pose_param.norm_dir()
        
    def check_new_part_pts(self):
        dist, _, _ = knn_points(self.nerf_part_pts.unsqueeze(0), self.src_pts.unsqueeze(0), K=1)
        mask = dist < 1/128
        self.nerf_part_pts = self.nerf_part_pts[mask.view(-1)]
        
    def update_new_part_pts(self, new_part_pts: torch.Tensor):
        '''
        this takes the segmentaion estimation from nerf and save the points that's near the initial guess
        '''
        dist, _, _ = knn_points(new_part_pts.unsqueeze(0), self.nerf_part_pts.unsqueeze(0))
        mask = dist < 1 / 128
        self.nerf_part_pts = new_part_pts[mask.view(-1)]
        

    def nerf_pts_check(self):
        # mask out point that are far from objects
        total_num = len(self.idx_list)
        dist_mask_list = []
        print(f'-----------removing outliers from nerf scan-----------')
        for idx in tqdm(range(total_num)):
            cur_idx = self.idx_list[idx]
            
            c2w = self.dataset.poses[cur_idx].view(4, 4)
            render_batch = self.renderer.gen_img_given_pose(c2w)
            pred_mask = render_batch['acc'] > 0.5
            pred_mask_scale = self.resize(pred_mask.permute(2, 0, 1)).view(self.op_w, self.op_h)
            
            nerf_proj_2d = self.project_point_cloud(self.src_pts, c2w.to(self.device), self.K.to(self.src_pts)).T
            nerf_mask_2d = self.get_pts_from_mask(pred_mask_scale).to(nerf_proj_2d)
            dist, _, _ = knn_points(nerf_proj_2d.unsqueeze(0), nerf_mask_2d.unsqueeze(0), K=1)
            dist_mask = dist < 3 # [1, N, 1]
            dist_mask_list += [dist_mask]
            
        # 3d points that are valid in all images
        final_mask = torch.cat(dist_mask_list, dim=-1).squeeze(0).float().sum(dim=-1) == total_num 
        
        self.src_pts = self.src_pts[final_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.src_pts.cpu().numpy())
        # remove floaters
        
        label = pcd.cluster_dbscan(eps=0.5, min_points=100)
        label_np = np.asarray(label)
        label_mask = label_np >=0
        cluster_pts = np.asarray(pcd.points)[label_mask]
        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(cluster_pts)
        
        
        # inliers, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
        
        
        fname = self.output_path / 'nerf_scan_checked.ply'
        o3d.io.write_point_cloud(str(fname), filtered)
        filter_scr_pts = torch.Tensor(np.asarray(cluster_pts)).to(self.src_pts)
        self.src_pts = filter_scr_pts
        
        

    def proj_3d_to_2d(self, idx, vis=False):
        cur_idx = self.idx_list[idx]
        cur_pose = self.dataset.poses[cur_idx].view(4, 4) # [1, 4, 4]
        # tgt_key = 'r_' + str(idx)
        # tgt_pts = self.ref_pts[tgt_key]
        # w, h = self.K[0,2], self.K[1, 2]
        tgt_mask = self.dataset.mask[cur_idx].view(1, self.w, self.h)
        tgt_mask_scale = self.resize(tgt_mask).view(self.op_w, self.op_h)
        tgt_pts = self.get_pts_from_mask(tgt_mask_scale)
        # step 1 proj dynamic part
        # tr = self.get_transform()
        # R = tr[:3, :3]
        # t = tr[:3, -1]
        # T = self.construct_transform(R, t)
        # T = self.pose_param.construct_transform()
        # dy_c2w = T.to(self.device) @ cur_pose.to(self.device)
        dy_c2w = self.pose_param.transform_c2w(cur_pose.to(self.device))
        dy_pts = self.nerf_part_pts
        dy_2d_pts = self.project_point_cloud(dy_pts, dy_c2w, self.K.to(dy_pts))
        # step 2 proj static part
        # st_mask = 1 - self.part_mask.float()
        # st_pts = self.src_ref_3d_torch[st_mask == 1]
        # st_2d_pts = self.project_point_cloud(st_pts, cur_pose, self.K.to(st_pts))
        # pts_2d = torch.cat([dy_2d_pts, st_2d_pts], dim=-1).T
        co_mask = self.co_mask_list[idx]
        co_mask_scale = self.resize(co_mask.unsqueeze(0)).view(self.op_w, self.op_h)
        co_2d_pts = self.get_pts_from_mask(co_mask_scale).to(dy_2d_pts)
        final_2d_pts = torch.cat([co_2d_pts, dy_2d_pts.T], dim=0)
        if vis:
            return final_2d_pts, tgt_pts.to(dy_2d_pts), dy_2d_pts.T, co_2d_pts
        else:
            return final_2d_pts, tgt_pts.to(dy_2d_pts), dy_2d_pts.T

    @staticmethod
    def get_pts_from_mask(mask):
        y_coords, x_coords = torch.where(mask == 1)
        pts = torch.stack([x_coords, y_coords]).T
        return pts 
    
    def project_point_cloud(self, pts, c2w, K):
        '''
        return [2, N]
        '''
        # Add a dimension of ones to the point cloud to make it homogeneous
        ones = torch.ones((pts.shape[0], 1), device=pts.device)
        homogeneous_point_cloud = torch.cat((pts, ones), dim=1)
        
        # Transform the point cloud from world coordinates to camera coordinates
        points_in_camera_coordinates = torch.inverse(c2w) @ homogeneous_point_cloud.t()
        
        # Normalize the coordinates
        points_in_camera_coordinates /= points_in_camera_coordinates[3, :].clone()
        
        # Project the points onto the 2D plane using the intrinsic matrix
        projected_points = K @ points_in_camera_coordinates[:3, :]
        
        # Normalize the projected points
        projected_points /= projected_points[2, :].clone()
        
        return projected_points[:2, :]
    
    def pts_2d_to_img(self, pts_2d):
        '''
        pts_2d in shape [N, 2]
        '''
        coord_2d = torch.round(pts_2d).long().T
        x_coords, y_coords = coord_2d
        # Estimate image size from intrinsic matrix K
        image_width = self.op_w
        image_height = self.op_h

        # Keep points within image boundaries
        x_coords = torch.clamp(x_coords, 0, image_width - 1)
        y_coords = torch.clamp(y_coords, 0, image_height - 1)

        # Create an image initially with all zeros (black)
        image = torch.zeros((image_height, image_width), dtype=torch.uint8)
        
        # Paint our points on the blank canvas as white pixels
        image[y_coords, x_coords] = 255

        return image
    
    def collect_co_opacity(self):
        co_masks = []
        for cnt, i in enumerate(self.idx_list):
            # gt_mask = self.dataset.mask[i].view(self.w, self.h, 1).bool().to(self.device)
            cur_pose = self.dataset.poses[i]
            tr = self.get_transform()
            R = tr[:3, :3]
            t = tr[:3, -1]
            T = self.construct_transform(R, t)
            dy_c2w = T.to(self.device) @ cur_pose.to(self.device)
            dy_render = self.renderer.gen_img_given_pose(dy_c2w.view(4, 4))
            dy_mask = dy_render['acc'] > 0.5
            old_render = self.renderer.gen_img_given_pose(cur_pose.view(4, 4).to(self.device))
            old_mask = old_render['acc'] > 0.5
            co_mask = torch.logical_or(dy_mask, old_mask)
            co_masks += [co_mask]
        return co_masks
    
    
class PoseEstimator_multipart(PoseEstimator):
    def __init__(self, renderer, dataset, output_dir, use_num_frames=100, device='cpu', scaling=0.5, use_se3=False, motion_type='r', idx_list=None, eps=0.5, select_frame=4, N=128, num_dy_parts=2) -> None:
        self.motion_type = motion_type
        self.num_dy_part = num_dy_parts
        super().__init__(renderer, dataset, output_dir, use_num_frames, device, scaling, use_se3, motion_type, idx_list, eps, select_frame, N)
    # def __init__(self, renderer, dataset, output_dir, use_num_frames=100, device='cpu', scaling=0.5, use_se3=False, motion_type='r', num_dy_parts=2) -> None:
    #     self.motion_type = motion_type
    #     self.num_dy_part = num_dy_parts
    #     super().__init__(renderer, dataset, output_dir, use_num_frames, device, scaling, use_se3)
    
    def preprocess(self, select_frame):
        print('------------------------------------------')
        print('collecting points for parts')
        print('------------------------------------------')
        self.collect_part()
        print(f'current training image samples {self.idx_list}')
        self.frame_selection(num_frames=select_frame)
        
        print('------------------------------------------')
        print('collecting points for parts again after frame selection')
        print('------------------------------------------')
        
        self.scan_nerf()
        self.nerf_pts_check()
        self.collect_part()
        self.process_part_pts()
        self.process_multi_part()
        self.construct_learnable_param()
        self.configure_optimizer()
    
    def update_part_pts_list(self, part_pts_list):
        '''
        part_pts: list of tensors
        '''
        # self.part_pts_list = part_pts_list
        checked_part_pts = []
        for part_pts in part_pts_list:
            if part_pts.shape[0] == 0:
                pass
            cur_part_pts = self.check_new_part_pts(part_pts)
            checked_part_pts += [cur_part_pts]
        self.part_pts_list = checked_part_pts
    
    def process_multi_part(self):
        whole_part_pcd = o3d.geometry.PointCloud()
        whole_part_pcd.points = o3d.utility.Vector3dVector(self.nerf_part_pts.cpu().numpy())
        label = whole_part_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points)
        label_np = np.asarray(label)
        part_pts_list = []
        max_part = label_np.max() + 1
        part_pts_cnt = []
        for i in range(max_part):
            part_pts_cnt += [(label_np == i).sum()]
        _, top_i = torch.topk(torch.Tensor(part_pts_cnt), k=self.num_dy_part)
        
        for i in top_i:
            part_pts_list += [self.nerf_part_pts[label_np == i.item()]]
        
        self.part_pts_list = part_pts_list
        
        for p_idx, pts in enumerate(part_pts_list):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
            fname = self.output_path / f'part_{p_idx}.ply'
            o3d.io.write_point_cloud(str(fname), pcd)
    
    # def get_base_part(self):
        
    #     pass
    
    def check_new_part_pts(self, new_part_pts: torch.Tensor):
        '''
        this takes the segmentaion estimation from nerf and save the points that's near the initial guess
        '''
        dist, _, _ = knn_points(new_part_pts.unsqueeze(0), self.nerf_part_pts.unsqueeze(0))
        mask = dist < 1 / 64
        return new_part_pts[mask.view(-1)]
    
    def scan_nerf(self):
        N = self.N
        dist = 3/N
                
        # Create i, j, k tensors
        i = torch.arange(N).view(-1,1,1,1).expand(N, N, N, 1)
        j = torch.arange(N).view(1,-1,1,1).expand(N, N, N, 1)
        k = torch.arange(N).view(1,1,-1,1).expand(N, N, N, 1)

        # Concatenate i, j, k to a 128x128x128x3 tensor
        pos = torch.cat((i, j, k), dim=3)
        pos = pos * dist + torch.Tensor([-1.5, -1.5, -1.5])
        pos = pos.to(self.device)
        query_pos = pos.view(-1, 3)
        _, sigma = self.renderer.model.ngp(query_pos, query_pos)
        sigma = sigma.view(N, N, N, 1)
        opacity = 1 - torch.exp(-sigma * dist)
        valid_grid = opacity > 0.1
        pts = pos[valid_grid.view(N, N, N)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        # pcd_final, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
        fname = self.output_path / 'nerf_scan.ply'
        down_pcd = pcd.voxel_down_sample(3/128)
        down_pts = torch.Tensor(np.asarray(down_pcd.points)).to(pts)
        
        o3d.io.write_point_cloud(str(fname), down_pcd)
        self.src_pts = down_pts
        
    def construct_learnable_param(self):
        if self.use_se3:
            self.pose_param = [PoseModule_se3().cuda() for i in range(self.num_dy_part)]
        else:
            self.pose_param = [PoseModule().cuda() for i in range(self.num_dy_part)]
    
    def configure_optimizer(self, lr_Q=1e-1, lr_T=1e-1, lr_scale=1e-2, lr_dir=1e-1, lr=1e-1, scheduler_step=150, gamma=0.1):
        if self.motion_type == 'r':
            q_dict = {
                "params": [self.pose_param[i].Q for i in range(self.num_dy_part)],
                "lr": lr_Q,
                "name": "Q"
            }
            origin_dict = {
                "params": [self.pose_param[i].axis_origin for i in range(self.num_dy_part)],
                "lr": lr_T,
                "name": "axis_origin"
            }
            param_list = [q_dict, origin_dict]
        else:
            dir_dict = {
                "params": [self.pose_param[i].dir for i in range(self.num_dy_part)],
                "lr": lr_dir,
                "name": "pose_dir"
            }
            scale_dict = {
                "params": [self.pose_param[i].scale for i in range(self.num_dy_part)],
                "lr": lr_scale,
                "name": "pose_scale"
            }
            param_list = [dir_dict, scale_dict]
        
        self.optimizer = torch.optim.Adam(
            param_list,
            lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
    
    
    def configure_optimizer_separate(self, lr_Q=1e-1, lr_T=1e-1, lr_scale=1e-2, lr_dir=1e-1, lr=1e-1, scheduler_step=150, gamma=0.1, pos_idx=0):
        if self.motion_type == 'r':
            q_dict = {
                "params": [self.pose_param[pos_idx].Q],
                "lr": lr_Q,
                "name": "Q"
            }
            origin_dict = {
                "params": [self.pose_param[pos_idx].axis_origin],
                "lr": lr_T,
                "name": "axis_origin"
            }
            param_list = [q_dict, origin_dict]
        else:
            dir_dict = {
                "params": [self.pose_param[pos_idx].dir],
                "lr": lr_dir,
                "name": "pose_dir"
            }
            scale_dict = {
                "params": [self.pose_param[pos_idx].scale],
                "lr": lr_scale,
                "name": "pose_scale"
            }
            param_list = [dir_dict, scale_dict]
        
        self.optimizer = torch.optim.Adam(
            param_list,
            lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=gamma)
    
    
    def norm_Q(self):
        for p in self.pose_param:
            p.norm_Q()
    
    def norm_dir(self):
        for p in self.pose_param:
            p.norm_dir()
    
    def proj_3d_to_2d(self, idx):
        cur_idx = self.idx_list[idx]
        cur_pose = self.dataset.poses[cur_idx].view(4, 4) # [1, 4, 4]
        # get gt
        tgt_mask = self.dataset.mask[cur_idx].view(1, self.w, self.h)
        tgt_mask_scale = self.resize(tgt_mask).view(self.op_w, self.op_h)
        tgt_pts = self.get_pts_from_mask(tgt_mask_scale)
        
        # proj prediction
        # dy_c2w = self.pose_param.transform_c2w(cur_pose.to(self.device))
        # dy_pts = self.nerf_part_pts
        # dy_2d_pts = self.project_point_cloud(dy_pts, dy_c2w, self.K.to(dy_pts))
        proj_dy_pts_list = []
        for i in range(self.num_dy_part):
            cur_c2w = self.pose_param[i].transform_c2w(cur_pose.to(self.device))
            dy_pts = self.part_pts_list[i]
            cur_2d_pts = self.project_point_cloud(dy_pts, cur_c2w, self.K.to(dy_pts))
            proj_dy_pts_list += [cur_2d_pts]
    
        dy_2d_pts = torch.cat(proj_dy_pts_list, dim=1)
        
        # step 2 proj static part
        
        co_mask = self.co_mask_list[idx]
        co_mask_scale = self.resize(co_mask.unsqueeze(0)).view(self.op_w, self.op_h)
        co_2d_pts = self.get_pts_from_mask(co_mask_scale).to(dy_2d_pts)
        final_2d_pts = torch.cat([co_2d_pts, dy_2d_pts.T], dim=0)
        
        return final_2d_pts, tgt_pts.to(dy_2d_pts), dy_2d_pts.T
    
    def shuffle_idx_list(self):
        np.random.shuffle(self.idx_list)
    
    
    def estimate_pose_accum_grad(self, step, accum_iter=4, norm_Q=True):
        # idx_float = random.uniform(0, 1)
        # idx = int(idx_float * len(self.idx_list))
        train_bar = tqdm(range(int(step * accum_iter + 1)))
        loss = 0
        total_train_num = len(self.idx_list)
        cur_step = 0
        
        # self.optimizer.zero_grad()
        for s in train_bar:
            with torch.set_grad_enabled(True):
                # idx_float = random.uniform(0, 1)
                # idx = int(idx_float * len(self.idx_list))
                idx = cur_step % total_train_num
                pred_pts, tgt_pts, _ = self.proj_3d_to_2d(idx)
                
                chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction='mean', batch_reduction='mean')
                
                # add outlier rejection to CD
                # chamfer_loss, _ = chamfer_distance(pred_pts.unsqueeze(0), tgt_pts.unsqueeze(0), point_reduction=None, batch_reduction=None)
                
                loss += chamfer_loss
                # final_loss = chamfer_loss / 
                if ((s + 1) % accum_iter == 0) or (s == step):
                    loss = loss / accum_iter
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_bar.set_description(f'trainig at step: {step}, current loss: {loss:.2f}')
                    cur_loss = loss.detach()
                    loss = 0
                    
                    # self.pose_param.norm_Q()
            # if s % 1000 == 0:
            #     self.pose_param.norm_Q()
            cur_step += 1
        if norm_Q:
            self.norm_Q()
            self.norm_dir()
        return cur_loss
    
if __name__ == "__main__":
    
    from test_ngp import NGPevaluator
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ngp_string = ['--config', "configs/test_stapler_prop.json"]
    opts = get_opts(ngp_string)
    setattr(opts, 'device', device)

    renderer = NGPevaluator(opts)
    src_ply = './results/stapler_rotation_0/voxel_pcd.ply'
    data_path = 'data/stapler_103111_art/train'
    estimator = CoarsePoseEstimator(ref_pts_fname=src_ply, renderer=renderer, root_dir=data_path, use_frame_num=100)
    estimator.estimate_pose_accum_grad(1000, accum_iter=8)