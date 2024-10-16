import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
import open3d as o3d
from PIL import Image
from torchvision import transforms as T
from pathlib import Path as P
from .ray_utils import *
import torch.nn.functional as F
from .pose_utils import radius_to_pose
from .sapien import SapienArtSegDataset_nerfacc
from models.ngp_wrapper import NGP_wrapper
from models.triplet_network import VP2PMatchNet
import random
from test_ngp import NGPevaluator
from torchsparse import SparseTensor

class SapienRegisterDataset(Dataset):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts
        # load and process voxel file
        
        self.renderer = NGPevaluator(opts)
        # self.match_net = VP2PMatchNet(input_dim=opts.input_dim)
        
        self.load_pcd()
        self.len = 1000    
        self.low_bound = torch.Tensor([-1.5, -1.5, -1.5]).cuda()
        self.vsize = 3/128
        self.sample_num = 512
        pass
    
    def load_pcd(self):
        voxel_fname = self.opts.voxel_fname
        pcd = o3d.io.read_point_cloud(voxel_fname)
        self.pts = np.asarray(pcd.points)
        self.voxel_tensor = torch.Tensor(self.pts).to(self.opts.device)
        
    def set_len(self, len):
        self.len = len
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        # radius = random.uniform(4, 7)
        # eval_data = 
        render_batch = self.renderer.gen_random_img()
        render_batch['src_pt'] = self.voxel_tensor
        idx_sample = self.process_mask(render_batch)
        render_batch['idx_sample'] = idx_sample
        inframe_gt = self.get_inframe_gt(render_batch)
        render_batch['inframe_gt'] = inframe_gt
        render_batch.pop('points')
        return render_batch
    
    def get_inframe_gt(self, data):
        points = data['points'].unsqueeze(0) # 1, N, 3
        src_pt =self.voxel_tensor.unsqueeze(0) # 1, k, 3
        pdist = torch.cdist(points, src_pt)
        pdist_test = pdist < self.vsize
        inframe_cnt = pdist_test.float().sum(dim=1).unsqueeze(0).view(-1)
        inframe_gt = inframe_cnt > 0
        return inframe_gt.float()
    
    def get_matching_matrix(self, data):
        pass
    
    
    def get_batch_data(self, batch_size):
        render_list = []
        for i in range(batch_size):
            render_list += [self.__getitem__(i)]
            
        # collect batch
        ret_dict = torch.utils.data.dataloader.default_collate(render_list)
        ret_dict['input_voxel'] = self.construct_sparsetensor(ret_dict)
        return ret_dict

    def process_mask(self, batch):
        mask = batch['scale_mask'].squeeze(0)
        idx = torch.stack(torch.where(mask>0)).T
        total_num = idx.shape[0]
        replace = self.sample_num > total_num
        samples = np.random.choice(total_num, self.sample_num, replace=replace)
        idx_sample = idx[samples]
        return idx_sample
        

    def construct_sparsetensor(self, batch):
        pc = batch['src_pt'] # [B, N, 3]
        input_voxel = pc.reshape(-1, 3)
        batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[1]).reshape(-1, 1).cuda()
        corrds = pc - self.low_bound
        corrds = corrds.reshape(-1, 3)
        corrds = torch.round(corrds / self.vsize)
        corrds = torch.cat((batch_inds, corrds), dim=-1)
        ip_tensor = SparseTensor(feats=input_voxel, coords=corrds)
        return ip_tensor
    