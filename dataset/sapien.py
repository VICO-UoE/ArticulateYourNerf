import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from pathlib import Path as P
from .ray_utils import *
import torch.nn.functional as F
from .pose_utils import radius_to_pose
'''
Sapien input coordinate system:

x left, y up, z back

openGL:

x forward, y left, z up
'''


class SapienDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type = None, white_back = None, eval_inference= None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = white_back

        w,h = self.img_wh
        if eval_inference is not None:
            num = len(self.img_files_val)
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

    def read_meta(self):

        base_dir = self.root_dir
        instance_dir = 'laptop'
        instance_id = '10211'
        degree_id = '80_degree'
        
        if self.split == 'train':
            # base_dir_train = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'train')
            base_dir_train = os.path.join(base_dir, 'train')
            img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
            pose_path_train = os.path.join(base_dir_train, 'transforms.json')
            self.meta = json.load(open(pose_path_train))
        
        elif self.split =='val':

            # self.base_dir_val = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'val')
            self.base_dir_val = os.path.join(base_dir, 'val')
            self.img_files_val = os.listdir(os.path.join(self.base_dir_val, 'rgb'))
            sorted_indices = np.argsort([int(filename.split('_')[1].split('.')[0]) for filename in self.img_files_val])
            self.img_files_val = [self.img_files_val[i] for i in sorted_indices]
            pose_path_val = os.path.join(self.base_dir_val, 'transforms.json')
            self.meta = json.load(open(pose_path_val))
        else:
            # self.base_dir_val = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'test')
            self.base_dir_val = os.path.join(base_dir, 'test')
            self.img_files_val = os.listdir(os.path.join(self.base_dir_val, 'rgb'))
            sorted_indices = np.argsort([int(filename.split('_')[1].split('.')[0]) for filename in self.img_files_val])
            self.img_files_val = [self.img_files_val[i] for i in sorted_indices]
            pose_path_val = os.path.join(self.base_dir_val, 'transforms.json')
            self.meta = json.load(open(pose_path_val))
            
        w, h = self.img_wh

        cam_x = self.meta.get('camera_angle_x', False)
        if cam_x:
            self.focal = 0.5*h/np.tan(0.5*self.meta['camera_angle_x'])
            self.focal *= self.img_wh[0]/320 # modify focal length to match size self.img_wh
        else:
            self.focal = self.meta.get('focal', None)
            if self.focal is None:
                raise ValueError('focal length not found in transforms.json')

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        # pose_scale_factor =  0.2512323810155881
        #obtained after pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))

        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rays_d = []
            self.all_rgbs = []

            for img_file in img_files_train:
                pose = np.array(self.meta['frames'][img_file.split('.')[0]])
                self.poses += [pose]
                # c2w = torch.FloatTensor(pose)

                image_path = os.path.join(base_dir_train, 'rgb', img_file)
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                c2w = torch.FloatTensor(pose)[:3, :4]
                rays_o, view_dirs, rays_d, radii = get_rays(self.directions, c2w, output_view_dirs=True, output_radii=True)
                #rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                self.all_rays_d+=[view_dirs]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rays_d = torch.cat(self.all_rays_d, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        else:
            return len(self.img_files_val) # return for testset

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays_o': self.all_rays[idx][:3],
                      'rays_d': self.all_rays_d[idx],
                      'viewdirs': self.all_rays[idx][3:6],
                      'target' : self.all_rgbs[idx]}

        else: # create data for each image separately
            img_file = self.img_files_val[idx]
            c2w = np.array(self.meta['frames'][img_file.split('.')[0]])
            c2w = torch.FloatTensor(c2w)[:3, :4]
            img = Image.open(os.path.join(self.base_dir_val, 'rgb', img_file))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            # rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, view_dirs, rays_d, radii = get_rays(self.directions, c2w, output_view_dirs=True, output_radii=True)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {
                    'rays_o' :rays[:,:3],
                    'rays_d' : view_dirs,
                    'viewdirs' : rays[:,3:6],
                    'instance_mask': valid_mask,             
                    'target': img,
                    'directions': self.directions,
                    'c2w':c2w}

        return sample



class SapienStaticSegDataset(SapienDataset):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type=None, white_back=None, eval_inference=None, near=2.0, far=6.0):
        # super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference)
        self.root_dir = root_dir
        self.split = split
        # self.read_meta()
        self.white_back = white_back

        self.meta_dict = {}
        self.num_img = None
        self.num_art = None
        
        self.near = near
        self.far = far
        
        self.bounds = np.array([self.near, self.far])
        self.dataset_path = P(self.root_dir) / split
        self.meta = json.load(open(str(self.dataset_path/'transforms.json')))
        
        self.focal = self.meta.get('focal', None)
        self.fnames = sorted(os.listdir(str(self.dataset_path/'rgb')))
        self.img_wh = img_wh
        w, h = self.img_wh
        self.directions = \
            get_ray_directions(h, w, self.focal)
        self.define_transforms()
        if eval_inference is not None:
            num = len(self.img_files_val)
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])
        # self.read_meta()
        if split == 'train':
            self.cache_data()

    def cache_data(self):
        # cache all the images into directions during training, save c2w for all available directions
        img_list = []
        seg_list = []
        c2w_list = []
        directions_list = []
        mask_list = []
        one_hot_list = []
        for fname in self.fnames:
            # gather directions, c2w, rgb, seg
            frame_id = fname.split('.')[0]
            rgb_fname = self.dataset_path / 'rgb' / fname
            seg_fname = self.dataset_path / 'seg' / fname
            img, mask = self.load_img(rgb_fname)
            seg, one_hot = self.load_seg(seg_fname)
            c2w = self.transform(np.array(self.meta['frames'][frame_id])).repeat(img.shape[0], 1, 1)

            img_list += [img]
            mask_list += [mask]
            seg_list += [seg]
            one_hot_list += [one_hot]
            c2w_list += [c2w]
            directions_list += [self.directions.view([-1, 3])]

        self.all_rgbs = torch.cat(img_list)
        self.all_segs = torch.cat(seg_list)
        self.all_dirs = torch.cat(directions_list)
        self.all_mask = torch.cat(mask_list)
        self.all_one_hots = torch.cat(one_hot_list)
        self.all_c2w = torch.cat(c2w_list)
    
    def load_img(self, fname):
        """
        load image and return a flatten image tensor [H*W, 3] and a vlida mask
        """
        img = self.transform(Image.open(str(fname)))
        valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
        return img, valid_mask

    def load_seg(self, fname):
        seg = np.array(Image.open(str(fname)))
        seg = torch.from_numpy(seg)
        seg = seg.type(torch.LongTensor)
        seg = seg - 1 # part label start from 2
        seg[seg<0] = 0 # recover background
        seg = seg.view([-1])
        seg_one_hot = F.one_hot(seg, 3).to(torch.float32)

        return seg, seg_one_hot

    def __getitem__(self, idx):
        # get the directions, c2w, rgb, and seg for each ray, and apply get ray for each collected directions during forwarding

        if self.split == 'train':
            ret_dict = {
                'c2w': self.all_c2w[idx],
                'rgb': self.all_rgbs[idx],
                'seg': self.all_segs[idx],
                'seg_one_hot': self.all_one_hots[idx],
                'dirs': self.all_dirs[idx],
                'mask': self.all_mask[idx]
            }
        else:
            frame_id = 'r_' + str(idx)
            c2w = self.transform(np.array(self.meta['frames'][frame_id])).squeeze(0)
            rgb_fname = self.dataset_path / 'rgb' / (frame_id + '.png')
            seg_fname = self.dataset_path / 'seg' / (frame_id + '.png')
            img = self.transform(Image.open(rgb_fname))
            seg = np.array(Image.open(seg_fname))
            seg = torch.from_numpy(seg)
            seg = seg.type(torch.LongTensor)
            seg = seg - 1 # starts with 2
            seg[seg<0] = 0
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            seg = seg.view([-1])
            seg_one_hot = F.one_hot(seg, 3)
            w, h = self.img_wh
            rays_o, view_dirs, rays_d = get_rays(self.directions, c2w[:3, :4], True)
            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            ret_dict = {
                "img": img,
                "seg": seg,
                "seg_one_hot": seg_one_hot.to(torch.float32),
                "c2w": c2w,
                "mask": valid_mask,
                "w": w,
                "h": h,
                "directions":self.directions,
                'original_rays_o' :rays[:,:3],
                'original_rays_d' : view_dirs,
                'original_viewdirs' : rays[:,3:6],
            }
        return ret_dict

    def __len__(self):
        if self.split == 'train':
            return self.all_rgbs.shape[0]
        else:
            return len(self.fnames)

    
class SapienPartDataset(SapienDataset):
    '''
    Need to load:
        1. Images and view point
        2. Object articulation status and type
        3. Segment mask for evaluation
    '''
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type=None, white_back=None, eval_inference=None, near=2.0, far=6.0):
        super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference)
        self.meta_dict = {}
        self.num_img = None
        self.num_art = None
        
        self.near = near
        self.far = far
        
        self.bounds = np.array([self.near, self.far])
        self.read_meta()

    def read_meta(self):
        dataset_path = P(self.root_dir)
        meta_dict = {}
        cur_path = dataset_path / self.split
        art_dirs = sorted([item for item in cur_path.iterdir() if item.is_dir()])
        for art_dir in art_dirs:
            cur_meta = json.load(open(str(art_dir/'transforms.json')))
            cur_img_fnames = os.listdir(str(art_dir/'rgb'))
            cur_img_files = [str(art_dir / 'rgb' / fname) for fname in cur_img_fnames]
            cur_seg_files = [str(art_dir / 'seg' / fname) for fname in cur_img_fnames]
            
            meta_dict[art_dir.name] = {
                "meta": cur_meta,
                "img_fnames": cur_img_files,
                "seg_fnames": cur_seg_files
            }
        self.meta_dict = meta_dict
        self.num_img = len(cur_img_files)
        self.num_art = len(art_dirs)

    def __len__(self):
        return int(self.num_art * self.num_art)

    def __getitem__(self, idx):
        w, h = self.img_wh
        img_idx = idx % self.num_art
        art_idx = idx // self.num_art
        art_id = 'idx_' + str(art_idx + 1) # articulation 0 is skipped
        cur_meta = self.meta_dict[art_id]
        img_fname = cur_meta['img_fnames'][img_idx]
        seg_fname = cur_meta['seg_fnames'][img_idx]
        frame_id = P(img_fname).name.split('.')[0]
        c2w = np.array(cur_meta['meta']['frame'][frame_id])
        focal = np.array(cur_meta['meta']['focal'])
        directions = get_ray_directions(h, w, focal)
        img = self.transform(Image.open(img_fname))
        seg = np.array(Image.open(seg_fname))
        # img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)
        seg = seg.type(torch.LongTensor)
        seg = seg - 1 # starts with 2
        seg[seg<0] = 0
        valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
        seg = seg.view([-1])
        seg_one_hot = F.one_hot(seg, 3)
        art_pose = np.array(cur_meta['meta']['qpos'])
        ret_dict = {
            "img": img,
            "seg": seg,
            "seg_one_hot": seg_one_hot.to(torch.float32),
            "c2w": c2w,
            "art_idx": art_idx,
            "art_pose": art_pose,
            "mask": valid_mask,
            "directions": directions,
            'w': w,
            'h': h,
            'focal': focal
        }

        return ret_dict

class SapienArtSegDataset(SapienDataset):

    def __init__(self, root_dir, split='train', img_wh=(320, 240), 
                 model_type=None, white_back=None, eval_inference=None,
                 record_hard_sample=False, near=2.0, far=6.0):
        # super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference)
        self.root_dir = root_dir
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = False
        self.num_img = None
        self.num_art = None
        self.record_hard_sample = record_hard_sample
        if self.record_hard_sample:
            self.sample_list = []
        self.use_sample_list = False
        self.near = near
        self.far = far
        
        self.split = split
        # if split == 'train':
        #     self.split = 'train'
        # else:
        #     self.split = 'test'

        self.bounds = np.array([self.near, self.far])
        self.meta_dict = self.read_meta()
        
        self.image_list = []
        self.seg_list = []
        self.get_img_list()
        self.poses = []
        self.dirs = []
        self.c2w = []
        self.rgb = []
        self.mask = []
        self.seg = []
        self.focal = None
        w, h = img_wh
        # self.focal = self.meta_dict['focal']
        self.get_focal()
        # self.directions = \
        #     get_ray_directions(h, w, self.focal) # (h, w, 3)
        self.directions = self.get_ray_directions()
        # if self.split == 'train':
            # self.cache_data_fg_only()
        if split == 'train' or split == 'val':
            self.cache_data()

    def get_focal(self):
        self.focal = self.meta_dict['focal']

    def get_ray_directions(self):
        w, h = self.img_wh
        return get_ray_directions(h, w, self.focal)


    def read_meta(self):
        dataset_path = P(self.root_dir)
        meta_dict = {}
        cur_path = dataset_path / self.split
        meta_dict = json.load(open(str(cur_path/'transforms.json')))
        return meta_dict

    def get_img_list(self):
        dataset_path = P(self.root_dir)
        cur_path = dataset_path / self.split
        frames = self.meta_dict['frames']
        for k, v in frames.items():
            img_name = str(cur_path/'rgb'/(k+'.png'))
            self.image_list += [img_name]
            self.seg_list += [str(cur_path/'seg'/(k+'.png'))]
        return

    def load_seg(self, fname):
        seg = np.array(Image.open(str(fname)))
        seg = torch.from_numpy(seg)
        seg = seg.type(torch.LongTensor)
        seg = seg - 1 # part label start from 2
        seg[seg<0] = 0 # recover background
        seg = seg.view([-1])
        # seg_one_hot = F.one_hot(seg, 3).to(torch.float32)

        return seg

    def cache_data(self):
        dataset_path = P(self.root_dir)
        cur_path = dataset_path / self.split
        frames = self.meta_dict['frames']
        for k, v in frames.items():
            img_name = str(cur_path/'rgb'/(k+'.png'))
            img = self.transform(Image.open(img_name)).view(4, -1).permute(1, 0)
            seg_name = str(cur_path/'seg'/(k+'.png'))
            self.seg += [self.load_seg(seg_name)]
            valid_mask = (img[:, -1]).view([-1, 1])
            img = img[:, :3]*img[:, -1:] # use black background
            self.poses += [torch.Tensor(np.array(v)).unsqueeze(0)]
            self.c2w += [torch.Tensor(np.array(v)).unsqueeze(0)] * img.shape[0]
            self.rgb += [img]
            self.mask += [valid_mask]
        
        num_img = len(frames.keys())
        self.dirs = [self.directions.view(-1, 3)] * num_img

        self.dirs = torch.cat(self.dirs, dim=0)
        self.c2w = torch.cat(self.c2w, dim=0)
        self.rgb = torch.cat(self.rgb, dim=0)
        self.mask = torch.cat(self.mask, dim=0)
        self.seg = torch.cat(self.seg, dim=0)
        return
    
    def cache_data_fg_only(self):
        dataset_path = P(self.root_dir)
        cur_path = dataset_path / self.split
        frames = self.meta_dict['frames']
        for k, v in frames.items():
            img_name = str(cur_path/'rgb'/(k+'.png'))
            img = self.transform(Image.open(img_name)).view(4, -1).permute(1, 0)
            valid_mask = (img[:, -1])
            img = img[:, :3]*img[:, -1:] # use black background
            img = img[valid_mask == 1]
            self.c2w += [torch.Tensor(np.array(v)).unsqueeze(0)] * img.shape[0]
            self.rgb += [img]
            self.mask += [valid_mask[valid_mask == 1]]
            self.dirs += [self.directions.view(-1, 3)[valid_mask == 1]] 
        

        self.dirs = torch.cat(self.dirs, dim=0)
        self.c2w = torch.cat(self.c2w, dim=0)
        self.rgb = torch.cat(self.rgb, dim=0)
        self.mask = torch.cat(self.mask, dim=0)
        return

    def __len__(self):
        if self.split == 'train':
            if self.use_sample_list:
                return len(self.sample_list)
            else:
                return len(self.rgb)
        else:
            return len(self.image_list)

    def _get_train_item(self, idx):
        if self.use_sample_list:
            sample_idx = self.sample_list[idx]
        else:
            sample_idx = idx
        ret_dict = {
            'rgb': self.rgb[sample_idx],
            'dirs': self.dirs[sample_idx],
            'c2w': self.c2w[sample_idx],
            'mask': self.mask[sample_idx],
            'idx': sample_idx,
            'seg_gt': self.seg[sample_idx]
        }
        return ret_dict

    def _get_test_item(self, idx):
        img = self.transform(Image.open(self.image_list[idx])).view(4, -1).permute(1, 0)
        valid_mask = img[:, -1]
        img = img[:, :3] * img[:, -1:]
        img = img.view([-1, 3])
        img_name = 'r_' + str(idx)
        c2w = self.meta_dict['frames'][img_name]
        c2w = torch.Tensor(np.array(c2w)).unsqueeze(0).repeat(img.shape[0], 1, 1)
        ret_dict = {
            'img': img,
            'c2w': c2w,
            'dirs': self.directions.view([-1, 3]),
            'valid_mask': valid_mask
            # 'seg_gt': self.load_seg(self.seg_list[idx])
        }
        return ret_dict

    def __getitem__(self, idx):
        if self.split == 'train':
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)
        

class SapienArtSegDataset_v2(SapienArtSegDataset):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type=None, white_back=None, eval_inference=None, record_hard_sample=False, near=2.0, far=6.0, use_keypoints=False,
                 batch_size=2048):
        
        self.rays_o = []
        self.rays_d = []
        super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference, record_hard_sample, near, far)
        self.ray_sampling_strategy = 'all_images'
        self.batch_size = batch_size
        self.use_keypoints = use_keypoints
        self.kp_idx = None 
        self.pose_regress = False
        self.kpt = None
        is_train = (self.split == 'train')
        if self.use_keypoints & is_train:
            kpt_fname = P(self.root_dir) / self.split / 'keypoitns.npy'
            kpt_fname = str(kpt_fname)
            kpt = np.load(kpt_fname, allow_pickle=True)
            self.kpt = kpt
            self.kps_idx = self.get_keypoint_idx()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
            
            
    def cache_data(self):
        dataset_path = P(self.root_dir)
        cur_path = dataset_path / self.split
        frames = self.meta_dict['frames']
        for k, v in frames.items():
            img_name = str(cur_path/'rgb'/(k+'.png'))
            img = self.transform(Image.open(img_name)).view(4, -1).permute(1, 0)
            seg_name = str(cur_path/'seg'/(k+'.png'))
            self.seg += [self.load_seg(seg_name)]
            valid_mask = (img[:, -1]).view([-1, 1])
            img = img[:, :3]*img[:, -1:] # use black background
            self.poses += [torch.Tensor(np.array(v)).unsqueeze(0)]
            self.c2w += [torch.Tensor(np.array(v)).unsqueeze(0).repeat(img.shape[0], 1, 1)]
            self.rgb += [img]
            self.mask += [valid_mask]
            
            # for Rays:
            c2w = torch.Tensor(np.array(v))[:3, :4]
            rays_o, view_dirs, rays_d = get_rays(self.directions, c2w, output_view_dirs=True) # [H*W, 3]
            
            view_dirs = view_dirs / torch.linalg.norm(
                    view_dirs, dim=-1, keepdims=True
                ) # [H*W, 3]
            # c2w = torch.FloatTensor(c2w)[:3, :4]
            self.rays_o += [rays_o] 
            self.rays_d += [view_dirs]
        
        #  get rays_o and view_dirs for Rays
        
        num_img = len(frames.keys())
        self.dirs = [self.directions.view(-1, 3)] * num_img

        self.dirs = torch.stack(self.dirs, dim=0)
        self.c2w = torch.stack(self.c2w, dim=0)
        self.rgb = torch.stack(self.rgb, dim=0)
        self.mask = torch.stack(self.mask, dim=0)
        self.seg = torch.stack(self.seg, dim=0)
        self.poses = torch.stack(self.poses, dim=0)
        return
    
    def __len__(self):
        if self.split == 'train':
            return 1000
        else:
            return len(self.image_list)

    def __get_idx(self):
        if self.ray_sampling_strategy == 'all_images': # randomly select images
            img_idxs = np.random.choice(len(self.poses), self.batch_size)
        elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            img_idxs = np.random.choice(len(self.poses), 1)[0]
        # randomly select pixels
        pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
        return img_idxs, pix_idxs

    def __get_seq_idx(self, idx):
        idx_start = idx * self.batch_size
        idx_end = (idx+1) * self.batch_size
        idxs = np.arange(idx_start, idx_end)
        N = self.rgb.shape[0] * self.rgb.shape[1]
        idxs = idxs % N
        img_size = self.rgb.shape[1]
        img_idxs = idxs // img_size
        pix_idxs = idxs % img_size
        return img_idxs, pix_idxs

    def get_keypoint_idx(self):
        kp_idx = []
        w, h = self.img_wh
        for idx, kpt in enumerate(self.kpt):
            kpt_i = kpt[:, 1]
            kpt_j = kpt[:, 0]
            cur_kp_idx = kpt_i * w + kpt_j + w * h * idx
            kp_idx += [torch.LongTensor(cur_kp_idx)]
        return torch.cat(kp_idx)
    
    def __get_train_keypoint_idx(self):
        idxs = np.random.choice(len(self.kps_idx), self.batch_size)
        kp_idx = self.kps_idx[idxs]
        w, h = self.img_wh
        img_idx = kp_idx // (w * h)
        pix_idx = kp_idx % (w * h)
        
        return img_idx, pix_idx
    
    def set_regress_pose(self, pose_regress=True):
        self.pose_regress = pose_regress
        return

    def _get_train_item(self, idx):
        if self.pose_regress & self.use_keypoints:
            img_idxs, pix_idxs = self.__get_train_keypoint_idx()
        else:
            # img_idxs, pix_idxs = self.__get_seq_idx(idx)
            img_idxs, pix_idxs = self.__get_idx()
        ret_dict = {
            'rgb': self.rgb[img_idxs, pix_idxs],
            'dirs': self.dirs[img_idxs, pix_idxs],
            'c2w': self.c2w[img_idxs, pix_idxs],
            'mask': self.mask[img_idxs, pix_idxs],
            'idx': [img_idxs, pix_idxs],
            'seg_gt': self.seg[img_idxs, pix_idxs]
        }
        return ret_dict

class SapienArtSegDataset_nerfacc(SapienArtSegDataset_v2):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type=None, 
                 white_back=None, eval_inference=None, record_hard_sample=False, 
                 near=2, far=6, use_keypoints=False, batch_size=2048, render_bkgd='white',
                 ignore_empty=False, co_mask=None):
        if render_bkgd == 'white':
            self.color_bkgd = torch.ones(3)
        else:
            self.color_bkgd = torch.zeros(3)
        super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference, record_hard_sample, near, far, use_keypoints, batch_size)
        if self.split == 'test':
            self.cache_data()
        if self.split == 'train' or self.split == 'val':
            self.rays_o = torch.stack(self.rays_o, dim=0)
            self.rays_d = torch.stack(self.rays_d, dim=0)
        # w, h = self.img_wh
        # self.K = torch.tensor(
        #     [
        #         [self.focal, 0, w / 2.0],
        #         [0, self.focal, h / 2.0],
        #         [0, 0, 1],
        #     ],
        #     dtype=torch.float32,
        # )  # (3, 3)
        # ===============================================
        self.ignore_empty = ignore_empty
        self.__init_non_empty_mask()
        if co_mask is not None:
            self.load_co_masks(co_mask)
        self.cur_chunk = 0
        self.imp_sampling_rate = 0.9
        
    def get_ray_directions(self):
        w, h = self.img_wh
        self.K = torch.tensor(
            [
                [self.focal, 0, w / 2.0],
                [0, self.focal, h / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        return get_ray_directions_ngp(h, w, self.K, flatten=False)
    
    def __retrive_img_pix_idx(self, global_idxs):
        w, h = self.img_wh
        img_gap = w * h
        img_idxs = global_idxs // img_gap
        pix_idxs = global_idxs % img_gap
        return img_idxs, pix_idxs
    
    def __init_non_empty_mask(self):
        w, h = self.img_wh
        if self.idx_list is not None:
            img_num = len(self.idx_list)
        else:
            img_num = self.poses.shape[0]
        total_rays = w * h * img_num
        self.global_pix_idx = torch.arange(total_rays).to(self.device)
        self.non_empty_mask = torch.ones(total_rays).to(self.device)
        self.cur_valid_idx = torch.arange(total_rays).to(self.device)
        pass
    
    def load_co_masks(self, fname):
        self.__init_non_empty_mask()
        co_masks = torch.load(fname).to(self.non_empty_mask)
        self.non_empty_mask = co_masks
        self.cur_valid_idx = self.global_pix_idx[self.non_empty_mask.bool().view(-1)]
    
    def update_non_empty_mask(self, img_idxs, pix_idxs):
        w, h = self.img_wh
        img_gap = w * h
        global_pix_idx = img_idxs * img_gap + pix_idxs
        global_pix_idx = torch.Tensor(global_pix_idx).long()
        self.non_empty_mask[global_pix_idx] = 0
        self.cur_valid_idx = self.global_pix_idx[self.non_empty_mask.bool()]
        
        pass
    
    def _get_non_empty_idx(self, idx):
        '''
        return img_idxs [b, ], pix_idxs [b, ]
        '''
        replace = self.batch_size > self.cur_valid_idx.shape[0]
        if replace:
            global_idxs = np.random.choice(self.cur_valid_idx.shape[0], self.batch_size, replace=replace)
        else:
            # get chunk
            
            start_idx = int(idx * self.batch_size) % self.cur_valid_idx.shape[0]
            next_idx = start_idx + self.batch_size
            end_idx = min(next_idx, self.cur_valid_idx.shape[0])
            
            global_idxs = np.arange(start_idx, end_idx, step=1)
        
        return self.__retrive_img_pix_idx(global_idxs)
    
        
    def __get_idx(self):
        if self.ray_sampling_strategy == 'all_images': # randomly select images
            img_idxs = np.random.choice(len(self.poses), self.batch_size)
        elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            img_idxs = np.random.choice(len(self.poses), 1)[0]
        # randomly select pixels
        pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
        return img_idxs, pix_idxs
    
    def _get_train_item(self, idx):
        # use black background
        if self.ignore_empty:
            img_idxs, pix_idxs = self._get_non_empty_idx(idx)
        else:
            img_idxs, pix_idxs = self.__get_idx()
        rays = Rays(origins=self.rays_o[img_idxs, pix_idxs], viewdirs=self.rays_d[img_idxs, pix_idxs])
        ret_dict = {
            'c2w': self.c2w[img_idxs, pix_idxs],
            'mask': self.mask[img_idxs, pix_idxs],
            'idx': [img_idxs, pix_idxs],
            # 'seg_gt': self.seg[img_idxs, pix_idxs],
            'rays': rays,
            # nerfacc required
            'pixels': self.rgb[img_idxs, pix_idxs],
            'dirs': self.dirs[img_idxs, pix_idxs],
            'color_bkgd': self.color_bkgd.to(self.rgb),
            # original_directions
            'directions': self.directions,
            'img_idxs': img_idxs,
            'pix_idxs': pix_idxs
        }
        
        return ret_dict
    
    def adjust_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    @torch.inference_mode
    def gather_rays_given_radius(self, radius):
        pass
    
    def img_bkgd_blending(self, img):
        alpha = img[:, -1:]
        pixels = img[:, :3]
        rgb = pixels * alpha + self.color_bkgd * (1.0 - alpha)
        # img = img[:, :3]*img[:, -1:] # use black background 
        # pass
        return rgb
    
    def cache_data(self):
        dataset_path = P(self.root_dir)
        cur_path = dataset_path / self.split
        frames = self.meta_dict['frames']
        for k, v in frames.items():
            img_name = str(cur_path/'rgb'/(k+'.png'))
            img = self.transform(Image.open(img_name)).view(4, -1).permute(1, 0)
            seg_name = str(cur_path/'seg'/(k+'.png'))
            self.seg += [self.load_seg(seg_name)]
            valid_mask = (img[:, -1]).view([-1, 1])
            # img = img[:, :3]*img[:, -1:] # use black background
            img = self.img_bkgd_blending(img)
            # sapien pose
            # c2w = torch.Tensor(np.array(v))
            # ngp pose
            c2w = torch.Tensor(transfrom_to_NGP(np.array(v)))
            self.poses += [c2w.unsqueeze(0)]
            self.c2w += [c2w.unsqueeze(0).repeat(img.shape[0], 1, 1)]
            self.rgb += [img]
            self.mask += [valid_mask]
            
            # for Rays:
            # c2w = torch.Tensor(np.array(v))[:3, :4]
            # rays_o, view_dirs, rays_d = get_rays(self.directions, c2w[:3, :4], output_view_dirs=True) # [H*W, 3]
            # use ngp pose
            rays_o, view_dirs = get_rays_ngp(self.directions, c2w[:3, :4])
            view_dirs = view_dirs / torch.linalg.norm(
                    view_dirs, dim=-1, keepdims=True
                ) # [H*W, 3]
            # c2w = torch.FloatTensor(c2w)[:3, :4]
            self.rays_o += [rays_o] 
            self.rays_d += [view_dirs]
        
        #  get rays_o and view_dirs for Rays
        
        num_img = len(frames.keys())
        self.dirs = [self.directions.view(-1, 3)] * num_img

        self.dirs = torch.stack(self.dirs, dim=0)
        self.c2w = torch.stack(self.c2w, dim=0)
        self.rgb = torch.stack(self.rgb, dim=0)
        self.mask = torch.stack(self.mask, dim=0)
        self.seg = torch.stack(self.seg, dim=0)
        self.poses = torch.stack(self.poses, dim=0)
        return
    
    def _get_test_item(self, idx):
        rays_o = self.rays_o[idx].reshape(self.directions.shape)
        rays_d = self.rays_d[idx].reshape(self.directions.shape)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        pixels = self.rgb[idx].reshape(self.directions.shape)
        ret_dict = {
            'c2w': self.poses[idx],
            'pixels': pixels,
            'rays': rays,
            'color_bkgd': self.color_bkgd.to(self.rgb),
            'dirs': self.directions,
            'mask': self.mask[idx]
        }
        return ret_dict
    
    def get_rays_given_radius(self, radius):
        pose = radius_to_pose(radius=radius)
        return self.get_rays_given_pose(pose)
    
    def get_rays_given_pose(self, pose):
        '''
        pose: np.array in shape [4, 4] or [3, 4]
        '''
        c2w = torch.Tensor(pose).to(self.directions)
        rays_o, view_dirs = get_rays_ngp(self.directions, c2w[:3, :4], flatten=False)
        view_dirs = view_dirs / torch.linalg.norm(
                view_dirs, dim=-1, keepdims=True
            ) # [H*W, 3]
        rays = Rays(origins=rays_o, viewdirs=view_dirs)
        ret_dict = {
            'rays': rays,
            'color_bkgd': self.color_bkgd.to(self.directions),
            'c2w': c2w,
            'dirs': self.directions
        }
        return ret_dict
        
class SapienParisDataset(SapienArtSegDataset_nerfacc):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type=None, state='start',
                 white_back=None, eval_inference=None, record_hard_sample=False, near=2, 
                 far=6, use_keypoints=False, batch_size=2048, render_bkgd='white', 
                 ignore_empty=False, co_mask=None, idx_list=None):
        self.state = state
        if idx_list is not None:
            self.idx_list = torch.Tensor(idx_list)
        else:
            self.idx_list = None
        super().__init__(root_dir, split, img_wh, model_type, white_back, eval_inference, record_hard_sample, near, far, use_keypoints, batch_size, render_bkgd, ignore_empty, co_mask)
        self.importance_pixels = None
        self.importance_sampling = False
        
    def load_transform(self):
        pass
        
    def cache_data(self):
        cur_path = P(self.root_dir) / self.state / self.split
        if self.split == 'test':
            seg_path = P(self.root_dir) / self.state / 'seg_test'
        else:
            seg_path = None
        frame_num = len(self.meta_dict.keys()) - 1 # exclude K
        self.seg = []
        for i in range(frame_num):
            k = str(i).zfill(4)
            img_name = str(cur_path / (k + '.png'))
            self.image_list += [img_name]
            img = self.transform(Image.open(img_name)).view(4, -1).permute(1, 0)
            valid_mask = (img[:, -1]).view([-1, 1])
            img = self.img_bkgd_blending(img)
            v = self.meta_dict[k]
            c2w = torch.Tensor(transfrom_to_NGP(np.array(v)))
            self.poses += [c2w.unsqueeze(0)]
            self.c2w += [c2w.unsqueeze(0).repeat(img.shape[0], 1, 1)]
            self.rgb += [img]
            self.mask += [valid_mask]
            rays_o, view_dirs = get_rays_ngp(self.directions, c2w[:3, :4])
            view_dirs = view_dirs / torch.linalg.norm(
                    view_dirs, dim=-1, keepdims=True
                ) # [H*W, 3]
            # c2w = torch.FloatTensor(c2w)[:3, :4]
            self.rays_o += [rays_o] 
            self.rays_d += [view_dirs]
            
            # load seg
            if seg_path is not None:
                seg_fname = seg_path / (k + '.png')
                seg = self.transform(Image.open(str(seg_fname)))
                self.seg += [seg]
            
        self.dirs = [self.directions.view(-1, 3)] * frame_num
        if seg_path is not None:
            self.seg = torch.cat(self.seg, dim=0)
        self.dirs = torch.stack(self.dirs, dim=0)
        self.c2w = torch.stack(self.c2w, dim=0)
        self.rgb = torch.stack(self.rgb, dim=0)
        self.mask = torch.stack(self.mask, dim=0)
        self.poses = torch.stack(self.poses, dim=0)
        return 
    
    def read_meta(self):
        # with open(os.path.join(self.config.root_dir, state, f"camera_{self.split}.json"), 'r') as f:
        #     cam_dict = json.load(f)
        #     f.close()
        fname = P(self.root_dir) / self.state / f'camera_{self.split}.json'
        meta_dict = json.load(open(str(fname)))
        return meta_dict

    def get_focal(self):
        self.K = np.asarray(self.meta_dict['K'])
        self.focal = self.K[0,0]
        return 
    
    def get_img_list(self):
        return
    
    def _get_train_item(self, idx):
        # use black background
        if self.ignore_empty:
            img_idxs, pix_idxs = self._get_non_empty_idx(idx)
        elif self.importance_sampling:
            img_idxs, pix_idxs = self._importance_sampling()
        else:
            img_idxs, pix_idxs = self.__get_idx()
        
        rays = Rays(origins=self.rays_o[img_idxs, pix_idxs], viewdirs=self.rays_d[img_idxs, pix_idxs])
        ret_dict = {
            'c2w': self.c2w[img_idxs, pix_idxs],
            'mask': self.mask[img_idxs, pix_idxs],
            'idx': [img_idxs, pix_idxs],
            'rays': rays,
            # nerfacc required
            'pixels': self.rgb[img_idxs, pix_idxs],
            'dirs': self.dirs[img_idxs, pix_idxs],
            'color_bkgd': self.color_bkgd.to(self.rgb),
            # original_directions
            'directions': self.directions,
            'img_idxs': img_idxs,
            'pix_idxs': pix_idxs
        }
        return ret_dict
    
    def __get_idx(self, ray_num=None):
        
        if ray_num is None:
            sample_num = self.batch_size
        else:
            sample_num = ray_num
            
        img_num = self.poses.shape[0]
        if self.idx_list is not None:
            img_num = len(self.idx_list)
        if self.ray_sampling_strategy == 'all_images': # randomly select images
            cur_idxs = np.random.choice(img_num, sample_num)
        elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
            cur_idxs = np.random.choice(img_num, 1)[0]
        # randomly select pixels
        pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], sample_num)
        if self.idx_list is not None:
            img_idxs = self.idx_list[cur_idxs]
        else:
            img_idxs = cur_idxs
        return img_idxs, pix_idxs
    
    def _importance_sampling(self):
        if self.importance_pixels is None:
            img_idxs, pix_idxs = self.__get_idx()
        else:
            ray_num = int(self.batch_size * self.imp_sampling_rate)
            important_ray_num = self.batch_size - ray_num
            img_idxs, pix_idxs = self.__get_idx(ray_num=ray_num)
            random_idx = torch.cat(
                [torch.Tensor(img_idxs).view(-1, 1), torch.Tensor(pix_idxs).view(-1, 1)], 
                dim=1)
            imp_total = self.importance_pixels.shape[0]
            replace = important_ray_num > imp_total
            imp_choices = np.random.choice(imp_total, important_ray_num, replace=replace)
            imp_idxs = self.importance_pixels[imp_choices]
            all_idxs = torch.cat([random_idx.to(imp_idxs), imp_idxs], dim=0)
            img_idxs = all_idxs[:, 0].cpu().numpy()
            pix_idxs = all_idxs[:, 1].cpu().numpy()
        return img_idxs, pix_idxs
    
    
    def get_rays_directions_customs(h, w, K):
        return get_ray_directions_ngp(h, w, K)