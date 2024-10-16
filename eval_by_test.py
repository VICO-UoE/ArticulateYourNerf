from models.ngp_wrapper import NGP_Prop_Wrapper
import torch
import torchvision.transforms.functional as tvF
from models.nerf.ngp import NGPradianceSegField
from models.utils import Rays
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
    render_transmittance_from_alpha,
    render_weight_from_alpha,
    render_transmittance_from_density,
)
from einops import rearrange

class NGP_Prop_Art_Wrapper(NGP_Prop_Wrapper):
    def __init__(self, config, training=False) -> None:
        self.seg_classes = config.seg_classes
        # revolute only
        self.art_type = 'revolute'
        # initialize pose param
        from pose_estimation import PoseModule
        self.pose_module_list = []
        for i in range(self.seg_classes):
            self.pose_module_list += [PoseModule().to(config.device)]
        super().__init__(config, training)
        
    
    def load_ckpt(self, ckpt_path):
        if self.training:
            self.ngp.train()
            self.estimator.train()
            self.config_optimizer()
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'], strict=False)
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            prop_networks = ckpt_dict['prop_networks']
            for idx, p in enumerate(self.proposal_networks):
                p.load_state_dict(prop_networks[idx])
            pose_dict = {
                "Q": torch.Tensor([ 0.9486, -0.3434,  0.0015, -0.0012]).to(self.device),
                "axis_origin": torch.Tensor([0.0073, 0.7525, 0.1010])
            }
            self.pose_module_list[0].load_state_dict(pose_dict)
            if self.config.resume_training:
                # load optimizer
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'], strict=False)
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            prop_networks = ckpt_dict['prop_networks']
            for idx, p in enumerate(self.proposal_networks):
                p.load_state_dict(prop_networks[idx])
            pose_dict = {
                "Q": torch.Tensor([ 0.9486, -0.3434,  0.0015, -0.0012]).to(self.device),
                "axis_origin": torch.Tensor([0.0073, 0.7525, 0.1010]).to(self.device)
            }
            self.pose_module_list[0].load_state_dict(pose_dict)
                
    def config_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ngp.seg_mlp.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        return 
    
    
    def initialize_ngp(self):
        self.ngp = NGPradianceSegField(
            aabb=self.config.aabb,
            seg_classes=self.config.seg_classes
        ).to(self.device)
    
    
    def render_rays(self, rays: Rays, render_bkgd, ngp, estimator, proposal_requires_grad=False, test_chunk_size=8192):
        
        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2)
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            sigmas = proposal_network(positions)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_sigma_seg_fn(t_starts, t_ends, ray_indices=None):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2).repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            rgb, sigmas, seg = ngp(positions, t_dirs)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return rgb, sigmas.squeeze(-1), seg.squeeze(-1)
        
        def rgb_sigma_fn(t_starts, t_ends, ray_indices=None):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2).repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            rgb, sigmas, _ = ngp(positions, t_dirs)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return rgb, sigmas.squeeze(-1)
        
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in self.proposal_networks
            ],
            prop_samples=self.config.num_samples_per_prop,
            num_samples=self.config.num_samples,
            n_rays=rays.origins.shape[0],
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            sampling_type=self.config.sampling_type,
            stratified=ngp.training,
            requires_grad=proposal_requires_grad,
        )
        
        def composite_rendering(t_starts, t_ends):
            dist = t_ends - t_starts
            rgb, sigma, seg = rgb_sigma_seg_fn(t_starts, t_ends)
            bs = int(rgb.shape[0] / self.seg_classes)
            rgb = rgb.view(bs, self.seg_classes, -1, 3)
            sigma = sigma.view(bs, self.seg_classes, -1) # [B, seg_classes, nsamples]
            seg = seg.view(bs, self.seg_classes, -1, self.seg_classes) #  [B, seg_classes, samples, seg_classes]
            dist = dist.view(bs, self.seg_classes, -1) # [B, seg_classes, nsamples]
            act_seg = torch.nn.functional.softmax(seg, dim=-1)
            seg_mask = torch.cat([act_seg[:, i:i+1, :, i] for i in range(self.seg_classes)], dim=1) # [B, seg_classes, samples]
            
            lambda_i = seg_mask * sigma * dist
            alpha_i = 1 - torch.exp(-lambda_i)
            
            lambda_comp = lambda_i.sum(dim=1) # [B, nsamples]
            alpha_comp = 1 - torch.exp(-lambda_comp)
            transmittance = render_transmittance_from_alpha(alphas=alpha_comp) # [B, nsamples]
            weighted_color = alpha_i.unsqueeze(-1) * rgb # [B, seg_class, nsamples, 3]
            weighted_color = weighted_color.sum(dim=1) # [B, nsamples, 3]
            final_color = torch.sum(transmittance.unsqueeze(-1) * weighted_color, dim=1) # [B, 3]
            weight_comp, _ = render_weight_from_alpha(alpha_comp)
            # render depth like rgb?
            pos = 0.5 * (t_starts + t_ends)
            pos = pos.view(bs, self.seg_classes, -1)
            weighted_depth = torch.sum(alpha_i * pos, dim=1)
            depth = torch.sum(transmittance * weighted_depth, dim=1)
            # depth = torch.sum(weight_comp * (t_starts + t_ends) * 0.5)
            acc = weight_comp.sum(dim=1)
            
            weighted_seg = torch.sum(alpha_i.unsqueeze(-1) * act_seg, dim=1) # acc along seg_classes
            final_seg = torch.sum(transmittance.unsqueeze(-1) * weighted_seg, dim=1) # acc along sample? [B, seg_classes]
            final_color = final_color + (1 - acc.unsqueeze(-1)) * render_bkgd
            extras = {
                'trans': transmittance,
                'weight': weight_comp,
                'seg': final_seg
            }
            return final_color, depth, acc, extras
        
        rgb, depth, opacity, extras = composite_rendering(t_starts, t_ends)
        return rgb, opacity, depth, extras
    
    
    @torch.inference_mode()
    def gen_data(self, data):
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        # generate point cloud
        # mask = acc > (1 - 1e-3)
        # mask = mask.squeeze(-1)
        # pts = rays.origins + rays.viewdirs * depth
        # # pts_cam = data['directions'] * depth
        # points = pts[mask].view(-1, 3)
        # # points_color = rgb[mask].view(-1, 3)
        # # img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        # mask = acc > (1 - 1e-2)
        # scale_pts, scale_mask = self.scale_down(mask, pts)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth
            # 'raw_pts': pts,
            # 'scale_pts': scale_pts,
            # 'scale_mask': scale_mask,
            # 'points': points
        }
        return ret_dict
    
    def gather_rays(self, dirs, c2w, is_training=False):
        if is_training:
            gather_fn = self.get_rays_torch_multiple_c2w
        else:
            gather_fn = self.get_rays
        rays_o_list = []
        rays_d_list = []
        for i in range(self.seg_classes):
            if i == 0:
                if len(c2w.shape) == 3:
                    
                    rays_o, rays_d = gather_fn(dirs.view(-1, 3), c2w[:, :3, :])
                else:
                    raise NotImplementedError('len(c2w) == 2')
                rays_o_list += [rays_o]
                rays_d_list += [rays_d]
            else:
                pose_param = self.pose_module_list[i-1]
                new_c2w = pose_param.transform_c2w(c2w)
                if len(c2w.shape) == 3:
                    
                    rays_o, rays_d = gather_fn(dirs.view(-1, 3), new_c2w[:, :3, :])
                else:
                    raise NotImplementedError('len(c2w) == 2')
                rays_o_list += [rays_o]
                rays_d_list += [rays_d]
        all_rays_o = torch.stack(rays_o_list, dim=1).view(-1, 3)
        all_rays_d = torch.stack(rays_d_list, dim=1).view(-1, 3)
        rays = Rays(origins=all_rays_o, viewdirs=all_rays_d)
        return rays
    
    def get_rays(self, dirs, c2w, flatten=False):
        """
        Get ray origin and directions in world coordinate for all pixels in one image.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            directions: (N, 3) ray directions in camera coordinate
            c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

        Outputs:
            rays_o: (N, 3), the origin of the rays in world coordinate
            rays_d: (N, 3), the direction of the rays in world coordinate
        """
        if c2w.ndim==2:
            # Rotate ray directions from camera coordinate to the world coordinate
            rays_d = dirs @ c2w[:, :3].T
        else:
            rays_d = rearrange(dirs, 'n c -> n 1 c') @ \
                    rearrange(c2w[..., :3], 'n a b -> n b a')
            rays_d = rearrange(rays_d, 'n 1 c -> n c')
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[..., 3].expand_as(rays_d)
        rays_d = rays_d / torch.linalg.norm(
                rays_d.clone(), dim=-1, keepdims=True
            ) # [H*W, 3]
        if flatten:
            return rays_o.view(-1, 3), rays_d.view(-1, 3)
        else:
            return rays_o, rays_d
        
    def get_rays_torch_multiple_c2w(self, directions, c2w, output_view_dirs=False):
        """
        Get ray origin and normalized directions in world coordinates for all pixels in one image.

        Inputs:
            directions: (N, 3) precomputed ray directions in camera coordinates
            c2w: (N, 3, 4) transformation matrix from camera coordinates to world coordinates
            output_view_dirs: If True, also output view directions.

        Outputs:
            rays_o: (N, 3), the origin of the rays in world coordinates
            rays_d: (N, 3), the normalized direction of the rays in world coordinates
            viewdirs (optional): (N, 3), the view directions in world coordinates
        """
        # Calculate rays_d (directions in world coordinates)
        c2w_T = c2w[:, :, :3].transpose(1, 2) # (N, 3, 3)
        dirs = directions.unsqueeze(1) # (N, 1, 3)
        rays_d = torch.matmul(dirs, c2w_T)  # (N, 1, 3)
        rays_d = rays_d.squeeze(1) # (N, 3)

        # Calculate rays_o (ray origins in world coordinates)
        rays_o = c2w[:, :, 3]  # (N, 3)

        if output_view_dirs:
            # Normalize view directions
            viewdirs = rays_d.clone()
            viewdirs /= torch.norm(viewdirs.clone(), dim=-1, keepdim=True)  # (N, 3)
            return rays_o, viewdirs, rays_d
        else:
            # Normalize rays_d
            rays_d /= torch.norm(rays_d.clone(), dim=1, keepdim=True)  # (N, 3)
            return rays_o, rays_d
    
    @torch.inference_mode()
    def eval(self, data):
        for p in self.proposal_networks:
            p.eval()
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        pixels = data["pixels"].to(self.device)
        
        # duplicate rays
        rays = self.gather_rays(dirs, c2w)
        
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        h, w, _ = data['rays'].origins.shape
        rgb = rgb.view(h, w, 3)
        acc = acc.view(h, w, 1)
        depth = acc.view(h, w, 1)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        img_gt = tvF.to_pil_image(pixels.cpu().permute(2, 0, 1))
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'img_pred': img_pred,
            'img_gt': img_gt
        }
        return ret_dict
    
    def train(self, data, step):
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        # duplicate rays
        rays = self.gather_rays(dirs, c2w, is_training=True)
        render_bkgd = data["color_bkgd"].to(self.device)
        pixels = data["pixels"].to(self.device)
        rgb, _, _, _ = self.render_rays(rays, render_bkgd, self.ngp, self.estimator)
        rgb_loss = F.smooth_l1_loss(rgb, pixels)
        self.optimizer.zero_grad()
        rgb_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        with torch.no_grad():
            psnr_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
            
        ret_dict = {
            'loss': rgb_loss,
            'psnr': psnr
        }
        return ret_dict
    
    @torch.inference_mode()
    def test(self, data):
        for p in self.proposal_networks:
            p.eval()
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        # duplicate rays
        rays = self.gather_rays(dirs, c2w.unsqueeze(0))
        
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        h, w, _ = data['rays'].origins.shape
        rgb = rgb.view(h, w, 3)
        acc = acc.view(h, w, 1)
        depth = acc.view(h, w, 1)
        # mask = acc > (1 - 1e-2)
        # scale_pts, scale_mask = self.scale_down(mask, pts)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'img_pred': img_pred
        }
        return ret_dict
    
    def render_whole_img(self, rays, render_bkgd):
        from models.utils import namedtuple_map
        from torch.utils.data._utils.collate import collate, default_collate_fn_map
        rays_shape = rays.origins.shape
        valid_ray_num = int(rays_shape[0]/ self.seg_classes)
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape
            
        results = []
        chunk = 8192
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            rgb, opacity, depth, extras = self.render_rays(chunk_rays, render_bkgd=render_bkgd, 
                             ngp=self.ngp, estimator=self.estimator)
            chunk_out = [rgb, opacity, depth]
            results.append(chunk_out)
        
        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )
        return (
            colors.view((valid_ray_num, -1)),
            opacities.view((valid_ray_num, -1)),
            depths.view((valid_ray_num, -1)),
            extras,
        )
    
from config import get_opts   
import random
from dataset.sapien import SapienArtSegDataset_nerfacc 
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ngp_string = ['--config', "configs/train_stapler_prop_seg.json"]
    opts = get_opts(ngp_string)
    
    setattr(opts, 'device', device)
    test_dataset = SapienArtSegDataset_nerfacc(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='val'
    )
    model = NGP_Prop_Art_Wrapper(config=opts, training=False)
    # test_batch = test_dataset.get_rays_given_radius(random.uniform(4, 6))
    
    # render_batch = model.test(test_batch)
    
    eval_batch = test_dataset.__getitem__(0)
    
    render_batch = model.eval(eval_batch)
    shape = render_batch['rgb'].shape
    print(f'predict rgb shape: {shape}')
    render_batch['img_pred'].save('test_stapler_pred.png')
    render_batch['img_gt'].save('test_stapler_gt.png')
    print('finish test')
    
    