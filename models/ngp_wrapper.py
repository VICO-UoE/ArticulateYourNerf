from typing_extensions import Literal
from torch.optim.optimizer import Optimizer as Optimizer
from torch import Tensor
from models.nerf.ngp import NGPradianceField, NGPDensityField, NGPradianceSegField, NGPDensitySegField
from typing import Callable, List, Optional, Tuple, Union
from models.utils import Rays, namedtuple_map
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
    _transform_stot,
    _pdf_loss
    )
import torch
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
import numpy as np
from pathlib import Path as P
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
    render_transmittance_from_alpha,
    render_weight_from_alpha,
    render_transmittance_from_density,
    render_visibility_from_density,
    render_visibility_from_alpha
)
from nerfacc.data_specs import RayIntervals
from nerfacc.pdf import importance_sampling
from nerfacc import traverse_grids
from models.utils import Rays
import itertools
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from einops import rearrange
from pose_estimation import PoseModule, PoseModule_se3
from models.utils import entropy_loss
import time
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

class OccArtEstimator(OccGridEstimator):
    def __init__(self, roi_aabb, 
                 resolution = 128, 
                 levels = 1, **kwargs) -> None:
        super().__init__(roi_aabb, resolution, levels, **kwargs)
        
        
    # just change the sampling method not to skip empty grids
    def sampling(self, 
                rays_o, 
                rays_d, 
                sigma_fn: Optional[Callable] = None,
                alpha_fn: Optional[Callable] = None,
                near_plane = 0, 
                far_plane = 10000000000, 
                t_min = None, 
                t_max = None, 
                render_step_size = 0.001, 
                early_stop_eps = 0.0001, 
                alpha_thre = 0, 
                stratified = False, 
                cone_angle = 0,
                traverse_steps_limit = 128) -> torch.Tuple[torch.Tensor]:
        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        intervals, samples, _ = traverse_grids(
            rays_o,
            rays_d,
            self.binaries,
            self.aabbs,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
            traverse_steps_limit=traverse_steps_limit,
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices
        packed_info = samples.packed_info

        # skip invisible space
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
            sigma_fn is not None or alpha_fn is not None
        ):
            alpha_thre = min(alpha_thre, self.occs.mean().item())

            # Compute visibility of the samples, and filter out invisible samples
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert (
                    sigmas.shape == t_starts.shape
                ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
                masks = render_visibility_from_density(
                    t_starts=t_starts,
                    t_ends=t_ends,
                    sigmas=sigmas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert (
                    alphas.shape == t_starts.shape
                ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
                masks = render_visibility_from_alpha(
                    alphas=alphas,
                    packed_info=packed_info,
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends, masks
    
    
class PropArtEstimator(PropNetEstimator):
    def __init__(self, optimizer = None, scheduler = None) -> None:
        super().__init__(optimizer, scheduler)
        
    @torch.no_grad()
    def sampling(
        self,
        prop_sigma_fns: List[Callable],
        prop_samples: List[int],
        num_samples: int,
        # rendering options
        n_rays: int,
        near_plane: float,
        far_plane: float,
        seg_classes: int = 2,
        sampling_type: Literal["uniform", "lindisp"] = "lindisp",
        # training options
        stratified: bool = False,
        requires_grad: bool = False,
        return_sigmas: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sampling with CDFs from proposal networks.

        Note:
            When `requires_grad` is `True`, the gradients are allowed to flow
            through the proposal networks, and the outputs of the proposal
            networks are cached to update them later when calling `update_every_n_steps()`

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), (
            "The number of proposal networks and the number of samples "
            "should be the same."
        )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        for level_fn, level_samples in zip(prop_sigma_fns, prop_samples):
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, stratified
            )
            t_vals = _transform_stot(
                sampling_type, intervals.vals, near_plane, far_plane
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                bs = int(sigmas.shape[0] / seg_classes)
                sigmas_reshape = sigmas.view(bs, seg_classes, -1)
                sigmas_combine = sigmas_reshape.sum(dim=1).view(bs, 1, -1).repeat(1, 2, 1).view(sigmas.shape[0], -1)
                
                
                trans, _ = render_transmittance_from_density(
                    t_starts, t_ends, sigmas_combine
                )
                cdfs = 1.0 - torch.cat(
                    [trans, torch.zeros_like(trans[:, :1])], dim=-1
                )
                if requires_grad:
                    self.prop_cache.append((intervals, cdfs))

        intervals, _ = importance_sampling(
            intervals, cdfs, num_samples, stratified
        )
        t_vals = _transform_stot(
            sampling_type, intervals.vals, near_plane, far_plane
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        if requires_grad:
            self.prop_cache.append((intervals, None))
        if return_sigmas:
            return t_starts, t_ends, sigmas_combine, trans
        else:
            return t_starts, t_ends
    
    @torch.no_grad()
    def seg_sampling(
        self,
        prop_sigma_fns: List[Callable],
        prop_samples: List[int],
        num_samples: int,
        # rendering options
        n_rays: int,
        near_plane: float,
        far_plane: float,
        seg_classes: int = 2,
        sampling_type: Literal["uniform", "lindisp"] = "lindisp",
        # training options
        stratified: bool = False,
        requires_grad: bool = False,
        use_background: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sampling with CDFs from proposal networks.

        Note:
            When `requires_grad` is `True`, the gradients are allowed to flow
            through the proposal networks, and the outputs of the proposal
            networks are cached to update them later when calling `update_every_n_steps()`

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), (
            "The number of proposal networks and the number of samples "
            "should be the same."
        )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        for level_fn, level_samples in zip(prop_sigma_fns, prop_samples):
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, stratified
            )
            t_vals = _transform_stot(
                sampling_type, intervals.vals, near_plane, far_plane
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas, segs = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                bs = int(sigmas.shape[0] / seg_classes)
                sigmas_reshape = sigmas.view(bs, seg_classes, -1)
                segs_reshape = segs.view(bs, seg_classes, -1, seg_classes)
                
                collect_segs_list = []
                
                if use_background:
                    valid_seg_class = seg_classes - 1
                else:
                    valid_seg_class = seg_classes
                    
                for i in range(valid_seg_class):
                    collect_segs_list += [segs_reshape[:, i:i+1, :, i]]
                    
                collect_segs = torch.cat(collect_segs_list, dim=1)
                seg_sigmas = collect_segs * sigmas_reshape
                
                sigmas_combine = seg_sigmas.sum(dim=1).view(bs, 1, -1).repeat(1, seg_classes, 1).view(sigmas.shape[0], -1)
                
                
                trans, _ = render_transmittance_from_density(
                    t_starts, t_ends, sigmas_combine
                )
                cdfs = 1.0 - torch.cat(
                    [trans, torch.zeros_like(trans[:, :1])], dim=-1
                )
                
                # seg_cdfs_list = []
                
                # for i in range(valid_seg_class):
                #     cur_seg_sigmas = seg_sigmas[:, i, ]
                #     cur_trans, _ = render_transmittance_from_density(
                #         t_starts[i*bs:(i+1)*bs, :], t_ends[i*bs:(i+1)*bs, :], cur_seg_sigmas
                #     )
                #     cur_cdfs = 1.0 - torch.cat(
                #         [cur_trans, torch.zeros_like(trans[:, :1])], dim=-1
                #     )
                #     seg_cdfs_list += [cur_cdfs]
                # seg_cdfs = torch.cat(seg_cdfs_list, dim=0)
                ray_num = t_starts.shape[0]
                valid_seg_sigmas = seg_sigmas[:, :valid_seg_class, :].view(ray_num, -1)
                seg_trans, _ = render_transmittance_from_density(t_starts, t_ends, valid_seg_sigmas)
                seg_cdfs = 1.0  - torch.cat(
                        [seg_trans, torch.zeros_like(trans[:, :1])], dim=-1
                    )
                
                if requires_grad:
                    self.prop_cache.append((intervals, seg_cdfs))

        intervals, _ = importance_sampling(
            intervals, cdfs, num_samples, stratified
        )
        t_vals = _transform_stot(
            sampling_type, intervals.vals, near_plane, far_plane
        )
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        if requires_grad:
            self.prop_cache.append((intervals, None))
        return t_starts, t_ends
    
    @torch.enable_grad()
    def compute_loss(self, trans: Tensor, loss_scaler: float = 1.0) -> Tensor:
        """Compute the loss for the proposal networks.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            loss_scaler: The loss scaler. Default to 1.0.

        Returns:
            The loss for the proposal networks.
        """
        if len(self.prop_cache) == 0:
            return torch.zeros((), device=self.device)

        intervals, _ = self.prop_cache.pop()
        # get cdfs at all edges of intervals
        cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)
        cdfs = cdfs.detach()

        loss = 0.0
        while self.prop_cache:
            prop_intervals, prop_cdfs = self.prop_cache.pop()
            loss += _pdf_loss(intervals, cdfs, prop_intervals, prop_cdfs).mean()
        return loss * loss_scaler
    
class NGP_wrapper():
    def __init__(self, config, training=False, use_timestamp=False) -> None:
        '''
        config params:
        -- trew34aining parameters
            max_steps = 20000
            init_batch_size = 1024
            target_sample_batch_size = 1 << 18
            weight_decay = (
                1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
            )
        -- scene parameters
            aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
            near_plane = 0.0
            far_plane = 1.0e10
        -- dataset parameters
            train_dataset_kwargs = {}
            test_dataset_kwargs = {}
        -- model parameters
            grid_resolution = 128
            grid_nlvl = 1
        -- render parameters
            render_step_size = 5e-3
            alpha_thre = 0.0
            cone_angle = 0.0
        
        '''
        self.target_sample_batch_size = 1<< 18
        self.aabb = config.aabb
        # use pytroch lightning to handle the GPU movement
        self.config = config
        self.device = config.device
        
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        current_ts = str(time.time()).split('.')[0]
        self.output_path = P(config.output_dir) / config.exp_name 
        
        if use_timestamp:
            self.output_path = self.output_path / current_ts
        
        self.unfold = torch.nn.Unfold(kernel_size=(4,4), stride=4)
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.training = training
        self.cur_training_step = 0
        ckpt_path = config.pre_trained_weights
        self.initialize_ngp_estimator()
        self.load_ckpt(ckpt_path)
            
        self.eval_path = self.output_path / 'eval'
        self.eval_path.mkdir(parents=True, exist_ok=True)
        
    
    def initialize_ngp_estimator(self):
        self.ngp = NGPradianceField(
            aabb=self.aabb
        ).to(self.device)
        self.estimator = OccArtEstimator(
            roi_aabb=self.aabb, 
            resolution=self.config.grid_resolution, 
            levels=self.config.grid_nlvl
        ).to(self.device)
        
    
    def load_ckpt(self, ckpt_path):
        if self.training:
            self.ngp.train()
            self.estimator.train()
            self.config_optimizer()
            if self.config.resume_training:
                ckpt_dict = torch.load(ckpt_path)
                self.ngp.load_state_dict(ckpt_dict['model'])
                self.estimator.load_state_dict(ckpt_dict['estimator'])
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'])
            self.estimator.load_state_dict(ckpt_dict['estimator'])
    
    def config_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ngp.parameters(),
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
        
    
    def render_rays(self, 
            rays: Rays,
            render_bkgd,
            ngp,
            estimator
        ):
        
        rays_o = rays.origins
        rays_d = rays.viewdirs

        def sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                sigmas = ngp.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                
                rgbs, sigmas = ngp(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends, _ = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            render_step_size=self.config.render_step_size,
            stratified=self.ngp.training,
            cone_angle=self.config.cone_angle,
            alpha_thre=self.config.alpha_thre,
            traverse_steps_limit=self.config.traverse_steps_limit
        )
        
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        # chunk_results = [rgb, opacity, depth, len(t_starts)]
        return rgb, opacity, depth, len(t_starts)

    def render_whole_img(self,
                        rays: Rays, 
                        render_bkgd: torch.Tensor, 
                        chunk_size: int=8192):
        
        
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape

        results = []
        chunk = (
            chunk_size
        )
        render_bkgd = render_bkgd
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            chunk_out = self.render_rays(chunk_rays, render_bkgd=render_bkgd, ngp=self.ngp, estimator=self.estimator)

            
            results.append(chunk_out)
        
        colors, opacities, depths, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )
    
    def occ_eval_fn(self, x):
        d = self.ngp.query_density(x)
        return d * self.config.render_step_size
    
    def train(self, data):
        self.ngp.train()
        self.estimator.train()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        pixels = data["pixels"].to(self.device)
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        self.estimator.update_every_n_steps(step=self.cur_training_step, occ_eval_fn=self.occ_eval_fn, occ_thre=1e-2)
        
        rgb, acc, depth, n_rendering_samples = self.render_rays(rays, render_bkgd=render_bkgd, ngp=self.ngp, estimator=self.estimator)
        
        if n_rendering_samples == 0:
            return -1
        
        rgb_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(rgb_loss) / np.log(10.0)
        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(rgb_loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        
        ret_dict = {
            'loss': rgb_loss,
            'psnr': psnr
        }
        if self.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (self.target_sample_batch_size / float(n_rendering_samples))
            )
            ret_dict['new_batch_size'] = num_rays
        return ret_dict
    
    def eval(self, data):
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        pixels = data["pixels"].to(self.device)
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        rgb, acc, depth, n_rendering_samples = self.render_whole_img(rays, render_bkgd=render_bkgd)
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        img_gt = tvF.to_pil_image(pixels.cpu().permute(2, 0, 1))
        
        # generate point cloud
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        points = pts[mask].view(-1, 3)
        points_color = rgb[mask].view(-1, 3)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'psnr': psnr,
            'img_pred': img_pred,
            'img_gt': img_gt,
            'points': points,
            'points_color': points_color
        }
        return ret_dict
    
    @torch.inference_mode()
    def test(self, data):
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        rgb, acc, depth, n_rendering_samples = self.render_whole_img(rays, render_bkgd=render_bkgd)
        # generate point cloud
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        points = pts[mask].view(-1, 3)
        points_color = rgb[mask].view(-1, 3)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        mask = acc > (1 - 1e-2)
        scale_pts, scale_mask = self.scale_down(mask, pts)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'points': points,
            'points_color': points_color,
            'img_pred': img_pred,
            'raw_pts': pts,
            'scale_pts': scale_pts,
            'scale_mask': scale_mask
        }
        return ret_dict
    
    
    @torch.inference_mode()
    def gen_data(self, data):
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        rgb, acc, depth, n_rendering_samples = self.render_whole_img(rays, render_bkgd=render_bkgd)
        # generate point cloud
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        # pts_cam = data['directions'] * depth
        points = pts[mask].view(-1, 3)
        # points_color = rgb[mask].view(-1, 3)
        # img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        mask = acc > (1 - 1e-2)
        scale_pts, scale_mask = self.scale_down(mask, pts)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'raw_pts': pts,
            'scale_pts': scale_pts,
            'scale_mask': scale_mask,
            'points': points
        }
        return ret_dict
    
    def scale_down(self, mask_raw, pts_raw):
        points = pts_raw.permute(2, 0, 1).unsqueeze(0)
        mask = mask_raw.permute(2, 0, 1).unsqueeze(0)
        _, _, h, w = points.shape
        h_, w_ = int(h/4), int(w/4)
        unfold_pts = self.unfold(points).reshape(1, 3, 4, 4, -1)
        unfold_mask = self.unfold(mask.to(points)).reshape(1, 1, 4, 4, -1)
        final_pts = unfold_pts[:, :, 1, 1, :].view(3, h_, w_)
        final_mask = unfold_mask[:, :, 1, 1, :].view(1, h_, w_)
        return final_pts, final_mask
    
    def save_ckpt(self, step):
        output_path = self.output_path / 'ckpt'
        output_path.mkdir(exist_ok=True)
        fname = str(step).zfill(6) + '.pth'
        ckpt_fname = output_path / fname
        ckpt_fname = str(ckpt_fname)
        ckpt = {
            'estimator': self.estimator.state_dict(),
            'model': self.ngp.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt, ckpt_fname)
        
class NGP_Occ_Art_Wrapper(NGP_wrapper):
    def __init__(self, config, training=False) -> None:
        super().__init__(config, training)
        
    def initialize_ngp_estimator(self):
        self.ngp = NGPradianceSegField(
            aabb=self.aabb
        ).to(self.device)
        self.estimator = OccArtEstimator(
            roi_aabb=self.aabb, 
            resolution=self.config.grid_resolution, 
            levels=self.config.grid_nlvl
        ).to(self.device)
    
    def load_ckpt(self, ckpt_path):
        if self.training:
            self.ngp.train()
            self.estimator.train()
            self.config_optimizer()
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'], strict=False)
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            # self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'])
            self.estimator.load_state_dict(ckpt_dict['estimator'])
    
    def config_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ngp.seg_mlp.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=self.config.weight_decay,
        )
    
    def render_rays(self, rays: Rays, render_bkgd, ngp, estimator):
        
        rays_o = rays.origins
        rays_d = rays.viewdirs
        def sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                sigmas = ngp.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                
                rgbs, sigmas, _ = ngp(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)
        
        def rgb_sigma_seg_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                
                rgbs, sigmas, seg = ngp(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1), seg.squeeze(-1)
        
        def composite_rendering(t_starts, t_ends, ray_indices, masks):
            rgbs_part, sigmas_part, segs_part = rgb_sigma_seg_fn(t_starts, t_ends, ray_indices)
            
            # rgbs = torch.zeros_like()
            pass
        
        # return super().render_rays(rays, render_bkgd, ngp, estimator)
        ray_indices, t_starts, t_ends, masks = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            render_step_size=self.config.render_step_size,
            stratified=self.ngp.training,
            cone_angle=self.config.cone_angle,
            alpha_thre=self.config.alpha_thre,
        )
    
        return 
class NGP_Prop_Wrapper(NGP_wrapper):
    def __init__(self, config, training=False, use_timestamp=False, mkdir=True) -> None:
        '''
        max_steps = 20000
        init_batch_size = 4096
        weight_decay = (
            1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
        )
        # scene parameters
        unbounded = False
        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
        near_plane = 2.0
        far_plane = 6.0
        # dataset parameters
        train_dataset_kwargs = {}
        test_dataset_kwargs = {}
        # model parameters
        proposal_networks = [
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=128,
            ).to(device),
        ]
        # render parameters
        num_samples = 64
        num_samples_per_prop = [128]
        sampling_type = "uniform"
        opaque_bkgd = False
        
        '''
        # super().__init__(config, training)
        self.target_sample_batch_size = 1<< 18
        self.aabb = config.aabb
        # use pytroch lightning to handle the GPU movement
        self.config = config
        self.device = config.device
        self.initialize_ngp()
        # =======================================
        self.config_estimator()
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
        # self.proposal_networks = [
        #     NGPDensityField(
        #         aabb=aabb,
        #         unbounded=False,
        #         n_levels=5,
        #         max_resolution=128,
        #     ).to(device),
        # ]
        # # render parameters
        # self.num_samples = 64
        # self.num_samples_per_prop = [128]
        # self.sampling_type = "uniform"
        # self.opaque_bkgd = False
        # =======================================
        
        # self.estimator = OccGridEstimator(
        #     roi_aabb=aabb, 
        #     resolution=config.grid_resolution, 
        #     levels=config.grid_nlvl
        # ).to(device)
        # =======================================
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        current_ts = str(time.time()).split('.')[0]
        self.output_path = P(config.output_dir) / config.exp_name 
        if use_timestamp:
            self.output_path = self.output_path / current_ts
        self.unfold = torch.nn.Unfold(kernel_size=(4,4), stride=4)
        if mkdir:
            self.output_path.mkdir(exist_ok=True, parents=True)
        else:
            self.output_path = P(config.pre_trained_weights).parent.parent
        self.training = training
        self.cur_training_step = 0
        ckpt_path = config.pre_trained_weights
        if self.training:
            
            self.config_optimizer()
        if ckpt_path is not None:
            self.load_ckpt(ckpt_path)
            
        self.eval_path = self.output_path / 'eval'
        self.eval_path.mkdir(parents=True, exist_ok=True)
        pass
    
    def load_ckpt(self, ckpt_path):
        if self.training:
            self.ngp.train()
            self.estimator.train()
            if self.config.resume_training:
                ckpt_dict = torch.load(ckpt_path)
                self.ngp.load_state_dict(ckpt_dict['model'])
                self.estimator.load_state_dict(ckpt_dict['estimator'])
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
                prop_networks = ckpt_dict['prop_networks']
                for idx, p in enumerate(self.proposal_networks):
                    p.load_state_dict(prop_networks[idx])
                self.prop_optimizer.load_state_dict(ckpt_dict['prop_optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'])
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            prop_networks = ckpt_dict['prop_networks']
            for idx, p in enumerate(self.proposal_networks):
                p.load_state_dict(prop_networks[idx])
    
    def initialize_ngp(self):
        self.ngp = NGPradianceField(
            aabb=self.aabb
        ).to(self.device)
        
    def config_estimator(self):
        self.proposal_networks = [
            NGPDensityField(
                aabb=self.config.aabb,
                unbounded=False,
                n_levels=5,
                max_resolution=128,
            ).to(self.config.device),
        ]
        self.prop_optimizer = torch.optim.Adam(
                itertools.chain(
                    *[p.parameters() for p in self.proposal_networks],
                ),
                # self.proposal_networks[0].parameters(),
                lr=1e-2,
                eps=1e-15,
                weight_decay=self.config.weight_decay,
            )
        self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.prop_optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        self.estimator = PropNetEstimator(
            self.prop_optimizer, 
            self.prop_scheduler).to(self.config.device)
        
        
    def config_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.ngp.parameters(),
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
        
    def render_rays(self, rays: Rays, render_bkgd, ngp, estimator, proposal_requires_grad=False, test_chunk_size=8192):

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2)
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            sigmas = proposal_network(positions)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2).repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            rgb, sigmas = ngp(positions, t_dirs)
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
        
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        return rgb, opacity, depth, extras
    
        
    def train(self, data, step):
        self.ngp.train()
        self.estimator.train()
        for p in self.proposal_networks:
            p.train()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        pixels = data["pixels"].to(self.device)
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        
        proposal_requires_grad = self.proposal_requires_grad_fn(step)
        # proposal_requires_grad = True
        # if proposal_requires_grad:
        #     print('updating estimator...')
        rgb, opacity, depth, extras = self.render_rays(
            rays, 
            render_bkgd=render_bkgd, 
            ngp=self.ngp, 
            estimator=self.estimator, 
            proposal_requires_grad=proposal_requires_grad)
        
        self.estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
        )
        # compute loss
        rgb_loss = F.smooth_l1_loss(rgb, pixels)
        with torch.no_grad():
            psnr_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(rgb_loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        
        ret_dict = {
            'loss': rgb_loss,
            'psnr': psnr
        }
        return ret_dict
    
    def render_whole_img(self, rays, render_bkgd):
        
        rays_shape = rays.origins.shape
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
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )
    
    @torch.no_grad()
    def eval(self, data):
        for p in self.proposal_networks:
            p.eval()
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        pixels = data["pixels"].to(self.device)
        
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd)
        
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        img_gt = tvF.to_pil_image(pixels.cpu().permute(2, 0, 1))
        
        # generate point cloud
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        points = pts[mask].view(-1, 3)
        points_color = rgb[mask].view(-1, 3)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'psnr': psnr,
            'img_pred': img_pred,
            'img_gt': img_gt,
            'points': points,
            'points_color': points_color
        }
        return ret_dict
    
    @torch.inference_mode()
    def test(self, data):
        for p in self.proposal_networks:
            p.eval()
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        # generate point cloud
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        points = pts[mask].view(-1, 3)
        points_color = rgb[mask].view(-1, 3)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        mask = acc > (1 - 1e-2)
        scale_pts, scale_mask = self.scale_down(mask, pts)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'points': points,
            'points_color': points_color,
            'img_pred': img_pred,
            'raw_pts': pts,
            'scale_pts': scale_pts,
            'scale_mask': scale_mask
        }
        return ret_dict
    
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
        mask = acc > (1 - 1e-3)
        mask = mask.squeeze(-1)
        pts = rays.origins + rays.viewdirs * depth
        # pts_cam = data['directions'] * depth
        points = pts[mask].view(-1, 3)
        # points_color = rgb[mask].view(-1, 3)
        # img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        mask = acc > (1 - 1e-2)
        scale_pts, scale_mask = self.scale_down(mask, pts)
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'raw_pts': pts,
            'scale_pts': scale_pts,
            'scale_mask': scale_mask,
            'points': points
        }
        return ret_dict
    
    def save_ckpt(self, step):
        output_path = self.output_path / 'ckpt'
        output_path.mkdir(exist_ok=True)
        fname = str(step).zfill(6) + '.pth'
        ckpt_fname = output_path / fname
        ckpt_fname = str(ckpt_fname)
        ckpt = {
            'estimator': self.estimator.state_dict(),
            'model': self.ngp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'prop_networks': [p.state_dict() for p in self.proposal_networks],
            'prop_optimizer': self.prop_optimizer.state_dict()
        }
        torch.save(ckpt, ckpt_fname)
        return ckpt
        
        
class NGPPropWrapperV2():
    def __init__(self, config, training=False) -> None:
        self.config = config
        self.device = config.device
        max_steps = 20000
        init_batch_size = 4096
        weight_decay = 1e-6
        self.training = training
        # outputs
        current_ts = str(time.time()).split('.')[0]
        self.output_path = P(config.output_dir) / config.exp_name / current_ts
        # scene parameters
        unbounded = False
        aabb = config.aabb
        self.near_plane = config.near_plane
        self.far_plane = config.far_plane
        # dataset parameters
        # model parameters
        proposal_networks = [
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=128,
            ).to(self.device),
        ]
        # render parameters
        self.num_samples = 64
        self.num_samples_per_prop = [128]
        self.sampling_type = "uniform"
        self.opaque_bkgd = False
        
        # 
        # scene = 'lego'
        # data_root = "/home/dj/Downloads/project/nerfacc_ngp/third_party/nerfacc/data/nerf_synthetic"
        # from datasets.nerf_synthetic import SubjectLoader
        # self.train_dataset = SubjectLoader(
        #     subject_id=scene,
        #     root_fp=data_root,
        #     split="train",
        #     num_rays=init_batch_size,
        #     device=self.device,
        #     **train_dataset_kwargs,
        # )

        # self.test_dataset = SubjectLoader(
        #     subject_id=scene,
        #     root_fp=data_root,
        #     split="test",
        #     num_rays=None,
        #     device=self.device,
        #     **test_dataset_kwargs,
        # )
        
        # ========================
        
        prop_optimizer = torch.optim.Adam(
            itertools.chain(
                *[p.parameters() for p in proposal_networks],
            ),
            lr=1e-2,
            eps=1e-15,
            weight_decay=weight_decay,
        )
        prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    prop_optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field = NGPradianceField(aabb=aabb, unbounded=unbounded).to(self.device)
        optimizer = torch.optim.Adam(
            radiance_field.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        proposal_requires_grad_fn = get_proposal_requires_grad_fn()
        
        self.proposal_networks = proposal_networks
        self.estimator = estimator
        self.prop_optimizer = prop_optimizer
        self.prop_scheduler = prop_scheduler
        self.grad_scaler = grad_scaler
        self.radiance_field = radiance_field
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.proposal_requires_grad_fn = proposal_requires_grad_fn
        
        # load pretrain
        ckpt_path = config.pre_trained_weights
        if self.train:
            if config.resume_training:
                ckpt_dict = torch.load(ckpt_path)
                self.radiance_field.load_state_dict(ckpt_dict['model'])
                self.estimator.load_state_dict(ckpt_dict['estimator'])
                self.proposal_networks[0].load_state_dict(ckpt_dict['prop_network'])
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.radiance_field.load_state_dict(ckpt_dict['model'])
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            self.proposal_networks[0].load_state_dict(ckpt_dict['prop_network'])
        
    def render_image_with_propnet(self, rays, render_bkgd, proposal_requires_grad):
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = chunk_rays.origins[..., None, :]
            t_dirs = chunk_rays.viewdirs[..., None, :]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            sigmas = proposal_network(positions)
            if self.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[..., None, :]
            t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            rgb, sigmas = self.radiance_field(positions, t_dirs)
            if self.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return rgb, sigmas.squeeze(-1)

        results = []
        chunk = (
            torch.iinfo(torch.int32).max
            if self.radiance_field.training
            else 8192
        )
        
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            t_starts, t_ends = self.estimator.sampling(
                prop_sigma_fns=[
                    lambda *args: prop_sigma_fn(*args, p) for p in self.proposal_networks
                ],
                prop_samples=self.num_samples_per_prop,
                num_samples=self.num_samples,
                n_rays=chunk_rays.origins.shape[0],
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                sampling_type=self.sampling_type,
                stratified=self.radiance_field.training,
                requires_grad=proposal_requires_grad,
            )
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices=None,
                n_rays=None,
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [rgb, opacity, depth]
            results.append(chunk_results)

        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )
    
    def train(self, data, step):
        self.radiance_field.train()
        self.estimator.train()
        for p in self.proposal_networks:
            p.train()
        render_bkgd = data["color_bkgd"].to(self.device)
        raw_rays = data["rays"]
        rays_o = raw_rays.origins.to(self.device)
        rays_d = raw_rays.viewdirs.to(self.device)
        rays = Rays(origins=rays_o, viewdirs=rays_d)
        
        pixels = data["pixels"].to(self.device)
        proposal_requires_grad = self.proposal_requires_grad_fn(step)
        rgb, acc, depth, extras = self.render_image_with_propnet(rays, render_bkgd, proposal_requires_grad)
        self.estimator.update_every_n_steps(extras['trans'], proposal_requires_grad, loss_scaler=1024)
        loss = F.smooth_l1_loss(rgb, pixels)

        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        with torch.no_grad():
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
        return psnr
    
    
class NGP_Prop_Art_Wrapper(NGP_Prop_Wrapper):
    def __init__(self, config, training=False, ignore_empty=False, use_timestamp=False, mkdir=True) -> None:
        self.seg_classes = config.seg_classes
        # revolute only
        self.motion_type = 'r'
        self.ignore_empty = ignore_empty
        # initialize pose param
        self.configure_pose_module(config.device)
        
        super().__init__(config, training, use_timestamp=use_timestamp, mkdir=mkdir)
    
    def configure_pose_module(self, device):
        self.pose_module_list = []
        for i in range(self.seg_classes - 1):
            self.pose_module_list += [PoseModule().to(device)]
        self.opt_seg = True
    
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
            # pose_dict = {
            #     "Q": torch.Tensor([ 0.9486, -0.3434,  0.0015, -0.0012]).to(self.device),
            #     "axis_origin": torch.Tensor([0.0073, 0.7525, 0.1010])
            # }
            # self.pose_module_list[0].load_state_dict(pose_dict)
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
                    gamma=0.5,
                ),
            ]
        )
        pose_params = []
        for i in range(len(self.pose_module_list)):
            Q_param_dict = {}
            k = 'pose_' + str(i)
            Q_param_dict['name'] = k + '_Q'
            Q_param_dict['params'] = self.pose_module_list[i].Q
            Q_param_dict['lr'] = 1e-4
            T_param_dict = {}
            T_param_dict['name'] = k + '_origin'
            T_param_dict['params'] = self.pose_module_list[i].axis_origin
            T_param_dict['lr'] = 1e-4
            pose_params += [Q_param_dict, T_param_dict]
        self.pose_optimizer = torch.optim.Adam(
            pose_params, lr=1e-5,
            eps=1e-15,
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
            seg = seg.view(bs, self.seg_classes, -1, self.seg_classes) #  [B, seg_classes, samples, seg_classes+1] background class
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
    
    # def render_whole_img(self, rays, render_bkgd):
    #     return super().render_whole_img(rays, render_bkgd)
    
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
    
    def train(self, data, step):
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        # duplicate rays
        rays = self.gather_rays(dirs, c2w, is_training=True)
        render_bkgd = data["color_bkgd"].to(self.device)
        pixels = data["pixels"].to(self.device)
        rgb, opacity, _, extras = self.render_rays(rays, render_bkgd, self.ngp, self.estimator)
        rgb_loss = F.mse_loss(rgb, pixels)
        
        # loss = rgb_loss + 0.5 * opa_loss
        # loss = opa_loss
        loss = rgb_loss
        opa_gt = data['mask'].to(opacity)
        opa_loss = F.smooth_l1_loss(opacity, opa_gt)
        # add segmentation regularization
        seg = extras['seg']
        
        seg_sum = seg.sum(dim=-1) + 1e-8
        seg_max, _ = seg.max(dim=-1)
        # seg_mask = seg_sum > 0.5
        # seg_opa = seg_max[seg_mask] / seg_sum[seg_mask]
        seg_opa = seg_max / seg_sum
        seg_entropy = entropy_loss(seg_opa)
        if self.config.use_opa_entropy and self.opt_seg:
            loss += 0.1 * seg_entropy
        
        self.optimizer.zero_grad()
        if self.opt_seg:
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            if self.config.accum_steps != 0:
                loss = opa_loss / self.config.accum_steps
                loss.backward()
                if step % self.config.accum_steps == 0:
                    self.pose_optimizer.step()
            else:
                self.pose_optimizer.step()
        with torch.no_grad():
            psnr_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        
        ret_dict = {
            'loss': rgb_loss,
            'psnr': psnr
        }
        
        if self.ignore_empty:
            co_mask = self.collect_empty_idxs(data['mask'], opacity)
            ret_dict['co_mask'] = co_mask
        
        return ret_dict
    
    def collect_empty_idxs(self, acc_gt, acc_pred):
        acc_gt = acc_gt.to(acc_pred)
        acc_mask = acc_gt == 0
        pred_mask = acc_pred == 0
        co_mask = torch.logical_and(acc_mask.view(-1), pred_mask)
        
        return co_mask
    
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
    
    @torch.inference_mode()
    def collect_estimate_valid(self, data):
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        pixels = data["pixels"].to(self.device)
        rays = self.gather_rays(dirs, c2w)
        
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        h, w, _ = data['rays'].origins.shape
        acc = acc.view(h, w, 1)
        pass
    
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
        psnr_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        seg = extras['seg_all'].reshape(h, w, -1)
        if seg.shape[-1] == 2:
            extra_seg = torch.zeros(h, w, 1).to(seg)
            seg_vis = torch.cat([seg, extra_seg], dim=-1)
        else:
            seg_vis = seg
        seg_img = tvF.to_pil_image(seg_vis.cpu().permute(2, 0, 1))
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'img_pred': img_pred,
            'img_gt': img_gt,
            'psnr': psnr,
            'seg_img': seg_img
        }
        return ret_dict
    
    def get_seg_color(self, valid_seg):
        color_map = plt.get_cmap('Set3', self.seg_classes)
        cmap_tensor = torch.Tensor(color_map.colors[:, :3]).to(valid_seg)
        if valid_seg.dim() >2:
            seg_shape = valid_seg.shape
            seg_color = torch.einsum('ij,jk->ik', valid_seg.view(-1, self.seg_classes), cmap_tensor)
            
            seg_color = seg_color.view(*seg_shape[:-1], 3)
        else:
            seg_color = torch.einsum('ij,jk->ik', valid_seg, cmap_tensor)
        
        return seg_color
    
    @torch.inference_mode()
    def scan_nerf(self, step, N=256):
        
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
        _, sigma, seg = self.ngp(query_pos, query_pos)
        sigma = sigma.view(N, N, N, 1)
        act_seg = torch.nn.functional.softmax(seg, dim=-1)
        act_seg = act_seg.view(N, N, N, self.seg_classes)
        vis_seg = torch.cat([act_seg, torch.zeros_like(sigma)], dim=-1)
        opacity = 1 - torch.exp(-sigma * dist)
        valid_grid = opacity > 0.1
        pts = pos[valid_grid.view(N, N, N)]
        # seg_color = vis_seg[valid_grid.view(N, N, N)]
        valid_seg = act_seg[valid_grid.view(N, N, N)]
        seg_color = self.get_seg_color(valid_seg)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(seg_color.cpu().numpy())
        fname = str(self.eval_path.parent / f'seg_point_cloud_{step}.ply')
        o3d.io.write_point_cloud(fname, pcd)
        print('saved nerf seg scan!')
        
        if self.seg_classes == 2:
            seg_mask = valid_seg[:, 1] > 0.5
            part_pts = pts[seg_mask]
        else:
            _, seg_i = valid_seg.max(dim=1)
            part_list = []
            for seg_class in range(self.seg_classes):
                if seg_class == 0:
                    pass
                else:
                    part_list += [pts[seg_i == seg_class]]
            part_pts = part_list
        
        
        return part_pts
    
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
        seg = []
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            rgb, opacity, depth, extras = self.render_rays(chunk_rays, render_bkgd=render_bkgd, 
                             ngp=self.ngp, estimator=self.estimator)
            chunk_out = [rgb, opacity, depth]
            results.append(chunk_out)
            seg.append(extras['seg'])
        
        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )
        seg_all = torch.cat(seg, dim=0)
        extras['seg_all'] = seg_all
        return (
            colors.view((valid_ray_num, -1)),
            opacities.view((valid_ray_num, -1)),
            depths.view((valid_ray_num, -1)),
            extras,
        )

class NGP_Prop_Art_Seg_Wrapper(NGP_Prop_Art_Wrapper):
    def __init__(self, config, training=False, ignore_empty=False, use_timestamp=False, use_se3=False, motion_type='r', mkdir=True) -> None:
        self.seg_classes = config.seg_classes
        # revolute only
        self.motion_type = motion_type
        self.ignore_empty = ignore_empty
        # initialize pose param
        self.use_se3 = use_se3
        self.best_psnr = 0
        self.best_ckpt = None
        self.config = config
        self.configure_pose_module(config.device)
        self.part_list = None
        super().__init__(config, training, ignore_empty, use_timestamp=use_timestamp, mkdir=mkdir)
        
        
    def configure_pose_module(self, device):
        self.pose_module_list = []
        pose_m = PoseModule_se3 if self.use_se3 else PoseModule
        for i in range(self.seg_classes - 1):
            self.pose_module_list += [pose_m().to(device)]
        self.opt_seg = True
        self.config_pose_optimizer()
        # for p in self.pose_module_list:
        #     p.to(device)
        # self.pose_optimizer_se3.to(device)
        # self.pose_scheduler_se3.to(device)
        return 
    
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
                p.load_state_dict(prop_networks[idx], strict=False)
            # pose_dict = {
            #     "Q": torch.Tensor([ 0.9486, -0.3434,  0.0015, -0.0012]).to(self.device),
            #     "axis_origin": torch.Tensor([0.0073, 0.7525, 0.1010])
            # }
            # self.pose_module_list[0].load_state_dict(pose_dict)
            if self.config.resume_training:
                # load optimizer
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        else:
            ckpt_dict = torch.load(ckpt_path)
            self.ngp.load_state_dict(ckpt_dict['model'])
            self.estimator.load_state_dict(ckpt_dict['estimator'])
            prop_networks = ckpt_dict['prop_networks']
            for idx, p in enumerate(self.proposal_networks):
                p.load_state_dict(prop_networks[idx])
            # pose_dict = {
            #     "Q": torch.Tensor([ 0.9486, -0.3434,  0.0015, -0.0012]).to(self.device),
            #     "axis_origin": torch.Tensor([0.0073, 0.7525, 0.1010]).to(self.device)
            # }
            # self.pose_module_list[0].load_state_dict(pose_dict)
            pose_params = ckpt_dict['pose_params']
            for id, pose_param in enumerate(self.pose_module_list):
                pose_param.load_state_dict(pose_params[id])
                
    
    def config_estimator(self):
        # self.proposal_networks = [
        #     NGPDensityField(
        #         aabb=self.config.aabb,
        #         unbounded=False,
        #         n_levels=5,
        #         max_resolution=128
        #     ).to(self.config.device),
        # ]
        self.proposal_networks = [
            NGPDensitySegField(
                aabb=self.config.aabb,
                unbounded=False,
                n_levels=5,
                max_resolution=128,
                seg_classes=self.seg_classes
            ).to(self.config.device),
        ]
        
        self.prop_optimizer = torch.optim.Adam(
                itertools.chain(
                    *[p.mlp_seg.parameters() for p in self.proposal_networks],
                ),
                # self.proposal_networks[0].parameters(),
                lr=1e-2,
                eps=1e-15,
                weight_decay=self.config.weight_decay,
            )
        self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.prop_optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        self.estimator = PropArtEstimator(
            self.prop_optimizer, 
            self.prop_scheduler).to(self.config.device)
    
    def render_rays(self, rays: Rays, render_bkgd, ngp, estimator, proposal_requires_grad=False, test_chunk_size=8192):
        
        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = rays.origins.unsqueeze(-2)
            t_dirs = rays.viewdirs.unsqueeze(-2)
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            sigmas = proposal_network(positions)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)
        
        def prop_sigma_seg_fn(t_starts, t_ends, proposal_network):
            t_origins = rays.origins.unsqueeze(-2).detach()
            t_dirs = rays.viewdirs.unsqueeze(-2).detach()
            positions = t_origins + t_dirs * (t_starts + t_ends).unsqueeze(-1) / 2.0
            sigmas, segs = proposal_network(positions)
            if self.config.opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1), segs

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
        
        t_starts, t_ends = estimator.seg_sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_seg_fn(*args, p) for p in self.proposal_networks
            ],
            prop_samples=self.config.num_samples_per_prop,
            num_samples=self.config.num_samples,
            n_rays=rays.origins.shape[0],
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            sampling_type=self.config.sampling_type,
            stratified=ngp.training,
            requires_grad=proposal_requires_grad,
            use_background=self.config.use_background,
            seg_classes=self.seg_classes
        )
        
        def composite_rendering(t_starts, t_ends):
            dist = t_ends - t_starts
            
            if self.config.use_background:
                valid_seg = self.seg_classes - 1
            else:
                valid_seg = self.seg_classes
                
            rgb, sigma, seg = rgb_sigma_seg_fn(t_starts, t_ends)
            
            bs = int(rgb.shape[0] / valid_seg)
            
            seg = seg.view(bs, self.seg_classes, -1, self.seg_classes) #  [B, seg_classes, samples, seg_classes+1] background class
            act_seg = torch.nn.functional.softmax(seg, dim=-1)
            
            rgb = rgb.view(bs, valid_seg, -1, 3)
            sigma = sigma.view(bs, valid_seg, -1) # [B, seg_classes, nsamples]
            dist = dist.view(bs, valid_seg, -1) # [B, seg_classes, nsamples]
            
            
            seg_mask = torch.cat([act_seg[:, i:i+1, :, i] for i in range(valid_seg)], dim=1) # [B, seg_classes, samples]
            
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
            pos = pos.view(bs, valid_seg, -1)
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
            
            # gather seg_masked trans to train the estimator
            ray_num = t_starts.shape[0]
            seg_sigmas = seg_mask * sigma
            seg_sigmas_reshape = seg_sigmas.view(ray_num, -1)
            trans_seg, _ = render_transmittance_from_density(t_starts, t_ends, seg_sigmas_reshape)
            extras['trans_seg'] = trans_seg
            
            return final_color, depth, acc, extras
        
        rgb, depth, opacity, extras = composite_rendering(t_starts, t_ends)
        
        # extras['est_sigmas'] = sigmas_combine
        # extras['est_trans'] = est_trans
        
        return rgb, opacity, depth, extras
    
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
        chunk = int(4096 * self.seg_classes)
        seg = []
        est_sigmas = []
        est_trans = []
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            rgb, opacity, depth, extras = self.render_rays(chunk_rays, render_bkgd=render_bkgd, 
                             ngp=self.ngp, estimator=self.estimator)
            chunk_out = [rgb, opacity, depth]
            results.append(chunk_out)
            seg.append(extras['seg'])
            # est_sigmas.append(extras['est_sigmas'])
            # est_trans.append(extras['est_trans'])
            
        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )
        seg_all = torch.cat(seg, dim=0)
        extras['seg_all'] = seg_all
        # extras['sigmas_vis'] = torch.cat(est_sigmas, dim=0)
        # extras['trans_vis'] = torch.cat(est_trans, dim=0)
        return (
            colors.view((valid_ray_num, -1)),
            opacities.view((valid_ray_num, -1)),
            depths.view((valid_ray_num, -1)),
            extras,
        )
    
    def config_init_optimizer(self, lr=1e-3):
        self.init_optimizer = torch.optim.Adam(
            self.ngp.seg_mlp.parameters(),
            lr=lr,
            eps=1e-15,
            weight_decay=self.config.weight_decay,
        )
        pass
    
    def init_seg(self, part_list, init_steps=1000, optimizer=None, lr=1e-4):
        # self.
        # self.ngp()
        seg_labels_list = []
        label_cnt = [0]
        
        self.config_init_optimizer(lr=lr)
        if optimizer is None:
            cur_optimizer = self.init_optimizer
        else:
            cur_optimizer = optimizer
        for i, p in enumerate(part_list):
            seg_label = torch.ones([p.shape[0]]) * (i + 1)
            seg_labels_list += [seg_label.to(part_list[0]).long()]
            label_cnt += [p.shape[0]]
        
        seg_labels = torch.cat(seg_labels_list, dim=0).to(part_list[0]).long()
        # weights = torch.Tensor([0, ])
        label_cnt = torch.Tensor(label_cnt)
        weights = label_cnt / label_cnt.sum()
        weights = weights.view(self.seg_classes).to(part_list[0])
        for s in range(init_steps):
            query_pos = torch.cat(part_list, dim=0)
            _, _, seg = self.ngp(query_pos, query_pos)
            seg_act = torch.nn.functional.softmax(seg, dim=-1)
            seg_loss = torch.nn.functional.nll_loss(seg_act, seg_labels, weight=weights)
            cur_optimizer.zero_grad()
            seg_loss.backward()
            cur_optimizer.step()
        pass
    
    
    @torch.inference_mode()
    def forward_check(self):
        pass
    
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
        psnr_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        seg = extras['seg_all'].reshape(h, w, -1)
        if seg.shape[-1] == 2:
            extra_seg = torch.zeros(h, w, 1).to(seg)
            seg_vis = torch.cat([seg, extra_seg], dim=-1)
        else:
            seg_vis = seg
            seg_vis = self.get_seg_color(seg)
        seg_img = tvF.to_pil_image(seg_vis.cpu().permute(2, 0, 1))
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'img_pred': img_pred,
            'img_gt': img_gt,
            'psnr': psnr,
            'seg_img': seg_img
        }
        return ret_dict
    
    def config_optimizer_multi_part(self):
        self.optimizer = torch.optim.Adam(
            self.ngp.seg_mlp.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                # torch.optim.lr_scheduler.LinearLR(
                #     self.optimizer, start_factor=0.01, total_iters=100
                # ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        self.config.max_steps // 4,
                        # self.config.max_steps * 2 // 4,
                        # self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=0.1,
                ),
            ]
        )
        pose_params = []
        for i in range(len(self.pose_module_list)):
            Q_param_dict = {}
            k = 'pose_' + str(i)
            Q_param_dict['name'] = k + '_Q'
            Q_param_dict['params'] = self.pose_module_list[i].Q
            Q_param_dict['lr'] = 1e-4
            T_param_dict = {}
            T_param_dict['name'] = k + '_origin'
            T_param_dict['params'] = self.pose_module_list[i].axis_origin
            T_param_dict['lr'] = 1e-4
            pose_params += [Q_param_dict, T_param_dict]
        self.pose_optimizer = torch.optim.Adam(
            pose_params, lr=1e-5,
            eps=1e-15,
        )
        return 
    
    @torch.inference_mode()
    def test(self, data):
        for p in self.proposal_networks:
            p.eval()
        self.ngp.eval()
        self.estimator.eval()
        render_bkgd = data["color_bkgd"].to(self.device)
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        # pixels = data["pixels"].to(self.device)
        
        # duplicate rays
        rays = self.gather_rays(dirs, c2w)
        
        rgb, acc, depth, extras = self.render_whole_img(rays, render_bkgd=render_bkgd)
        h, w, _ = data['rays'].origins.shape
        rgb = rgb.view(h, w, 3)
        acc = acc.view(h, w, 1)
        depth = acc.view(h, w, 1)
        img_pred = tvF.to_pil_image(rgb.cpu().permute(2, 0, 1))
        # img_gt = tvF.to_pil_image(pixels.cpu().permute(2, 0, 1))
        # psnr_loss = F.mse_loss(rgb, pixels)
        # psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        seg = extras['seg_all'].reshape(h, w, -1)
        if seg.shape[-1] == 2:
            extra_seg = torch.zeros(h, w, 1).to(seg)
            seg_vis = torch.cat([seg, extra_seg], dim=-1)
        else:
            seg_vis = seg
            seg_vis = self.get_seg_color(seg)
        seg_img = tvF.to_pil_image(seg_vis.cpu().permute(2, 0, 1))
        ret_dict = {
            'rgb': rgb,
            'acc': acc,
            'depth': depth,
            'img_pred': img_pred,
            # 'img_gt': img_gt,
            # 'psnr': psnr,
            'seg_img': seg_img,
            'seg_label': seg
        }
        return ret_dict
    
    def train(self, data, step, use_init=False, percentage=0.5):
        dirs = data['dirs'].to(self.device)
        c2w = data['c2w'].to(self.device)
        # duplicate rays
        rays = self.gather_rays(dirs, c2w, is_training=True)
        render_bkgd = data["color_bkgd"].to(self.device)
        pixels = data["pixels"].to(self.device)
        proposal_requires_grad = self.proposal_requires_grad_fn(step)
        rgb, opacity, _, extras = self.render_rays(rays, render_bkgd, self.ngp, self.estimator, proposal_requires_grad=proposal_requires_grad)
        self.estimator.update_every_n_steps(
        extras["trans_seg"], proposal_requires_grad, loss_scaler=1024
        )
        rgb_loss = F.smooth_l1_loss(rgb, pixels)
        
        # loss = rgb_loss + 0.5 * opa_loss
        # loss = opa_loss
        loss = rgb_loss
        opa_gt = data['mask'].to(opacity)
        opa_loss = F.smooth_l1_loss(opacity.view(-1), opa_gt.view(-1))
        # add segmentation regularization
        seg = extras['seg']
        
        seg_sum = seg.sum(dim=-1) + 1e-8
        seg_max, _ = seg.max(dim=-1)
        # seg_mask = seg_sum > 0.5
        # seg_opa = seg_max[seg_mask] / seg_sum[seg_mask]
        seg_opa = seg_max / seg_sum
        seg_entropy = entropy_loss(seg_opa)
        if self.config.use_opa_entropy and self.opt_seg:
            loss += 0.1 * seg_entropy
        
        self.optimizer.zero_grad()
        if self.opt_seg:
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            if self.config.accum_steps != 0:
                loss = opa_loss / self.config.accum_steps
                loss.backward()
                if step % self.config.accum_steps == 0:
                    self.pose_optimizer_se3.step()
                    self.pose_scheduler_se3.step()
                    self.pose_optimizer_se3.zero_grad()
            else:
                self.pose_optimizer_se3.step()
        with torch.no_grad():
            psnr_loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(psnr_loss) / np.log(10.0)
        
        ret_dict = {
            'loss': rgb_loss,
            'psnr': psnr
        }
        
        if self.ignore_empty:
            co_mask = self.collect_empty_idxs(data['mask'], opacity)
            ret_dict['co_mask'] = co_mask
        
        if use_init:
            if torch.rand(1) < percentage:
                self.init_seg(self.part_list, init_steps=1)
        
        return ret_dict
    
    def config_pose_optimizer(self):
        q_param_list = []
        origin_param_list = []
        dir_param_list = []
        scale_param_list = []
        # for p in self.pose_module_list:
        #     q_param_list += [{
        #         "params": p.Q
        #     }]
        #     origin_param_list += [{
        #         "params":p.axis_origin
        #     }]
        #     dir_param_list += [{
        #         "params": p.dir
        #     }]
        #     scale_param_list += [
        #         {
        #             "param": p.scale
        #         }
        #     ]
        
        if self.motion_type == 'r':
            # Q_params = [
            #     {"params":  q_param_list}
            # ]
            # origin_params = [
            #     {"params": o } for o in origin_param_list
            # ]
            # opt_params = [*Q_params, *origin_params]
            # opt_params = q_param_list + origin_param_list
            for p in self.pose_module_list:
                p.scale.requires_grad=False
                p.dir.requires_grad=False
        else:
            for p in self.pose_module_list:
                p.Q.requires_grad=False
                p.axis_origin.requires_grad=False
            # dir_params = [
            #     {"params": d} for d in dir_param_list
            # ]
            # scale_params = [
            #     {"params":s} for s in scale_param_list
            # ]
            # opt_params = [*dir_params, *scale_params]
            # opt_params = dir_param_list + scale_param_list
        self.pose_optimizer_se3 = torch.optim.Adam(
            [
                {"params": p.parameters()} for p in self.pose_module_list
                ],
            lr=self.config.pose_lr
        )
        self.pose_scheduler_se3 = torch.optim.lr_scheduler.StepLR(self.pose_optimizer_se3, step_size=50, gamma=0.1)
        
    def save_ckpt(self, step, psnr):
        output_path = self.output_path / 'ckpt'
        output_path.mkdir(exist_ok=True)
        fname = str(step).zfill(6) + '.pth'
        ckpt_fname = output_path / fname
        ckpt_fname = str(ckpt_fname)
        ckpt = {
            'estimator': self.estimator.state_dict(),
            'model': self.ngp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'prop_networks': [p.state_dict() for p in self.proposal_networks],
            'prop_optimizer': self.prop_optimizer.state_dict()
        }
        pose_param = [i.state_dict() for i in self.pose_module_list]
        ckpt['pose_params'] = pose_param
        torch.save(ckpt, ckpt_fname)
        
        if psnr > self.best_psnr:
            best_ckpt_fname = output_path / 'best_ckpt.pth'
            self.best_ckpt = ckpt
            torch.save(self.best_ckpt, str(best_ckpt_fname))
        
        return ckpt