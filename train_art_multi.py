import os
from config import get_opts
opts = get_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)
from models.ngp_wrapper import NGP_Prop_Art_Wrapper, NGP_Prop_Wrapper, NGP_Prop_Art_Seg_Wrapper
import torch
from torch.utils.data import DataLoader
from dataset.sapien import SapienParisDataset
from tqdm import tqdm
import open3d as o3d
import sys
import traceback
from pose_estimation import PoseEstimator_multipart
from test_ngp import NGPevaluator
from dataset.pose_utils import quaternion_to_axis_angle, get_rotation_axis_angle
from dataset.io_utils import load_gt_from_json, load_multipart_gt
from models.utils import axis_metrics, geodesic_distance, translational_error
import numpy as np
import torch.nn.functional as F
import shutil

# from dataset.ray_utils import *
class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)


if __name__ == '__main__':
    # load config
    opts = get_opts()
    
    print("=" * 100)
    print(f"running exp: {opts.exp_name}")
    print("=" * 100)
    
    # sys.stdout = TracePrints()
    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    setattr(opts, 'device', device)
    
    # gt_info = load_gt_from_json(opts.motion_gt_json, opts.state, opts.motion_type)
    gt_infos = load_multipart_gt(opts.motion_gt_json, state=opts.state, motion_type=opts.motion_type)
    
    # load model
    ignore_empty = False
    if opts.use_art_seg_estimator:
        model = NGP_Prop_Art_Seg_Wrapper(config=opts, training=True, 
                                         ignore_empty=ignore_empty, use_timestamp=True, use_se3=opts.use_se3)
    else:
        model = NGP_Prop_Art_Wrapper(config=opts, training=True, ignore_empty=ignore_empty)
    # co_mask = 'stapler_art_mask.pth'
    # load dataset
    train_dataset = SapienParisDataset(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='train',
        render_bkgd='white',
        ignore_empty=ignore_empty,
        co_mask=None,
        state=opts.state
    )
    
    test_dataset = SapienParisDataset(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='val',
        render_bkgd='white',
        state=opts.state
    )
    pose_lr = 1e-1
    # load pose estimator
    print('=' * 100)
    print('=' * 40, 'loading coarse pose estimator', '=' * 40)
    print('=' * 100)
    # pretrain_cfg = opts.pretrained_config
    # pretrain_strs = ["--config", pretrain_cfg]
    # pretrain_opts = get_opts(pretrain_strs)
    
        
    pretran_model = NGP_Prop_Wrapper(opts, training=False)
    
    renderer = NGPevaluator(opts, dataset=train_dataset, model=pretran_model)
    print('=' * 100)
    print('=' * 40, 'runing coarse pose estimation', '=' * 40)
    print('=' * 100)
    if opts.idx_list is None:
        idx_list = None
    else:
        idx_list = np.asarray(opts.idx_list)
    estimator = PoseEstimator_multipart(renderer=renderer, dataset=train_dataset, output_dir=model.eval_path, use_num_frames=opts.use_num_frames, device=device, scaling=0.5, use_se3=opts.use_se3, motion_type=opts.motion_type, idx_list=idx_list, eps=opts.eps, select_frame=opts.use_num_frames, N=opts.voxel_res, num_dy_parts=opts.seg_classes-1)
    print(f'current training image samples {estimator.idx_list}')
    # if 'laptop' in opts.exp_name:
    #     estimator.configure_optimizer(gamma=0.5, lr=1e-2, scheduler_step=2000)
    # if opts.use_num_frames < 8:
    for p in estimator.pose_param: p.init_param()
    estimator.configure_optimizer(scheduler_step=500)
            
    # estimator.shuffle_idx_list()
    # print(f'current training image samples {estimator.idx_list}')
    cur_loss = estimator.estimate_pose_accum_grad(200, accum_iter=32, norm_Q=False)
    for i in range(2):
        if cur_loss < 1 :
            break
        else:
            for p in estimator.pose_param:
                p.init_param()
            # estimator.shuffle_idx_list()
            estimator.configure_optimizer(lr_Q=1e-2, lr_T=1e-2, lr_dir=1e-1, lr_scale=1e-2, scheduler_step=500, gamma=0.5)
            cur_loss = estimator.estimate_pose_accum_grad(1000, accum_iter=4, norm_Q=False)
    # for pos_idx in range(estimator.num_dy_part):
        
    #     estimator.configure_optimizer_separate(scheduler_step=500, pos_idx=pos_idx)
            
    #     estimator.estimate_pose_accum_grad(400, accum_iter=32)
    
    # pose_dict = estimator.pose_param.state_dict()
    # model.pose_module_list[0].load_state_dict(pose_dict)
    for i in range(opts.seg_classes - 1):
        cur_dict = estimator.pose_param[i].state_dict()
        model.pose_module_list[i].load_state_dict(cur_dict)
        pass
    
    
    print('=' * 100)
    print('=' * 40, 'Coarse pose estimation results:', '=' * 40)
    for p in estimator.pose_param:
        print(f'estimated quaternion: {p.Q.detach().cpu().numpy()}')
        axis_dir, radian, angles = quaternion_to_axis_angle(p.Q)
        norm_dir = torch.nn.functional.normalize(p.dir.view(1, -1))
        translation = norm_dir * p.scale
    
        print(f'axis_dir: {axis_dir.detach().cpu().numpy()}, angles: {angles.detach().cpu().numpy()}, axis origin: {p.axis_origin.detach().cpu().numpy()}, translation: {translation.detach().cpu().view(-1).numpy()}')
    
    print('=' * 100)
    # ============================================================================================================
    train_dataset.idx_list = np.asarray(estimator.idx_list)
    idx_list_fname = model.eval_path / 'train_set.txt'
    with open(str(idx_list_fname), 'w') as f:
        f.write(str(estimator.idx_list.tolist()))
    if opts.ignore_empty:
        train_dataset.ignore_empty = True
        non_empty_fname = str(estimator.occu_mask_fname)
        train_dataset.load_co_masks(non_empty_fname)
    # ============================================================================================================
    # use importance sampling
    imp_sampling = opts.imp_sampling
    if imp_sampling:
        train_dataset.importance_pixels = estimator.cache_idxs
        train_dataset.importance_sampling = True
        train_dataset.imp_sampling_rate = 0.5

    
    # ============================================================================================================
    
    model.part_list = estimator.part_pts_list
    if opts.use_init_seg:
        print("=" * 50)
        print(f"start init_seg optimizing")
        # model.part_list = estimator.part_pts_list
        model.init_seg(part_list=estimator.part_pts_list, init_steps=1000, lr=1e-4)
        print("=" * 50)
    model.scan_nerf(0)
    # ============================================================================================================
    # i = 10
    # eval_data = test_dataset.__getitem__(i)
    # eval_dict = model.eval(eval_data)
    # # save images 
    # img_gt = eval_dict['img_gt']
    # img_pred = eval_dict['img_pred']
    # img_seg = eval_dict['seg_img']
    # # eval_pcd += [eval_dict['points']]
    # # eval_pcd_color += [eval_dict['points_color']]
    # img_gt.save(model.eval_path / f'img_gt_{i:04d}.png')
    # img_pred.save(model.eval_path / f'img_pred_{i:04d}.png')
    # img_seg.save(model.eval_path / f'img_seg_{i:04d}.png')
    train_bar = tqdm(range(opts.max_steps + 2))
    # model.config_optimizer_multi_part()
    percentage = 1
    use_init = False
    shutil.copy(opts.config, str(model.eval_path / 'config.json'))
    init_lr = opts.init_lr
    # ============================================================================================================
    for step in train_bar:
        # print('training step: %d' % step, end='\r')
        if step > opts.init_start_step:
            use_init = True
            
        data = train_dataset.__getitem__(step)
        ret_dcit = model.train(data, step, use_init=False, percentage=percentage)
        # train_dataset.adjust_batch_size(ret_dcit['new_batch_size'])
        # cur_bs = ret_dcit['new_batch_size']
        if ret_dcit == -1:
            psnr = -1
        else:
            psnr = ret_dcit['psnr'].item()
        loss = ret_dcit['loss']
        # if (step < 10000) & ignore_empty:
        #     co_mask = ret_dcit['co_mask']
        #     img_idxs = data['img_idxs'][co_mask.cpu()]
        #     pix_idxs = data['pix_idxs'][co_mask.cpu()]
        #     train_dataset.update_non_empty_mask(img_idxs, pix_idxs)
        #     pass
        valid_pix = train_dataset.cur_valid_idx.shape[0]
        train_bar.set_description(f'trainig at step: {step}, current psnr: {psnr:.2f}, current loss: {loss:.2E}, current valid pix: {valid_pix}')
        
        if step > 3500:
            model.ignore_empty = False
            train_dataset.ignore_empty = False
        
        # if step > 4000:
        #     model.opt_seg = False
        # elif step > 8000:
        #     model.opt_seg = True
        
        
        
        if step % opts.init_interval_steps == 0:
            if use_init:
                # percentage *= 0.5
                print("=" * 50)
                print(f"start init_seg optimizing")
                model.part_list = estimator.part_pts_list
                model.init_seg(part_list=estimator.part_pts_list, init_steps=opts.init_step, lr=init_lr)
                init_lr *= opts.init_lr_decay
                init_lr = max(1e-6, init_lr)
                print("=" * 50)
            model.scan_nerf(step)
        if step > 0: 
            
            if step % 1000 == 0:
                model.scan_nerf(step)
                # train_dataset.importance_sampling = False
                # train_dataset.imp_sampling_rate -= 0.05
            # if step % 1000 == 0:
            #     cache_idxs = []
            #     for i in estimator.idx_list:
            #         pose = train_dataset.poses[i]
            #         batch = train_dataset.get_rays_given_pose(pose.view(4, 4))
            #         batch['c2w'] = batch['c2w'].view(1, 4, 4)
            #         test_dict = model.test(batch)
            #         rgb_gt = train_dataset.rgb[i] # [800 * 800, 3]
            #         rgb_pred = test_dict['rgb'].view(-1, 3)
            #         rgb_diff = (rgb_pred - rgb_gt.to(rgb_pred)).sum(dim=-1)
            #         valid_diff = rgb_diff[rgb_diff > 0]
            #         diff_thres = valid_diff.max() * 0.5
            #         gt_mask = train_dataset.mask[i]
            #         pred_mask = test_dict['acc'] > 0.5
            #         xor = torch.logical_xor(gt_mask.to(pred_mask), pred_mask.view(gt_mask.shape))
            #         # pix_idx = torch.arange(rgb_gt.shape[0]).to(rgb_pred)
            #         # valid_pix_idx = pix_idx[rgb_diff > diff_thres]
            #         valid_pix_idx = xor
            #         img_idx = torch.ones_like(valid_pix_idx).to(rgb_pred) * i
            #         cache_idx = torch.cat([img_idx.view(-1, 1), valid_pix_idx.view(-1, 1)], dim=1)
            #         cache_idxs += [cache_idx]
            #         pass
                    
            #     cache_idxs = torch.cat(cache_idxs, dim=0)
            #     imp_sampling = True
            #     if imp_sampling:
            #         train_dataset.importance_pixels = cache_idxs
            #         train_dataset.importance_sampling = True
                
            
            if (step % opts.eval_step == 0) or (step == opts.max_steps):
                
                print(f'evaluating at step {step}')
                with torch.no_grad():
                    save_eval_dir = model.eval_path /f'eval_step_{step}' 
                    save_eval_dir.mkdir(exist_ok = True, parents=True)
                    psnrs = []
                    eval_pcd = []
                    eval_pcd_color = []
                    part_pts = model.scan_nerf(step)
                    # part_pts_num = part_pts.shape[0]
                    # no update
                    # if ignore_empty == False:
                    # if part_pts_num > 0:
                        # if step >= 2 * opts.eval_step:
                        # if step %  (2 * opts.eval_step) == 0:
                        # estimator.nerf_part_pts = part_pts
                        # estimator.check_new_part_pts()
                    # estimator.nerf_part_pts = part_pts
                    # estimator.check_new_part_pts()
                        # estimator.update_new_part_pts(part_pts)
                    estimator.update_part_pts_list(part_pts)
                    
                    
                    points = estimator.nerf_part_pts.detach().cpu().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd_name = str(save_eval_dir / f'part_estimation_{step}.ply')
                    o3d.io.write_point_cloud(pcd_name, pcd)
                    
                    for i in tqdm(range(len(test_dataset))):
                        eval_data = test_dataset.__getitem__(i)
                        eval_dict = model.eval(eval_data)
                        psnrs += [eval_dict['psnr'].item()]
                        # save images 
                        img_gt = eval_dict['img_gt']
                        img_pred = eval_dict['img_pred']
                        
                        img_seg = eval_dict['seg_img']
                        # eval_pcd += [eval_dict['points']]
                        # eval_pcd_color += [eval_dict['points_color']]
                        img_gt.save(save_eval_dir / f'img_gt_{i:04d}.png')
                        img_pred.save(save_eval_dir / f'img_pred_{i:04d}.png')
                        img_seg.save(save_eval_dir / f'img_seg_{i:04d}.png')
                    avg_psnr = sum(psnrs) / len(psnrs)
                    # multiview_pcd = torch.cat(eval_pcd, dim=0)
                    # multiview_color = torch.cat(eval_pcd_color, dim=0)
                    pcd_fname = f'eval_pcd_step_{step}.ply'
                    # pcd = o3d.geometry.PointCloud()
                    # pcd_pts = multiview_pcd.float().cpu().numpy()
                    # pcd_colors = multiview_color.float().cpu().numpy()
                    # pcd.points = o3d.utility.Vector3dVector(pcd_pts)
                    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                    # o3d.io.write_point_cloud(str(model.eval_path/pcd_fname), pcd)
                    
                    # print('=' * 40, 'runing coarse pose estimation', '=' * 40)
                    # print('=' * 100)
                    # # pose_lr = 1e-2
                    # estimator.configure_optimizer(lr_Q=1e-2, lr_T=1e-2)
                    # estimator.estimate_pose_accum_grad(600, accum_iter=32)
                    # pose_dict = estimator.pose_param.state_dict()
                    # model.pose_module_list[0].load_state_dict(pose_dict)
                    eval_dict_total = {}
                    pred_infos = []
                    for p_idx, p in enumerate(model.pose_module_list):
                        axis_dir, radian, angles = quaternion_to_axis_angle(p.Q)
                        pred_R = get_rotation_axis_angle(axis_dir.cpu().numpy(), radian.cpu().numpy())
                        
                        if opts.motion_type == 'r':
                            pred_axis_dir = axis_dir.detach()
                        else:
                            pred_axis_dir = p.dir.detach()
                        
                        pred_info = {
                            "axis_o": p.axis_origin.detach(),
                            "axis_d": F.normalize(pred_axis_dir.view(1, -1)).view(-1),
                            "R": torch.Tensor(pred_R),
                            "theta": radian.detach(),
                            "dist": p.scale.detach()
                        }
                        pred_infos += [pred_info]
                        
                        axis_origin = p.axis_origin.detach().cpu().numpy().tolist()
                        axis_dir = axis_dir.detach().cpu().numpy().tolist()
                        angles = angles.detach().cpu().numpy().tolist()
                        eval_dicts = []
                        for gt_info in gt_infos:
                            ang_err, pos_err = axis_metrics(pred_info, gt_info)
                            trans_err = translational_error(pred_info, gt_info)
                            geo_dist = geodesic_distance(torch.Tensor(pred_R), gt_info['R'])
                            metric_fname = model.eval_path / f'motion_metric_{step}.json'
                            
                            
                            norm_dir = torch.nn.functional.normalize(p.dir.view(1, -1))
                            
                            translation = norm_dir * p.scale
                    # translation = estimator.pose_param.dir * estimator.pose_param.scale
                            print(f'\ncurrent estiamted pose: axis direction = {axis_dir}, angles = {angles}, axis origin = {axis_origin}, translation: {translation.detach().cpu().view(-1).numpy()}')
                            
                            
                            eval_metric_dict = {
                                "ang_err": ang_err.item(),
                                "pos_err": pos_err.item(),
                                "geo_dist": geo_dist.item(),
                                "avg_psnr": avg_psnr,
                                "trans_err": trans_err.item()
                            }
                            eval_dicts += [eval_metric_dict]
                            print(f'evaluation results')
                            for k, v in eval_metric_dict.items():
                                print(f'{k} : {v}')
                        
                        cur_key = 'pred_' + str(p_idx)
                        eval_dict_total[cur_key] = eval_dicts

                    import json
                    with open(str(metric_fname), 'w') as fp:
                        json.dump(eval_dict_total, fp=fp)
                    print('=' * 100)
                    print(f'\nevaluation: avg_psnr = {avg_psnr:.2f}\n')
                    print('=' * 100)
                    print('=' * 40, 'runing coarse pose estimation', '=' * 40)
                    print('=' * 100)
                    
                        
                    ckpt = model.save_ckpt(step, avg_psnr)
                    print('save ckpt at step: %d'%step)
                    
                    # pose_lr = 1e-2
                    if step < opts.max_steps:
                        
                        estimator.configure_optimizer(lr_Q=1e-2, lr_T=1e-2, lr_dir=1e-1, lr_scale=1e-2, scheduler_step=500)
                        estimator.estimate_pose_accum_grad(opts.pose_accum_step, accum_iter=opts.pose_accum_iter)
                        # pose_dict = estimator.pose_param.state_dict()
                        for i in range(opts.seg_classes - 1):
                            cur_dict = estimator.pose_param[i].state_dict()
                            model.pose_module_list[i].load_state_dict(cur_dict)
                            pass
                    
                    
                    
                    
                    
                    
                    
                    # print(f'axis_dir: {axis_dir.detach().cpu().numpy()}, angles: {angles.detach().cpu().numpy()}, axis origin: {estimator.pose_param.axis_origin.detach().cpu().numpy()}, translation: {translation.detach().cpu().view(-1).numpy()}')
                                        

    


