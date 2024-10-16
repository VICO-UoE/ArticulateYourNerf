import os
from config import get_opts
opts = get_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)
import torch
from models.ngp_wrapper import NGP_Prop_Art_Wrapper, NGP_Prop_Wrapper, NGP_Prop_Art_Seg_Wrapper
from torch.utils.data import DataLoader
from dataset.sapien import SapienParisDataset
from tqdm import tqdm
import open3d as o3d
import sys
import traceback
from pose_estimation import PoseEstimator
from test_ngp import NGPevaluator
from dataset.pose_utils import quaternion_to_axis_angle, get_rotation_axis_angle
from dataset.io_utils import load_gt_from_json
from models.utils import axis_metrics, geodesic_distance, translational_error
import numpy as np
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
    
    gt_info = load_gt_from_json(opts.motion_gt_json, opts.state, opts.motion_type)
    
    # load model
    ignore_empty = False
    if opts.use_art_seg_estimator:
        model = NGP_Prop_Art_Seg_Wrapper(config=opts, training=True, 
                                         ignore_empty=ignore_empty, use_timestamp=True, use_se3=opts.use_se3)
    else:
        model = NGP_Prop_Art_Wrapper(config=opts, training=True, ignore_empty=ignore_empty)
    co_mask = 'stapler_art_mask.pth'
    shutil.copy(opts.config, str(model.eval_path / 'config.json'))
    
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
    estimator = PoseEstimator(renderer=renderer, dataset=train_dataset, output_dir=model.eval_path, use_num_frames=opts.use_num_frames, device=device, scaling=0.5, use_se3=opts.use_se3, motion_type=opts.motion_type, idx_list=idx_list, eps=opts.eps, select_frame=opts.use_num_frames)
        # if 'laptop' in opts.exp_name:
        #     estimator.configure_optimizer(gamma=0.5, lr=1e-2, scheduler_step=2000)
        # if opts.use_num_frames < 8:
    estimator.configure_optimizer(scheduler_step=500)
    if not opts.skip_init:
        
        
        accu_iter = opts.use_num_frames if opts.use_num_frames <=32 else 32
        steps = 6000 // accu_iter
        
        estimator.estimate_pose_accum_grad(steps, accum_iter=accu_iter)
        
        pose_dict = estimator.pose_param.state_dict()
        model.pose_module_list[0].load_state_dict(pose_dict)
        print('=' * 100)
        print('=' * 40, 'Coarse pose estimation results:', '=' * 40)
        print(f'estimated quaternion: {estimator.pose_param.Q.detach().cpu().numpy()}')
        axis_dir, radian, angles = quaternion_to_axis_angle(estimator.pose_param.Q)
        norm_dir = torch.nn.functional.normalize(estimator.pose_param.dir.view(1, -1))
        translation = norm_dir * estimator.pose_param.scale
        
        print(f'axis_dir: {axis_dir.detach().cpu().numpy()}, angles: {angles.detach().cpu().numpy()}, axis origin: {estimator.pose_param.axis_origin.detach().cpu().numpy()}, translation: {translation.detach().cpu().view(-1).numpy()}')
        print(f'current training image samples {estimator.idx_list}')
        print('=' * 100)
        if opts.ignore_empty:
            train_dataset.ignore_empty = True
            non_empty_fname = str(estimator.occu_mask_fname)
            train_dataset.load_co_masks(non_empty_fname)
        
        if opts.ignore_empty:
            train_dataset.ignore_empty = True
            non_empty_fname = str(estimator.occu_mask_fname)
            train_dataset.load_co_masks(non_empty_fname)
            
        imp_sampling = opts.imp_sampling
        if imp_sampling:
            train_dataset.importance_pixels = estimator.cache_idxs
            train_dataset.importance_sampling = True
    
    assign_part_pts = not opts.skip_init
    # ============================================================================================================
    train_dataset.idx_list = np.asarray(estimator.idx_list)
    idx_list_fname = model.eval_path / 'train_set.txt'
    with open(str(idx_list_fname), 'w') as f:
        f.write(str(estimator.idx_list.tolist()))
    

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
    # ============================================================================================================
    for step in train_bar:
        # print('training step: %d' % step, end='\r')
        
        data = train_dataset.__getitem__(step)
        ret_dcit = model.train(data, step)
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
        
        
        
        if step > 0: 
            
            if step % opts.eval_step == 0:
                print(f'evaluating at step {step}')
                with torch.no_grad():
                    save_eval_dir = model.eval_path /f'eval_step_{step}' 
                    save_eval_dir.mkdir(exist_ok = True, parents=True)
                    psnrs = []
                    eval_pcd = []
                    eval_pcd_color = []
                    part_pts = model.scan_nerf(step)
                    part_pts_num = part_pts.shape[0]
                    # no update
                    # if ignore_empty == False:
                    if part_pts_num > 0:
                        # if step >= 2 * opts.eval_step:
                        # if step %  (2 * opts.eval_step) == 0:
                        # estimator.nerf_part_pts = part_pts
                        # estimator.check_new_part_pts()
                        # estimator.nerf_part_pts = part_pts
                        # estimator.check_new_part_pts()
                        if assign_part_pts:
                            estimator.nerf_part_pts = part_pts
                            assign_part_pts = False
                        else:
                            estimator.update_new_part_pts(part_pts)
                    
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
                    axis_dir, radian, angles = quaternion_to_axis_angle(model.pose_module_list[0].Q)
                    pred_R = get_rotation_axis_angle(axis_dir.cpu().numpy(), radian.cpu().numpy())
                    
                    if opts.motion_type == 'r':
                        pred_axis_dir = axis_dir.detach()
                    else:
                        pred_axis_dir = model.pose_module_list[0].dir.detach()
                    
                    pred_info = {
                        "axis_o": model.pose_module_list[0].axis_origin.detach(),
                        "axis_d": pred_axis_dir,
                        "R": torch.Tensor(pred_R),
                        "theta": radian.detach(),
                        "dist": model.pose_module_list[0].scale.detach()
                    }
                    ang_err, pos_err = axis_metrics(pred_info, gt_info)
                    trans_err = translational_error(pred_info, gt_info)
                    geo_dist = geodesic_distance(torch.Tensor(pred_R), gt_info['R'])
                    metric_fname = model.eval_path / f'motion_metric_{step}.json'
                    
                    axis_origin = model.pose_module_list[0].axis_origin.detach().cpu().numpy().tolist()
                    axis_dir = axis_dir.detach().cpu().numpy().tolist()
                    angles = angles.detach().cpu().numpy().tolist()
                    
                    norm_dir = torch.nn.functional.normalize(model.pose_module_list[0].dir.view(1, -1))
                    
                    translation = norm_dir * model.pose_module_list[0].scale
                    # translation = estimator.pose_param.dir * estimator.pose_param.scale
                    print(f'\ncurrent estiamted pose: axis direction = {axis_dir}, angles = {angles:}, axis origin = {axis_origin}, translation: {translation.detach().cpu().view(-1).numpy()}')
                    print('=' * 100)
                    print(f'\nevaluation: avg_psnr = {avg_psnr:.2f}\n')
                    print('=' * 100)
                    
                    eval_metric_dict = {
                        "ang_err": ang_err.item(),
                        "pos_err": pos_err.item(),
                        "geo_dist": geo_dist.item(),
                        "avg_psnr": avg_psnr,
                        "trans_err": trans_err.item()
                    }
                    print(f'evaluation results')
                    for k, v in eval_metric_dict.items():
                        print(f'{k} : {v}')
                    import json
                    with open(str(metric_fname), 'w') as fp:
                        json.dump(eval_metric_dict, fp=fp)
                        
                    print('=' * 40, 'runing coarse pose estimation', '=' * 40)
                    print('=' * 100)
                    
                        
                    ckpt = model.save_ckpt(step, avg_psnr)
                    print('save ckpt at step: %d'%step)
                    
                    
                    # pose_lr = 1e-2
                    if step < 10000:
                        estimator.configure_optimizer(lr_Q=1e-2, lr_T=1e-2, lr_dir=1e-1, lr_scale=1e-2, scheduler_step=opts.pose_scheduler_step)
                        accu_iter = opts.use_num_frames if opts.use_num_frames <=8 else 8
                        steps = 6000 // accu_iter
                        estimator.estimate_pose_accum_grad(steps, accum_iter=accu_iter)
                        pose_dict = estimator.pose_param.state_dict()
                        model.pose_module_list[0].load_state_dict(pose_dict)
                        axis_dir, radian, angles = quaternion_to_axis_angle(estimator.pose_param.Q)
                        translation = estimator.pose_param.dir * estimator.pose_param.scale
                    
                    # print(f'axis_dir: {axis_dir.detach().cpu().numpy()}, angles: {angles.detach().cpu().numpy()}, axis origin: {estimator.pose_param.axis_origin.detach().cpu().numpy()}, translation: {translation.detach().cpu().view(-1).numpy()}')
                                        

    


