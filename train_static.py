import os
from config import get_opts
opts = get_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)
from models.ngp_wrapper import NGP_wrapper, NGP_Prop_Wrapper
# from config import get_opts
import torch
from torch.utils.data import DataLoader
from dataset.sapien import SapienParisDataset
from tqdm import tqdm
import open3d as o3d
import sys
import traceback
# from dataset.ray_utils import *
class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)


if __name__ == '__main__':
    # load config
    # opts = get_opts()
    # sys.stdout = TracePrints()
    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    setattr(opts, 'device', device)
    
    # load model
    
    model = NGP_Prop_Wrapper(config=opts, training=True)
    
    # load dataset
    train_dataset = SapienParisDataset(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='train',
        state=opts.state
    )
    
    test_dataset = SapienParisDataset(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='val',
        state=opts.state
    )
    # train loop
    
    # c2w = train_dataset.poses[0] # [1, 4, 4]
    # directions_ngp = train_dataset.directions_ngp # [h*w, 3]
    # rays_o = train_dataset.rays_o[0] # [h*w, 3]
    # rays_d = train_dataset.rays_o[0] # [h*w, 3]
    # c2w_ngp = torch.Tensor(transfrom_to_NGP(c2w[0])).to(directions_ngp) # [4, 4]
    # rays_o_n, rays_d_n = get_rays_ngp(directions_ngp, c2w_ngp[:3, :])
    
    train_bar = tqdm(range(opts.max_steps + 1))
    
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
            
        train_bar.set_description(f'trainig at step: {step}, current psnr: {psnr:.2f}')
        if step > 0: 
            if step % opts.ckpt_step == 0:
                ckpt = model.save_ckpt(step)
                print('save ckpt at step: %d'%step)
                
            if step % opts.eval_step == 0:
                with torch.no_grad():
                    psnrs = []
                    eval_pcd = []
                    eval_pcd_color = []
                    for i in tqdm(range(len(test_dataset))):
                        eval_data = test_dataset.__getitem__(i)
                        eval_dict = model.eval(eval_data)
                        psnrs += [eval_dict['psnr'].item()]
                        # save images 
                        img_gt = eval_dict['img_gt']
                        img_pred = eval_dict['img_pred']
                        eval_pcd += [eval_dict['points']]
                        eval_pcd_color += [eval_dict['points_color']]
                        img_gt.save(model.eval_path / f'img_gt_{i:04d}.png')
                        img_pred.save(model.eval_path / f'img_pred_{i:04d}.png')
                    avg_psnr = sum(psnrs) / len(psnrs)
                    multiview_pcd = torch.cat(eval_pcd, dim=0)
                    multiview_color = torch.cat(eval_pcd_color, dim=0)
                    pcd_fname = f'eval_pcd_step_{step}.ply'
                    pcd = o3d.geometry.PointCloud()
                    pcd_pts = multiview_pcd.float().cpu().numpy()
                    pcd_colors = multiview_color.float().cpu().numpy()
                    pcd.points = o3d.utility.Vector3dVector(pcd_pts)
                    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                    o3d.io.write_point_cloud(str(model.eval_path.parent/pcd_fname), pcd)
                    print('evaluation: avg_psnr = %.2f'%avg_psnr)
            
        

    


