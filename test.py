from models.ngp_wrapper import NGP_wrapper
from config import get_opts
import torch
from torch.utils.data import DataLoader
from dataset.sapien import SapienArtSegDataset_nerfacc
from tqdm import tqdm
import open3d as o3d
import sys
import traceback
import random
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
    # sys.stdout = TracePrints()
    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    setattr(opts, 'device', device)
    
    # load model
    
    model = NGP_wrapper(config=opts, training=True)
    
    test_dataset = SapienArtSegDataset_nerfacc(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='test'
    )
    
    radius = random.uniform(4, 6)
    data = test_dataset.get_rays_given_radius(radius)
    render_dict = model.test(data)
    
    

    


