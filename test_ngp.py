import torch
from config import get_opts

import random

class NGPevaluator():
    def __init__(self, opts, radius_range=[4, 6], model=None, dataset=None) -> None:
        
    # if dataset is None:
    #     self.dataset = SapienArtSegDataset_nerfacc(
    #     root_dir = opts.root_dir,
    #     near = opts.near_plane,
    #     far = opts.far_plane,
    #     img_wh = opts.img_wh, 
    #     batch_size=opts.batch_size,
    #     split='test'
    #     )
    # else:
        self.dataset = dataset
            
        # if model is None:
        #     self.model = NGP_Prop_Wrapper(config=opts, training=False)
        # else:
        self.model = model
        self.radius_range = radius_range
    
    def get_img_given_radius(self, radius):
        batch = self.dataset.get_rays_given_radius(radius)
        render_batch = self.model.gen_data(batch)
        return render_batch

    def gen_img_given_pose(self, pose):
        '''
        pose: [4, 4] numpy array or torch tensor
        
        '''
        batch = self.dataset.get_rays_given_pose(pose)
        render_batch = self.model.gen_data(batch)
        return render_batch

    def gen_random_img(self):
        radius = random.uniform(*self.radius_range)
        
        return self.get_img_given_radius(radius)
    
if __name__ == '__main__':
    from dataset.sapien import SapienArtSegDataset_nerfacc
    from models.ngp_wrapper import NGP_wrapper, NGP_Prop_Wrapper
    config_str = ["--config", "configs/test_laptop_prop_debug.json"]
    opts = get_opts(config_str)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    
    setattr(opts, 'device', device)
    
    # load model
    
    test_dataset = SapienArtSegDataset_nerfacc(
        root_dir = opts.root_dir,
        near = opts.near_plane,
        far = opts.far_plane,
        img_wh = opts.img_wh, 
        batch_size=opts.batch_size,
        split='test'
    )
    
    model = NGP_Prop_Wrapper(config=opts, training=False)
    test_batch = test_dataset.get_rays_given_radius(random.uniform(4, 6))
    
    render_batch = model.test(test_batch)
    shape = render_batch['rgb'].shape
    print(f'predict rgb shape: {shape}')
    render_batch['img_pred'].save('test_laptop.png')
    