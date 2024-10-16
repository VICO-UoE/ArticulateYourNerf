from dataset.sapien_register_static import SapienRegisterDataset
from models.triplet_network import VP2PMatchNet
import torch
import sys
sys.argv = ["--config", "configs/test_laptop.json"]
from config import get_opts
opts = get_opts(sys.argv)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
setattr(opts, 'device', device)
test_network = VP2PMatchNet(input_dim=3).cuda()
test_dataset = SapienRegisterDataset(opts)
test_data = test_dataset.get_batch_data(8)
test_network.forward_sapien(test_data)