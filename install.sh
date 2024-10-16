# torch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# nerfacc & tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfacc

# pytorch3d
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install -r requirements.txt