import torch
import numpy as np

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import AutoEncoder



device="cuda"



model=AutoEncoder(
    latent_dim=256
)



model.load_state_dict(
    torch.load(
        "/home/william29/visual-world-model/checkpoints/autoencoder.pt"
    )
)



model.to(device)

model.eval()



images=np.load(
    "/home/william29/visual-world-model/scripts/datasets/random/images.npy"
)



img=images[0,0]


x=torch.tensor(
    img/255,
    dtype=torch.float32
)



x=x.permute(
    2,0,1
)


x=x.unsqueeze(0)

x=x.to(device)



with torch.no_grad():

    recon,z=model(x)



print(
    "clean latent mean:",
    z.mean().item()
)


print(
    "clean latent std:",
    z.std().item()
)


print(
    "clean recon mse:",
    ((recon-x)**2).mean().item()
)