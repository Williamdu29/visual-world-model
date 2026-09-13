

import torch
import numpy as np

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("加入的搜索路径:", project_root)  # 必须打印出 /home/william29/visual-world-model 才对
sys.path.append(project_root)

from models.autoencoder import AutoEncoder


device="cuda"


model=AutoEncoder(
    latent_dim=256
)


model.load_state_dict(
    torch.load(
        "/home/william29/visual-world-model/checkpoints/autoencoder_distractor.pt"
    )
)


model.to(device)

model.eval()



images=np.load(
    "/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)


x=torch.tensor(
    images[0,0],
    dtype=torch.float32
)


x=x.permute(2,0,1)

x=x.unsqueeze(0)

x=x.to(device)



with torch.no_grad():

    recon,z=model(x)


print(
    "before:",
    z.mean().item(),
    z.std().item()
)


z_norm = (
    z-z.mean()
)/(
    z.std()+1e-8
)


with torch.no_grad():

    recon_norm=model.decoder(
        z_norm
    )


print(
    "after normalization"
)

print(
    recon_norm.min().item(),
    recon_norm.max().item()
)