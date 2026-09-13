import torch

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import AutoEncoder


device="cuda"


model1=AutoEncoder(
    latent_dim=256
).to(device)



# random input

x=torch.randn(
    1,3,64,64
).to(device)



# save current random output

with torch.no_grad():

    y1,z1=model1(x)



# load checkpoint

model2=AutoEncoder(
    latent_dim=256
).to(device)


model2.load_state_dict(
    torch.load(
        "/home/william29/visual-world-model/checkpoints/autoencoder.pt"
    )
)


model2.eval()


with torch.no_grad():

    y2,z2=model2(x)



print(
"random model output:",
y1.mean().item()
)


print(
"loaded model output:",
y2.mean().item()
)


print(
"latent:",
z2.mean().item(),
z2.std().item()
)