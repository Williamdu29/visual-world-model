import torch

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamics import LatentDynamics



device="cuda"



model = LatentDynamics(
    latent_dim=256,
    action_dim=6
).to(device)



z = torch.randn(
    8,
    256
).to(device)



action = torch.randn(
    8,
    6
).to(device)



next_z = model(
    z,
    action
)



print(
    "input latent:",
    z.shape
)


print(
    "action:",
    action.shape
)


print(
    "predicted latent:",
    next_z.shape
)