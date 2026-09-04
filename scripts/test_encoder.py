import torch

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import CNNEncoder



device = "cuda"


encoder = CNNEncoder(
    latent_dim=256
).to(device)



x = torch.randn(
    8,
    3,
    64,
    64
).to(device)



z = encoder(x)


print(
    "input:",
    x.shape
)


print(
    "latent:",
    z.shape
)