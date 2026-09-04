import torch
import numpy as np


import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import CNNEncoder



device="cuda"


encoder=CNNEncoder(
    latent_dim=256
)


encoder.load_state_dict(
    torch.load(
        "/home/william29/visual-world-model/checkpoints/autoencoder.pt"
    ),
    strict=False
)


encoder.to(device)

encoder.eval()



images=np.load(
    "datasets/random/images.npy"
)



images = (
    images.reshape(
        -1,
        64,
        64,
        3
    )
)


images = (
    images.astype(np.float32)
    /255.0
)



images=torch.from_numpy(
    images
).permute(
    0,3,1,2
)


images=images.to(device)



with torch.no_grad():

    z=encoder(images)



z=z.cpu().numpy()


print(
    z.shape
)



np.save(
    "/home/william29/visual-world-model/scripts/datasets/random/latent.npy",
    z
)