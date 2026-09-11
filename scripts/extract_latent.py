import torch
import numpy as np

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import CNNEncoder



device = "cuda"



# =========================
# load encoder
# =========================

encoder = CNNEncoder(
    latent_dim=256
)


checkpoint = torch.load(
    "/home/william29/visual-world-model/checkpoints/autoencoder.pt"
)


# only load encoder part

encoder_state = {}


for k,v in checkpoint.items():

    if k.startswith("encoder.encoder."):

        new_key = k.replace(
            "encoder.encoder.",
            "encoder."
        )

        encoder_state[new_key] = v


    elif k.startswith("encoder.fc."):

        new_key = k.replace(
            "encoder.",
            ""
        )

        encoder_state[new_key] = v



print(
    "loaded keys:"
)

for k in encoder_state.keys():

    print(k)

encoder.load_state_dict(
    encoder_state
)


encoder.to(device)

encoder.eval()



# =========================
# load images
# =========================

images = np.load(
    "datasets/random/images.npy"
)


print(
    "images:",
    images.shape
)


# images:

# episode,time,H,W,C


episodes = images.shape[0]

steps = images.shape[1]



latents=[]



for ep in range(episodes):


    print(
        "processing episode",
        ep
    )


    imgs = images[ep]


    imgs = (
        imgs.astype(
            np.float32
        )
        /255.0
    )


    imgs = torch.from_numpy(
        imgs
    )


    imgs = imgs.permute(
        0,3,1,2
    )


    imgs = imgs.to(device)



    with torch.no_grad():

        z = encoder(imgs)



    latents.append(
        z.cpu().numpy()
    )



latents=np.array(
    latents
)


print(
    "latent:",
    latents.shape
)



np.save(
    "datasets/random/latent_episode.npy",
    latents
)