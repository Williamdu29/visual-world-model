import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import AutoEncoder



device = "cuda"



# =====================
# load model
# =====================

model = AutoEncoder(
    latent_dim=256
)


model.load_state_dict(
    torch.load(
        "/home/william29/visual-world-model/checkpoints/autoencoder.pt"
    )
)


model.to(device)

model.eval()



# =====================
# load image
# =====================

images = np.load(
    "datasets/random/images.npy"
)



print(
    "dataset:",
    images.shape
)



# choose one image

img = images[0,0]


# H,W,C
# -> C,H,W

x = (
    img.astype(np.float32)
    /255.0
)


x = torch.from_numpy(
    x
)


x = x.permute(
    2,0,1
)


x = x.unsqueeze(0)


x = x.to(device)



# =====================
# reconstruction
# =====================

with torch.no_grad():

    recon,z = model(x)



print(
    "latent:",
    z.shape
)



# move back cpu

recon = (
    recon
    .cpu()
    .squeeze(0)
    .permute(1,2,0)
    .numpy()
)



original = (
    x
    .cpu()
    .squeeze(0)
    .permute(1,2,0)
    .numpy()
)



# =====================
# plot
# =====================

plt.figure(figsize=(8,4))


plt.subplot(1,2,1)

plt.title(
    "Original"
)

plt.imshow(
    original
)

plt.axis(
    "off"
)



plt.subplot(1,2,2)

plt.title(
    "Reconstruction"
)


plt.imshow(
    recon
)

plt.axis(
    "off"
)



# 定位到项目根目录下的 experiments 文件夹（不受运行目录影响）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_root, "experiments")
# 自动创建目录，已存在也不会报错
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "reconstruction_result.png")

plt.savefig(
    save_path,
    dpi=300
)
# WSL 无图形界面时 plt.show() 会弹出警告，不需要可以注释掉
# plt.show()