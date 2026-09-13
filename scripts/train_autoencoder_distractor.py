import torch
import torch.nn as nn

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader


from models.autoencoder import AutoEncoder

from datasets.dataset import WorldModelDataset



device="cuda"


dataset = WorldModelDataset(
    "/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)


loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)



model = AutoEncoder(
    latent_dim=256
).to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


criterion = nn.MSELoss()



epochs=30



for epoch in range(epochs):

    total_loss=0


    for images in loader:

        images = images.to(device)


        recon,z = model(images)


        loss = criterion(
            recon,
            images
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()



    print(
        f"epoch {epoch+1}, loss {total_loss/len(loader)}"
    )



# 定位到项目根目录下的 checkpoints 文件夹（基于脚本自身位置，不受运行目录影响）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_root, "checkpoints")
# 自动创建目录，已存在也不会报错
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "autoencoder_distractor.pt")


with torch.no_grad():

    test_img = next(iter(loader))

    test_img = test_img.to(device)

    recon,z = model(test_img)


    print(
        "TRAIN END latent mean:",
        z.mean().item()
    )

    print(
        "TRAIN END latent std:",
        z.std().item()
    )


    print(
        "TRAIN END recon mse:",
        ((recon-test_img)**2).mean().item()
    )

torch.save(
    model.state_dict(),
    save_path
)