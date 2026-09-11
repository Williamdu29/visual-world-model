import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.dynamics import LatentDynamics

from datasets.latent_transition_dataset import (
    LatentTransitionDataset
)



device = "cuda"



# ======================
# Dataset
# ======================

dataset = LatentTransitionDataset(
    "datasets/random/latent_episode.npy",
    "datasets/random/actions.npy"
)


loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)



print(
    "number of transitions:",
    len(dataset)
)



# ======================
# Model
# ======================

model = LatentDynamics(
    latent_dim=256,
    action_dim=6,
    hidden_dim=512
)


model.to(device)



# ======================
# Optimizer
# ======================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


criterion = nn.MSELoss()



# ======================
# Training
# ======================

epochs = 100



for epoch in range(epochs):


    total_loss = 0



    for z, action, next_z in loader:


        z = z.to(device)

        action = action.to(device)

        next_z = next_z.to(device)



        pred_next_z = model(
            z,
            action
        )



        loss = criterion(
            pred_next_z,
            next_z
        )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        total_loss += loss.item()



    avg_loss = (
        total_loss / len(loader)
    )



    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {avg_loss:.6f}"
    )



# ======================
# Save checkpoint
# ======================

checkpoint = {

    "epoch": epochs,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "loss":
        avg_loss
}



torch.save(
    checkpoint,
    "/home/william29/visual-world-model/checkpoints/dynamics.pt"
)


print(
    "Dynamics model saved."
)