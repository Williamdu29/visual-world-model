import torch
import numpy as np

from torch.utils.data import DataLoader



import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.dynamics import LatentDynamics

from datasets.latent_transition_dataset import (
    LatentTransitionDataset
)



device="cuda"



# ======================
# Load dataset
# ======================

dataset = LatentTransitionDataset(
    "datasets/random/latent_episode.npy",
    "datasets/random/actions.npy"
)


loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)



# ======================
# Load model
# ======================

model = LatentDynamics(
    latent_dim=256,
    action_dim=6
)


checkpoint = torch.load(
    "/home/william29/visual-world-model/checkpoints/dynamics.pt"
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(device)

model.eval()



# ======================
# Evaluation
# ======================

mse = torch.nn.MSELoss()


total_loss = 0

count = 0



with torch.no_grad():


    for z, action, next_z in loader:


        z = z.to(device)

        action = action.to(device)

        next_z = next_z.to(device)



        pred_z = model(
            z,
            action
        )


        loss = mse(
            pred_z,
            next_z
        )


        total_loss += loss.item()

        count += 1



avg_loss = total_loss / count



print(
    "One-step latent prediction MSE:",
    avg_loss
)