import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# 把当前脚本的上一级目录（项目根目录）加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamics import LatentDynamics



device="cuda"



model = LatentDynamics(
    latent_dim=256,
    action_dim=6
)



checkpoint=torch.load(
    "/home/william29/visual-world-model/checkpoints/dynamics.pt"
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(device)

model.eval()



latents=np.load(
    "datasets/random/latent_episode.npy"
)


actions=np.load(
    "datasets/random/actions.npy"
)



episode=0


start=0


horizon=20 # imagine 的步数



true_z = latents[
    episode,
    start:start+horizon
]


pred=[]



z=torch.tensor(
    true_z[0],
    dtype=torch.float32
).unsqueeze(0).to(device)



with torch.no_grad():


    for t in range(horizon-1):


        action=torch.tensor(
            actions[
                episode,
                start+t
            ],
            dtype=torch.float32
        ).unsqueeze(0).to(device)



        z=model(
            z,
            action
        )


        pred.append(
            z.cpu().numpy()[0]
        )



pred=np.array(pred)



true_z=true_z[1:]



errors=np.mean(
    (pred-true_z)**2,
    axis=1
)



print(
    "rollout errors:"
)

print(errors)



plt.plot(
    errors
)

plt.xlabel(
    "prediction step"
)

plt.ylabel(
    "latent MSE"
)

plt.savefig(
    "/home/william29/visual-world-model/experiments/rollout_error.png"
)

plt.show()