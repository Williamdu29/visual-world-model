import torch
import torch.nn as nn



class LatentDynamics(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        action_dim=6,
        hidden_dim=512
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                latent_dim + action_dim,
                hidden_dim
            ),

            nn.ReLU(),


            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),


            nn.Linear(
                hidden_dim,
                latent_dim
            )

        )


    def forward(
        self,
        z,
        action
    ):


        x = torch.cat(
            [
                z,
                action
            ],
            dim=-1
        )


        next_z = self.network(x)


        return next_z