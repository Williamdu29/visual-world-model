import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )


        self.fc = nn.Linear(
            256 * 4 * 4,
            latent_dim
        )


    def forward(self,x):

        x = self.encoder(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        z = self.fc(x)

        return z