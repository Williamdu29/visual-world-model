import torch
import torch.nn as nn


class CNNDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):
        super().__init__()


        self.fc = nn.Linear(
            latent_dim,
            256 * 4 * 4
        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),


            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()

        )


    def forward(self,z):

        x = self.fc(z)

        x = x.view(
            -1,
            256,
            4,
            4
        )


        x = self.decoder(x)

        return x