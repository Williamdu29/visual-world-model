import torch.nn as nn

from models.encoder import CNNEncoder
from models.decoder import CNNDecoder



class AutoEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=256
    ):

        super().__init__()


        self.encoder = CNNEncoder(
            latent_dim
        )


        self.decoder = CNNDecoder(
            latent_dim
        )


    def forward(self,x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        return reconstruction, z