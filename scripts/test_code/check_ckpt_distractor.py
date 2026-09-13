import torch

ckpt=torch.load(
"/home/william29/visual-world-model/checkpoints/autoencoder_distractor.pt"
)

print(type(ckpt))

print(ckpt.keys())