import numpy as np

images=np.load(
"/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)

print(images.shape)

print(images.dtype)

print(images.min())

print(images.max())

print(images[0,0,0,0])