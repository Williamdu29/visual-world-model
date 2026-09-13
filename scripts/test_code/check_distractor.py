import numpy as np


clean = np.load(
"/home/william29/visual-world-model/scripts/datasets/random/images.npy"
)


dist = np.load(
"/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)


print("clean:", clean.shape)

print("distractor:", dist.shape)