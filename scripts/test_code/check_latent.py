import numpy as np

z = np.load(
    "datasets/random/latent.npy"
)

a = np.load(
    "datasets/random/actions.npy"
)


print(z.shape)
print(a.shape)