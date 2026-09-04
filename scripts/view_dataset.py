import numpy as np
import matplotlib.pyplot as plt


images = np.load(
    "datasets/random/images.npy"
)


print(images.shape)


img = images[0,0]


plt.imshow(img)
plt.axis("off")
plt.show()
