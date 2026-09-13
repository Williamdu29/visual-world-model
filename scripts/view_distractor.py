import numpy as np
import matplotlib.pyplot as plt


images=np.load(
"/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)


img=images[0,0]


plt.imshow(img)

plt.axis("off")

plt.savefig(
"/home/william29/visual-world-model/experiments/distractor_example.png"
)

plt.show()