from datasets.dataset import WorldModelDataset



clean_dataset = WorldModelDataset(
    "/home/william29/visual-world-model/scripts/datasets/random/images.npy"
)


dist_dataset = WorldModelDataset(
    "/home/william29/visual-world-model/scripts/datasets/distractor/images.npy"
)



print(
    "clean:",
    len(clean_dataset)
)


print(
    "distractor:",
    len(dist_dataset)
)



x1=clean_dataset[0]

x2=dist_dataset[0]


print(
    x1.shape
)

print(
    x2.shape
)