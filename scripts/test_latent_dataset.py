from datasets.latent_transition_dataset import LatentTransitionDataset



dataset = LatentTransitionDataset(
    "datasets/random/latent_episode.npy",
    "datasets/random/actions.npy"
)


print(
    "dataset size:",
    len(dataset)
)


z,a,next_z = dataset[0]


print(
    z.shape
)

print(
    a.shape
)

print(
    next_z.shape
)