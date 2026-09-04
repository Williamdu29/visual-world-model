import numpy as np
from dm_control import suite


env = suite.load(
    "walker",
    "walk"
)


time_step = env.reset()


image = env.physics.render(
    height=64,
    width=64,
    camera_id=0
)


print(image.shape)
print(image.dtype)
