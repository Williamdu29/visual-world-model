from dm_control import suite


env = suite.load(
    "walker",
    "walk"
)

print(env)


time_step = env.reset()

print(time_step)
