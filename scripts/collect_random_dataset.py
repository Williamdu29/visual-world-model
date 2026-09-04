import os
import numpy as np
from tqdm import tqdm

from dm_control import suite



def collect_episode(
    domain="walker",
    task="walk",
    max_steps=500
):

    env = suite.load(
        domain,
        task
    )


    timestep = env.reset()


    images = []
    actions = []
    rewards = []


    action_spec = env.action_spec()


    for step in range(max_steps):

        # random action
        action = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape
        )


        timestep = env.step(action)


        # RGB observation
        image = env.physics.render(
            height=64,
            width=64,
            camera_id=0
        )


        images.append(image)
        actions.append(action)
        rewards.append(
            timestep.reward
            if timestep.reward is not None
            else 0.0
        )


        if timestep.last():
            break


    return (
        np.array(images),
        np.array(actions),
        np.array(rewards)
    )




def main():


    num_episodes = 10


    save_dir = (
        "datasets/random"
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    all_images = []
    all_actions = []
    all_rewards = []


    for ep in tqdm(
        range(num_episodes)
    ):

        images, actions, rewards = collect_episode()


        all_images.append(images)
        all_actions.append(actions)
        all_rewards.append(rewards)



    all_images = np.array(
        all_images,
        dtype=np.uint8
    )


    all_actions = np.array(
        all_actions,
        dtype=np.float32
    )


    all_rewards = np.array(
        all_rewards,
        dtype=np.float32
    )


    np.save(
        f"{save_dir}/images.npy",
        all_images
    )


    np.save(
        f"{save_dir}/actions.npy",
        all_actions
    )


    np.save(
        f"{save_dir}/rewards.npy",
        all_rewards
    )


    print("Dataset saved!")
    print(
        "images:",
        all_images.shape
    )

    print(
        "actions:",
        all_actions.shape
    )

    print(
        "rewards:",
        all_rewards.shape
    )



if __name__ == "__main__":
    main()
