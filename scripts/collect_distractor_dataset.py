import os
import numpy as np
from tqdm import tqdm

from dm_control import suite



def random_background():

    color = np.random.uniform(
        0,
        1,
        size=(64,64,3)
    )

    return color



def collect_episode(
    max_steps=500
):

    env = suite.load(
        "walker",
        "walk"
    )


    timestep = env.reset()


    images=[]
    actions=[]
    rewards=[]


    action_spec = env.action_spec()


    bg = random_background()


    for step in range(max_steps):


        action=np.random.uniform(
            action_spec.minimum,
            action_spec.maximum,
            size=action_spec.shape
        )


        timestep=env.step(action)



        image=env.physics.render(
            height=64,
            width=64,
            camera_id=0
        )


        image=image.astype(
            np.float32
        )/255.0


        # blend background
        image = (
            0.7*image
            +
            0.3*bg
        )


        image=np.clip(
            image,
            0,
            1
        )


        images.append(
            image
        )

        actions.append(
            action
        )

        rewards.append(
            timestep.reward
            if timestep.reward
            else 0
        )



        if timestep.last():
            break



    return (
        np.array(images),
        np.array(actions),
        np.array(rewards)
    )



def main():


    episodes=10


    save_dir=(
        "/home/william29/visual-world-model/scripts/datasets/distractor"
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    images=[]
    actions=[]
    rewards=[]


    for ep in tqdm(range(episodes)):


        img,act,r=collect_episode()


        images.append(img)
        actions.append(act)
        rewards.append(r)



    np.save(
        f"{save_dir}/images.npy",
        np.array(images)
    )


    np.save(
        f"{save_dir}/actions.npy",
        np.array(actions)
    )


    np.save(
        f"{save_dir}/rewards.npy",
        np.array(rewards)
    )



    print(
        "saved"
    )



if __name__=="__main__":
    main()