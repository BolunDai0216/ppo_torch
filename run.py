import gym
import numpy as np

from train import Train


def main():
    # env = gym.make("CartPole-v1")
    # agent = Train(env, name="cartpolev1", save_freq=100)

    env = gym.make("LunarLander-v2")
    agent = Train(env, name="lunardisv2", save_freq=100)

    # env = gym.make("HalfCheetah-v2")
    # agent = Train(env, name="halfcheetahv2")

    # env = gym.make("LunarLanderContinuous-v2")
    # agent = Train(env, name="lunarv2")

    # env = gym.make("Hopper-v2")
    # agent = Train(env, name="hopperv2")

    # agent.train()

    # ----- Testing -----
    render = False
    model_path = "trained_models/lunardisv2/20211111-181353/model_1000.pth"
    # model_path = "trained_models/halfcheetahv2/20211109-161223/model_1000.pth"
    # model_path = "trained_models/lunarv2/20211108-231546/model_1000.pth"
    # model_path = "trained_models/cartpolev1/20211111-170918/model_100.pth"

    reward, history = agent.test(render=render, path=model_path)

    reward_list = []
    for _ in range(50):
        reward, history = agent.test(render=False)
        print(f"reward: {reward}")
        reward_list.append(reward)

    reward_array = np.array(reward_list)
    reward_mean = np.mean(reward_array)
    reward_max_diff = np.amax(reward_array) - reward_mean
    reward_min_diff = reward_mean - np.amin(reward_array)

    print(
        f"mean: {reward_mean}, max_diff: {reward_max_diff}, min_diff: {reward_min_diff}"
    )


if __name__ == "__main__":
    main()
