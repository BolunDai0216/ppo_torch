from datetime import datetime
from pathlib import Path
from pdb import set_trace
from time import time

import gym
import numpy as np
import tensorflow as tf
import torch
from gym.spaces import Box, Discrete
from torch import nn

from ppo import PPO
from replay_buffer import ReplayBuffer


class Train:
    def __init__(
        self,
        env,
        name,
        actor_hidden_dim=[64, 64],
        critic_hidden_dim=[64, 64],
        actor_activations=[nn.Tanh, nn.Tanh, nn.Tanh],
        critic_activations=[nn.Tanh, nn.Tanh, "None"],
        actor_update_iters=80,
        critic_update_iters=80,
        epoch_num=10000,
    ):
        if isinstance(env.action_space, Box):
            continuous_env = True
            self.act_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            continuous_env = False
            self.act_dim = env.action_space.n

        self.obs_dim = env.observation_space.shape[0]

        self.ppo = PPO(
            self.obs_dim,
            self.act_dim,
            actor_hidden_dim,
            critic_hidden_dim,
            actor_activations,
            critic_activations,
            continuous=continuous_env,
        )

        self.env = env
        self.max_size = self.env._max_episode_steps
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, 4 * self.max_size)

        self.actor_update_iters = actor_update_iters
        self.critic_update_iters = critic_update_iters

        self.name = name
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        Path("trained_models/{}/{}".format(self.name, self.stamp)).mkdir(
            parents=True, exist_ok=True
        )
        self.epoch_num = epoch_num

    def train(self):
        train_log_dir = "logs/" + self.name + "/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        len_sum = 0

        for i in range(self.epoch_num):
            while True:
                reward, buffer_full = self.play()

                if buffer_full:
                    break

            loss_pi, loss_vf = self.update()

            if (i + 1) % 1000 == 0:
                filename = "trained_models/{}/{}/model_{}.pth".format(
                    self.name, self.stamp, i + 1
                )
                self.ppo.save(filename)

            with train_summary_writer.as_default():
                tf.summary.scalar("Reward", reward, step=i)
                tf.summary.scalar("Actor Loss", loss_pi, step=i)
                tf.summary.scalar("Critic Loss", loss_vf, step=i)

            print(f"Iter: {i}, Reward: {reward}")

    def play(self, render=False):
        state = self.env.reset()
        episode_reward = 0
        buffer_full = False

        for t in range(self.max_size):
            if render:
                self.env.render()

            # make state tensor have correct shape and move to device
            action, logp_a = self.ppo.get_action(state)
            value = self.ppo.get_value(state)

            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            # store experience in buffer
            self.buffer.add(state, action, reward, value, logp_a)

            # update state
            state = next_state

            if self.buffer.ptr == self.buffer.max_size:
                buffer_full = True
                break

            if done:
                break

        if buffer_full or t == self.max_size - 1:
            last_v = self.ppo.get_value(state)
        else:
            last_v = 0.0

        self.buffer.end_of_episode(last_val=last_v)

        return episode_reward, buffer_full

    def update(self):
        data = self.buffer.sample()

        for i in range(self.actor_update_iters):
            self.ppo.actor_opt.zero_grad()
            loss_pi, kl = self.ppo.actor_loss(data)
            if kl > 1.5 * self.ppo.target_kl:
                break
            loss_pi.backward()
            self.ppo.actor_opt.step()

        # print(f"Actor has been updated {i} times, the KL is {kl}")

        for j in range(self.critic_update_iters):
            self.ppo.critic_opt.zero_grad()
            loss_vf = self.ppo.critic_loss(data)
            loss_vf.backward()
            self.ppo.critic_opt.step()

        return loss_pi.detach().item(), loss_vf.detach().item()


def main():
    # env = gym.make("Reacher-v2")
    # agent = Train(env, name="reacherv2")

    env = gym.make("LunarLanderContinuous-v2")
    agent = Train(env, name="lunarv2")

    # env = gym.make("Pendulum-v1")
    # agent = Train(env, name="pendulumv1")

    agent.train()


if __name__ == "__main__":
    main()
