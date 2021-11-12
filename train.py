from datetime import datetime
from pathlib import Path
from pdb import set_trace
from time import time

import gym
import numpy as np
import tensorflow as tf
import torch
from gym.spaces import Box, Discrete
from gym.wrappers import Monitor
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
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
        save_freq=1000,
    ):
        # get action space from env
        if isinstance(env.action_space, Box):
            continuous_env = True
            self.act_dim = env.action_space.shape[0]
            self.buffer_act_dim = self.act_dim
        elif isinstance(env.action_space, Discrete):
            continuous_env = False
            self.act_dim = env.action_space.n
            self.buffer_act_dim = 1

        # get observation space from env
        self.obs_dim = env.observation_space.shape[0]

        # create PPO object
        self.ppo = PPO(
            self.obs_dim,
            self.act_dim,
            actor_hidden_dim,
            critic_hidden_dim,
            actor_activations,
            critic_activations,
            continuous=continuous_env,
            clip_ratio=clip_ratio,
            target_kl=target_kl,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )

        # create replay buffer
        self.env = env
        self.max_size = self.env._max_episode_steps
        self.buffer = ReplayBuffer(
            self.obs_dim, self.buffer_act_dim, 4 * self.max_size, gamma=gamma, lam=lam
        )

        # define variables from saving and updating model parameters
        self.name = name
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        Path("trained_models/{}/{}".format(self.name, self.stamp)).mkdir(
            parents=True, exist_ok=True
        )
        self.epoch_num = epoch_num
        self.save_freq = save_freq
        self.actor_update_iters = actor_update_iters
        self.critic_update_iters = critic_update_iters

    def train(self):
        # create tensorboard logger
        train_log_dir = "logs/" + self.name + "/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # main training loop
        for i in range(self.epoch_num):
            # wait till replay buffer is full
            while True:
                reward, buffer_full = self.play()

                if buffer_full:
                    break

            # update model parameters
            loss_pi, loss_vf = self.update()

            # save model parameters
            if (i + 1) % self.save_freq == 0:
                filename = "trained_models/{}/{}/model_{}.pth".format(
                    self.name, self.stamp, i + 1
                )
                self.ppo.save(filename)

            # log training process to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar("Reward", reward, step=i)
                tf.summary.scalar("Actor Loss", loss_pi, step=i)
                tf.summary.scalar("Critic Loss", loss_vf, step=i)

            # print training process to console
            print(f"Iter: {i}, Reward: {reward}")

    def test(self, render=False, path=None, mode="human"):
        if path is not None:
            self.ppo.load(path)

        state_list = []
        action_list = []

        episode_reward = 0
        state = self.env.reset()

        for t in range(self.max_size):
            if render:
                self.env.render(mode=mode)

            action, _ = self.ppo.get_action(state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            state_list.append(state[:, np.newaxis])
            if self.ppo.continuous:
                action_list.append(action[:, np.newaxis])
            else:
                action_list.append(action)

            state = next_state

            if done:
                break

        if render:
            self.env.close()

        history = {"state": state_list, "action": action_list}

        return episode_reward, history

    def play(self, render=False):
        state = self.env.reset()
        episode_reward = 0
        buffer_full = False

        for t in range(self.max_size):
            if render:
                self.env.render()

            # get action and value function
            action, logp_a = self.ppo.get_action(state)
            value = self.ppo.get_value(state)

            # step and record reward
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            # store experience in buffer
            self.buffer.add(state, action, reward, value, logp_a)

            # update state
            state = next_state

            # update when the buffer is full
            if self.buffer.ptr == self.buffer.max_size:
                buffer_full = True
                break

            if done:
                break

        # if the agent did not reach terminal state value function of last state is not zero
        if buffer_full or t == self.max_size - 1:
            last_v = self.ppo.get_value(state)
        # value function for terminal state is zero
        else:
            last_v = 0.0

        self.buffer.end_of_episode(last_val=last_v)

        return episode_reward, buffer_full

    def update(self):
        data = self.buffer.sample()

        # update actor weights
        for i in range(self.actor_update_iters):
            self.ppo.actor_opt.zero_grad()
            loss_pi, kl = self.ppo.actor_loss(data)
            # stop updating after large KL divergence
            if kl > 1.5 * self.ppo.target_kl:
                break
            loss_pi.backward()
            self.ppo.actor_opt.step()

        # update critic weights
        for j in range(self.critic_update_iters):
            self.ppo.critic_opt.zero_grad()
            loss_vf = self.ppo.critic_loss(data)
            loss_vf.backward()
            self.ppo.critic_opt.step()

        return loss_pi.detach().item(), loss_vf.detach().item()
