from pdb import set_trace

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import nn

from ppo import PPO
from replay_buffer import ReplayBuffer


class Train:
    def __init__(
        self,
        env,
        device,
        actor_hidden_dim=[64, 64],
        critic_hidden_dim=[64, 64],
        actor_activations=[nn.Tanh, nn.Tanh, nn.Tanh],
        critic_activations=[nn.Tanh, nn.Tanh, nn.Tanh],
        actor_update_iters=80,
        critic_update_iters=80,
    ):
        if isinstance(env.action_space, Box):
            continuous_env = True
        elif isinstance(env.action_space, Discrete):
            continuous_env = False

        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        self.ppo = PPO(
            self.obs_dim,
            self.act_dim,
            actor_hidden_dim,
            critic_hidden_dim,
            actor_activations,
            critic_activations,
            device,
            continuous=continuous_env,
        )

        self.env = env
        self.device = device
        self.max_size = self.env._max_episode_steps
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.max_size, device)

        self.actor_update_iters = actor_update_iters
        self.critic_update_iters = critic_update_iters

    def evaluate(self):
        pass

    def train(self):
        pass

    def play(self, render=False) -> float:
        state = self.env.reset()
        episode_reward = 0

        for t in range(self.max_size):
            if render:
                self.env.render()

            # make state tensor have correct shape and move to device
            state = self.state_manip(state)
            action, logp_a = self.ppo.get_action(state)
            value = self.ppo.get_value(state)[0, 0]

            action_np = self.action_manip(action)
            next_state, reward, done, _ = self.env.step(action_np)
            episode_reward += reward

            # store experience in buffer
            value_np = self.value_manip(value)
            self.buffer.add(state, action, reward, value_np, logp_a)

            # update state
            state = next_state

            if done:
                if t == self.max_size - 1:
                    state = self.state_manip(state)
                    last_v = self.ppo.get_value(state)[0, 0]
                    last_v = self.value_manip(last_v)
                else:
                    last_v = 0.0

                self.buffer.end_of_episode(last_val=last_v)

        return episode_reward

    def update(self):
        for i in range(self.actor_update_iters):
            self.ppo.actor_opt.zero_grad()
            data = self.buffer.sample()
            loss_pi, pi_info = self.ppo.actor_loss(data)
            kl = pi_info["kl"]

            if kl > 1.5 * self.ppo.target_kl:
                break
            loss_pi.backward()
            self.ppo.actor_opt.step()

        for j in range(self.critic_update_iters):
            self.ppo.critic_opt.zero_grad()
            data = self.buffer.sample()
            loss_vf = self.ppo.critic_loss(data)
            loss_vf.backward()
            self.ppo.critic_opt.step()

    def state_manip(self, state: np.ndarray) -> torch.Tensor:
        _state = state.reshape(1, -1)
        _state = torch.as_tensor(_state, dtype=torch.float32)
        state = _state.to(device=self.device, non_blocking=True)

        return state

    def action_manip(self, action: torch.Tensor) -> np.ndarray:
        action = action[0, :].cpu().numpy()

        return action

    def value_manip(self, value: torch.Tensor) -> float:
        value = value.cpu().item()

        return value


def main():
    env = gym.make("Walker2d-v2")
    device = "cuda:2"
    agent = Train(env, device)
    for i in range(10):
        reward = agent.play()
        agent.update()
        print(f"i, reward: {reward}")


if __name__ == "__main__":
    main()
