import numpy as np
import torch

from mlp import mlp


class Critic(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, activations):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [1]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations)

    def forward(self, obs):
        val = self.net(obs)

        return val
