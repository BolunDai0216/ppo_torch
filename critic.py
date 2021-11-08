from pdb import set_trace

import numpy as np
import torch

from mlp import mlp


class Critic(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: list, activations: list, device: str):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [1]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        val = self.net(obs)
        val = torch.squeeze(val, -1)

        return val
