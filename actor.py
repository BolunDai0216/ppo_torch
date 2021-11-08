from pdb import set_trace

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from mlp import mlp


class Actor(torch.nn.Module):
    def forward(self, obs: torch.Tensor, act: torch.Tensor = None) -> torch.Tensor:
        pi = self._distribution(obs)
        log_prob = None

        if act is not None:
            # used only during gradient calculation
            log_prob = self._log_prob_from_distribution(pi, act)

        return pi, log_prob

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def _log_prob_from_distribution(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ActorContinuous(Actor):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: list,
        activations: list,
        device: str,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [self.act_dim]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.std = torch.exp(log_std).to(device=device, non_blocking=True).detach()

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        mean = self.net(obs)
        distribution = Normal(mean, self.std)

        return distribution

    def _log_prob_from_distribution(
        self, pi: torch.distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        log_prob = pi.log_prob(act).sum(axis=-1)

        return log_prob


class ActorDiscrete(Actor):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: list,
        activations: list,
        device: str,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [self.act_dim]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        logits = self.net(obs)
        distribution = Categorical(logits=logits)

        return distribution

    def _log_prob_from_distribution(
        self, pi: torch.distributions.Distribution, act: torch.Tensor
    ) -> torch.Tensor:
        log_prob = pi.log_prob(act)

        return log_prob
