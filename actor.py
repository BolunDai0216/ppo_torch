from pdb import set_trace

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from mlp import mlp


class Actor(torch.nn.Module):
    def forward(self, obs, act=None):
        pi = self._distribution(obs)

        if act is not None:
            log_prob = self._log_prob_from_distribution(pi, act)
        else:
            log_prob = None

        return pi, log_prob

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, obs, act):
        raise NotImplementedError


class ActorContinuous(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim,
        activations,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [self.act_dim]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations)

        # fixed standard deviation with variance of 0.2
        std = 0.04 * np.ones(act_dim, dtype=np.float32)
        std = torch.as_tensor(std)
        self.cov_mat = torch.diag(std).detach()

    def _distribution(self, obs):
        mean = self.net(obs)
        distribution = MultivariateNormal(mean, self.cov_mat)

        return distribution

    def _log_prob_from_distribution(self, pi, act):
        log_prob = pi.log_prob(act)

        return log_prob


class ActorDiscrete(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim,
        activations,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_sizes = [self.obs_dim] + hidden_dim + [self.act_dim]
        self.activations = activations
        self.net = mlp(self.feature_sizes, self.activations)

    def _distribution(self, obs):
        logits = self.net(obs)
        distribution = Categorical(logits=logits)

        return distribution

    def _log_prob_from_distribution(self, pi, act):
        log_prob = pi.log_prob(act)

        return log_prob
