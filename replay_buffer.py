import numpy as np
import scipy.signal
import torch
from pdb import set_trace


class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Borrowed from: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int,
        device: str,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.obs_buf = torch.zeros(max_size, obs_dim, dtype=torch.float32).to(
            device=device, non_blocking=True
        )
        self.act_buf = torch.zeros(max_size, act_dim, dtype=torch.float32).to(
            device=device, non_blocking=True
        )
        self.logp_buf = torch.zeros(max_size, dtype=torch.float32).to(
            device=device, non_blocking=True
        )
        self.adv_buf = torch.zeros(max_size, dtype=torch.float32).to(
            device=device, non_blocking=True
        )
        self.ret_buf = torch.zeros(max_size, dtype=torch.float32).to(
            device=device, non_blocking=True
        )

        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = max_size
        self.device = device

    def add(
        self, obs: torch.Tensor, act: torch.Tensor, rew: float, val: float, logp: float
    ):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_of_episode(self, last_val: float = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv_slice = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[path_slice] = torch.as_tensor(adv_slice.copy()).to(
            device=self.device, non_blocking=True
        )

        # the next line computes rewards-to-go, to be targets for the value function
        ret_slice = discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = torch.as_tensor(ret_slice.copy()).to(
            device=self.device, non_blocking=True
        )
        self.path_start_idx = self.ptr

    def sample(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean = torch.mean(self.adv_buf)
        adv_std = torch.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(
            obs=self.obs_buf.detach(),
            act=self.act_buf.detach(),
            ret=self.ret_buf.detach(),
            adv=self.adv_buf.detach(),
            logp=self.logp_buf.detach(),
        )

        return data


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
