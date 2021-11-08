from pdb import set_trace

import torch

from actor import ActorContinuous, ActorDiscrete
from critic import Critic
from torch.optim import Adam


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dim: list,
        critic_hidden_dim: list,
        actor_activations: list,
        critic_activations: list,
        device: str,
        continuous: bool = False,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
    ):
        self.critic = Critic(obs_dim, critic_hidden_dim, critic_activations, device)

        if continuous:
            self.actor = ActorContinuous(
                obs_dim, act_dim, actor_hidden_dim, actor_activations, device
            )
        else:
            self.actor = ActorDiscrete(
                obs_dim, act_dim, actor_hidden_dim, actor_activations, device
            )

        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        self.actor_opt = Adam(self.actor.net.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.net.parameters(), lr=critic_lr)

    def save(self, path: str):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_opt_state_dict": self.actor_opt.state_dict(),
                "critic_opt_state_dict": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        policy, _ = self.actor(obs)
        action = policy.sample()
        logp_a = self.actor._log_prob_from_distribution(policy, action)

        return action, logp_a

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        value = self.critic(obs)

        return value

    def actor_loss(self, data: dict) -> (torch.Tensor, dict):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def critic_loss(self, data: dict):
        obs, ret = data["obs"], data["ret"]
        loss_vf = ((self.critic(obs) - ret) ** 2).mean()

        return loss_vf
