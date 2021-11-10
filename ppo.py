from pdb import set_trace

import torch

from actor import ActorContinuous, ActorDiscrete
from critic import Critic
from torch.optim import Adam


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        actor_activations,
        critic_activations,
        continuous=False,
        clip_ratio=0.2,
        target_kl=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
    ):
        self.critic = Critic(obs_dim, critic_hidden_dim, critic_activations)

        if continuous:
            self.actor = ActorContinuous(
                obs_dim, act_dim, actor_hidden_dim, actor_activations
            )
        else:
            self.actor = ActorDiscrete(
                obs_dim, act_dim, actor_hidden_dim, actor_activations
            )

        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

        self.actor_opt = Adam(self.actor.net.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.net.parameters(), lr=critic_lr)

    def save(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_opt_state_dict": self.actor_opt.state_dict(),
                "critic_opt_state_dict": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])

    def get_action(self, obs, test=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32)
            policy, _ = self.actor(obs)

            if test:
                action = policy.mean
            else:
                action = policy.sample()

            logp_a = self.actor._log_prob_from_distribution(policy, action)

        return action.numpy(), logp_a.item()

    def get_value(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32)
            value = self.critic(obs)

        return value.item()

    def actor_loss(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old.detach())
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        # set_trace()

        return loss_pi, approx_kl

    def critic_loss(self, data):
        obs, ret = data["obs"], data["ret"]
        loss_vf = ((self.critic(obs).squeeze() - ret.detach()) ** 2).mean()

        return loss_vf
