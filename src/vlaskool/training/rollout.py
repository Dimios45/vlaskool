"""Rollout collection for FM-GRPO.

GRPO rollout structure (paper Appendix F):
  - group_size G = 8: for each initial state, 8 parallel trajectories
  - Each trajectory: H = 80 steps, binary terminal reward
  - groups_per_update = global_batch_size // (G × H) ≈ 12

Flow-Matching GRPO:
  For each step, we also collect (fm_noise, fm_time) used during rollout.
  These are passed to compute_fm_log_prob() during the GRPO update to ensure
  the importance ratio uses the SAME noise sample:
    r(θ) = exp(-fm_loss_θ_new + fm_loss_θ_old)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import numpy as np

from vlaskool.envs.wrappers import VecEnvWrapper
from vlaskool.models.policy import SmolVLALoRAPolicy


@dataclasses.dataclass
class RolloutBatch:
    """Container for G × N_groups episodes.

    Shapes (after flattening):  B = n_groups × G × T
      obs_images:    (B, C, H, W)
      obs_states:    (B, state_dim)
      actions:       (B, action_dim)        — noisy actions executed
      mus:           (B, action_dim)        — deterministic ODE actions
      old_log_probs: (B,)                   — FM surrogate log_p at collection time
      fm_noise:      (B, 1, action_dim)     — FM noise sample (for consistent IR)
      fm_time:       (B,)                   — FM time sample
      advantages:    (B,)                   — group-normalized advantages
      tasks:         list[str]
    """

    obs_images: torch.Tensor      # (B, C, H, W)
    obs_states: torch.Tensor      # (B, state_dim)
    actions: torch.Tensor         # (B, action_dim)
    mus: torch.Tensor             # (B, action_dim)
    old_log_probs: torch.Tensor   # (B,)
    fm_noise: torch.Tensor        # (B, 1, action_dim) for importance ratio
    fm_time: torch.Tensor         # (B,)
    advantages: torch.Tensor      # (B,)
    tasks: list[str]

    # Episode-level info
    episode_successes: torch.Tensor  # (n_groups × G,)
    mean_reward: float
    n_episodes: int

    def to(self, device: torch.device | str) -> "RolloutBatch":
        return dataclasses.replace(
            self,
            obs_images=self.obs_images.to(device),
            obs_states=self.obs_states.to(device),
            actions=self.actions.to(device),
            mus=self.mus.to(device),
            old_log_probs=self.old_log_probs.to(device),
            fm_noise=self.fm_noise.to(device),
            fm_time=self.fm_time.to(device),
            advantages=self.advantages.to(device),
            episode_successes=self.episode_successes.to(device),
        )

    def __len__(self) -> int:
        return self.actions.shape[0]

    def minibatch_iter(self, minibatch_size: int):
        """Yield shuffled minibatches for multiple rollout epochs."""
        idx = torch.randperm(len(self))
        for start in range(0, len(self), minibatch_size):
            mb_idx = idx[start : start + minibatch_size]
            yield RolloutBatch(
                obs_images=self.obs_images[mb_idx],
                obs_states=self.obs_states[mb_idx],
                actions=self.actions[mb_idx],
                mus=self.mus[mb_idx],
                old_log_probs=self.old_log_probs[mb_idx],
                fm_noise=self.fm_noise[mb_idx],
                fm_time=self.fm_time[mb_idx],
                advantages=self.advantages[mb_idx],
                tasks=[self.tasks[i] for i in mb_idx.tolist()],
                episode_successes=self.episode_successes,
                mean_reward=self.mean_reward,
                n_episodes=self.n_episodes,
            )


@torch.no_grad()
def collect_rollouts(
    policy: SmolVLALoRAPolicy,
    env: VecEnvWrapper,
    n_groups: int,
    episode_length: int,
    exploration_sigma: float,
    device: torch.device,
    instruction: str,
    skip_fm_log_prob: bool = False,
) -> RolloutBatch:
    """Collect n_groups × G episodes using the stochastic policy.

    For each group:
      1. Reset all G envs to new initial states
      2. Roll out for episode_length steps
      3. Get binary terminal reward (success)
      4. Compute group-normalized advantages

    Also collects FM noise/time for use in importance ratio computation.
    """
    G = env.num_envs
    policy.eval()

    all_images: list[torch.Tensor] = []
    all_states: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_mus: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []
    all_fm_noise: list[torch.Tensor] = []
    all_fm_time: list[torch.Tensor] = []
    all_successes: list[torch.Tensor] = []

    action_dim = policy.action_dim
    cfg = getattr(policy.base_policy, "config", None)
    max_action_dim = getattr(cfg, "max_action_dim", action_dim)
    chunk_size = getattr(cfg, "chunk_size", 50)

    for group_idx in range(n_groups):
        obs, _ = env.reset()

        g_images, g_states, g_actions = [], [], []
        g_mus, g_log_probs, g_fm_noise, g_fm_time = [], [], [], []

        for step in range(episode_length):
            img = obs["observation.images.camera1"].to(device)    # (G, C, H, W)
            state = obs["observation.state"].to(device)           # (G, state_dim)

            # Deterministic action from flow matching ODE
            action = policy.select_action_deterministic(obs)       # (G, action_dim)
            mu = action.clone()

            # Exploration noise (Gaussian wrapper for stochastic rollout)
            noise_eps = torch.randn_like(action) * exploration_sigma
            action_noisy = action + noise_eps                      # (G, action_dim)

            if skip_fm_log_prob:
                # REINFORCE mode: no importance ratio needed, skip VLM forward
                fm_noise_sample = torch.zeros(G, 1, max_action_dim, device=device)
                fm_time_sample = torch.zeros(G, device=device)
                lp = torch.zeros(G, device=device)
            else:
                # PPO mode: sample FM noise/time for importance ratio computation
                fm_noise_sample = torch.randn(G, 1, max_action_dim, device=device)  # (G,1,D)
                fm_time_sample = torch.rand(G, device=device)                        # (G,)

                obs_for_lp = {
                    "observation.images.camera1": img,
                    "observation.state": state,
                    "task": obs["task"],
                }
                try:
                    lp = policy.compute_fm_log_prob(
                        obs_for_lp, mu, fm_noise=fm_noise_sample, fm_time=fm_time_sample
                    )  # (G,)
                except Exception:
                    lp = SmolVLALoRAPolicy._gaussian_log_prob(action_noisy, mu, exploration_sigma)

            g_images.append(img)
            g_states.append(state)
            g_actions.append(action_noisy)
            g_mus.append(mu)
            g_log_probs.append(lp)
            g_fm_noise.append(fm_noise_sample)
            g_fm_time.append(fm_time_sample)

            obs, reward, done, info = env.step(action_noisy)

        success = env.get_success().float().to(device)  # (G,)

        all_images.append(torch.stack(g_images, dim=1))      # (G, T, C, H, W)
        all_states.append(torch.stack(g_states, dim=1))      # (G, T, state_dim)
        all_actions.append(torch.stack(g_actions, dim=1))    # (G, T, action_dim)
        all_mus.append(torch.stack(g_mus, dim=1))            # (G, T, action_dim)
        all_log_probs.append(torch.stack(g_log_probs, dim=1))# (G, T)
        all_fm_noise.append(torch.stack(g_fm_noise, dim=1))  # (G, T, 1, max_D)
        all_fm_time.append(torch.stack(g_fm_time, dim=1))    # (G, T)
        all_successes.append(success)                         # (G,)

    # ── Group-normalized advantages ───────────────────────────────────────────
    successes = torch.stack(all_successes, dim=0)             # (n_groups, G)
    group_mean = successes.mean(dim=1, keepdim=True)
    group_std = successes.std(dim=1, keepdim=True) + 1e-8
    group_adv = (successes - group_mean) / group_std          # (n_groups, G)
    T = episode_length
    adv_per_step = group_adv.unsqueeze(-1).expand(-1, -1, T)  # (n_groups, G, T)

    # ── Flatten: (n_groups, G, T, ...) → (B, ...) ────────────────────────────
    n_g = n_groups

    imgs_s = torch.stack(all_images, dim=0)      # (n_g, G, T, C, H, W)
    B_total = n_g * G * T
    C, H, W = imgs_s.shape[3], imgs_s.shape[4], imgs_s.shape[5]
    flat_images = imgs_s.reshape(B_total, C, H, W)

    states_s = torch.stack(all_states, dim=0)
    sd = states_s.shape[-1]
    flat_states = states_s.reshape(B_total, sd)

    actions_s = torch.stack(all_actions, dim=0)
    ad = actions_s.shape[-1]
    flat_actions = actions_s.reshape(B_total, ad)

    mus_s = torch.stack(all_mus, dim=0)
    flat_mus = mus_s.reshape(B_total, ad)

    lp_s = torch.stack(all_log_probs, dim=0)
    flat_lp = lp_s.reshape(B_total)

    fn_s = torch.stack(all_fm_noise, dim=0)      # (n_g, G, T, 1, max_D)
    max_D = fn_s.shape[-1]
    flat_fm_noise = fn_s.reshape(B_total, 1, max_D)

    ft_s = torch.stack(all_fm_time, dim=0)
    flat_fm_time = ft_s.reshape(B_total)

    flat_adv = adv_per_step.reshape(B_total)
    flat_tasks = [instruction] * B_total

    return RolloutBatch(
        obs_images=flat_images.cpu(),
        obs_states=flat_states.cpu(),
        actions=flat_actions.cpu(),
        mus=flat_mus.cpu(),
        old_log_probs=flat_lp.cpu(),
        fm_noise=flat_fm_noise.cpu(),
        fm_time=flat_fm_time.cpu(),
        advantages=flat_adv.cpu(),
        tasks=flat_tasks,
        episode_successes=successes.reshape(-1).cpu(),
        mean_reward=successes.mean().item(),
        n_episodes=n_g * G,
    )
