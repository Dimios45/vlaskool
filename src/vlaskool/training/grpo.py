"""GRPO (Group Relative Policy Optimization) trainer.

Implements the training loop from the paper (Appendix F, Table 6):
  - Optimizer:      AdamW, lr=2e-5, β=(0.9, 0.999), ε=1e-5
  - Batch size:     8192 transitions per update
  - Rollout epochs: 16 optimization passes per collected batch
  - Group size:     8 parallel envs per initial state
  - Clip ratio:     (0.20, 0.28)
  - KL coef:        0.0 (no KL penalty)
  - Entropy bonus:  0.0

The GRPO loss (importance-weighted + clipping):
  L = -E[ min(r(θ) · Â, clip(r(θ), 1-ε_lo, 1+ε_hi) · Â) ]
  where r(θ) = p_θ(a|s) / p_θ_old(a|s)

For SmolVLA (flow matching) we use the Gaussian wrapper:
  log p(a|s) = -||a - μ_θ(s)||² / (2σ²) - d/2·log(2πσ²)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from vlaskool.models.policy import SmolVLALoRAPolicy
from vlaskool.training.rollout import RolloutBatch, collect_rollouts
from vlaskool.envs.wrappers import VecEnvWrapper

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Trains SmolVLALoRAPolicy on a single task using GRPO.

    Usage:
        trainer = GRPOTrainer(policy, env, cfg)
        stats = trainer.train(task_cfg)
    """

    def __init__(
        self,
        policy: SmolVLALoRAPolicy,
        env: VecEnvWrapper,
        # Hyperparams (paper defaults)
        learning_rate: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip_norm: float = 1.0,
        global_batch_size: int = 8192,
        rollout_epochs: int = 16,
        group_size: int = 8,
        clip_low: float = 0.20,
        clip_high: float = 0.28,
        total_episodes: int = 10240,
        episode_length: int = 80,
        exploration_sigma: float = 0.1,
        sigma_decay: float = 0.995,
        sigma_min: float = 0.01,
        # VLM micro-batch size: number of samples forwarded through SmolVLA at once.
        # Gradient accumulation is used so effective batch = vla_minibatch_size × accum_steps.
        # RTX 4090 fits ~4 samples per VLM forward with bfloat16.
        vla_minibatch_size: int = 4,
        device: str = "cuda",
        wandb_run=None,
        checkpoint_dir: Path = Path("checkpoints"),
        log_interval: int = 10,
        grpo_mode: str = "reinforce",
    ) -> None:
        self.policy = policy
        self.env = env
        self.lr = learning_rate
        self.betas = betas
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip_norm
        self.global_batch_size = global_batch_size
        self.rollout_epochs = rollout_epochs
        self.group_size = group_size
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.sigma = exploration_sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.vla_minibatch_size = vla_minibatch_size
        self.device = torch.device(device)
        self.wandb_run = wandb_run
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.grpo_mode = grpo_mode

        # Optimizer is created fresh per task (paper: reset between tasks)
        self.optimizer: Optional[AdamW] = None
        self._update_step = 0

    def _build_optimizer(self) -> AdamW:
        params = self.policy.trainable_parameters()
        logger.info(f"GRPO optimizer: {sum(p.numel() for p in params):,} trainable params")
        return AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

    def train(
        self,
        task_id: str,
        instruction: str,
        save_path: Optional[Path] = None,
    ) -> dict[str, list]:
        """Run GRPO training on one task.

        Args:
            task_id:     Task identifier (for logging/checkpointing).
            instruction: Language instruction for this task.
            save_path:   Directory to save final LoRA checkpoint.

        Returns:
            stats dict with 'success_rate', 'loss', 'update_steps'.
        """
        self.optimizer = self._build_optimizer()
        self.policy.train()

        # Derived quantities
        G = self.group_size
        T = self.episode_length
        transitions_per_group = G * T
        n_groups_per_update = max(1, self.global_batch_size // transitions_per_group)
        total_groups = self.total_episodes // G
        n_updates = max(1, total_groups // n_groups_per_update)

        logger.info(
            f"\n{'='*60}\n"
            f"GRPO training: task={task_id!r}\n"
            f"  instruction:      {instruction!r}\n"
            f"  total_episodes:   {self.total_episodes}\n"
            f"  n_updates:        {n_updates}\n"
            f"  groups_per_update:{n_groups_per_update}\n"
            f"  rollout_epochs:   {self.rollout_epochs} (effective: {1 if self.grpo_mode == 'reinforce' else self.rollout_epochs})\n"
            f"  grpo_mode:        {self.grpo_mode}\n"
            f"  exploration σ:    {self.sigma:.4f}\n"
            f"{'='*60}"
        )

        stats = {
            "success_rate": [],
            "loss": [],
            "pg_loss": [],
            "update_steps": [],
            "sigma": [],
        }

        for update_idx in range(n_updates):
            # ── Collect rollouts ──────────────────────────────────────────────
            batch = collect_rollouts(
                policy=self.policy,
                env=self.env,
                n_groups=n_groups_per_update,
                episode_length=T,
                exploration_sigma=self.sigma,
                device=self.device,
                instruction=instruction,
                skip_fm_log_prob=(self.grpo_mode == "reinforce"),
            )
            batch = batch.to(self.device)

            # ── GRPO update (rollout_epochs passes) ───────────────────────────
            # SmolVLA is a VLM — forward a small micro-batch and accumulate.
            # Logical batch per epoch step = vla_minibatch_size × accum_steps.
            total_loss = 0.0
            n_mb = 0
            B = len(batch)
            # Accumulate over as many micro-batches as fit in one logical step.
            # We process the full collected batch once per epoch.
            accum_steps = max(1, B // self.vla_minibatch_size)

            # REINFORCE mode: no off-policy reuse (fresh noise each forward)
            effective_rollout_epochs = 1 if self.grpo_mode == "reinforce" else self.rollout_epochs

            for epoch in range(effective_rollout_epochs):
                self.optimizer.zero_grad()
                epoch_loss = 0.0
                for mb in batch.minibatch_iter(self.vla_minibatch_size):
                    if self.grpo_mode == "reinforce":
                        loss, _pg = self._grpo_loss_reinforce(mb)
                    else:
                        loss, _pg = self._grpo_loss(mb)
                    # Scale loss for gradient accumulation
                    (loss / accum_steps).backward()
                    epoch_loss += loss.item()

                nn.utils.clip_grad_norm_(
                    self.policy.trainable_parameters(), self.grad_clip
                )
                self.optimizer.step()
                total_loss += epoch_loss / max(accum_steps, 1)
                n_mb += 1

            avg_loss = total_loss / max(n_mb, 1)
            self._update_step += 1

            # Decay exploration noise
            self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

            # ── Logging ───────────────────────────────────────────────────────
            sr = batch.mean_reward
            stats["success_rate"].append(sr)
            stats["loss"].append(avg_loss)
            stats["update_steps"].append(self._update_step)
            stats["sigma"].append(self.sigma)

            if update_idx % min(self.log_interval, max(1, n_updates // 10)) == 0 or update_idx == n_updates - 1:
                logger.info(
                    f"  [{task_id}] update {update_idx+1}/{n_updates} | "
                    f"success={sr:.3f} | loss={avg_loss:.4f} | σ={self.sigma:.4f}"
                )

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        f"train/{task_id}/success_rate": sr,
                        f"train/{task_id}/loss": avg_loss,
                        f"train/{task_id}/sigma": self.sigma,
                        "update_step": self._update_step,
                    }
                )

        # ── Save checkpoint ───────────────────────────────────────────────────
        if save_path is not None:
            self.policy.save_lora(save_path)
            logger.info(f"Saved LoRA checkpoint → {save_path}")

        return stats

    # ── GRPO loss ─────────────────────────────────────────────────────────────

    def _grpo_loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FM-GRPO objective for one minibatch.

        Uses flow-matching surrogate log probability:
          log p_θ(a|s) ≈ -E_t[||v_θ(x_t,t,s) - u_t||²]

        Importance ratio (same FM noise/time as rollout for consistency):
          r(θ) = exp(log p_θ_new - log p_θ_old)

        Clipped objective:
          L = -mean[ min(r·Â, clip(r, 1-ε_lo, 1+ε_hi)·Â) ]
        """
        obs = {
            "observation.images.camera1": batch.obs_images,
            "observation.state": batch.obs_states,
            "task": batch.tasks,
        }

        # FM-surrogate log prob with current θ (requires grad)
        curr_log_probs = self.policy.compute_log_prob(
            obs=obs,
            action=batch.mus,       # use deterministic actions (no exploration noise)
            sigma=self.sigma,
            fm_noise=batch.fm_noise,
            fm_time=batch.fm_time,
        )  # (B,)

        # Importance ratio
        old_log_probs = batch.old_log_probs.to(curr_log_probs.device)
        ratio = torch.exp(curr_log_probs - old_log_probs)  # (B,)

        adv = batch.advantages.to(curr_log_probs.device)

        # Clipped GRPO objective
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        pg_loss = -torch.min(unclipped, clipped).mean()

        return pg_loss, pg_loss

    def _grpo_loss_reinforce(self, batch: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """REINFORCE-style GRPO: advantage-weighted FM loss.

        Uses the same gradient pathway as SFT (which demonstrably works):
          L = mean(FM_loss_per_sample · advantage)

        Positive advantage → minimize FM loss (reinforce success).
        Negative advantage → maximize FM loss (push away from failure).
        No importance ratio needed — fresh noise/time each forward.
        """
        obs = {
            "observation.images.camera1": batch.obs_images,
            "observation.state": batch.obs_states,
            "task": batch.tasks,
        }

        # Per-sample FM loss (positive MSE, fresh noise/time, with gradients)
        fm_loss = self.policy.compute_fm_loss_per_sample(
            obs=obs, action=batch.mus,
        )  # (B,)

        adv = batch.advantages.to(fm_loss.device)

        # Advantage-weighted FM loss
        loss = (fm_loss * adv).mean()
        return loss, loss
