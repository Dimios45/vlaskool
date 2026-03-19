"""SmolVLA + LoRA policy wrapper with Flow-SDE GRPO.

Architecture:
  - Base: SmolVLAPolicy from LeRobot (SmolVLM2-500M backbone + flow-matching head)
  - Adaptation: LoRA rank 32 on attention + MLP layers (paper Appendix F)
  - Training: Flow-SDE / FM-surrogate GRPO (same approach as Pi-0 in the paper)

Flow-Matching GRPO (Paper Flow-SDE section, also Pi-0 approach):
  SmolVLA is a flow-matching model: v_θ(x_t, t, s) predicts velocity.
  Training forward: x_t = t·noise + (1-t)·action, u_t = noise - action
                    loss = MSE(v_θ(x_t,t,s), u_t)
  Policy gradient uses FM loss as surrogate for log probability:
    log p_θ(a|s) ≈ -E_t[||v_θ(x_t,t,s) - u_t||²]
  Importance ratio:
    r(θ) = exp(log p_θ_new - log p_θ_old)
         = exp(-fm_loss_new + fm_loss_old)

This is the correct, differentiable GRPO formulation for flow-matching VLAs.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SmolVLA constants (matching LeRobot's lerobot/policies/smolvla/modeling_smolvla.py)
_ACTION_KEY = "action"
_STATE_KEY = "observation.state"
_IMG_KEY = "observation.images.camera1"


class SmolVLALoRAPolicy(nn.Module):
    """SmolVLA with LoRA adaptation for continual GRPO training.

    Supports three modes:
      - deterministic: ODE-decoded action μ_θ(s)           — evaluation
      - rollout:       deterministic + rollout noise ε      — GRPO collection
      - log_prob:      FM-surrogate log p(a|s) via forward()— GRPO update
    """

    def __init__(
        self,
        base_policy: nn.Module,
        action_dim: int = 6,
        exploration_sigma: float = 0.1,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.action_dim = action_dim
        self.exploration_sigma = exploration_sigma
        self.device = torch.device(device)
        self.dtype = dtype

    # ── Inference (no grad) ───────────────────────────────────────────────────

    @torch.no_grad()
    def select_action_deterministic(self, obs: dict) -> torch.Tensor:
        """ODE-decoded action for evaluation. Returns (B, action_dim) float32."""
        obs = self._to_device(obs)
        with self._autocast():
            action = self.base_policy.select_action(obs)
        if isinstance(action, dict):
            action = action.get("action", list(action.values())[0])
        # SmolVLA returns (B, T, action_dim) — take first step
        if action.ndim == 3:
            action = action[:, 0, :]
        return action.float()

    @torch.no_grad()
    def select_action_stochastic(
        self, obs: dict, sigma: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stochastic action for GRPO rollouts.

        Adds Gaussian noise to the deterministic action and computes
        the Gaussian log probability (for the old log_prob stored in rollout).

        Returns:
          action:   (B, action_dim) noisy action (what gets executed)
          mu:       (B, action_dim) deterministic mean (saved for FM log_prob)
          log_prob: (B,) Gaussian log p(action | mu, sigma) — used as placeholder
        """
        sigma = sigma if sigma is not None else self.exploration_sigma
        mu = self.select_action_deterministic(obs)        # (B, action_dim)
        noise = torch.randn_like(mu) * sigma
        action = mu + noise
        log_prob = self._gaussian_log_prob(action, mu, sigma)
        return action, mu, log_prob

    # ── Differentiable log probability (for GRPO update) ─────────────────────

    def compute_fm_log_prob(
        self,
        obs: dict,
        action: torch.Tensor,
        fm_noise: Optional[torch.Tensor] = None,
        fm_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample FM surrogate log p(a|s) using SmolVLA.forward().

        log p_θ(a|s) ≈ -E_t[||v_θ(x_t,t,s) - u_t||²]

        Uses the SAME noise/time as rollout collection for consistent importance ratios.

        Args:
          obs:      observation dict (matching SmolVLA input keys)
          action:   (B, action_dim) clean actions from rollout
          fm_noise: (B, 1, action_dim) the noise sample used in rollout (for consistency)
          fm_time:  (B,) the time sample used in rollout (for consistency)

        Returns:
          (B,) per-sample FM log probability (negative MSE, larger = more likely)
        """
        obs = self._to_device(obs)
        action = action.to(self.device).float()
        B = action.shape[0]

        # SmolVLA forward expects action shape: (B, chunk_size, action_dim)
        # We use the model's chunk_size; action is repeated across the chunk.
        config = getattr(self.base_policy, "config", None)
        chunk_size = getattr(config, "chunk_size", 50)
        action_input_dim = action.shape[-1]  # typically 6

        if action.ndim == 2:
            # (B, action_dim) → (B, chunk_size, action_dim)
            action_3d = action.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
        else:
            action_3d = action  # (B, T, action_dim)
            if action_3d.shape[1] != chunk_size:
                # Repeat or trim to chunk_size
                action_3d = action_3d[:, :chunk_size, :]
                if action_3d.shape[1] < chunk_size:
                    repeat_times = (chunk_size + action_3d.shape[1] - 1) // action_3d.shape[1]
                    action_3d = action_3d.repeat(1, repeat_times, 1)[:, :chunk_size, :]

        # Build full batch dict
        batch = {_ACTION_KEY: action_3d.to(self.device)}
        batch.update(obs)

        # Use fixed noise/time if provided (for consistent importance ratios).
        # Noise shape must match (B, chunk_size, max_action_dim).
        if fm_noise is not None:
            fn = fm_noise.to(self.device).float()
            # Expand to (B, chunk_size, max_action_dim)
            if fn.ndim == 3 and fn.shape[1] == 1:
                max_action_dim = getattr(config, "max_action_dim", fn.shape[-1])
                if fn.shape[-1] < max_action_dim:
                    fn = torch.cat([fn, torch.zeros(B, 1, max_action_dim - fn.shape[-1],
                                    device=self.device)], dim=-1)
                fn = fn.expand(-1, chunk_size, -1).contiguous()
            noise_arg = fn
        else:
            noise_arg = None
        time_arg = fm_time.to(self.device).float() if fm_time is not None else None

        with self._autocast():
            loss_scalar, loss_dict = self.base_policy.forward(
                batch, noise=noise_arg, time=time_arg
            )

        # Per-sample loss: (B, T, action_dim) → mean → (B,)
        if "losses_after_rm_padding" in loss_dict:
            per_sample = loss_dict["losses_after_rm_padding"]  # (B, T, D)
        else:
            # Fallback: recompute with scalar
            per_sample = loss_scalar.expand(B, 1, self.action_dim)

        fm_loss = per_sample.float().mean(dim=(1, 2))  # (B,)
        return -fm_loss  # log p ≈ -MSE (larger = more likely)

    def compute_log_prob(
        self,
        obs: dict,
        action: torch.Tensor,
        sigma: float | None = None,
        fm_noise: Optional[torch.Tensor] = None,
        fm_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unified log probability: uses FM surrogate for SmolVLA.

        Falls back to Gaussian wrapper if FM forward fails.
        """
        try:
            return self.compute_fm_log_prob(obs, action, fm_noise, fm_time)
        except Exception as e:
            logger.warning(f"FM log_prob failed ({e}), falling back to Gaussian wrapper")
            sigma = sigma if sigma is not None else self.exploration_sigma
            obs = self._to_device(obs)
            action = action.to(self.device).float()
            mu = self._forward_with_grad_gaussian(obs)
            return self._gaussian_log_prob(action, mu, sigma)

    def _forward_with_grad_gaussian(self, obs: dict) -> torch.Tensor:
        """Gaussian fallback: get mean action with gradient tracking."""
        with torch.enable_grad(), self._autocast():
            action = self.base_policy.select_action(obs)
        if isinstance(action, dict):
            action = action.get("action", list(action.values())[0])
        if action.ndim == 3:
            action = action[:, 0, :]
        return action.float()

    # ── Log probability helpers ───────────────────────────────────────────────

    @staticmethod
    def _gaussian_log_prob(action: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
        d = action.shape[-1]
        sq_error = ((action - mu) ** 2).sum(dim=-1)
        return -sq_error / (2 * sigma**2) - d / 2 * math.log(2 * math.pi * sigma**2)

    # ── Autocast helper ───────────────────────────────────────────────────────

    def _autocast(self):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        if self.dtype == torch.bfloat16 and device_type == "cuda":
            return torch.amp.autocast(device_type=device_type, dtype=self.dtype)
        return nullcontext()

    # ── LoRA management ───────────────────────────────────────────────────────

    def save_lora(self, path: str | Path) -> None:
        """Save only the trainable (LoRA) parameters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        lora_sd = {
            name: param.data.cpu()
            for name, param in self.base_policy.named_parameters()
            if param.requires_grad
        }
        torch.save(lora_sd, path / "lora_weights.pt")
        logger.info(f"LoRA weights saved → {path} ({len(lora_sd)} tensors)")

    def load_lora(self, path: str | Path) -> None:
        """Load trainable (LoRA) parameters from checkpoint."""
        path = Path(path)
        ckpt_file = path / "lora_weights.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"No LoRA checkpoint at {ckpt_file}")
        lora_sd = torch.load(ckpt_file, map_location=self.device, weights_only=True)
        # Load only the matching LoRA params
        model_sd = self.base_policy.state_dict()
        model_sd.update(lora_sd)
        self.base_policy.load_state_dict(model_sd)
        logger.info(f"LoRA weights loaded ← {path} ({len(lora_sd)} tensors)")

    # ── Param accessors ───────────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.base_policy.parameters() if p.requires_grad]

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.base_policy.parameters())

    # ── Standard nn.Module overrides ──────────────────────────────────────────

    def _to_device(self, obs: dict) -> dict:
        out = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device,
                              dtype=self.dtype if v.dtype.is_floating_point else v.dtype)
            else:
                out[k] = v
        return out

    def train(self, mode: bool = True):
        self.base_policy.train(mode)
        return self

    def eval(self):
        self.base_policy.eval()
        return self

    def to(self, *args, **kwargs):
        self.base_policy = self.base_policy.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        return self.base_policy.parameters(recurse=recurse)

    def state_dict(self, **kwargs):
        return self.base_policy.state_dict(**kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self.base_policy.load_state_dict(state_dict, **kwargs)


# ── Factory ───────────────────────────────────────────────────────────────────

def load_smolvla_with_lora(
    pretrained_name: str = "lerobot/smolvla_base",
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
    action_dim: int = 6,
    exploration_sigma: float = 0.1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
) -> SmolVLALoRAPolicy:
    """Load SmolVLA, apply LoRA (rank 32), return wrapped policy.

    Paper hyperparams (Appendix F): rank=32, alpha=32, dropout=0.

    Uses manual LoRA injection (not PEFT get_peft_model) to avoid
    attribute collisions with SmolVLA's internal model hierarchy.
    """
    from vlaskool.models.lora_manual import inject_lora

    logger.info(f"Loading SmolVLA from {pretrained_name!r}...")
    base_policy = _load_base_policy(pretrained_name, device=device, dtype=dtype)

    if target_modules is None:
        target_modules = _auto_detect_target_modules(base_policy)

    # Manual LoRA injection: avoids PEFT wrapping issues with SmolVLA's
    # nested model structure (SmolVLMForConditionalGeneration.model conflict)
    base_policy = inject_lora(
        base_policy, rank=lora_rank, alpha=lora_alpha,
        target_names=target_modules, dropout=lora_dropout,
    )
    # Move trainable LoRA params to target device + dtype
    for param in base_policy.parameters():
        if param.requires_grad:
            param.data = param.data.to(device=device, dtype=dtype)

    n_train = sum(p.numel() for p in base_policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in base_policy.parameters())
    logger.info(f"LoRA applied: {n_train:,} / {n_total:,} trainable ({100*n_train/n_total:.2f}%)")

    policy = SmolVLALoRAPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        exploration_sigma=exploration_sigma,
        device=device,
        dtype=dtype,
    )

    if checkpoint_path is not None:
        policy.load_lora(checkpoint_path)

    return policy


def _load_base_policy(pretrained_name: str, device: str, dtype: torch.dtype) -> nn.Module:
    """Load SmolVLAPolicy from LeRobot hub and configure for single-camera setup."""
    for module_path in [
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.common.policies.smolvla.modeling_smolvla",
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            policy = mod.SmolVLAPolicy.from_pretrained(pretrained_name)
            policy = policy.to(device=device, dtype=dtype)
            _fix_smolvla_normalization(policy)

            # We use 1 camera (overhead) but SmolVLA expects 3.
            # Allow remaining cameras to be empty (black, masked out).
            n_cameras = len(policy.config.image_features)
            policy.config.empty_cameras = n_cameras - 1  # allow all-but-one to be empty
            logger.info(
                f"Loaded SmolVLAPolicy via {module_path} "
                f"(empty_cameras={policy.config.empty_cameras})"
            )
            return policy
        except ImportError:
            continue
        except Exception as e:
            logger.warning(f"LeRobot load via {module_path} failed: {e}")
            break

    logger.warning("Could not load SmolVLA. Using random stub for testing.")
    return _StubPolicy(action_dim=6, device=device, dtype=dtype)


def _auto_detect_target_modules(model: nn.Module) -> list[str]:
    """Find LoRA target Linear modules in SmolVLA (SmolVLM2 backbone)."""
    CANONICAL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    FALLBACK = ["out_proj", "fc1", "fc2"]

    found = {name.split(".")[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    targets = [m for m in CANONICAL if m in found] or [m for m in FALLBACK if m in found]
    if not targets:
        targets = sorted(found - {"embed_tokens", "lm_head"})
    logger.info(f"LoRA target modules: {targets}")
    return targets


def _fix_smolvla_normalization(policy: nn.Module) -> None:
    """Fix uninitialized (inf) normalization stats → identity (mean=0, std=1)."""
    for mod_name in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"]:
        if not hasattr(policy, mod_name):
            continue
        mod = getattr(policy, mod_name)
        for name, param in list(mod.named_parameters()):
            if torch.any(torch.isinf(param)):
                if "mean" in name or "min" in name:
                    param.data.zero_()
                else:
                    param.data.fill_(1.0)


# ── Stub for testing ──────────────────────────────────────────────────────────

class _StubPolicy(nn.Module):
    """Random-action stub policy for testing without a real checkpoint."""

    def __init__(self, action_dim: int = 6, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.action_dim = action_dim
        self._dev = torch.device(device)
        self._dtype = dtype
        # Minimal config for compatibility
        from types import SimpleNamespace
        self.config = SimpleNamespace(
            max_action_dim=action_dim,
            chunk_size=1,
        )
        self.head = nn.Linear(64, action_dim)

    def select_action(self, batch: dict) -> torch.Tensor:
        B = next(
            (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim >= 2),
            1,
        )
        return torch.randn(B, self.action_dim, device=self._dev, dtype=self._dtype)

    def forward(self, batch: dict, noise=None, time=None) -> tuple:
        B = next(
            (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim >= 2),
            1,
        )
        loss = torch.zeros(B, requires_grad=True).mean()
        per_sample = torch.zeros(B, 1, self.action_dim, requires_grad=True)
        return loss, {"losses_after_rm_padding": per_sample, "loss": 0.0}

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model.pt")
