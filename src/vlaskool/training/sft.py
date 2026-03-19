"""Supervised Fine-Tuning (Behavior Cloning) trainer.

Used for:
  1. The SFT baseline (comparison #2: normal finetuning on seen tasks)
  2. Initial warm-start before GRPO (required by paper)

Uses SmolVLA's native flow-matching forward loss for training,
NOT MSE on decoded actions (which would require no_grad removal).

Demo format (HDF5):
  demos/<task_id>/episode_<N>.h5:
    - observations/images/overhead: (T, H, W, C) uint8
    - observations/state:           (T, state_dim) float32
    - actions:                      (T, action_dim) float32
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from vlaskool.models.policy import SmolVLALoRAPolicy

logger = logging.getLogger(__name__)


class DemoDataset(Dataset):
    """Loads demonstration episodes from HDF5 files for behavior cloning.

    Each sample returns a single timestep's observation + an action chunk
    of length `chunk_size` starting from that timestep (zero-padded if
    fewer steps remain in the episode).
    """

    def __init__(
        self,
        demo_dir: Path | str,
        task_id: str,
        instruction: str,
        chunk_size: int = 50,
        action_dim: int = 6,
    ) -> None:
        self.instruction = instruction
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.episodes: list[dict] = []

        demo_path = Path(demo_dir) / task_id
        if not demo_path.exists():
            logger.warning(f"Demo directory not found: {demo_path}")
            return

        for ep_file in sorted(demo_path.glob("episode_*.h5")):
            try:
                ep = self._load_episode(ep_file)
                if ep["actions"].shape[-1] != action_dim:
                    logger.warning(
                        f"Action dim mismatch in {ep_file}: "
                        f"got {ep['actions'].shape[-1]}, expected {action_dim}"
                    )
                    continue
                self.episodes.append(ep)
            except Exception as e:
                logger.warning(f"Could not load {ep_file}: {e}")

        n_steps = sum(len(ep["actions"]) for ep in self.episodes)
        logger.info(
            f"DemoDataset: {len(self.episodes)} episodes, {n_steps} steps — {task_id!r}"
        )

    def _load_episode(self, path: Path) -> dict:
        with h5py.File(path, "r") as f:
            images = f["observations/images/overhead"][:]  # (T, H, W, C)
            state = f["observations/state"][:]             # (T, state_dim)
            actions = f["actions"][:]                       # (T, action_dim)
        return {"images": images, "state": state, "actions": actions}

    def __len__(self) -> int:
        return sum(len(ep["actions"]) for ep in self.episodes)

    def __getitem__(self, idx: int) -> dict:
        for ep in self.episodes:
            T = len(ep["actions"])
            if idx < T:
                img = ep["images"][idx]       # (H, W, C) uint8
                state = ep["state"][idx]      # (state_dim,) float32

                # Build action chunk: chunk_size steps starting from idx
                remaining = ep["actions"][idx:]  # (T-idx, action_dim)
                if len(remaining) >= self.chunk_size:
                    action_chunk = remaining[:self.chunk_size]
                else:
                    # Pad with last action repeated
                    pad_len = self.chunk_size - len(remaining)
                    last = remaining[-1:]
                    action_chunk = np.concatenate(
                        [remaining, np.repeat(last, pad_len, axis=0)], axis=0
                    )

                # Normalize image: (H, W, C) uint8 → (C, H, W) float32 [0,1]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                return {
                    "observation.images.camera1": img_t,
                    "observation.state": torch.from_numpy(state).float(),
                    "action": torch.from_numpy(action_chunk).float(),  # (chunk_size, action_dim)
                    "task": self.instruction,
                }
            idx -= T
        raise IndexError(f"Index out of range: {idx}")


class SFTTrainer:
    """Behavior cloning trainer using SmolVLA's native flow-matching loss.

    Instead of MSE on decoded actions (which requires disabling no_grad),
    we call base_policy.forward(batch) which computes the proper FM loss:
      loss = E_t[||v_θ(x_t, t, s) - u_t||²]
    This is differentiable and matches SmolVLA's pretraining objective.
    """

    def __init__(
        self,
        policy: SmolVLALoRAPolicy,
        demo_dir: Path | str,
        # Training hyperparams
        learning_rate: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        batch_size: int = 32,
        micro_batch_size: int = 4,
        num_epochs: int = 100,
        warmup_steps: int = 100,
        device: str = "cuda",
        wandb_run=None,
        checkpoint_dir: Path = Path("checkpoints"),
        log_interval: int = 10,
    ) -> None:
        self.policy = policy
        self.demo_dir = Path(demo_dir)
        self.lr = learning_rate
        self.betas = betas
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip_norm
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)
        self.wandb_run = wandb_run
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

    def train(
        self,
        task_id: str,
        instruction: str,
        save_path: Optional[Path] = None,
    ) -> dict[str, list]:
        """Train via flow-matching BC on demonstrations for one task.

        Args:
            task_id:     Task identifier.
            instruction: Language instruction.
            save_path:   Optional path to save LoRA weights.

        Returns:
            Stats dict with 'loss', 'epoch'.
        """
        # Get chunk_size from SmolVLA config
        config = getattr(self.policy.base_policy, "config", None)
        chunk_size = getattr(config, "chunk_size", 50)

        dataset = DemoDataset(
            demo_dir=self.demo_dir,
            task_id=task_id,
            instruction=instruction,
            chunk_size=chunk_size,
            action_dim=self.policy.action_dim,
        )
        if len(dataset) == 0:
            logger.warning(f"No demos found for {task_id!r} — skipping SFT")
            return {"loss": [], "epoch": []}

        # Use micro_batch_size for the DataLoader (fits in GPU memory).
        # Accumulate gradients to simulate the logical batch_size.
        accum_steps = max(1, self.batch_size // self.micro_batch_size)
        loader = DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        optimizer = AdamW(
            self.policy.trainable_parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        # Linear warmup scheduler
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            return 1.0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        self.policy.train()
        stats = {"loss": [], "epoch": []}
        global_step = 0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for mb_idx, batch in enumerate(loader):
                # Build batch dict for SmolVLA's forward()
                fm_batch = {
                    "observation.images.camera1": batch["observation.images.camera1"].to(
                        self.device, dtype=self.policy.dtype
                    ),
                    "observation.state": batch["observation.state"].to(
                        self.device, dtype=self.policy.dtype
                    ),
                    "action": batch["action"].to(
                        self.device, dtype=self.policy.dtype
                    ),  # (B, chunk_size, action_dim)
                    "task": batch["task"],
                }

                # SmolVLA forward: flow-matching loss (differentiable)
                with torch.autocast("cuda", dtype=self.policy.dtype):
                    loss, loss_dict = self.policy.base_policy.forward(fm_batch)

                # Scale for gradient accumulation
                (loss / accum_steps).backward()
                epoch_loss += loss.item()
                n_batches += 1

                # Step optimizer every accum_steps micro-batches
                if (mb_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.policy.trainable_parameters(), self.grad_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Handle any remaining accumulated gradients
            if n_batches % accum_steps != 0:
                nn.utils.clip_grad_norm_(
                    self.policy.trainable_parameters(), self.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stats["loss"].append(avg_loss)
            stats["epoch"].append(epoch)

            if epoch % self.log_interval == 0:
                logger.info(f"  [{task_id}] SFT epoch {epoch+1}/{self.num_epochs} | loss={avg_loss:.6f}")

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        f"sft/{task_id}/loss": avg_loss,
                        f"sft/{task_id}/epoch": epoch,
                        "sft_step": global_step,
                    }
                )

        if save_path is not None:
            self.policy.save_lora(save_path)
            logger.info(f"SFT checkpoint saved → {save_path}")

        return stats
