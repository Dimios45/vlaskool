"""Continual RL Trainer — the paper's "simple recipe".

Recipe (Section 3 + Appendix F):
  Large pretrained VLA + LoRA (rank 32) + GRPO (on-policy RL)
  Sequential fine-tuning, no replay, no regularization.

This trainer orchestrates:
  1. GRPO training on each task in order
  2. Evaluation after each task (building success matrix)
  3. Final held-out evaluation
  4. Metric computation: AVG, NBT, FWT, ZS

Three experimental modes (paper Table 1):
  Mode 1: zero_shot   — no training; eval pretrained SmolVLA on all tasks
  Mode 2: sft         — sequential BC finetuning on seen tasks
  Mode 3: continual_rl — sequential GRPO+LoRA on seen tasks (main contribution)
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vlaskool.evaluation.evaluator import SuccessMatrix, TaskEvaluator
from vlaskool.evaluation.metrics import MetricsResult, aggregate_metrics_over_seeds
from vlaskool.models.policy import SmolVLALoRAPolicy, load_smolvla_with_lora
from vlaskool.training.grpo import GRPOTrainer
from vlaskool.training.sft import SFTTrainer
from vlaskool.envs.wrappers import VecEnvWrapper, make_maniskill_env

logger = logging.getLogger(__name__)


class ContinualRLTrainer:
    """Orchestrates the full CRL experiment.

    Args:
        training_tasks:  List of task dicts in the order they are trained.
        held_out_tasks:  List of held-out task dicts.
        model_cfg:       Model config (pretrained_name, lora settings, etc.)
        train_cfg:       Training hyperparams.
        eval_cfg:        Evaluation settings.
        mode:            'zero_shot', 'sft', or 'continual_rl'.
        seed:            Random seed.
        output_dir:      Root directory for checkpoints + results.
        wandb_run:       Optional W&B run handle.
    """

    def __init__(
        self,
        training_tasks: list[dict],
        held_out_tasks: list[dict],
        mode: str = "continual_rl",
        # Model
        pretrained_name: str = "lerobot/smolvla_base",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        action_dim: int = 7,
        # Training (GRPO)
        learning_rate: float = 2e-5,
        global_batch_size: int = 8192,
        rollout_epochs: int = 16,
        group_size: int = 8,
        total_episodes: int = 10240,
        episode_length: int = 80,
        exploration_sigma: float = 0.1,
        sigma_decay: float = 0.995,
        # Training (SFT)
        sft_learning_rate: float = 1e-4,
        sft_batch_size: int = 32,
        sft_num_epochs: int = 100,
        sft_demo_dir: str = "data/demos",
        # Evaluation
        n_eval_episodes: int = 100,
        eval_after_each_task: bool = True,
        # System
        device: str = "cuda",
        use_bf16: bool = True,
        sim_backend: str = "gpu",
        robot_uids: str = "panda",
        seed: int = 0,
        output_dir: Path = Path("results"),
        wandb_run=None,
        lora_checkpoint: Optional[str | Path] = None,
    ) -> None:
        self.training_tasks = training_tasks
        self.held_out_tasks = held_out_tasks
        self.mode = mode
        self.pretrained_name = pretrained_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.action_dim = action_dim
        self.lr = learning_rate
        self.global_batch_size = global_batch_size
        self.rollout_epochs = rollout_epochs
        self.group_size = group_size
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.exploration_sigma = exploration_sigma
        self.sigma_decay = sigma_decay
        self.sft_lr = sft_learning_rate
        self.sft_batch_size = sft_batch_size
        self.sft_epochs = sft_num_epochs
        self.sft_demo_dir = Path(sft_demo_dir)
        self.n_eval_episodes = n_eval_episodes
        self.eval_after_each_task = eval_after_each_task
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.sim_backend = sim_backend
        self.robot_uids = robot_uids
        self.seed = seed
        self.output_dir = Path(output_dir) / mode / f"seed_{seed}"
        self.wandb_run = wandb_run
        self.lora_checkpoint = Path(lora_checkpoint) if lora_checkpoint else None

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

    def run(self) -> MetricsResult:
        """Execute the full experiment and return final metrics."""
        start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: mode={self.mode!r}, seed={self.seed}")
        logger.info(f"  Training tasks: {[t['id'] for t in self.training_tasks]}")
        logger.info(f"  Held-out tasks: {[t['id'] for t in self.held_out_tasks]}")
        logger.info(f"{'='*60}")

        # ── Load model ────────────────────────────────────────────────────────
        policy = load_smolvla_with_lora(
            pretrained_name=self.pretrained_name,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            action_dim=self.action_dim,
            exploration_sigma=self.exploration_sigma,
            device=str(self.device),
            dtype=self.dtype,
            checkpoint_path=self.lora_checkpoint,
        )

        # ── Build evaluator ───────────────────────────────────────────────────
        evaluator = TaskEvaluator(
            policy=policy,
            training_tasks=self.training_tasks,
            held_out_tasks=self.held_out_tasks,
            n_eval_episodes=self.n_eval_episodes,
            episode_length=self.episode_length,
            num_envs=self.group_size,
            sim_backend=self.sim_backend,
            robot_uids=self.robot_uids,
            device=str(self.device),
            wandb_run=self.wandb_run,
        )

        # ── Initialize success matrix ─────────────────────────────────────────
        sm = SuccessMatrix.empty(
            task_ids=[t["id"] for t in self.training_tasks],
            held_out_ids=[t["id"] for t in self.held_out_tasks],
        )

        # ── Row 0: zero-shot evaluation (training + held-out) ────────────────
        logger.info("\nStep 0: Zero-shot evaluation (before any training)")
        evaluator.evaluate_all(after_task_idx=0, success_matrix=sm)
        evaluator.evaluate_held_out(sm, label="zero-shot")
        sm.save(self.output_dir / "success_matrix_partial.json")

        if self.mode == "zero_shot":
            sm.save(self.output_dir / "success_matrix_final.json")
            sm.print()
            metrics = sm.get_metrics()
            logger.info(f"\nFinal metrics [{self.mode}]: {metrics}")
            return metrics

        # ── Sequential training ───────────────────────────────────────────────
        # For continual_rl: create ONE GPU env and reuse it across tasks via switch_task().
        # This avoids the SAPIEN GPU render system lifecycle bug (fails on 2nd+ create).
        grpo_env_wrapper: Optional[VecEnvWrapper] = None
        if self.mode == "continual_rl":
            first_task = self.training_tasks[0]
            grpo_env_wrapper = VecEnvWrapper(
                make_maniskill_env(
                    object_name=first_task["object_name"],
                    instruction=first_task["instruction"],
                    num_envs=self.group_size,
                    sim_backend=self.sim_backend,
                    robot_uids=self.robot_uids,
                )
            )

        try:
          for task_idx, task_cfg in enumerate(self.training_tasks):
            task_id = task_cfg["id"]
            instruction = task_cfg["instruction"]
            logger.info(f"\n{'─'*60}")
            logger.info(f"Training task {task_idx+1}/{len(self.training_tasks)}: {task_id!r}")

            ckpt_path = self.output_dir / "checkpoints" / f"task_{task_idx:02d}_{task_id}"

            if self.mode == "sft":
                self._train_sft(policy, task_cfg, ckpt_path)

            elif self.mode == "continual_rl":
                # Switch the existing GPU env to the new task object
                if task_idx > 0:
                    grpo_env_wrapper.switch_task(task_cfg["object_name"], instruction)
                self._train_grpo(policy, task_cfg, ckpt_path, env_wrapper=grpo_env_wrapper)

            else:
                raise ValueError(f"Unknown mode: {self.mode!r}")

            # ── Evaluate all tasks after training this one ────────────────────
            if self.eval_after_each_task:
                evaluator.evaluate_all(
                    after_task_idx=task_idx + 1,
                    success_matrix=sm,
                )
                sm.save(self.output_dir / "success_matrix_partial.json")
                sm.print()

        finally:
            if grpo_env_wrapper is not None:
                grpo_env_wrapper.close()

        # ── Final held-out evaluation ─────────────────────────────────────────
        logger.info("\nFinal: evaluating held-out tasks")
        evaluator.evaluate_held_out(sm)
        sm.save(self.output_dir / "success_matrix_final.json")
        sm.print()

        metrics = sm.get_metrics()
        elapsed = (time.time() - start) / 3600
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment done in {elapsed:.1f}h")
        logger.info(f"Final metrics [{self.mode}]: {metrics}")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(metrics.to_dict())

        return metrics

    def _train_grpo(
        self,
        policy: SmolVLALoRAPolicy,
        task_cfg: dict,
        ckpt_path: Path,
        env_wrapper: Optional[VecEnvWrapper] = None,
    ) -> None:
        """Run GRPO on one task (continual RL mode).

        If env_wrapper is provided (reused GPU env), we don't close it here.
        """
        owns_env = env_wrapper is None
        if env_wrapper is None:
            env_wrapper = VecEnvWrapper(
                make_maniskill_env(
                    object_name=task_cfg["object_name"],
                    instruction=task_cfg["instruction"],
                    num_envs=self.group_size,
                    sim_backend=self.sim_backend,
                    robot_uids=self.robot_uids,
                )
            )

        trainer = GRPOTrainer(
            policy=policy,
            env=env_wrapper,
            learning_rate=self.lr,
            global_batch_size=self.global_batch_size,
            rollout_epochs=self.rollout_epochs,
            group_size=self.group_size,
            total_episodes=self.total_episodes,
            episode_length=self.episode_length,
            exploration_sigma=self.exploration_sigma,
            sigma_decay=self.sigma_decay,
            # RTX 4090: SmolVLA fits ~4 samples per forward in bfloat16
            vla_minibatch_size=getattr(self, "vla_minibatch_size", 4),
            device=str(self.device),
            wandb_run=self.wandb_run,
            checkpoint_dir=self.output_dir / "checkpoints",
        )

        try:
            trainer.train(
                task_id=task_cfg["id"],
                instruction=task_cfg["instruction"],
                save_path=ckpt_path,
            )
        finally:
            if owns_env:
                env_wrapper.close()

    def _train_sft(
        self, policy: SmolVLALoRAPolicy, task_cfg: dict, ckpt_path: Path
    ) -> None:
        """Run behavior cloning on one task (SFT mode)."""
        trainer = SFTTrainer(
            policy=policy,
            demo_dir=self.sft_demo_dir,
            learning_rate=self.sft_lr,
            batch_size=self.sft_batch_size,
            num_epochs=self.sft_epochs,
            device=str(self.device),
            wandb_run=self.wandb_run,
        )
        trainer.train(
            task_id=task_cfg["id"],
            instruction=task_cfg["instruction"],
            save_path=ckpt_path,
        )


def run_all_comparisons(
    training_tasks: list[dict],
    held_out_tasks: list[dict],
    common_cfg: dict,
    seeds: list[int] = [0, 1, 2],
) -> dict[str, MetricsResult]:
    """Run all three comparison modes and return aggregated metrics.

    Modes:
      1. zero_shot    — pretrained SmolVLA, no finetuning
      2. sft          — sequential BC finetuning on seen tasks
      3. continual_rl — sequential GRPO+LoRA (paper's recipe)

    Returns dict mapping mode name → aggregated MetricsResult.
    """
    modes = ["zero_shot", "sft", "continual_rl"]
    all_results: dict[str, list[MetricsResult]] = {m: [] for m in modes}

    for mode in modes:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# MODE: {mode}")
        logger.info(f"{'#'*60}")

        for seed in seeds:
            trainer = ContinualRLTrainer(
                training_tasks=training_tasks,
                held_out_tasks=held_out_tasks,
                mode=mode,
                seed=seed,
                **common_cfg,
            )
            result = trainer.run()
            all_results[mode].append(result)

    # Aggregate over seeds
    aggregated = {
        mode: aggregate_metrics_over_seeds(results)
        for mode, results in all_results.items()
    }
    return aggregated
