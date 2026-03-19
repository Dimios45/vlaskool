"""Task evaluator: runs episodes and builds the success matrix.

The success matrix S[i, j] is the core data structure for CRL evaluation:
  - Row 0:    zero-shot success (before any finetuning)
  - Row i+1:  success after training through task i

After all tasks trained:
  - Evaluate held-out tasks → ZS metric
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vlaskool.envs.wrappers import VecEnvWrapper, make_maniskill_env
from vlaskool.evaluation.metrics import MetricsResult, compute_metrics, print_success_matrix
from vlaskool.models.policy import SmolVLALoRAPolicy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SuccessMatrix:
    """Stores and persists the full success matrix for one experiment."""

    task_ids: list[str]          # T training task IDs in order
    held_out_ids: list[str]      # H held-out task IDs

    # S[i, j] = success rate on training task j after training through task i
    # Shape: (T+1, T) — row 0 = zero-shot
    matrix: np.ndarray

    # held_out[j] = success rate on held-out task j after all training
    # Shape: (H,)
    held_out: np.ndarray

    # held_out_zero_shot[j] = success rate on held-out task j before any training
    # Shape: (H,)
    held_out_zero_shot: np.ndarray

    @classmethod
    def empty(cls, task_ids: list[str], held_out_ids: list[str]) -> "SuccessMatrix":
        T = len(task_ids)
        H = len(held_out_ids)
        return cls(
            task_ids=task_ids,
            held_out_ids=held_out_ids,
            matrix=np.full((T + 1, T), np.nan),
            held_out=np.full((H,), np.nan),
            held_out_zero_shot=np.full((H,), np.nan),
        )

    def set(self, after_task: int, eval_task: int, success_rate: float) -> None:
        """Set S[after_task, eval_task]. after_task=0 for zero-shot."""
        self.matrix[after_task, eval_task] = success_rate

    def get_metrics(self) -> MetricsResult:
        return compute_metrics(self.matrix, self.held_out)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task_ids": self.task_ids,
            "held_out_ids": self.held_out_ids,
            "matrix": self.matrix.tolist(),
            "held_out": self.held_out.tolist(),
            "held_out_zero_shot": self.held_out_zero_shot.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Success matrix saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "SuccessMatrix":
        with open(path) as f:
            data = json.load(f)
        H = len(data["held_out_ids"])
        return cls(
            task_ids=data["task_ids"],
            held_out_ids=data["held_out_ids"],
            matrix=np.array(data["matrix"]),
            held_out=np.array(data["held_out"]),
            held_out_zero_shot=np.array(data.get("held_out_zero_shot", [float("nan")] * H)),
        )

    def print(self) -> None:
        print_success_matrix(self.matrix, self.task_ids)
        if not np.all(np.isnan(self.held_out_zero_shot)):
            print("\nHeld-out tasks (zero-shot):")
            for tid, sr in zip(self.held_out_ids, self.held_out_zero_shot):
                print(f"  {tid}: {sr*100:.1f}%")
        if not np.all(np.isnan(self.held_out)):
            print("\nHeld-out tasks (after training):")
            for tid, sr in zip(self.held_out_ids, self.held_out):
                print(f"  {tid}: {sr*100:.1f}%")
        metrics = self.get_metrics()
        print(f"\nMetrics: {metrics}")


class TaskEvaluator:
    """Evaluates a policy on a list of tasks and populates a SuccessMatrix.

    Usage:
        evaluator = TaskEvaluator(policy, task_cfgs, held_out_cfgs, cfg)

        # Zero-shot row
        evaluator.evaluate_all(after_task_idx=0, success_matrix=sm)

        # After training task 0
        evaluator.evaluate_all(after_task_idx=1, success_matrix=sm)

        # After all training
        evaluator.evaluate_held_out(success_matrix=sm)
    """

    def __init__(
        self,
        policy: SmolVLALoRAPolicy,
        training_tasks: list[dict],
        held_out_tasks: list[dict],
        n_eval_episodes: int = 100,
        episode_length: int = 80,
        num_envs: int = 8,
        sim_backend: str = "gpu",
        robot_uids: str = "panda",
        device: str = "cuda",
        wandb_run=None,
        video_dir: Optional[Path] = None,
    ) -> None:
        self.policy = policy
        self.training_tasks = training_tasks
        self.held_out_tasks = held_out_tasks
        self.n_eval_episodes = n_eval_episodes
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.sim_backend = sim_backend
        self.robot_uids = robot_uids
        self.device = torch.device(device)
        self.wandb_run = wandb_run
        self.video_dir = Path(video_dir) if video_dir else None

    def evaluate_task(self, task_cfg: dict) -> float:
        """Evaluate policy on one task, return success rate ∈ [0, 1]."""
        object_name = task_cfg["object_name"]
        instruction = task_cfg["instruction"]
        task_id = task_cfg["id"]

        env = make_maniskill_env(
            object_name=object_name,
            instruction=instruction,
            num_envs=self.num_envs,
            sim_backend=self.sim_backend,
            robot_uids=self.robot_uids,
        )
        env_wrapper = VecEnvWrapper(env)

        n_success = 0
        n_total = 0
        self.policy.eval()

        try:
            while n_total < self.n_eval_episodes:
                obs, _ = env_wrapper.reset()
                for step in range(self.episode_length):
                    with torch.no_grad():
                        action = self.policy.select_action_deterministic(obs)
                    obs, reward, done, info = env_wrapper.step(action)
                success = env_wrapper.get_success()
                n_success += success.sum().item()
                n_total += self.num_envs
        finally:
            env_wrapper.close()

        sr = n_success / n_total
        logger.info(f"  Eval [{task_id}]: {n_success}/{n_total} = {sr:.3f}")
        return sr

    def evaluate_all(
        self,
        after_task_idx: int,
        success_matrix: SuccessMatrix,
    ) -> None:
        """Evaluate all training tasks and record in success matrix.

        Args:
            after_task_idx: Row index in success matrix (0 = zero-shot).
            success_matrix:  SuccessMatrix to fill.
        """
        row_label = "zero-shot" if after_task_idx == 0 else f"after task {after_task_idx}"
        logger.info(f"\n── Evaluation: {row_label} ──────────────────")

        for j, task_cfg in enumerate(self.training_tasks):
            sr = self.evaluate_task(task_cfg)
            success_matrix.set(after_task_idx, j, sr)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        f"eval/{task_cfg['id']}/success": sr,
                        f"eval/after_task": after_task_idx,
                    }
                )

    def evaluate_held_out(self, success_matrix: SuccessMatrix, label: str = "final") -> None:
        """Evaluate all held-out tasks.

        Args:
            label: "zero-shot" stores into held_out_zero_shot, otherwise into held_out.
        """
        logger.info(f"\n── Evaluation: held-out tasks ({label}) ──────────────────")
        for h, task_cfg in enumerate(self.held_out_tasks):
            sr = self.evaluate_task(task_cfg)
            if label == "zero-shot":
                success_matrix.held_out_zero_shot[h] = sr
            else:
                success_matrix.held_out[h] = sr

            if self.wandb_run is not None:
                self.wandb_run.log({f"eval/held_out/{label}/{task_cfg['id']}": sr})
