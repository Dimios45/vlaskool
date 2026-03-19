"""Experiment 3: Continual RL VLA — the paper's main recipe.

SmolVLA + LoRA (rank 32) + GRPO, sequential over seen tasks.
No replay, no regularization, just on-policy RL.

Paper hyperparams (Appendix F, Table 6):
  lr=2e-5, batch=8192, rollout_epochs=16, group_size=8,
  clip=(0.2, 0.28), episodes_per_task=10240, episode_length=80

Usage:
    uv run python scripts/run_continual_rl.py
    uv run python scripts/run_continual_rl.py --seed 0 --episodes 10240
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlaskool.continual.trainer import ContinualRLTrainer
from vlaskool.envs.put_object_on_plate import TRAINING_TASKS, HELD_OUT_TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Continual RL VLA: SmolVLA + LoRA + GRPO"
    )
    # Experiment
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--robot-uids", type=str, default="so100", choices=["so100", "panda"])
    parser.add_argument("--action-dim", type=int, default=6)
    # GRPO (paper values as defaults)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rollout-epochs", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10240)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=0.1, help="Exploration noise std")
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    # Eval
    parser.add_argument("--n-eval", type=int, default=100)
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlaskool")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"continual_rl_seed{args.seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["continual_rl", "smolvla", "grpo", "maniskill"],
        )

    trainer = ContinualRLTrainer(
        training_tasks=TRAINING_TASKS,
        held_out_tasks=HELD_OUT_TASKS,
        mode="continual_rl",
        pretrained_name=args.pretrained,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        global_batch_size=args.batch_size,
        rollout_epochs=args.rollout_epochs,
        group_size=args.group_size,
        total_episodes=args.episodes,
        episode_length=args.episode_length,
        exploration_sigma=args.sigma,
        action_dim=args.action_dim,
        n_eval_episodes=args.n_eval,
        device=args.device,
        sim_backend=args.sim_backend,
        robot_uids=args.robot_uids,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        wandb_run=wandb_run,
    )

    metrics = trainer.run()

    print(f"\n{'='*50}")
    print(f"Continual RL Results (seed={args.seed}):")
    print(f"  {metrics}")
    print(f"{'='*50}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
