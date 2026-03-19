"""Experiment 1: Zero-shot evaluation — no finetuning.

Evaluates pretrained SmolVLA on all seen + held-out tasks.
This is the baseline showing what the model can do out-of-the-box.

Usage:
    uv run python scripts/run_zero_shot.py
    uv run python scripts/run_zero_shot.py --seed 0 --n-eval 100
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
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of SmolVLA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=100, help="Episodes per task")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlaskool")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"zero_shot_seed{args.seed}",
            config=vars(args),
        )

    trainer = ContinualRLTrainer(
        training_tasks=TRAINING_TASKS,
        held_out_tasks=HELD_OUT_TASKS,
        mode="zero_shot",
        pretrained_name=args.pretrained,
        n_eval_episodes=args.n_eval,
        device=args.device,
        sim_backend=args.sim_backend,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        wandb_run=wandb_run,
    )

    metrics = trainer.run()

    print(f"\n{'='*50}")
    print(f"Zero-shot Results (seed={args.seed}):")
    print(f"  {metrics}")
    print(f"{'='*50}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
