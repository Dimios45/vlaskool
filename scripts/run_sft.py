"""Experiment 2: Sequential SFT — behavior cloning on seen tasks.

Fine-tunes SmolVLA sequentially via behavior cloning on 35 demos/task.
Evaluates on seen + held-out tasks after each training task.

Usage:
    uv run python scripts/run_sft.py --robot-uids so100 --demo-dir data/demos/so100
    uv run python scripts/run_sft.py --robot-uids so100 --demo-dir data/demos/so100 --seed 0
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
    parser = argparse.ArgumentParser(description="Sequential SFT on ManiSkill tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-dir", type=str, default="data/demos/so100")
    parser.add_argument("--robot-uids", type=str, default="so100", choices=["so100", "panda"])
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--sft-epochs", type=int, default=100)
    parser.add_argument("--sft-batch-size", type=int, default=32)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlaskool")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"sft_seed{args.seed}",
            config=vars(args),
        )

    trainer = ContinualRLTrainer(
        training_tasks=TRAINING_TASKS,
        held_out_tasks=HELD_OUT_TASKS,
        mode="sft",
        pretrained_name=args.pretrained,
        lora_rank=args.lora_rank,
        action_dim=args.action_dim,
        sft_learning_rate=args.sft_lr,
        sft_batch_size=args.sft_batch_size,
        sft_num_epochs=args.sft_epochs,
        sft_demo_dir=args.demo_dir,
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
    print(f"SFT Results (seed={args.seed}):")
    print(f"  {metrics}")
    print(f"{'='*50}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
