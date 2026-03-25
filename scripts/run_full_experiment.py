"""Full paper experiment: SFT warm-up → Sequential GRPO (continual RL).

Paper recipe (Section 3 + Appendix D/F):
  Phase 1: SFT on 35 demos/task × 4 seen tasks → ~51.6% initial success
  Phase 2: Sequential GRPO on 4 seen tasks, 10240 episodes each
  Evaluate: After each GRPO task + held-out tasks at the end

Usage:
    uv run python scripts/run_full_experiment.py
    uv run python scripts/run_full_experiment.py --seed 0 --episodes 10240
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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
        description="Full experiment: SFT warm-up + Continual RL (GRPO)"
    )
    # Experiment
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--robot-uids", type=str, default="so100", choices=["so100", "panda"])
    parser.add_argument("--action-dim", type=int, default=6)
    # SFT phase
    parser.add_argument("--sft-demo-dir", type=str, default="data/demos/so100")
    parser.add_argument("--sft-epochs", type=int, default=100)
    parser.add_argument("--sft-batch-size", type=int, default=32)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--sft-success-only", action="store_true", default=True,
                        help="Filter SFT demos to successes only (default: True)")
    parser.add_argument("--sft-all-demos", action="store_true",
                        help="Include failures in SFT (overrides --sft-success-only)")
    parser.add_argument("--sft-multitask", action="store_true",
                        help="Joint multi-task SFT instead of sequential")
    # GRPO phase (paper values)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rollout-epochs", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10240)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--grpo-mode", type=str, default="reinforce",
                        choices=["ppo", "reinforce"],
                        help="GRPO loss mode: 'reinforce' (advantage-weighted FM) or 'ppo' (importance ratio)")
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    # Eval
    parser.add_argument("--n-eval", type=int, default=100)
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlaskool")
    parser.add_argument("--wandb-name", type=str, default=None)
    # Skip phases
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT, use --lora-checkpoint instead")
    parser.add_argument("--lora-checkpoint", type=str, default=None, help="Pre-trained LoRA checkpoint to start GRPO from")
    args = parser.parse_args()

    # Resolve SFT demo filtering
    sft_success_only = args.sft_success_only and not args.sft_all_demos

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"full_experiment_seed{args.seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["full_experiment", "smolvla", "sft+grpo", "maniskill"],
        )

    output_dir = Path(args.output_dir)
    t0 = time.time()

    # ── Phase 1: SFT warm-up ────────────────────────────────────────────────
    sft_checkpoint = args.lora_checkpoint

    if not args.skip_sft and sft_checkpoint is None:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: SFT warm-up on all 4 seen tasks")
        logger.info("=" * 60)

        sft_trainer = ContinualRLTrainer(
            training_tasks=TRAINING_TASKS,
            held_out_tasks=HELD_OUT_TASKS,
            mode="sft",
            pretrained_name=args.pretrained,
            lora_rank=args.lora_rank,
            action_dim=args.action_dim,
            sft_learning_rate=args.sft_lr,
            sft_batch_size=args.sft_batch_size,
            sft_num_epochs=args.sft_epochs,
            sft_demo_dir=args.sft_demo_dir,
            sft_success_only=sft_success_only,
            sft_multitask=args.sft_multitask,
            n_eval_episodes=args.n_eval,
            episode_length=args.episode_length,
            device=args.device,
            sim_backend=args.sim_backend,
            robot_uids=args.robot_uids,
            seed=args.seed,
            output_dir=output_dir,
            wandb_run=wandb_run,
        )

        sft_metrics = sft_trainer.run()
        logger.info(f"SFT metrics: {sft_metrics}")

        # Find the last SFT checkpoint (after training all 4 tasks)
        if args.sft_multitask:
            sft_checkpoint = str(
                output_dir / "sft" / f"seed_{args.seed}" / "checkpoints" / "sft_multitask"
            )
        else:
            last_task = TRAINING_TASKS[-1]
            last_idx = len(TRAINING_TASKS) - 1
            sft_checkpoint = str(
                output_dir / "sft" / f"seed_{args.seed}" / "checkpoints"
                / f"task_{last_idx:02d}_{last_task['id']}"
            )
        logger.info(f"SFT checkpoint: {sft_checkpoint}")

        sft_elapsed = (time.time() - t0) / 60
        logger.info(f"SFT phase completed in {sft_elapsed:.1f} min")

    # ── Phase 2: Sequential GRPO ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Continual RL (Sequential GRPO)")
    logger.info(f"  Starting from checkpoint: {sft_checkpoint}")
    logger.info("=" * 60)

    grpo_trainer = ContinualRLTrainer(
        training_tasks=TRAINING_TASKS,
        held_out_tasks=HELD_OUT_TASKS,
        mode="continual_rl",
        pretrained_name=args.pretrained,
        lora_rank=args.lora_rank,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        global_batch_size=args.batch_size,
        rollout_epochs=args.rollout_epochs,
        group_size=args.group_size,
        total_episodes=args.episodes,
        episode_length=args.episode_length,
        exploration_sigma=args.sigma,
        grpo_mode=args.grpo_mode,
        n_eval_episodes=args.n_eval,
        device=args.device,
        sim_backend=args.sim_backend,
        robot_uids=args.robot_uids,
        seed=args.seed,
        output_dir=output_dir,
        wandb_run=wandb_run,
        lora_checkpoint=sft_checkpoint,
    )

    grpo_metrics = grpo_trainer.run()

    total_elapsed = (time.time() - t0) / 3600
    print(f"\n{'=' * 60}")
    print(f"Full Experiment Results (seed={args.seed}, {total_elapsed:.1f}h):")
    print(f"  GRPO metrics: {grpo_metrics}")
    print(f"{'=' * 60}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
