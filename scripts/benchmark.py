"""Benchmark: run all 3 comparisons and produce final results table.

Comparison modes (paper Table 1):
  1. zero_shot    — pretrained SmolVLA, no finetuning
  2. sft          — sequential BC finetuning on seen tasks
  3. continual_rl — sequential GRPO+LoRA (paper's recipe)

Results: AVG, NBT, FWT, ZS metrics (mean ± SE over 3 seeds).

Usage:
    uv run python scripts/benchmark.py                  # all modes, 3 seeds
    uv run python scripts/benchmark.py --mode all       # same
    uv run python scripts/benchmark.py --mode continual_rl --seeds 0
    uv run python scripts/benchmark.py --dry-run        # quick test with 1 ep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlaskool.continual.trainer import ContinualRLTrainer, run_all_comparisons
from vlaskool.envs.put_object_on_plate import TRAINING_TASKS, HELD_OUT_TASKS
from vlaskool.evaluation.metrics import MetricsResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_results_table(results: dict[str, MetricsResult]) -> None:
    """Pretty-print benchmark results table."""
    header = f"{'Method':<20} {'AVG':>12} {'NBT':>12} {'FWT':>12} {'ZS':>12}"
    sep = "─" * len(header)

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS: SmolVLA Continual RL on ManiSkill")
    print("Paper: arXiv 2603.11653")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for method, m in results.items():
        row = (
            f"{method:<20} "
            f"{m.avg*100:>6.1f}±{m.avg_se*100:<4.1f} "
            f"{m.nbt*100:>6.1f}±{m.nbt_se*100:<4.1f} "
            f"{m.fwt*100:>6.1f}±{m.fwt_se*100:<4.1f} "
            f"{m.zs*100:>6.1f}±{m.zs_se*100:<4.1f}"
        )
        print(row)
    print(sep)
    print("AVG: final avg success (↑), NBT: forgetting (↓), FWT: fwd transfer (↑), ZS: held-out (↑)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all 3 comparison modes")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "zero_shot", "sft", "continual_rl"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10240)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--global-batch-size", type=int, default=8192)
    parser.add_argument("--rollout-epochs", type=int, default=16)
    parser.add_argument("--robot-uids", type=str, default="panda",
                        choices=["panda", "so100"],
                        help="Robot to use: panda (default) or so100")
    parser.add_argument("--demo-dir", type=str, default="data/demos")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vlaskool")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick smoke test: 1 episode, 1 eval episode, 1 seed",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [0]
        args.episodes = 8          # minimal for testing
        args.episode_length = 5
        args.n_eval = 2
        # Reduce global_batch_size so n_groups_per_update=1 for quick smoke test
        args.global_batch_size = 40  # 8 envs × 5 steps = 40 transitions
        args.rollout_epochs = 2
        logger.info("DRY RUN mode — using minimal settings")

    modes = (
        ["zero_shot", "sft", "continual_rl"] if args.mode == "all" else [args.mode]
    )

    common_cfg = dict(
        pretrained_name=args.pretrained,
        lora_rank=args.lora_rank,
        total_episodes=args.episodes,
        episode_length=args.episode_length,
        n_eval_episodes=args.n_eval,
        global_batch_size=args.global_batch_size,
        rollout_epochs=args.rollout_epochs,
        sft_demo_dir=args.demo_dir,
        device=args.device,
        sim_backend=args.sim_backend,
        robot_uids=args.robot_uids,
        output_dir=Path(args.output_dir),
    )

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name="benchmark",
            config={**vars(args), "modes": modes},
            tags=["benchmark"],
        )
        common_cfg["wandb_run"] = wandb_run

    all_seed_results: dict[str, list[MetricsResult]] = {m: [] for m in modes}

    for mode in modes:
        for seed in args.seeds:
            logger.info(f"\nRunning mode={mode!r}, seed={seed}")
            trainer = ContinualRLTrainer(
                training_tasks=TRAINING_TASKS,
                held_out_tasks=HELD_OUT_TASKS,
                mode=mode,
                seed=seed,
                **common_cfg,
            )
            result = trainer.run()
            all_seed_results[mode].append(result)

    # Aggregate over seeds
    from vlaskool.evaluation.metrics import aggregate_metrics_over_seeds

    final_results = {
        mode: aggregate_metrics_over_seeds(results)
        for mode, results in all_seed_results.items()
    }

    # Print table
    print_results_table(final_results)

    # Save JSON
    output_path = Path(args.output_dir) / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {mode: m.to_dict() for mode, m in final_results.items()},
            f,
            indent=2,
        )
    logger.info(f"Results saved → {output_path}")

    if wandb_run:
        for mode, m in final_results.items():
            wandb_run.summary.update({f"{mode}/{k}": v for k, v in m.to_dict().items()})
        wandb_run.finish()


if __name__ == "__main__":
    main()
