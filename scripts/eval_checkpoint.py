"""Evaluate a single LoRA checkpoint on all training + held-out tasks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlaskool.evaluation.evaluator import SuccessMatrix, TaskEvaluator
from vlaskool.envs.put_object_on_plate import TRAINING_TASKS, HELD_OUT_TASKS
from vlaskool.models.policy import load_smolvla_with_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to save eval JSON")
    parser.add_argument("--n-eval", type=int, default=104)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--robot-uids", type=str, default="so100")
    args = parser.parse_args()

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    policy = load_smolvla_with_lora(
        lora_rank=args.lora_rank,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )

    sm = SuccessMatrix.empty(
        task_ids=[t["id"] for t in TRAINING_TASKS],
        held_out_ids=[t["id"] for t in HELD_OUT_TASKS],
    )

    evaluator = TaskEvaluator(
        policy=policy,
        training_tasks=TRAINING_TASKS,
        held_out_tasks=HELD_OUT_TASKS,
        n_eval_episodes=args.n_eval,
        sim_backend=args.sim_backend,
        robot_uids=args.robot_uids,
        device=args.device,
    )

    logger.info("Evaluating all training tasks...")
    evaluator.evaluate_all(after_task_idx=1, success_matrix=sm)

    logger.info("Evaluating held-out tasks...")
    evaluator.evaluate_held_out(sm, label="final")

    sm.save(args.output)
    sm.print()
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
