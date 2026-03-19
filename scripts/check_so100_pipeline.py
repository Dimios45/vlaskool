"""End-to-end pipeline check for SO100 arm.

Verifies each stage of the training pipeline:
  1. Environment: creation, reset, step (GPU parallel)
  2. Observation format: correct keys/shapes for SmolVLA
  3. Policy: SmolVLA inference with SO100 obs
  4. Rollout: GRPO rollout collection
  5. Training step: GRPO loss + backward pass

Usage:
    uv run python scripts/check_so100_pipeline.py
    uv run python scripts/check_so100_pipeline.py --sim-backend cpu  # slower, no GPU sim
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlaskool.envs.wrappers import VecEnvWrapper, make_maniskill_env
from vlaskool.models.policy import load_smolvla_with_lora

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(name: str, fn):
    try:
        result = fn()
        print(f"  {PASS}  {name}" + (f"  [{result}]" if result else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    num_envs = 1 if args.sim_backend == "cpu" else args.num_envs

    print("\n" + "=" * 60)
    print("SO100 Pipeline Check")
    print("=" * 60)

    all_ok = True
    # Use a dict as a mutable shared state (closures can't rebind outer-scope names)
    state = {"env_wrapper": None, "obs": None, "policy": None, "batch": None}

    # ── 1. Environment ─────────────────────────────────────────────
    print("\n[1] Environment")

    def make_env():
        state["env_wrapper"] = VecEnvWrapper(
            make_maniskill_env(
                "carrot", "put carrot on plate",
                num_envs=num_envs,
                sim_backend=args.sim_backend,
                robot_uids="so100",
            )
        )
        return f"num_envs={state['env_wrapper'].num_envs}"

    all_ok &= check("create PutObjectOnPlate-v1 with SO100", make_env)

    def reset_env():
        obs, _ = state["env_wrapper"].reset()
        state["obs"] = obs
        return f"keys={list(obs.keys())}"

    all_ok &= check("env.reset()", reset_env)

    def step_env():
        obs, _ = state["env_wrapper"].reset()
        state["obs"] = obs
        action = torch.zeros(num_envs, 6)
        obs2, reward, done, info = state["env_wrapper"].step(action)
        success = state["env_wrapper"].get_success()
        return f"reward.shape={reward.shape}, success.shape={success.shape}"

    all_ok &= check("env.step() with 6D action (no padding)", step_env)

    # ── 2. Observation Format ──────────────────────────────────────
    print("\n[2] Observation Format")

    if state["obs"] is None and state["env_wrapper"] is not None:
        state["obs"], _ = state["env_wrapper"].reset()

    def check_image_key():
        obs = state["obs"]
        assert "observation.images.camera1" in obs, f"Missing key, got: {list(obs.keys())}"
        img = obs["observation.images.camera1"]
        assert img.shape == (num_envs, 3, 224, 224), f"Bad shape: {img.shape}"
        assert img.dtype == torch.float32
        assert 0.0 <= img.min() and img.max() <= 1.0, "Not in [0,1]"
        return f"shape={img.shape} dtype={img.dtype}"

    all_ok &= check("observation.images.camera1: (B,3,224,224) float32 in [0,1]", check_image_key)

    def check_state_key():
        state_val = state["obs"]["observation.state"]
        assert state_val.shape == (num_envs, 6), f"Bad shape: {state_val.shape}"
        return f"shape={state_val.shape}"

    all_ok &= check("observation.state: (B,6) — matches SmolVLA state_dim", check_state_key)

    def check_task_key():
        tasks = state["obs"]["task"]
        assert len(tasks) == num_envs
        return f"task[0]={tasks[0]!r}"

    all_ok &= check("task: list[str] of length B", check_task_key)

    # ── 3. Policy ──────────────────────────────────────────────────
    print("\n[3] SmolVLA Policy")

    def load_policy():
        t = time.time()
        state["policy"] = load_smolvla_with_lora(lora_rank=32, device="cuda")
        p = state["policy"]
        trainable = sum(x.numel() for x in p.trainable_parameters())
        total = sum(x.numel() for x in p.base_policy.parameters())
        return f"{trainable/1e6:.1f}M / {total/1e6:.1f}M trainable, {time.time()-t:.1f}s"

    all_ok &= check("load SmolVLA + LoRA rank=32", load_policy)

    def infer_action():
        obs = state["obs"]
        obs_dev = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        action = state["policy"].select_action_deterministic(obs_dev)
        assert action.shape == (num_envs, 6), f"Bad action shape: {action.shape}"
        return f"action.shape={action.shape} dtype={action.dtype}"

    all_ok &= check("policy.select_action_deterministic() → (B,6)", infer_action)

    def check_log_prob():
        obs = state["obs"]
        obs_dev = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        action = torch.randn(num_envs, 6, device="cuda")
        fm_noise = torch.randn(num_envs, 1, 6, device="cuda")
        fm_time = torch.rand(num_envs, device="cuda")
        log_p = state["policy"].compute_log_prob(obs_dev, action, fm_noise=fm_noise, fm_time=fm_time)
        assert log_p.shape == (num_envs,), f"Bad shape: {log_p.shape}"
        assert torch.isfinite(log_p).all(), f"Non-finite log_probs: {log_p}"
        return f"log_p.shape={log_p.shape}, mean={log_p.mean():.3f}"

    all_ok &= check("policy.compute_log_prob() — FM surrogate", check_log_prob)

    # ── 4. Rollout Collection ──────────────────────────────────────
    print("\n[4] GRPO Rollout Collection")

    from vlaskool.training.rollout import collect_rollouts

    def do_collect():
        state["batch"] = collect_rollouts(
            policy=state["policy"],
            env=state["env_wrapper"],
            n_groups=1,
            episode_length=3,
            exploration_sigma=0.1,
            device=torch.device("cuda"),
            instruction="put carrot on plate",
        )
        b = state["batch"]
        return f"B={len(b)}, advantages.shape={b.advantages.shape}"

    all_ok &= check("collect_rollouts() — 1 group × 3 steps", do_collect)

    def check_batch():
        b = state["batch"]
        assert b.obs_images.shape[1:] == (3, 224, 224)
        assert b.obs_states.shape[1] == 6
        assert b.actions.shape[1] == 6
        assert b.fm_noise is not None
        assert b.fm_time is not None
        assert torch.isfinite(b.advantages).all()
        return f"images={b.obs_images.shape}, states={b.obs_states.shape}, actions={b.actions.shape}"

    all_ok &= check("batch shapes: images (B,3,H,W), states (B,6), actions (B,6)", check_batch)

    # ── 5. GRPO Training Step ─────────────────────────────────────
    print("\n[5] GRPO Training Step")

    from vlaskool.training.grpo import GRPOTrainer

    def training_step():
        trainer = GRPOTrainer(
            policy=state["policy"],
            env=state["env_wrapper"],
            learning_rate=2e-5,
            global_batch_size=num_envs * 3,
            rollout_epochs=1,
            group_size=num_envs,
            total_episodes=num_envs,
            episode_length=3,
            vla_minibatch_size=min(4, num_envs),
            device="cuda",
        )
        stats = trainer.train(task_id="test_so100", instruction="put carrot on plate")
        loss = stats["loss"][0] if stats["loss"] else float("nan")
        assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss: {loss}"
        return f"loss={loss:.4f}"

    all_ok &= check("GRPOTrainer.train() — 1 update step", training_step)

    # ── 6. Task Switching ─────────────────────────────────────────
    print("\n[6] Task Switching (GPU env reuse)")

    def switch_task():
        state["env_wrapper"].switch_task("bread_loaf", "put bread on plate")
        obs2, _ = state["env_wrapper"].reset()
        assert obs2["task"][0] == "put bread on plate"
        return "carrot → bread_loaf, reconfigure OK"

    all_ok &= check("VecEnvWrapper.switch_task() — rebuild scene in-place", switch_task)

    # ── Cleanup ────────────────────────────────────────────────────
    if state["env_wrapper"] is not None:
        state["env_wrapper"].close()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — SO100 pipeline is ready")
    else:
        print("SOME CHECKS FAILED — see above")
    print("=" * 60 + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
