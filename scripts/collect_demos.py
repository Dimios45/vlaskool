"""Collect demonstration episodes via a scripted oracle (no mplib).

Uses SAPIEN's pinocchio IK + scipy to plan waypoints for pick-and-place,
then executes them as pd_joint_pos actions.

Converts absolute joint-position actions → delta actions for SmolVLA.

Saves demos to:
  data/demos/<robot_uids>/<task_id>/episode_<N>.h5

Each HDF5 file contains:
  observations/images/overhead  (T, H, W, C)  uint8
  observations/state            (T, 6)         float32  — qpos[:6]
  actions                       (T, 6)         float32  — joint delta, clipped [-1,1]
  attrs: instruction, task_id, robot_uids, success=True

Usage:
    uv run python scripts/collect_demos.py --robot-uids so100 --all-tasks
    uv run python scripts/collect_demos.py --robot-uids so100 --all-training
    uv run python scripts/collect_demos.py --robot-uids panda --all-tasks --n-demos 50
    uv run python scripts/collect_demos.py --task-ids put_carrot_on_plate --n-demos 5 --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vlaskool.envs.put_object_on_plate  # noqa: F401 — triggers @register_env
from vlaskool.envs.put_object_on_plate import (
    ALL_TASKS, TRAINING_TASKS, HELD_OUT_TASKS, get_task_by_id,
    _ROBOT_SCENE_CONFIGS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DEMO_DIR = Path("data/demos")
ACTION_DIM = 6


# ── IK + scripted oracle ───────────────────────────────────────────────────────

# SO100 joint limits
_SO100_JOINT_LO = np.array([-2.0, -1.5708, -1.5708, -1.8, -3.14159, -1.1])
_SO100_JOINT_HI = np.array([ 2.0,  1.5708,  1.5708,  1.8,  3.14159,  1.1])
_SO100_GRIPPER_OPEN = 0.0
_SO100_GRIPPER_CLOSE = -0.8


def _get_home_orientation(model) -> list:
    """Return the quaternion of Fixed_Jaw_tip at the SO100 home configuration."""
    q_home = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    model.compute_forward_kinematics(q_home)
    return list(model.get_link_pose(7).q)  # [w, x, y, z]


def _solve_ik_so100(
    model,
    target_local: np.ndarray,
    target_quat: list | None = None,
    initial_qpos: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Find SO100 qpos (6D) that puts Fixed_Jaw_tip at target_local.

    Uses SAPIEN's CLIK pinocchio IK.  The orientation target is taken from
    the arm's home configuration so CLIK only has to solve for position.
    Gripper joint (index 5) is held at 0.

    Returns (qpos_6d, position_error_m).
    """
    q_home = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    if initial_qpos is None:
        initial_qpos = q_home.copy()
    else:
        initial_qpos = initial_qpos.copy()
    initial_qpos[5] = 0.0

    if target_quat is None:
        target_quat = _get_home_orientation(model)

    active = np.array([1, 1, 1, 1, 1, 0], dtype=np.int32)

    target_pose = sapien.Pose(p=target_local.tolist(), q=target_quat)

    result_q, success, error = model.compute_inverse_kinematics(
        7,
        target_pose,
        initial_qpos=initial_qpos,
        active_qmask=active,
        max_iterations=3000,
        dt=0.1,
        damp=1e-6,
    )
    result_q[5] = 0.0
    err_pos = float(np.linalg.norm(error[:3]))
    return result_q, err_pos


def _interpolate_joints(q_start: np.ndarray, q_end: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Linear interpolation in joint space, n_steps points (not including start)."""
    return [q_start + (q_end - q_start) * (i + 1) / n_steps for i in range(n_steps)]


def _collect_so100_trajectory(
    env_raw,
    model,
    debug: bool = False,
) -> tuple[bool, list[dict]]:
    """Run one SO100 pick-and-place episode.

    Uses pinocchio IK for key waypoints, then interpolates in joint space.
    Approach direction is from the side (along robot local -Y axis) since the
    SO100 arm extends horizontally — this keeps waypoints well within reach.

    Returns (success, trajectory_steps).
    Each step: {'img', 'state', 'qpos', 'action'}.
    """
    robot = env_raw.agent.robot
    R = robot.pose.sp.to_transformation_matrix()[:3, :3]
    t = robot.pose.sp.p

    def w2l(p_world: np.ndarray) -> np.ndarray:
        return R.T @ (np.asarray(p_world) - t)

    obj_world = env_raw.object_actor.pose.sp.p.copy()
    plate_world = env_raw.plate.pose.sp.p.copy()

    obj_local = w2l(obj_world)
    plate_local = w2l(plate_world)

    grasp_local = obj_local.copy()    # at the object center

    place_local = plate_local.copy()
    place_local[2] += 0.02           # slightly above plate surface

    # Home
    q_home = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])

    # Get home orientation for consistent IK targets
    home_quat = _get_home_orientation(model)

    # Solve IK for each waypoint using CLIK warm-started from previous solution
    waypoints_local = [grasp_local, place_local]
    waypoint_names  = ["grasp",     "place"]
    q_waypoints = [q_home.copy()]

    for name, target in zip(waypoint_names, waypoints_local):
        q_prev = q_waypoints[-1]
        q_sol, err = _solve_ik_so100(model, target, target_quat=home_quat, initial_qpos=q_prev)
        if err > 0.025:
            if debug:
                logger.warning(f"    IK failed for {name}: err={err:.4f}m  target_local={target}")
            return False, []
        q_waypoints.append(q_sol)

    if debug:
        logger.info("    IK waypoints computed")

    # Execute trajectory
    trajectory: list[dict] = []

    def _record_step(action_7d: np.ndarray) -> None:
        raw_obs = env_raw.get_obs()
        img = _extract_image_raw(raw_obs)
        state = _extract_state_raw(raw_obs)
        qpos = env_raw.agent.robot.get_qpos()[0].cpu().numpy().copy()
        trajectory.append({"img": img, "state": state, "qpos": qpos})
        env_raw.step(action_7d)

    STEPS = 12  # steps per phase

    # wp indices: 0=home, 1=grasp, 2=place
    phases = [
        (0, 1, STEPS, True),   # home → grasp (open gripper)
        (1, 1, 6,    False),   # hold at grasp, close gripper
        (1, 2, STEPS, False),  # grasp → place (closed)
        (2, 2, 6,    True),    # hold at place, open gripper
    ]

    for (from_idx, to_idx, n_steps, gripper_open) in phases:
        q_from = q_waypoints[from_idx][:5]
        q_to   = q_waypoints[to_idx][:5]
        g = _SO100_GRIPPER_OPEN if gripper_open else _SO100_GRIPPER_CLOSE

        for q_arm in _interpolate_joints(q_from, q_to, n_steps):
            _record_step(np.append(q_arm, g).astype(np.float32))

    success_info = env_raw.evaluate()
    success = bool(success_info["success"][0].item())

    if debug:
        logger.info(f"    Episode: success={success}, T={len(trajectory)}")

    return success, trajectory


# ── Demo collection ───────────────────────────────────────────────────────────

def collect_demos_for_task(
    task_cfg: dict,
    robot_uids: str,
    n_demos: int,
    min_success: int,
    episode_length: int,
    demo_dir: Path,
    debug: bool = False,
) -> tuple[int, int]:
    """Collect n_demos episodes with at least min_success successful ones.

    Stores ALL episodes (success and failure) tagged with success=True/False.
    Keeps collecting until both quotas are met:
      - total stored >= n_demos
      - successful stored >= min_success

    Returns (n_total_stored, n_success_stored).
    """
    task_id = task_cfg["id"]
    object_name = task_cfg["object_name"]
    instruction = task_cfg["instruction"]

    save_dir = demo_dir / robot_uids / task_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Count existing episodes
    existing = sorted(save_dir.glob("episode_*.h5"))
    n_existing = len(existing)
    n_success_existing = 0
    for ep in existing:
        with h5py.File(ep) as f:
            if f.attrs.get("success", False):
                n_success_existing += 1

    if n_existing >= n_demos and n_success_existing >= min_success:
        logger.info(
            f"  {task_id}: {n_existing} demos ({n_success_existing} success) already collected, skipping"
        )
        return n_existing, n_success_existing

    env = gym.make(
        "PutObjectOnPlate-v1",
        num_envs=1,
        obs_mode="rgbd",
        render_mode=None,
        sim_backend="cpu",
        control_mode="pd_joint_pos",
        robot_uids=robot_uids,
        object_name=object_name,
        instruction=instruction,
    )
    env_raw = env.unwrapped

    if robot_uids != "so100":
        raise NotImplementedError(
            f"Scripted oracle only implemented for so100 (got {robot_uids!r}). "
            "For panda, use the mplib motion planner."
        )

    model = env_raw.agent.robot.create_pinocchio_model()

    n_collected = n_existing
    n_success = n_success_existing
    n_attempts = 0
    max_attempts = n_demos * 50  # generous budget

    def _done() -> bool:
        return n_collected >= n_demos and n_success >= min_success

    try:
        with tqdm(total=n_demos - n_existing, desc=f"{robot_uids}/{task_id[:22]}") as pbar:
            while not _done() and n_attempts < max_attempts:
                n_attempts += 1
                env.reset()

                try:
                    success, traj = _collect_so100_trajectory(env_raw, model, debug=debug)
                except Exception as exc:
                    if debug:
                        logger.warning(f"    attempt {n_attempts}: oracle error: {exc}")
                    success = False
                    traj = []

                if len(traj) < 3:
                    continue

                # If we already have enough total demos, only save successes
                # (to top up the success quota without exceeding n_demos too much)
                if n_collected >= n_demos and not success:
                    continue

                # Convert absolute qpos → delta actions
                T = len(traj) - 1
                imgs   = [traj[t]["img"]   for t in range(T)]
                states = [traj[t]["state"] for t in range(T)]
                deltas = []
                for t in range(T):
                    q0 = traj[t]["qpos"][:ACTION_DIM]
                    q1 = traj[t + 1]["qpos"][:ACTION_DIM]
                    deltas.append(np.clip((q1 - q0).astype(np.float32), -1.0, 1.0))

                ep_path = save_dir / f"episode_{n_collected:04d}.h5"
                with h5py.File(ep_path, "w") as f:
                    f.create_dataset(
                        "observations/images/overhead",
                        data=np.stack(imgs), compression="gzip", compression_opts=4,
                    )
                    f.create_dataset("observations/state", data=np.stack(states))
                    f.create_dataset("actions", data=np.stack(deltas))
                    f.attrs["instruction"] = instruction
                    f.attrs["task_id"]     = task_id
                    f.attrs["robot_uids"]  = robot_uids
                    f.attrs["success"]     = bool(success)
                    f.attrs["n_steps"]     = T

                n_collected += 1
                if success:
                    n_success += 1
                pbar.update(1)

    finally:
        env.close()

    logger.info(
        f"  {task_id}: {n_collected} demos ({n_success} success, "
        f"{n_collected - n_success} fail) → {save_dir}"
    )
    return n_collected, n_success


# ── Observation helpers ───────────────────────────────────────────────────────

def _extract_image_raw(obs: dict) -> np.ndarray:
    """(H, W, 3) uint8 overhead RGB."""
    if "sensor_data" in obs and "overhead" in obs["sensor_data"]:
        rgb = obs["sensor_data"]["overhead"]["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        return (rgb[0] if rgb.ndim == 4 else rgb).astype(np.uint8)
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _extract_state_raw(obs: dict) -> np.ndarray:
    """(6,) float32 — first 6 qpos values."""
    if "agent" in obs and "qpos" in obs["agent"]:
        q = obs["agent"]["qpos"]
        if isinstance(q, torch.Tensor):
            q = q.cpu().numpy()
        q = q[0] if q.ndim == 2 else q
        return q[:ACTION_DIM].astype(np.float32)
    return np.zeros(ACTION_DIM, dtype=np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect demos via scripted oracle")
    parser.add_argument(
        "--robot-uids", type=str, default="so100",
        choices=["so100"],
        help="Robot to collect demos for (only so100 supported)",
    )

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--all-tasks", action="store_true",
                             help="All 8 tasks (4 training + 4 held-out)")
    task_group.add_argument("--all-training", action="store_true",
                             help="4 training tasks")
    task_group.add_argument("--all-held-out", action="store_true",
                             help="4 held-out tasks")
    task_group.add_argument("--task-ids", nargs="+")

    parser.add_argument("--n-demos", type=int, default=35,
                        help="Total demos per task (success + fail)")
    parser.add_argument("--min-success", type=int, default=15,
                        help="Minimum successful demos required per task")
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--demo-dir", type=str, default="data/demos")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.all_tasks:
        tasks = ALL_TASKS
    elif args.all_held_out:
        tasks = HELD_OUT_TASKS
    elif args.task_ids:
        tasks = [get_task_by_id(tid) for tid in args.task_ids]
    else:
        tasks = TRAINING_TASKS

    demo_dir = Path(args.demo_dir)
    logger.info(
        f"Collecting {args.n_demos} demos (≥{args.min_success} success) "
        f"× {len(tasks)} tasks (robot={args.robot_uids})"
    )
    logger.info(f"Tasks: {[t['id'] for t in tasks]}")

    total_demos = total_success = 0
    for task_cfg in tasks:
        n, s = collect_demos_for_task(
            task_cfg=task_cfg,
            robot_uids=args.robot_uids,
            n_demos=args.n_demos,
            min_success=args.min_success,
            episode_length=args.episode_length,
            demo_dir=demo_dir,
            debug=args.debug,
        )
        total_demos += n
        total_success += s

    logger.info(
        f"\nDone — {total_demos} total demos ({total_success} success) "
        f"in {demo_dir / args.robot_uids}/"
    )


if __name__ == "__main__":
    main()
