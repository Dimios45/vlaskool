"""Environment wrappers: LeRobot-compatible observation format + vec env helpers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Any


# Control mode per robot:
#   panda  → pd_ee_delta_pose (7D: 6 pose delta + 1 gripper); SmolVLA 6D is padded to 7D
#   so100  → pd_joint_delta_pos (6D: 5 joints + 1 gripper); exactly matches SmolVLA 6D
_ROBOT_CONTROL_MODE = {
    "panda": "pd_ee_delta_pose",
    "so100": "pd_joint_delta_pos",
}


def make_maniskill_env(
    object_name: str,
    instruction: str,
    num_envs: int = 8,
    sim_backend: str = "gpu",
    obs_mode: str = "rgbd",
    render_mode: str | None = None,
    robot_uids: str = "panda",
    seed: int = 0,
) -> gym.Env:
    """Create a vectorized PutObjectOnPlate environment.

    Args:
        robot_uids: ``"panda"`` (default) or ``"so100"``.  Pass via env_kwargs in
                    benchmark:  ``--env-kwargs robot_uids=so100``
    """
    # Register our custom task (import triggers @register_env decorator)
    import vlaskool.envs.put_object_on_plate  # noqa: F401

    control_mode = _ROBOT_CONTROL_MODE.get(robot_uids, "pd_ee_delta_pose")
    env = gym.make(
        "PutObjectOnPlate-v1",
        num_envs=num_envs,
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=sim_backend,
        control_mode=control_mode,
        robot_uids=robot_uids,
        object_name=object_name,
        instruction=instruction,
    )
    env = LeRobotObsWrapper(env, instruction=instruction, robot_uids=robot_uids)
    return env


class LeRobotObsWrapper(gym.Wrapper):
    """Reshape ManiSkill3 observations into LeRobot/SmolVLA policy input format.

    SmolVLA input keys (from config.input_features):
      - "observation.state":            (B, 6) float32 — proprioceptive state
      - "observation.images.camera1":   (B, C, H, W) float32 in [0, 1]
      - "task":                         list[str] of length B

    SmolVLA was pretrained with state_dim=6 and action_dim=6.
    We map ManiSkill3 end-effector pose (6D) → observation.state.
    """

    # SmolVLA native state/action dims
    STATE_DIM: int = 6   # matches SmolVLA pretrained config
    ACTION_DIM: int = 6  # 6-DOF end-effector delta

    def __init__(self, env: gym.Env, instruction: str, robot_uids: str = "panda") -> None:
        super().__init__(env)
        self.instruction = instruction
        self.robot_uids = robot_uids
        self._num_envs = getattr(env, "num_envs", 1)
        # panda uses pd_ee_delta_pose (7D); SmolVLA outputs 6D → pad gripper dim to 0
        # so100 uses pd_joint_delta_pos (6D); exact match, no padding needed
        self._env_action_dim = 7 if robot_uids == "panda" else self.ACTION_DIM

    def observation(self, raw_obs: dict) -> dict:
        """Convert raw ManiSkill obs to SmolVLA format."""
        images = self._extract_images(raw_obs)
        state = self._extract_state(raw_obs)
        B = images.shape[0] if images.ndim == 4 else 1
        return {
            # SmolVLA's expected key names
            "observation.images.camera1": images,
            "observation.state": state,
            "task": [self.instruction] * B,
            # Keep overhead alias for backward compat
            "observation.images.overhead": images,
        }

    def _extract_images(self, raw_obs: dict) -> torch.Tensor:
        """Extract overhead RGB, normalize to [0, 1] float32."""
        # ManiSkill3 stores sensor data in raw_obs["sensor_data"]["overhead"]["rgb"]
        if "sensor_data" in raw_obs:
            rgb = raw_obs["sensor_data"]["overhead"]["rgb"]  # (B, H, W, C) uint8
        elif "image" in raw_obs:
            rgb = raw_obs["image"]["overhead"]["rgb"]
        else:
            # Fallback: grey image
            rgb = torch.zeros(self._num_envs, 256, 256, 3, dtype=torch.uint8)

        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        # (B, H, W, C) uint8 → (B, C, H, W) float32 [0, 1]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        return rgb.permute(0, 3, 1, 2).float() / 255.0

    def _extract_state(self, raw_obs: dict) -> torch.Tensor:
        """Extract 6D end-effector state matching SmolVLA's state_dim=6.

        Priority: TCP pose [x,y,z,qx,qy,qz] → first 6 qpos joints → zeros.
        """
        if "extra" in raw_obs and "tcp_pose" in raw_obs.get("extra", {}):
            tcp = raw_obs["extra"]["tcp_pose"]
            if isinstance(tcp, np.ndarray):
                tcp = torch.from_numpy(tcp)
            if tcp.ndim == 1:
                tcp = tcp.unsqueeze(0)
            return tcp[:, :self.STATE_DIM].float()

        if "agent" in raw_obs and "qpos" in raw_obs["agent"]:
            qpos = raw_obs["agent"]["qpos"]
            if isinstance(qpos, np.ndarray):
                qpos = torch.from_numpy(qpos)
            if qpos.ndim == 1:
                qpos = qpos.unsqueeze(0)
            # Pad or truncate to STATE_DIM
            d = qpos.shape[-1]
            if d >= self.STATE_DIM:
                return qpos[:, :self.STATE_DIM].float()
            pad = torch.zeros(qpos.shape[0], self.STATE_DIM - d)
            return torch.cat([qpos.float(), pad], dim=-1)

        return torch.zeros(self._num_envs, self.STATE_DIM, dtype=torch.float32)

    def step(self, action: torch.Tensor | np.ndarray):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        # Pad SmolVLA's 6D output to the env's expected action dim if needed.
        # panda/pd_ee_delta_pose: 7D (6 pose delta + 1 gripper) → pad with 0
        # so100/pd_joint_delta_pos: 6D → no padding
        if action.shape[-1] < self._env_action_dim:
            pad = self._env_action_dim - action.shape[-1]
            action = np.concatenate(
                [action, np.zeros((*action.shape[:-1], pad))], axis=-1
            )
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(raw_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        return self.observation(raw_obs), info


class VecEnvWrapper:
    """Thin wrapper for collecting vectorized rollouts, compatible with both
    GPU and CPU ManiSkill backends.

    Usage:
        env = VecEnvWrapper(make_maniskill_env(...))
        obs, info = env.reset()
        obs, reward, done, trunc, info = env.step(actions)
        success = env.get_success()
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.num_envs: int = getattr(env, "num_envs", 1)
        self._last_info: dict = {}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._last_info = info
        return obs, info

    def step(self, action: torch.Tensor | np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        done = terminated | truncated
        return obs, reward, done, info

    def get_success(self) -> torch.Tensor:
        """Return boolean success tensor of shape (num_envs,)."""
        if "success" in self._last_info:
            s = self._last_info["success"]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.bool)
            return s.bool()
        return torch.zeros(self.num_envs, dtype=torch.bool)

    def switch_task(self, object_name: str, instruction: str) -> None:
        """Switch to a different object, reusing the same SAPIEN GPU environment.

        Calls env.switch_task() on the underlying ManiSkill env and then reconfigures
        the scene (rebuilds actors). This avoids creating a new SAPIEN render system,
        which fails on the 2nd+ instantiation in the same process.
        """
        # Walk through wrappers to find the raw ManiSkill env
        raw = self.env
        while hasattr(raw, "env"):
            # Update instruction on LeRobotObsWrapper layers
            if hasattr(raw, "instruction"):
                raw.instruction = instruction
            raw = raw.env
        # Call switch_task on the PutObjectOnPlateEnv
        if hasattr(raw, "switch_task"):
            raw.switch_task(object_name, instruction)
        # Reconfigure rebuilds _load_scene() with the new object
        self.reset(options={"reconfigure": True})

    def close(self) -> None:
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
