"""Custom ManiSkill3 task: Put Object On Plate.

Follows the ManiSkill benchmark described in the paper (Appendix D, H):
- Scene 25 Main: standardized tabletop with a plate target
- 4 training objects + 4 held-out objects
- Episode length: 80 steps
- Binary sparse reward: 1 if object is on plate, else 0
- Language-conditioned via task description
- 40 discrete object positions × 4 rotations
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat as _euler2quat

from mani_skill.agents.robots import Panda, SO100
from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.envs.utils.randomization as randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ─── Task registry ────────────────────────────────────────────────────────────

TRAINING_TASKS: list[dict] = [
    {
        "id": "put_banana_on_plate",
        "instruction": "put banana on plate",
        "object_name": "carrot",
    },
    {
        "id": "put_gelatin_box_on_plate",
        "instruction": "put gelatin box on plate",
        "object_name": "bread_loaf",
    },
    {
        "id": "put_mug_on_plate",
        "instruction": "put mug on plate",
        "object_name": "ketchup_bottle",
    },
    {
        "id": "put_cup_on_plate",
        "instruction": "put cup on plate",
        "object_name": "cup",
    },
]

HELD_OUT_TASKS: list[dict] = [
    {
        "id": "put_bowl_on_plate",
        "instruction": "put bowl on plate",
        "object_name": "watering_can",
    },
    {
        "id": "put_marker_on_plate",
        "instruction": "put marker on plate",
        "object_name": "pipe",
    },
    {
        "id": "put_foam_brick_on_plate",
        "instruction": "put foam brick on plate",
        "object_name": "toy_bear",
    },
    {
        "id": "put_tuna_can_on_plate",
        "instruction": "put tuna can on plate",
        "object_name": "hamburger",
    },
]

ALL_TASKS = TRAINING_TASKS + HELD_OUT_TASKS

# YCB model IDs for each object (best available matches from mani_skill2_ycb)
_OBJECT_YCB_IDS: dict[str, str] = {
    "carrot":         "011_banana",           # elongated food item
    "bread_loaf":     "009_gelatin_box",      # small flat box (cracker_box too tall)
    "ketchup_bottle": "025_mug",             # mug shape (mustard_bottle too tall)
    "cup":            "065-a_cups",           # actual cup
    "watering_can":   "024_bowl",            # bowl (pitcher_base too tall to grasp)
    "pipe":           "040_large_marker",     # cylindrical
    "toy_bear":       "061_foam_brick",       # toy block
    "hamburger":      "007_tuna_fish_can",    # flat round can
}

# Precomputed z_offsets (bottom of mesh to origin) so objects rest on the table.
# Avoids querying collision mesh at runtime which fails on GPU backend.
_OBJECT_Z_OFFSETS: dict[str, float] = {
    "carrot":         0.0187,
    "bread_loaf":     0.0150,
    "ketchup_bottle": 0.0407,
    "cup":            0.0308,
    "watering_can":   0.0275,
    "pipe":           0.0094,
    "toy_bear":       0.0256,
    "hamburger":      0.0168,
}

# Plate properties
_PLATE_RADIUS: float = 0.12
_PLATE_HALF_HEIGHT: float = 0.005
_PLATE_COLOR: tuple = (0.85, 0.85, 0.85, 1.0)

# Robot-specific scene configs (calibrated to each robot's workspace)
# Panda base at (-0.615, 0, 0), facing +X; workspace center ~(0, 0)
# SO100 base at (-0.725, 0, 0) rotated 90° about Z; workspace center ~(-0.46, 0)
_ROBOT_SCENE_CONFIGS: dict = {
    "panda": {
        "plate_xy": (0.0, 0.12),
        "spawn_center": (0.0, -0.12),
        "spawn_half": (0.12, 0.12),
        "cam_eye": [0.0, -0.4, 1.0],
        "cam_target": [0.0, 0.1, 0.6],
    },
    "so100": {
        "plate_xy": (-0.46, 0.08),
        "spawn_center": (-0.48, -0.06),
        "spawn_half": (0.04, 0.05),
        "cam_eye": [-0.3, -0.4, 0.5],
        "cam_target": [-0.5, 0.0, 0.05],
    },
}
_DEFAULT_ROBOT_CFG = _ROBOT_SCENE_CONFIGS["panda"]

# Legacy constants (used by collect_demos.py for plate world position)
_PLATE_XY: tuple[float, float] = (0.0, 0.12)  # panda default


@register_env("PutObjectOnPlate-v1", max_episode_steps=80)
class PutObjectOnPlateEnv(BaseEnv):
    """Language-conditioned pick-and-place: put a named object onto a plate.

    The object identity is set via ``object_name`` in the constructor.
    A language instruction is stored in ``self.instruction``.

    Binary sparse reward: 1 if object centroid is within plate radius at
    episode end.
    """

    SUPPORTED_ROBOTS = ["panda", "so100"]
    agent: Union[Panda, SO100]

    # Success threshold
    SUCCESS_DIST: float = _PLATE_RADIUS + 0.02  # horizontal
    SUCCESS_HEIGHT: float = -0.01               # object z >= table surface - eps

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        object_name: str = "carrot",
        instruction: str | None = None,
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ) -> None:
        self.object_name = object_name
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._scene_cfg = _ROBOT_SCENE_CONFIGS.get(
            robot_uids if isinstance(robot_uids, str) else robot_uids[0],
            _DEFAULT_ROBOT_CFG,
        )

        # Resolve instruction from task registry
        task_cfg = next(
            (t for t in ALL_TASKS if t["object_name"] == object_name), None
        )
        if instruction is not None:
            self.instruction = instruction
        elif task_cfg is not None:
            self.instruction = task_cfg["instruction"]
        else:
            self.instruction = f"put {object_name} on plate"

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def switch_task(self, object_name: str, instruction: str | None = None) -> None:
        """Switch to a different object without recreating the SAPIEN environment.

        Call env.reset(options={"reconfigure": True}) after this to rebuild the scene.
        This avoids the SAPIEN GPU render system lifecycle issue when creating multiple
        GPU environments sequentially in the same process.
        """
        self.object_name = object_name
        task_cfg = next((t for t in ALL_TASKS if t["object_name"] == object_name), None)
        if instruction is not None:
            self.instruction = instruction
        elif task_cfg is not None:
            self.instruction = task_cfg["instruction"]
        else:
            self.instruction = f"put {object_name} on plate"

    # ── Sensors ───────────────────────────────────────────────────────────────

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        cfg = self._scene_cfg
        pose = sapien_utils.look_at(eye=cfg["cam_eye"], target=cfg["cam_target"])
        return [CameraConfig("overhead", pose, 256, 256, np.deg2rad(60), 0.01, 10.0)]

    @property
    def _default_human_render_camera_configs(self) -> list[CameraConfig]:
        cfg = self._scene_cfg
        cx, cy = cfg["plate_xy"]
        pose = sapien_utils.look_at(
            eye=[cx + 0.5, cy - 0.5, 0.8],
            target=[cx, cy, 0.3],
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 10.0)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _load_agent(self, options: dict) -> None:
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict) -> None:
        # Standard tabletop scene (table + robot init)
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        cfg = self._scene_cfg
        px, py = cfg["plate_xy"]

        # Plate: flat cylinder, kinematic (doesn't move)
        self.plate = actors.build_cylinder(
            self.scene,
            radius=_PLATE_RADIUS,
            half_length=_PLATE_HALF_HEIGHT,
            color=list(_PLATE_COLOR),
            name="plate",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[px, py, _PLATE_HALF_HEIGHT],
                q=_euler2quat(0, np.pi / 2, 0),  # rotate X-axis cylinder → Z-axis (flat plate)
            ),
        )

        # Manipulation object — YCB mesh asset
        ycb_id = _OBJECT_YCB_IDS.get(self.object_name, "003_cracker_box")
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{ycb_id}")
        builder.initial_pose = sapien.Pose(p=[px, py - 0.1, 0])
        self.object_actor = builder.build(name=self.object_name)

        # Use precomputed z_offset (collision mesh query fails on GPU backend)
        self._obj_z_offset = _OBJECT_Z_OFFSETS.get(self.object_name, 0.03)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        """Randomize object position + rotation."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            cfg = self._scene_cfg
            cx, cy = cfg["spawn_center"]
            hx, hy = cfg["spawn_half"]

            xyz = torch.zeros((b, 3))
            xy = torch.rand((b, 2)) * torch.tensor([hx, hy]) * 2 - torch.tensor([hx, hy])
            xyz[:, 0] = xy[:, 0] + cx
            xyz[:, 1] = xy[:, 1] + cy
            xyz[:, 2] = self._obj_z_offset

            # Random yaw rotation only
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.object_actor.set_pose(Pose.create_from_pq(xyz, qs))

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict) -> dict:
        # Note: instruction/task string is injected by LeRobotObsWrapper, not here
        # Panda exposes agent.tcp.pose; SO100 exposes agent.tcp_pose directly
        if hasattr(self.agent, "tcp"):
            tcp = self.agent.tcp.pose.raw_pose
        else:
            tcp = self.agent.tcp_pose.raw_pose
        return dict(tcp_pose=tcp)

    # ── Reward & success ──────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Return batched success tensor — one bool per parallel env."""
        obj_pos = self.object_actor.pose.p   # (num_envs, 3)
        plate_pos = self.plate.pose.p         # (num_envs, 3)  or broadcast

        # Horizontal distance
        dist_xy = torch.linalg.norm(obj_pos[:, :2] - plate_pos[:, :2], dim=1)
        # Object must be roughly at table height or above
        height_ok = obj_pos[:, 2] >= self.SUCCESS_HEIGHT

        success = (dist_xy < self.SUCCESS_DIST) & height_ok
        return {"success": success}

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        # Paper uses sparse binary reward only
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)


def get_task_by_id(task_id: str) -> dict:
    """Look up task config by ID."""
    for t in ALL_TASKS:
        if t["id"] == task_id:
            return t
    raise KeyError(
        f"Unknown task: {task_id!r}. Available: {[t['id'] for t in ALL_TASKS]}"
    )
