from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import mujoco
import mujoco.viewer

from .arm_motion_executor import SimpleUR5eArmMotionExecutor
from .state_extractor import OBJECT_SIZES, TABLE_STATE, yaw_to_quat


MIN_ACTION_PAUSE_SECONDS = 0.65


def _find_object_freejoint(model: mujoco.MjModel, object_name: str) -> int:
    named_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_freejoint")
    if named_joint >= 0:
        return named_joint

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
        return -1

    start = int(model.body_jntadr[body_id])
    count = int(model.body_jntnum[body_id])
    for joint_id in range(start, start + count):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return -1


def set_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_name: str,
    x: float,
    y: float,
    z: float,
    theta: float,
) -> bool:
    joint_id = _find_object_freejoint(model, object_name)
    if joint_id < 0:
        print(f"[EXECUTOR] warning: {object_name} has no freejoint; pose update skipped.")
        return False

    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = [x, y, z]
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = yaw_to_quat(theta)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return True


def sync_viewer(viewer: Any, seconds: float = 0.5) -> None:
    if viewer is None:
        return

    end_time = time.time() + max(0.0, seconds)
    while time.time() < end_time:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return
        viewer.sync()
        time.sleep(min(0.03, max(0.0, end_time - time.time())))
    viewer.sync()
    print("[VIEWER] synced")


class MujocoExecutor:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer_enabled: bool = True,
        action_delay: float = 1.25,
        use_arm_motion: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer_enabled = viewer_enabled
        self.action_delay = action_delay
        self.viewer: Any = None
        self.arm = (
            SimpleUR5eArmMotionExecutor(model, data, action_delay=action_delay)
            if use_arm_motion
            else None
        )

    @contextmanager
    def viewer_session(self) -> Iterator[Any]:
        if not self.viewer_enabled:
            yield None
            return

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            if self.arm is not None:
                self.arm.set_viewer(viewer)
            sync_viewer(self.viewer, 0.25)
            yield viewer
            if self.arm is not None:
                self.arm.set_viewer(None)
            self.viewer = None

    def sync(self, seconds: float | None = None) -> None:
        sync_viewer(self.viewer, self.action_delay if seconds is None else seconds)

    def mark_held(self, object_name: str) -> None:
        print(f"[EXECUTOR] marked {object_name} as held")
        self.sync(max(self.action_delay * 0.5, MIN_ACTION_PAUSE_SECONDS))

    def pick_object(self, object_name: str) -> None:
        if self.arm is not None and self.viewer is not None and self.arm.available:
            self.arm.execute_pick(object_name)
        self.mark_held(object_name)

    def place_object(self, object_name: str, x: float, y: float, theta: float) -> bool:
        if self.arm is not None and self.viewer is not None and self.arm.available:
            self.arm.execute_place(object_name, x, y, theta)
            print(f"[EXECUTOR] moved {object_name} with UR5e joint arm replay")
            self.sync(max(self.action_delay * 0.25, MIN_ACTION_PAUSE_SECONDS))
            return True

        object_height = OBJECT_SIZES[object_name][2]
        z = TABLE_STATE["height"] + object_height / 2.0
        updated = set_object_pose(self.model, self.data, object_name, x, y, z, theta)
        if updated:
            print(f"[EXECUTOR] moved {object_name} to target pose in MuJoCo")
        self.sync(self.action_delay)
        return updated

    def move_arm_home(self) -> None:
        if self.arm is not None and self.arm.available:
            self.arm.move_home()
