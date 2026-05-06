from __future__ import annotations

import math
import time
from typing import Any

import mujoco
import numpy as np

from .state_extractor import OBJECT_SIZES, TABLE_STATE, quat_to_yaw, yaw_to_quat


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

MIN_VISIBLE_SEGMENT_SECONDS = 0.85
VIEWER_FRAME_SECONDS = 0.025
ATTACHMENT_SITE_NAME = "attachment_site"
SITE_TO_OBJECT_CENTER_Z = 0.11
PRE_GRASP_CLEARANCE_Z = 0.25
LIFT_CLEARANCE_Z = 0.24
IK_MAX_ITERATIONS = 120
IK_TOLERANCE = 0.012
IK_DAMPING = 1e-3
IK_STEP_SCALE = 0.65


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


class SimpleUR5eArmMotionExecutor:
    """Small visual joint-space executor.

    This is deliberately lightweight: it is not a numerical IK solver. It maps
    tabletop pick/place targets to plausible UR5e joint waypoints so the live
    replay shows the robot arm collaborating with the LLM plan.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer: Any = None,
        action_delay: float = 1.25,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer
        self.action_delay = action_delay
        self.joint_qpos_addr: list[int] = []
        self.joint_dof_addr: list[int] = []
        self.ctrl_index_by_joint: dict[int, int] = {}
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ATTACHMENT_SITE_NAME)
        self.available = self._load_arm_indices()

    def _load_arm_indices(self) -> bool:
        for joint_name in ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                print(f"[ARM] disabled: missing joint {joint_name}")
                return False
            self.joint_qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_dof_addr.append(int(self.model.jnt_dofadr[joint_id]))

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id >= 0:
                self.ctrl_index_by_joint[joint_id] = actuator_id

        if self.site_id < 0:
            print(f"[ARM] disabled: missing site {ATTACHMENT_SITE_NAME}")
            return False

        return True

    def set_viewer(self, viewer: Any) -> None:
        self.viewer = viewer

    def home_qpos(self) -> list[float]:
        if self.model.nkey > 0:
            return [float(value) for value in self.model.key_qpos[0][: len(ARM_JOINT_NAMES)]]
        return [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

    def current_qpos(self) -> list[float]:
        return [float(self.data.qpos[addr]) for addr in self.joint_qpos_addr]

    def move_home(self) -> None:
        if not self.available:
            return
        self._move_to(self.home_qpos(), "[ARM] moving to home")

    def execute_pick(self, object_name: str) -> None:
        if not self.available:
            print("[ARM] skipped: joint arm executor unavailable.")
            return

        pose = self._get_object_pose(object_name)
        x, y, object_z, theta = pose
        print(f"[ARM] joint planner: pick {object_name} at x={x:.3f}, y={y:.3f}")

        pre_site = (x, y, object_z + PRE_GRASP_CLEARANCE_Z)
        grasp_site = (x, y, object_z + SITE_TO_OBJECT_CENTER_Z)
        lift_site = (x, y, object_z + LIFT_CLEARANCE_Z)

        pre_qpos = self._solve_site_ik(pre_site, theta, self._target_qpos(x, y, theta, level="pre"))
        self._move_to(pre_qpos, "[ARM] moving gripper above object")

        grasp_qpos = self._solve_site_ik(grasp_site, theta, pre_qpos)
        self._move_to(grasp_qpos, "[ARM] lowering gripper to grasp object")
        print(f"[ARM] symbolic gripper close on {object_name}")

        self._snap_object_to_gripper(object_name, theta)
        lift_qpos = self._solve_site_ik(lift_site, theta, grasp_qpos)
        self._move_to_with_attached_object(
            lift_qpos,
            object_name,
            theta,
            "[ARM] lifting object attached to gripper",
        )

    def execute_place(self, object_name: str, x: float, y: float, theta: float) -> None:
        if not self.available:
            print("[ARM] skipped: joint arm executor unavailable.")
            return

        table_z = TABLE_STATE["height"] + OBJECT_SIZES[object_name][2] / 2.0
        print(f"[ARM] joint planner: place {object_name} at x={x:.3f}, y={y:.3f}, theta={theta:.3f}")

        pre_site = (x, y, table_z + LIFT_CLEARANCE_Z)
        place_site = (x, y, table_z + SITE_TO_OBJECT_CENTER_Z)

        pre_qpos = self._solve_site_ik(pre_site, theta, self._target_qpos(x, y, theta, level="pre"))
        self._move_to_with_attached_object(
            pre_qpos,
            object_name,
            theta,
            "[ARM] carrying object attached to gripper",
        )

        place_qpos = self._solve_site_ik(place_site, theta, pre_qpos)
        self._move_to_with_attached_object(
            place_qpos,
            object_name,
            theta,
            "[ARM] lowering attached object to goal",
        )
        self._set_object_pose(object_name, x, y, table_z, theta)
        print(f"[ARM] symbolic gripper open release {object_name}")
        retreat_qpos = self._solve_site_ik(pre_site, theta, place_qpos)
        self._move_to(retreat_qpos, "[ARM] retreating after release")

    def _target_qpos(self, x: float, y: float, theta: float, level: str) -> list[float]:
        reach = math.sqrt(x * x + y * y)
        reach_alpha = (_clamp(reach, 0.25, 0.75) - 0.25) / 0.50
        pan = -math.pi / 2.0 + math.atan2(y, max(x, 0.05))

        if level == "low":
            shoulder_lift = -1.12 + 0.24 * reach_alpha
            elbow = 1.12 - 0.16 * reach_alpha
            wrist_1 = -1.62
        elif level == "lift":
            shoulder_lift = -1.42 + 0.18 * reach_alpha
            elbow = 1.48 - 0.20 * reach_alpha
            wrist_1 = -1.54
        else:
            shoulder_lift = -1.34 + 0.20 * reach_alpha
            elbow = 1.38 - 0.18 * reach_alpha
            wrist_1 = -1.56

        return [
            _clamp(pan, -3.14, 3.14),
            _clamp(shoulder_lift, -3.14, 3.14),
            _clamp(elbow, -3.14, 3.14),
            wrist_1,
            -1.5708,
            _clamp(theta, -3.14, 3.14),
        ]

    def _move_to(self, target_qpos: list[float], label: str) -> None:
        print(label)
        start_qpos = self.current_qpos()
        steps = self._step_count()
        for step in range(1, steps + 1):
            alpha = step / steps
            qpos = [_lerp(start, target, alpha) for start, target in zip(start_qpos, target_qpos)]
            self._set_arm_qpos(qpos)
            self._sync_step()

    def _move_to_with_attached_object(
        self,
        target_qpos: list[float],
        object_name: str,
        theta: float,
        label: str,
    ) -> None:
        print(label)
        start_qpos = self.current_qpos()
        steps = self._step_count()
        for step in range(1, steps + 1):
            alpha = step / steps
            qpos = [_lerp(start, target, alpha) for start, target in zip(start_qpos, target_qpos)]
            self._set_arm_qpos(qpos)
            self._snap_object_to_gripper(object_name, theta)
            self._sync_step()

    def _set_arm_qpos(self, qpos: list[float]) -> None:
        for index, value in enumerate(qpos):
            qpos_addr = self.joint_qpos_addr[index]
            dof_addr = self.joint_dof_addr[index]
            self.data.qpos[qpos_addr] = value
            self.data.qvel[dof_addr] = 0.0
            if index < self.model.nu:
                self.data.ctrl[index] = value
        mujoco.mj_forward(self.model, self.data)

    def _solve_site_ik(
        self,
        target_pos: tuple[float, float, float],
        theta: float,
        seed_qpos: list[float] | None = None,
    ) -> list[float]:
        original_qpos = self.current_qpos()
        qpos = np.array(seed_qpos if seed_qpos is not None else self.current_qpos(), dtype=float)
        target = np.array(target_pos, dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        best_qpos = qpos.copy()
        best_error_norm = float("inf")

        for _iteration in range(IK_MAX_ITERATIONS):
            self._set_arm_qpos(qpos.tolist())
            current = np.array(self.data.site_xpos[self.site_id], dtype=float)
            error = target - current
            error_norm = float(np.linalg.norm(error))
            if error_norm < best_error_norm:
                best_error_norm = error_norm
                best_qpos = qpos.copy()
            if error_norm <= IK_TOLERANCE:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = jacp[:, self.joint_dof_addr]
            lhs = jac @ jac.T + IK_DAMPING * np.eye(3)
            dq = jac.T @ np.linalg.solve(lhs, error)
            qpos = qpos + IK_STEP_SCALE * dq
            qpos = self._clamp_arm_qpos(qpos)

        best_qpos = self._clamp_arm_qpos(best_qpos)
        best_qpos[5] = _clamp(theta, -3.14, 3.14)
        if best_error_norm > IK_TOLERANCE:
            print(f"[ARM] IK residual {best_error_norm:.3f} m for target {tuple(round(v, 3) for v in target_pos)}")
        self._set_arm_qpos(original_qpos)
        return best_qpos.tolist()

    def _clamp_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        clamped = qpos.copy()
        for index, joint_name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0 or not bool(self.model.jnt_limited[joint_id]):
                continue
            lower, upper = self.model.jnt_range[joint_id]
            clamped[index] = _clamp(float(clamped[index]), float(lower), float(upper))
        return clamped

    def _snap_object_to_gripper(self, object_name: str, theta: float) -> None:
        site_pos = self.data.site_xpos[self.site_id]
        object_center = (
            float(site_pos[0]),
            float(site_pos[1]),
            float(site_pos[2] - SITE_TO_OBJECT_CENTER_Z),
        )
        self._set_object_pose(object_name, object_center[0], object_center[1], object_center[2], theta)

    def _step_count(self) -> int:
        return max(16, int(self._segment_seconds() / VIEWER_FRAME_SECONDS))

    def _sync_step(self) -> None:
        if self.viewer is not None:
            if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
                return
            self.viewer.sync()
        time.sleep(min(VIEWER_FRAME_SECONDS, self._segment_seconds() / max(self._step_count(), 1)))

    def _segment_seconds(self) -> float:
        return max(self.action_delay, MIN_VISIBLE_SEGMENT_SECONDS)

    def _get_object_pose(self, object_name: str) -> tuple[float, float, float, float]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_freejoint")
        if joint_id >= 0:
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            quat = self.data.qpos[qpos_addr + 3 : qpos_addr + 7].tolist()
            return (
                float(self.data.qpos[qpos_addr]),
                float(self.data.qpos[qpos_addr + 1]),
                float(self.data.qpos[qpos_addr + 2]),
                float(quat_to_yaw(quat)),
            )

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            raise ValueError(f"{object_name} does not exist in the MuJoCo model.")
        return (
            float(self.data.xpos[body_id][0]),
            float(self.data.xpos[body_id][1]),
            float(self.data.xpos[body_id][2]),
            0.0,
        )

    def _set_object_pose(self, object_name: str, x: float, y: float, z: float, theta: float) -> None:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_freejoint")
        if joint_id < 0:
            print(f"[ARM] warning: {object_name} has no freejoint; object carry skipped.")
            return
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        qvel_addr = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_addr : qpos_addr + 3] = [x, y, z]
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = yaw_to_quat(theta)
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
