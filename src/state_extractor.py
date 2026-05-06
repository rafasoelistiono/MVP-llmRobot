from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import mujoco


OBJECT_NAMES = ("red_box", "blue_box", "green_box")

OBJECT_SIZES: dict[str, list[float]] = {
    "red_box": [0.08, 0.08, 0.04],
    "blue_box": [0.07, 0.07, 0.04],
    "green_box": [0.14, 0.07, 0.04],
}

ROBOT_STATE: dict[str, Any] = {
    "name": "UR5e",
    "base": [0.00, 0.00, 0.00],
    "max_reach": 0.85,
    "effective_reach": 0.75,
    "workspace": "tabletop",
    "gripper": "two_finger_gripper",
}

TABLE_STATE: dict[str, Any] = {
    "name": "target_table",
    "height": 0.40,
    "valid_placement_area": {
        "x_min": 0.35,
        "x_max": 0.70,
        "y_min": -0.25,
        "y_max": 0.25,
    },
}

GOAL_TEXT = (
    "Place all objects inside the valid placement area on the target table "
    "without collision and with feasible UR5e motions."
)


def yaw_to_quat(theta: float) -> list[float]:
    """Return a MuJoCo quaternion [w, x, y, z] for a world-z yaw."""
    half_theta = theta * 0.5
    return [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)]


def quat_to_yaw(quat: list[float]) -> float:
    """Extract world-z yaw from a MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _object_pose(model: mujoco.MjModel, data: mujoco.MjData, object_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
        raise ValueError(f"Object body '{object_name}' does not exist in the MuJoCo model.")

    x, y = data.xpos[body_id][0], data.xpos[body_id][1]
    theta = 0.0
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_freejoint")
    if joint_id >= 0:
        qpos_addr = model.jnt_qposadr[joint_id]
        x = data.qpos[qpos_addr]
        y = data.qpos[qpos_addr + 1]
        theta = quat_to_yaw(data.qpos[qpos_addr + 3 : qpos_addr + 7].tolist())

    return [round(float(x), 4), round(float(y), 4), round(float(theta), 4)]


def extract_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract the JSON-compatible planning state from the MuJoCo scene."""
    mujoco.mj_forward(model, data)
    statuses = statuses or {}

    objects = []
    for object_name in OBJECT_NAMES:
        objects.append(
            {
                "name": object_name,
                "size": OBJECT_SIZES[object_name],
                "pose": _object_pose(model, data, object_name),
                "status": statuses.get(object_name, "on_table"),
            }
        )

    return {
        "robot": deepcopy(ROBOT_STATE),
        "table": deepcopy(TABLE_STATE),
        "objects": objects,
        "goal": GOAL_TEXT,
    }


def extract_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper with the name used by the LLM3 demo."""
    return extract_state(model, data, statuses)


def object_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {obj["name"]: obj for obj in state["objects"]}


def save_initial_qpos(data: mujoco.MjData) -> Any:
    """Snapshot the full qpos vector so every LLM attempt starts identically."""
    return data.qpos.copy()


def reset_to_initial_qpos(model: mujoco.MjModel, data: mujoco.MjData, initial_qpos: Any) -> None:
    """Reset MuJoCo dynamics and restore the saved object and robot configuration."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    if data.ctrl is not None and data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
