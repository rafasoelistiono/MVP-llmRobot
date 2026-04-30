import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_ROOT / "swarm_scene.xml"

GROUND_Z = 0.10
HOVER_Z = 1.0
SPLIT_DELTA_Z = 0.28
SIMULATION_END = 32.0
IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

COMPACT_OFFSETS = np.array(
    [
        [-0.35, -0.35],
        [0.35, -0.35],
        [-0.35, 0.35],
        [0.35, 0.35],
    ],
    dtype=float,
)

WIDE_OFFSETS = np.array(
    [
        [-0.75, -0.75],
        [0.75, -0.75],
        [-0.75, 0.75],
        [0.75, 0.75],
    ],
    dtype=float,
)

HOVER_ALTITUDES = np.full(4, HOVER_Z, dtype=float)
SPLIT_A_ALTITUDES = HOVER_ALTITUDES + np.array([SPLIT_DELTA_Z, -SPLIT_DELTA_Z, SPLIT_DELTA_Z, -SPLIT_DELTA_Z], dtype=float)
SPLIT_B_ALTITUDES = HOVER_ALTITUDES + np.array([-SPLIT_DELTA_Z, SPLIT_DELTA_Z, -SPLIT_DELTA_Z, SPLIT_DELTA_Z], dtype=float)


@dataclass(frozen=True)
class DroneLayout:
    drone_id: str
    qpos_slice: slice
    qvel_slice: slice
    qacc_slice: slice
    ctrl_indices: np.ndarray


@dataclass
class DroneState:
    layout: DroneLayout
    target: np.ndarray
    pid_x: "PositionPID"
    pid_y: "PositionPID"
    pid_z: "PositionPID"


@dataclass
class PositionPID:
    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    setpoint: float = 0.0
    integral: float = 0.0
    previous_error: float = 0.0

    def update(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = 0.0 if dt <= 0.0 else (error - self.previous_error) / dt
        self.previous_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return float(np.clip(output, self.output_min, self.output_max))


def save_csv_log(filename: str | Path, rows: list[dict[str, float | str]]) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", "drone_id", "target_x", "target_y", "target_z", "actual_x", "actual_y", "actual_z"]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def get_drone_layout(model: mujoco.MjModel, drone_name: str) -> DroneLayout:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{drone_name}_freejoint")
    if joint_id < 0:
        raise ValueError(f"Could not find free joint for {drone_name}")

    qpos_start = int(model.jnt_qposadr[joint_id])
    qvel_start = int(model.jnt_dofadr[joint_id])
    ctrl_indices = []

    for motor_index in range(1, 5):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{drone_name}_thrust{motor_index}",
        )
        if actuator_id < 0:
            raise ValueError(f"Could not find actuator {drone_name}_thrust{motor_index}")
        ctrl_indices.append(actuator_id)

    return DroneLayout(
        drone_id=drone_name,
        qpos_slice=slice(qpos_start, qpos_start + 7),
        qvel_slice=slice(qvel_start, qvel_start + 6),
        qacc_slice=slice(qvel_start, qvel_start + 6),
        ctrl_indices=np.array(ctrl_indices, dtype=int),
    )


def smooth_progress(current_time: float, start_time: float, end_time: float) -> float:
    raw = 1.0 if end_time <= start_time else (current_time - start_time) / (end_time - start_time)
    clamped = float(np.clip(raw, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * start + alpha * end


def get_micro_motion(drone_index: int, sim_time: float) -> np.ndarray:
    if sim_time < 10.0 or sim_time >= SIMULATION_END:
        return np.zeros(3, dtype=float)

    if sim_time < 13.0:
        envelope = smooth_progress(sim_time, 10.0, 11.5)
    elif sim_time < 28.0:
        envelope = 1.0
    else:
        envelope = 1.0 - smooth_progress(sim_time, 28.0, SIMULATION_END)

    phase = drone_index * (np.pi / 2.0)
    return envelope * np.array(
        [
            0.012 * np.sin(2.0 * np.pi * 0.22 * sim_time + phase),
            0.012 * np.cos(2.0 * np.pi * 0.18 * sim_time + 0.7 * phase),
            0.008 * np.sin(2.0 * np.pi * 0.15 * sim_time + 0.5 * phase),
        ],
        dtype=float,
    )


def build_targets(offsets_xy: np.ndarray, altitudes: np.ndarray, sim_time: float) -> np.ndarray:
    targets = np.column_stack((offsets_xy, altitudes))

    for drone_index in range(len(targets)):
        targets[drone_index] += get_micro_motion(drone_index, sim_time)
        targets[drone_index, 2] = max(GROUND_Z, targets[drone_index, 2])

    return targets


def get_targets_for_time(sim_time: float) -> np.ndarray:
    if sim_time < 10.0:
        ground_altitudes = np.full(4, GROUND_Z, dtype=float)
        return build_targets(COMPACT_OFFSETS, ground_altitudes, sim_time)

    if sim_time < 13.0:
        alpha = smooth_progress(sim_time, 10.0, 13.0)
        altitudes = lerp(np.full(4, GROUND_Z, dtype=float), HOVER_ALTITUDES, alpha)
        return build_targets(COMPACT_OFFSETS, altitudes, sim_time)

    if sim_time < 17.0:
        alpha = smooth_progress(sim_time, 13.0, 17.0)
        offsets = lerp(COMPACT_OFFSETS, WIDE_OFFSETS, alpha)
        return build_targets(offsets, HOVER_ALTITUDES, sim_time)

    if sim_time < 20.0:
        alpha = smooth_progress(sim_time, 17.0, 20.0)
        offsets = lerp(WIDE_OFFSETS, COMPACT_OFFSETS, alpha)
        return build_targets(offsets, HOVER_ALTITUDES, sim_time)

    if sim_time < 23.0:
        return build_targets(COMPACT_OFFSETS, SPLIT_A_ALTITUDES, sim_time)

    if sim_time < 26.0:
        return build_targets(COMPACT_OFFSETS, SPLIT_B_ALTITUDES, sim_time)

    if sim_time < 28.0:
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    alpha = smooth_progress(sim_time, 28.0, SIMULATION_END)
    altitudes = lerp(HOVER_ALTITUDES, np.full(4, GROUND_Z, dtype=float), alpha)
    return build_targets(COMPACT_OFFSETS, altitudes, sim_time)


class SwarmSimulation:
    def __init__(self, log_every_n_steps: int = 1) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.log_rows: list[dict[str, float | str]] = []
        self.drones: list[DroneState] = []
        self.step_count = 0

        for index, target in enumerate(get_targets_for_time(0.0)):
            layout = get_drone_layout(self.model, f"drone{index}")
            self.drones.append(
                DroneState(
                    layout=layout,
                    target=np.array(target, dtype=float),
                    pid_x=PositionPID(kp=4.0, ki=0.05, kd=0.35, output_min=-2.0, output_max=2.0),
                    pid_y=PositionPID(kp=4.0, ki=0.05, kd=0.35, output_min=-2.0, output_max=2.0),
                    pid_z=PositionPID(kp=3.0, ki=0.03, kd=0.25, output_min=-1.0, output_max=1.0),
                )
            )
            self.set_drone_pose(layout, np.array([target[0], target[1], GROUND_Z], dtype=float), np.zeros(3, dtype=float))

        mujoco.mj_forward(self.model, self.data)

    def set_drone_pose(self, layout: DroneLayout, position: np.ndarray, velocity: np.ndarray) -> None:
        qpos = self.data.qpos[layout.qpos_slice]
        qvel = self.data.qvel[layout.qvel_slice]
        qacc = self.data.qacc[layout.qacc_slice]

        qpos[:3] = position
        qpos[3:7] = IDENTITY_QUATERNION
        qvel[:3] = velocity
        qvel[3:6] = 0.0
        qacc[:] = 0.0
        self.data.qfrc_applied[layout.qvel_slice] = 0.0
        self.data.ctrl[layout.ctrl_indices] = 0.0

    def update_targets(self) -> None:
        for drone, target in zip(self.drones, get_targets_for_time(float(self.data.time))):
            drone.target = np.array(target, dtype=float)
            drone.pid_x.setpoint = float(target[0])
            drone.pid_y.setpoint = float(target[1])
            drone.pid_z.setpoint = float(target[2])

    def move_drones(self) -> None:
        dt = self.model.opt.timestep

        for drone in self.drones:
            current_position = self.data.qpos[drone.layout.qpos_slice][:3].copy()
            velocity = np.array(
                [
                    drone.pid_x.update(float(current_position[0]), dt),
                    drone.pid_y.update(float(current_position[1]), dt),
                    drone.pid_z.update(float(current_position[2]), dt),
                ],
                dtype=float,
            )
            next_position = current_position + velocity * dt
            self.set_drone_pose(drone.layout, next_position, velocity)

    def log_step(self) -> None:
        for drone in self.drones:
            actual = np.array(self.data.qpos[drone.layout.qpos_slice][:3], dtype=float)
            self.log_rows.append(
                {
                    "time": round(float(self.data.time), 4),
                    "drone_id": drone.layout.drone_id,
                    "target_x": round(float(drone.target[0]), 4),
                    "target_y": round(float(drone.target[1]), 4),
                    "target_z": round(float(drone.target[2]), 4),
                    "actual_x": round(float(actual[0]), 4),
                    "actual_y": round(float(actual[1]), 4),
                    "actual_z": round(float(actual[2]), 4),
                }
            )

    def step(self) -> None:
        self.update_targets()
        self.move_drones()
        self.data.time = float(self.data.time + self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1
        if self.step_count % self.log_every_n_steps == 0:
            self.log_step()

    def run_with_viewer(self, duration: float = SIMULATION_END) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < duration:
                step_start = time.time()
                self.step()

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    simulation = SwarmSimulation()
    simulation.run_with_viewer()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"swarm_sync_{timestamp}.csv"
    saved_path = save_csv_log(log_path, simulation.log_rows)
    print(f"Saved swarm log to {saved_path}")


if __name__ == "__main__":
    main()
