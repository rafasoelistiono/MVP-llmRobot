import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ----------------------------- Configuration/Const Variable ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_ROOT / "swarm_scene.xml"

GROUND_Z = 0.10
HOVER_Z = 1.0
SPLIT_DELTA_Z = 0.28
SIMULATION_END = 32.0
DEFAULT_HOVER_THRUST = 3.2495625
OUTER_LOOP_DIVIDER = 20
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


# ------------------------------- Data Model ----------------------------------
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
        # TODO : Implement one PID update step.
        # Inputs:
        # - measurement: the current measured position on one axis
        # - dt: simulation timestep in seconds
        #
        # Output:
        # - one scalar control output, typically a velocity command for this axis
        #
        # Expected idea:
        # - compute error = setpoint - measurement
        # - accumulate integral using dt
        # - compute derivative from the change in error
        # - combine kp, ki, kd
        # - clip the result between output_min and output_max
        #
        # Relevant state stored in this object:
        # - self.setpoint
        # - self.integral
        # - self.previous_error
        #
        # Hint:
        # - start with:
        #   error = self.setpoint - measurement
        # - update:
        #   self.integral += error * dt
        # - derivative is usually:
        #   (error - self.previous_error) / dt
        # - remember to save self.previous_error = error
        # - use np.clip(...) before returning

        placeholder_output = 0.0
        return float(np.clip(placeholder_output, self.output_min, self.output_max))


# ------------------------------- I/O Helpers ---------------------------------
def save_csv_log(filename: str | Path, rows: list[dict[str, float | str]]) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", "drone_id", "target_x", "target_y", "target_z", "actual_x", "actual_y", "actual_z"]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


# -------------------------- MuJoCo Layout Mapping ----------------------------
def get_drone_layout(model: mujoco.MjModel, drone_name: str) -> DroneLayout:
    # TODO : Map one named drone in the XML scene to slices in MuJoCo's
    # shared state arrays.
    #
    # Inputs:
    # - model: the loaded MuJoCo model
    # - drone_name: e.g. "drone0", "drone1", ...
    #
    # Output:
    # - DroneLayout containing:
    #   qpos_slice: where this drone's freejoint pose lives
    #   qvel_slice: where this drone's joint velocity lives
    #   qacc_slice: where this drone's acceleration lives
    #   ctrl_indices: the 4 actuator ids for this drone
    #
    # Relevant MuJoCo fields:
    # - mujoco.mj_name2id(...)
    # - model.jnt_qposadr
    # - model.jnt_dofadr
    #
    # Hint:
    # - locate the joint named f"{drone_name}_freejoint"
    # - read its qpos start from model.jnt_qposadr[joint_id]
    # - read its qvel start from model.jnt_dofadr[joint_id]
    # - collect actuator ids for:
    #   f"{drone_name}_thrust1" ... f"{drone_name}_thrust4"
    #
    # Temporary fallback below:
    # - this keeps the lab file syntactically valid
    # - it assumes the drones are stored in simple contiguous order
    # - you should replace it with name-based lookup using MuJoCo APIs

    drone_index = int(drone_name.replace("drone", ""))
    qpos_start = 7 * drone_index
    qvel_start = 6 * drone_index
    ctrl_start = 4 * drone_index

    return DroneLayout(
        drone_id=drone_name,
        qpos_slice=slice(qpos_start, qpos_start + 7),
        qvel_slice=slice(qvel_start, qvel_start + 6),
        qacc_slice=slice(qvel_start, qvel_start + 6),
        ctrl_indices=np.arange(ctrl_start, ctrl_start + 4, dtype=int),
    )


# --------------------------- Planning / Choreography -------------------------
def smooth_progress(current_time: float, start_time: float, end_time: float) -> float:
    raw = 1.0 if end_time <= start_time else (current_time - start_time) / (end_time - start_time)
    clamped = float(np.clip(raw, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * start + alpha * end


def get_micro_motion(drone_index: int, sim_time: float) -> np.ndarray:
    # TODO : Add a small deterministic motion offset for each drone so
    # the formation does not look perfectly rigid.
    #
    # Inputs:
    # - drone_index: 0..3
    # - sim_time: current simulation time
    #
    # Output:
    # - np.ndarray with shape (3,) containing [dx, dy, dz]
    #
    # Expected idea:
    # - use small sin/cos terms
    # - apply different phase shifts per drone
    # - keep amplitudes small so the formation remains safe
    # - fade the motion in after takeoff and out before landing
    #
    # Hint:
    # - np.sin and np.cos are enough
    # - a common pattern is:
    #   phase = drone_index * (np.pi / 2.0)
    # - return zeros outside the flying phase

    return np.zeros(3, dtype=float)


def build_targets(offsets_xy: np.ndarray, altitudes: np.ndarray, sim_time: float) -> np.ndarray:
    # TODO : Combine XY formation offsets and Z altitudes into a single
    # (4, 3) target array for all drones.
    #
    # Inputs:
    # - offsets_xy: shape (4, 2)
    # - altitudes: shape (4,)
    # - sim_time: current simulation time
    #
    # Output:
    # - targets: shape (4, 3), one [x, y, z] row per drone
    #
    # Expected idea:
    # - place offsets into x/y columns
    # - place altitudes into the z column
    # - add get_micro_motion(...) per drone
    # - clamp z so it does not go below GROUND_Z
    #
    # Hint:
    # - np.column_stack((offsets_xy, altitudes)) is a convenient start

    return np.column_stack((offsets_xy, altitudes))


def get_targets_for_time(sim_time: float) -> np.ndarray:
    # TODO : Return the synchronized (4, 3) target array for the
    # current choreography phase.
    #
    # Input:
    # - sim_time: current simulation time
    #
    # Output:
    # - np.ndarray of shape (4, 3)
    #
    # Choreography to implement:
    # - grounded compact
    # - takeoff
    # - spread
    # - regroup / shrink
    # - split A
    # - split B
    # - equalize
    # - landing
    #
    # Expected math:
    # - use smooth_progress(...) to compute alpha values
    # - use lerp(...) to interpolate offsets or altitude vectors
    # - use build_targets(...) to assemble the final target matrix
    #
    # Hint:
    # - keep the phase structure visible with sequential if blocks
    # - the function should always return a (4, 3) array

    ground_altitudes = np.full(4, GROUND_Z, dtype=float)

    if sim_time < 10.0:
        return build_targets(COMPACT_OFFSETS, ground_altitudes, sim_time)

    if sim_time < 13.0:
        # TODO: replace this placeholder with altitude interpolation
        return build_targets(COMPACT_OFFSETS, ground_altitudes, sim_time)

    if sim_time < 17.0:
        # TODO: replace this placeholder with formation spreading
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    if sim_time < 20.0:
        # TODO: replace this placeholder with regroup / shrink logic
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    if sim_time < 23.0:
        # TODO: replace this placeholder with split A altitudes
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    if sim_time < 26.0:
        # TODO: replace this placeholder with split B altitudes
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    if sim_time < 28.0:
        # TODO: replace this placeholder with equalized hover
        return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)

    # TODO: replace this placeholder with landing interpolation
    return build_targets(COMPACT_OFFSETS, HOVER_ALTITUDES, sim_time)


# ----------------------------- Main Simulation -------------------------------
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
        # TODO : Push the current choreography targets into each drone.
        #
        # Inputs available here:
        # - self.data.time for current sim time
        # - self.drones list containing DroneState objects
        #
        # Expected updates:
        # - drone.target should store the current [x, y, z] target
        # - pid_x.setpoint, pid_y.setpoint, pid_z.setpoint should be updated
        #
        # Hint:
        # - call get_targets_for_time(float(self.data.time))
        # - loop with zip(self.drones, targets)

        targets = get_targets_for_time(float(self.data.time))
        for drone, target in zip(self.drones, targets):
            drone.target = np.array(target, dtype=float)
            # TODO : set the PID setpoints here

    def move_drones(self) -> None:
        # TODO : Move each drone one simulation step toward its target
        # using the per-axis PID controllers.
        #
        # Relevant fields:
        # - self.data.qpos[drone.layout.qpos_slice][:3] gives current position
        # - self.model.opt.timestep gives dt
        # - drone.pid_x / pid_y / pid_z produce velocity commands
        #
        # Expected math:
        # - read current_position
        # - call PID.update(...) for x, y, z
        # - form velocity = [vx, vy, vz]
        # - integrate:
        #   next_position = current_position + velocity * dt
        # - call self.set_drone_pose(...)
        #
        # Hint:
        # - keep the loop per drone
        # - use np.array([...], dtype=float) for velocity

        zero_velocity = np.zeros(3, dtype=float)
        for drone in self.drones:
            current_position = self.data.qpos[drone.layout.qpos_slice][:3].copy()
            self.set_drone_pose(drone.layout, current_position, zero_velocity)

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
