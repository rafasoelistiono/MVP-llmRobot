"""Simple MuJoCo trajectory player for 4 freejoint drone bodies."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from scene_utils import find_body_ids, load_model_and_data, try_import_mujoco

NUM_DRONES = 4
DRONE_IDS = [0, 1, 2, 3]
IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


class TrajectoryPlayer:
    """Read trajectory.csv, write positions to MuJoCo freejoints, then run."""

    def __init__(
        self,
        scene_path: Path,
        trajectory_path: Path,
        body_names: list[str],
        playback_speed: float = 1.0,
    ) -> None:
        self.scene_path = scene_path
        self.trajectory_path = trajectory_path
        self.body_names = body_names
        self.playback_speed = max(float(playback_speed), 1e-6)
        self.mujoco: Any | None = None
        self.model: Any | None = None
        self.data: Any | None = None
        self.freejoint_qpos_addrs: list[int] = []
        self.timestamps: np.ndarray | None = None
        self.trajectory: np.ndarray | None = None

    def load_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Load outputs/trajectory.csv into arrays."""
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory CSV not found: {self.trajectory_path}")

        rows: list[dict[str, float | int]] = []
        with self.trajectory_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required = {"time", "drone_id", "x", "y", "z"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Trajectory CSV must contain columns: time, drone_id, x, y, z")

            for row in reader:
                rows.append(
                    {
                        "time": float(row["time"]),
                        "drone_id": int(row["drone_id"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]),
                    }
                )

        if not rows:
            raise ValueError("Trajectory CSV is empty")

        timestamps = np.array(sorted({float(row["time"]) for row in rows}), dtype=float)
        drone_ids = sorted({int(row["drone_id"]) for row in rows})
        if drone_ids != DRONE_IDS:
            raise ValueError(f"Trajectory must contain drone_id 0, 1, 2, 3. Found: {drone_ids}")

        trajectory = np.zeros((len(timestamps), NUM_DRONES, 3), dtype=float)
        seen: set[tuple[float, int]] = set()
        time_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}

        for row in rows:
            timestamp = float(row["time"])
            drone_id = int(row["drone_id"])
            seen.add((timestamp, drone_id))
            trajectory[time_to_index[timestamp], drone_id] = [float(row["x"]), float(row["y"]), float(row["z"])]

        for timestamp in timestamps:
            for drone_id in DRONE_IDS:
                if (float(timestamp), drone_id) not in seen:
                    raise ValueError(f"Missing sample for time={timestamp:.3f}, drone_id={drone_id}")

        self.timestamps = timestamps
        self.trajectory = trajectory
        return trajectory, timestamps

    def setup_mujoco(self) -> None:
        """Load scene and find freejoint qpos addresses for the drone bodies."""
        if len(self.body_names) != NUM_DRONES:
            raise ValueError("Exactly 4 body names are required")

        self.mujoco = try_import_mujoco()
        self.model, self.data = load_model_and_data(self.scene_path)
        body_ids = find_body_ids(self.model, self.body_names)
        self.freejoint_qpos_addrs = []

        for body_name in self.body_names:
            qpos_addr = self._find_freejoint_qpos_addr(body_ids[body_name], body_name)
            self.freejoint_qpos_addrs.append(qpos_addr)

    def play(self, render: bool = True) -> None:
        """Apply trajectory frames to MuJoCo and optionally show the viewer."""
        if self.trajectory is None or self.timestamps is None:
            self.load_trajectory()
        if self.model is None or self.data is None:
            self.setup_mujoco()

        if render:
            self._play_with_viewer()
            return

        assert self.trajectory is not None
        assert self.mujoco is not None
        for positions in self.trajectory:
            self._apply_positions(positions)
            self.mujoco.mj_forward(self.model, self.data)

    def _play_with_viewer(self) -> None:
        """Run playback with MuJoCo passive viewer."""
        assert self.trajectory is not None
        assert self.timestamps is not None
        assert self.mujoco is not None

        try:
            import mujoco.viewer
        except ImportError as exc:
            raise ImportError("mujoco.viewer is not available in this environment") from exc

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            previous_time = float(self.timestamps[0])
            for index, positions in enumerate(self.trajectory):
                if not viewer.is_running():
                    break

                self._apply_positions(positions)
                self.mujoco.mj_forward(self.model, self.data)
                viewer.sync()

                current_time = float(self.timestamps[index])
                dt = max(current_time - previous_time, 0.0)
                previous_time = current_time
                if dt > 0.0:
                    time.sleep(dt / self.playback_speed)

    def _apply_positions(self, positions: np.ndarray) -> None:
        """Write one frame of drone positions into freejoint qpos."""
        assert self.data is not None

        for drone_id, qpos_addr in enumerate(self.freejoint_qpos_addrs):
            self.data.qpos[qpos_addr : qpos_addr + 3] = positions[drone_id]
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = IDENTITY_QUATERNION

    def _find_freejoint_qpos_addr(self, body_id: int, body_name: str) -> int:
        """Find the qpos address of a body's freejoint."""
        assert self.model is not None
        assert self.mujoco is not None

        joint_count = int(self.model.body_jntnum[body_id])
        joint_start = int(self.model.body_jntadr[body_id])
        for offset in range(joint_count):
            joint_id = joint_start + offset
            if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[joint_id])

        raise ValueError(
            f"Body '{body_name}' does not have a freejoint. "
            "Use drone bodies with <freejoint> so trajectory playback can set qpos."
        )
