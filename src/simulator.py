"""Trajectory export helper."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def save_trajectory_csv(trajectory: np.ndarray, timestamps: np.ndarray, output_path: Path) -> None:
    """Save trajectory as time, drone_id, x, y, z rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "drone_id", "x", "y", "z"])
        for time_index, timestamp in enumerate(timestamps):
            for drone_id in range(trajectory.shape[1]):
                x, y, z = trajectory[time_index, drone_id]
                writer.writerow([f"{timestamp:.3f}", drone_id, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
