"""Trajectory generation for 4-drone motion primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

NUM_DRONES = 4


def get_initial_positions(plan: dict[str, Any]) -> np.ndarray:
    """Return initial positions with shape (4, 3) for the requested formation."""
    formation = str(plan.get("formation", "square"))
    height = float(plan.get("height", 0.8))
    spacing = float(plan.get("spacing", 0.8))
    radius = float(plan.get("radius", 1.0))

    if formation == "circle":
        angles = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
        xy = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    elif formation == "line":
        x = (np.arange(NUM_DRONES) - 1.5) * spacing
        xy = np.column_stack((x, np.zeros(NUM_DRONES)))
    elif formation == "diamond":
        xy = np.array([[0.0, spacing], [spacing, 0.0], [0.0, -spacing], [-spacing, 0.0]])
    else:
        half = spacing / 2.0
        xy = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])

    z = np.full((NUM_DRONES, 1), height)
    return np.column_stack((xy, z))


def generate_trajectory(plan: dict[str, Any], dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory array with shape (T, 4, 3)."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    duration = float(plan.get("duration", 8.0))
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    initial = get_initial_positions(plan)
    primitive = str(plan.get("primitive", "hover"))

    if primitive == "rotate":
        trajectory = _rotate(initial, timestamps, plan)
    elif primitive == "rise":
        trajectory = _rise(initial, timestamps, plan)
    elif primitive == "wave":
        trajectory = _wave(initial, timestamps, plan)
    elif primitive == "spiral":
        trajectory = _spiral(initial, timestamps, plan)
    elif primitive == "move_forward":
        trajectory = _move_forward(initial, timestamps, plan)
    else:
        trajectory = np.repeat(initial[None, :, :], len(timestamps), axis=0)

    return trajectory, timestamps


def _rotate(initial: np.ndarray, timestamps: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    speed_multiplier = float(plan.get("speed_multiplier", 1.0))
    angular_speed = 0.45 * speed_multiplier
    trajectory = np.zeros((len(timestamps), NUM_DRONES, 3), dtype=float)

    radii = np.linalg.norm(initial[:, :2], axis=1)
    angles0 = np.arctan2(initial[:, 1], initial[:, 0])
    for index, timestamp in enumerate(timestamps):
        angles = angles0 + angular_speed * timestamp
        trajectory[index, :, 0] = radii * np.cos(angles)
        trajectory[index, :, 1] = radii * np.sin(angles)
        trajectory[index, :, 2] = initial[:, 2]
    return trajectory


def _rise(initial: np.ndarray, timestamps: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    target_height = min(float(plan.get("height", 0.8)) + 0.5, 2.0)
    progress = _smooth_progress(timestamps / max(timestamps[-1], 1e-6))
    trajectory = np.repeat(initial[None, :, :], len(timestamps), axis=0)
    trajectory[:, :, 2] = 0.5 + (target_height - 0.5) * progress[:, None]
    return trajectory


def _wave(initial: np.ndarray, timestamps: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    amplitude = min(float(plan.get("amplitude", 0.25)), 0.3)
    height = float(plan.get("height", 0.8))
    speed_multiplier = float(plan.get("speed_multiplier", 1.0))
    omega = 1.0 * speed_multiplier
    phases = np.linspace(0.0, 2.0 * np.pi, NUM_DRONES, endpoint=False)
    trajectory = np.repeat(initial[None, :, :], len(timestamps), axis=0)

    for drone_id, phase in enumerate(phases):
        trajectory[:, drone_id, 2] = height + amplitude * np.sin(omega * timestamps + phase)
    return trajectory


def _spiral(initial: np.ndarray, timestamps: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    speed_multiplier = float(plan.get("speed_multiplier", 1.0))
    height = float(plan.get("height", 0.8))
    angular_speed = 0.35 * speed_multiplier
    base_radii = np.maximum(np.linalg.norm(initial[:, :2], axis=1), 0.5)
    angles0 = np.arctan2(initial[:, 1], initial[:, 0])
    progress = timestamps / max(timestamps[-1], 1e-6)
    radius_scale = 0.85 + 0.25 * progress
    z_profile = np.clip(height + 0.2 * np.sin(2.0 * np.pi * progress), 0.3, 2.0)

    trajectory = np.zeros((len(timestamps), NUM_DRONES, 3), dtype=float)
    for index, timestamp in enumerate(timestamps):
        angles = angles0 + angular_speed * timestamp
        radii = base_radii * radius_scale[index]
        trajectory[index, :, 0] = radii * np.cos(angles)
        trajectory[index, :, 1] = radii * np.sin(angles)
        trajectory[index, :, 2] = z_profile[index]
    return trajectory


def _move_forward(initial: np.ndarray, timestamps: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    speed_multiplier = float(plan.get("speed_multiplier", 1.0))
    forward_speed = 0.12 * speed_multiplier
    trajectory = np.repeat(initial[None, :, :], len(timestamps), axis=0)
    trajectory[:, :, 0] += forward_speed * timestamps[:, None]
    return trajectory


def _smooth_progress(progress: np.ndarray) -> np.ndarray:
    """Smoothly map progress from 0..1 to 0..1."""
    clipped = np.clip(progress, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)
