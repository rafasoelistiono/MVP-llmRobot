"""MuJoCo scene helpers with clear validation errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def try_import_mujoco() -> Any:
    """Import MuJoCo or raise a clear optional-dependency message."""
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("MuJoCo is not installed. Install it with: pip install mujoco") from exc
    return mujoco


def list_body_names(model: Any) -> list[str]:
    """Return all named bodies in a MuJoCo model."""
    mujoco = try_import_mujoco()
    names: list[str] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            names.append(name)
    return names


def find_body_ids(model: Any, body_names: list[str]) -> dict[str, int]:
    """Find MuJoCo body IDs by name and report available names on failure."""
    mujoco = try_import_mujoco()
    body_ids: dict[str, int] = {}
    missing: list[str] = []

    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            missing.append(body_name)
        else:
            body_ids[body_name] = int(body_id)

    if missing:
        available = ", ".join(list_body_names(model)) or "<no named bodies>"
        raise ValueError(
            "Could not find requested MuJoCo body name(s): "
            f"{', '.join(missing)}. Available body names: {available}"
        )

    return body_ids


def load_model_and_data(scene_path: Path) -> tuple[Any, Any]:
    """Load a MuJoCo model and data from an XML scene path."""
    mujoco = try_import_mujoco()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    return model, data
