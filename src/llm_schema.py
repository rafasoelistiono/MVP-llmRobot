"""Schema helpers for the Mini SwarmGPT rule-based planner."""

from __future__ import annotations

from typing import Any

NUM_DRONES = 4
DRONE_IDS = [0, 1, 2, 3]

SUPPORTED_FORMATIONS = ["square", "circle", "line", "diamond"]
SUPPORTED_PRIMITIVES = ["hover", "rotate", "rise", "wave", "spiral", "move_forward"]
SUPPORTED_SPEEDS = ["slow", "normal", "fast"]


def get_default_plan() -> dict[str, Any]:
    """Return the default 4-drone planning dictionary."""
    return {
        "num_drones": NUM_DRONES,
        "drone_ids": DRONE_IDS.copy(),
        "formation": "square",
        "primitive": "hover",
        "speed": "normal",
        "speed_multiplier": 1.0,
        "height": 0.8,
        "duration": 8.0,
        "radius": 1.0,
        "spacing": 0.8,
        "amplitude": 0.25,
    }


def build_llm_prompt(user_prompt: str) -> str:
    """Build a future LLM prompt template while keeping this MVP rule-based."""
    return f"""
You are a drone swarm planning assistant. Convert the user request into valid JSON only.

Rules:
- The swarm must always contain exactly 4 drones.
- Drone IDs must be 0, 1, 2, and 3.
- Keep x and y inside [-3.0, 3.0].
- Keep z inside [0.3, 2.0].
- Minimum inter-drone distance must be at least 0.25 meter.
- Supported formations: {SUPPORTED_FORMATIONS}.
- Supported primitives: {SUPPORTED_PRIMITIVES}.
- Supported speeds: {SUPPORTED_SPEEDS}.
- Do not include text outside the JSON object.

User prompt:
{user_prompt}
""".strip()


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields and clamp values into the supported planning schema."""
    cleaned = get_default_plan()
    cleaned.update(plan)

    cleaned["num_drones"] = NUM_DRONES
    cleaned["drone_ids"] = DRONE_IDS.copy()

    if cleaned.get("formation") not in SUPPORTED_FORMATIONS:
        cleaned["formation"] = "square"
    if cleaned.get("primitive") not in SUPPORTED_PRIMITIVES:
        cleaned["primitive"] = "hover"
    if cleaned.get("speed") not in SUPPORTED_SPEEDS:
        cleaned["speed"] = "normal"

    speed_multipliers = {"slow": 0.5, "normal": 1.0, "fast": 1.5}
    cleaned["speed_multiplier"] = speed_multipliers[cleaned["speed"]]

    cleaned["duration"] = _positive_float(cleaned.get("duration"), 8.0)
    cleaned["height"] = float(min(max(_positive_float(cleaned.get("height"), 0.8), 0.3), 2.0))
    cleaned["radius"] = _positive_float(cleaned.get("radius"), 1.0)
    cleaned["spacing"] = _positive_float(cleaned.get("spacing"), 0.8)
    cleaned["amplitude"] = _positive_float(cleaned.get("amplitude"), 0.25)
    return cleaned


def _positive_float(value: Any, fallback: float) -> float:
    """Convert a value to positive float or return fallback."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return fallback
    return converted if converted > 0.0 else fallback
