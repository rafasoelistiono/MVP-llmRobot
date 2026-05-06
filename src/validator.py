from __future__ import annotations

from typing import Any

from .motion_validator import (
    CLEARANCE,
    EPSILON,
    MotionValidator as _MotionValidator,
    RolloutResult,
    RuntimeState,
    check_collision_with_placed_objects,
    compute_footprint,
    goal_satisfied,
    inside_target_area,
    is_reachable,
    runtime_state_to_state,
    validate_pick,
    validate_place,
)


class MotionValidator(_MotionValidator):
    """Backward-compatible constructor for the original module name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) >= 3 and not isinstance(args[0], dict):
            model, data, state, *rest = args
            super().__init__(state, *rest, model=model, data=data, **kwargs)
        else:
            super().__init__(*args, **kwargs)


__all__ = [
    "CLEARANCE",
    "EPSILON",
    "MotionValidator",
    "RolloutResult",
    "RuntimeState",
    "check_collision_with_placed_objects",
    "compute_footprint",
    "goal_satisfied",
    "inside_target_area",
    "is_reachable",
    "runtime_state_to_state",
    "validate_pick",
    "validate_place",
]
