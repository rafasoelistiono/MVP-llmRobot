from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from .state_extractor import OBJECT_NAMES, object_map


CLEARANCE = 0.01
EPSILON = 1e-9


@dataclass
class RuntimeState:
    held_object: str | None
    object_status: dict[str, str]
    placed_objects: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RuntimeState":
        objects = object_map(state)
        return cls(
            held_object=None,
            object_status={name: objects[name].get("status", "on_table") for name in OBJECT_NAMES},
            placed_objects=[],
        )


@dataclass
class RolloutResult:
    success: bool
    executed_successfully: list[dict[str, Any]]
    failed_action: dict[str, Any] | None
    final_state: dict[str, Any]
    goal_details: dict[str, bool] = field(default_factory=dict)


def compute_footprint(length: float, width: float, theta: float) -> tuple[float, float]:
    footprint_x = abs(length * math.cos(theta)) + abs(width * math.sin(theta))
    footprint_y = abs(length * math.sin(theta)) + abs(width * math.cos(theta))
    return footprint_x, footprint_y


def _object_size(state: dict[str, Any], object_name: str) -> list[float]:
    objects = object_map(state)
    if object_name not in objects:
        raise KeyError(object_name)
    return objects[object_name]["size"]


def _object_pose(state: dict[str, Any], object_name: str) -> list[float]:
    objects = object_map(state)
    if object_name not in objects:
        raise KeyError(object_name)
    return objects[object_name]["pose"]


def inside_target_area(
    object_name: str,
    x: float,
    y: float,
    theta: float,
    state: dict[str, Any],
) -> tuple[bool, float, float]:
    length, width, _height = _object_size(state, object_name)
    footprint_x, footprint_y = compute_footprint(length, width, theta)
    area = state["table"]["valid_placement_area"]
    inside = (
        x - footprint_x / 2 >= area["x_min"] - EPSILON
        and x + footprint_x / 2 <= area["x_max"] + EPSILON
        and y - footprint_y / 2 >= area["y_min"] - EPSILON
        and y + footprint_y / 2 <= area["y_max"] + EPSILON
    )
    return inside, footprint_x, footprint_y


def is_reachable(x: float, y: float, state: dict[str, Any]) -> tuple[bool, float, float]:
    base_x, base_y, _base_z = state["robot"]["base"]
    effective_reach = float(state["robot"]["effective_reach"])
    distance = math.sqrt((x - base_x) ** 2 + (y - base_y) ** 2)
    return distance <= effective_reach + EPSILON, distance, effective_reach


def _placed_footprint(placed_object: dict[str, Any], state: dict[str, Any]) -> tuple[float, float]:
    length, width, _height = _object_size(state, placed_object["object"])
    return compute_footprint(length, width, float(placed_object["theta"]))


def check_collision_with_placed_objects(
    object_name: str,
    x: float,
    y: float,
    theta: float,
    state: dict[str, Any],
    runtime_state: RuntimeState,
    clearance: float = CLEARANCE,
) -> tuple[bool, str | None]:
    length, width, _height = _object_size(state, object_name)
    footprint_x, footprint_y = compute_footprint(length, width, theta)
    for other in runtime_state.placed_objects:
        other_fx, other_fy = _placed_footprint(other, state)
        collides = (
            abs(x - float(other["x"])) < (footprint_x + other_fx) / 2 + clearance
            and abs(y - float(other["y"])) < (footprint_y + other_fy) / 2 + clearance
        )
        if collides:
            return True, str(other["object"])
    return False, None


def validate_pick(
    object_name: str,
    state: dict[str, Any],
    runtime_state: RuntimeState,
    step: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    objects = object_map(state)
    action_spec = {"step": step, "action": "pick", "object": object_name, "parameters": {}}
    if object_name not in objects:
        return _failure(action_spec, "invalid_action_sequence", f"{object_name} does not exist.")

    if runtime_state.object_status.get(object_name) == "placed":
        return _failure(
            action_spec,
            "invalid_action_sequence",
            f"Cannot pick {object_name} because it is already placed.",
        )

    if runtime_state.held_object is not None:
        return _failure(
            action_spec,
            "invalid_action_sequence",
            f"Cannot pick {object_name} because another object is already held.",
        )

    if runtime_state.object_status.get(object_name) == "held":
        return _failure(
            action_spec,
            "invalid_action_sequence",
            f"Cannot pick {object_name} because it is already held.",
        )

    pose = objects[object_name]["pose"]
    reachable, distance, effective_reach = is_reachable(float(pose[0]), float(pose[1]), state)
    if verbose:
        status = "OK" if reachable else "FAILED"
        print(f"[VALIDATOR] reachability: distance={distance:.3f} <= {effective_reach:.3f} {status}")
    if not reachable:
        return _failure(
            action_spec,
            "unreachable",
            f"{object_name} is outside the UR5e effective reach.",
        )

    runtime_state.held_object = object_name
    runtime_state.object_status[object_name] = "held"
    return _success(action_spec)


def validate_place(
    object_name: str,
    x: float,
    y: float,
    theta: float,
    state: dict[str, Any],
    runtime_state: RuntimeState,
    step: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    action_spec = {
        "step": step,
        "action": "place",
        "object": object_name,
        "parameters": {"x": x, "y": y, "theta": theta},
    }

    if runtime_state.held_object != object_name:
        return _failure(
            action_spec,
            "invalid_action_sequence",
            f"Cannot place {object_name} because it is not currently held.",
        )

    try:
        inside, footprint_x, footprint_y = inside_target_area(object_name, x, y, theta, state)
    except KeyError:
        return _failure(action_spec, "invalid_action_sequence", f"{object_name} does not exist.")

    if verbose:
        print(f"[VALIDATOR] footprint_x={footprint_x:.3f}, footprint_y={footprint_y:.3f}")
        print(f"[VALIDATOR] target area: {'OK' if inside else 'FAILED'}")
    if not inside:
        return _failure(
            action_spec,
            "outside_table_area",
            f"{object_name} exceeds the valid target placement area.",
        )

    reachable, distance, effective_reach = is_reachable(x, y, state)
    if verbose:
        print(
            f"[VALIDATOR] reachability: distance={distance:.3f} <= {effective_reach:.3f} "
            f"{'OK' if reachable else 'FAILED'}"
        )
    if not reachable:
        return _failure(
            action_spec,
            "unreachable",
            f"{object_name} target position is outside the UR5e effective reach. Move it closer to the robot base.",
        )

    collides, other_name = check_collision_with_placed_objects(object_name, x, y, theta, state, runtime_state)
    if verbose:
        print(f"[VALIDATOR] collision: {'FAILED' if collides else 'OK'}")
    if collides:
        return _failure(
            action_spec,
            "collision",
            f"{object_name} placement is in collision with {other_name}.",
        )

    if verbose:
        print("[VALIDATOR] IK check skipped: no IK utility found.")
        print("[VALIDATOR] Path check skipped: no planner utility found.")

    runtime_state.object_status[object_name] = "placed"
    runtime_state.held_object = None
    runtime_state.placed_objects.append(
        {
            "object": object_name,
            "x": x,
            "y": y,
            "theta": theta,
            "footprint_x": footprint_x,
            "footprint_y": footprint_y,
        }
    )
    return _success(action_spec)


def evaluate_goal(state: dict[str, Any], runtime_state: RuntimeState) -> dict[str, bool]:
    placed_status = {name: runtime_state.object_status.get(name) == "placed" for name in OBJECT_NAMES}
    inside_all = True
    reachable_all = True

    for placed_object in runtime_state.placed_objects:
        inside, _fx, _fy = inside_target_area(
            placed_object["object"],
            float(placed_object["x"]),
            float(placed_object["y"]),
            float(placed_object["theta"]),
            state,
        )
        reachable, _distance, _reach = is_reachable(float(placed_object["x"]), float(placed_object["y"]), state)
        inside_all = inside_all and inside
        reachable_all = reachable_all and reachable

    no_collision = True
    for index, obj_i in enumerate(runtime_state.placed_objects):
        fx_i, fy_i = _placed_footprint(obj_i, state)
        for obj_j in runtime_state.placed_objects[index + 1 :]:
            fx_j, fy_j = _placed_footprint(obj_j, state)
            collides = (
                abs(float(obj_i["x"]) - float(obj_j["x"])) < (fx_i + fx_j) / 2 + CLEARANCE
                and abs(float(obj_i["y"]) - float(obj_j["y"])) < (fy_i + fy_j) / 2 + CLEARANCE
            )
            no_collision = no_collision and not collides

    return {
        **{f"{name}_placed": placed for name, placed in placed_status.items()},
        "inside_target_area": inside_all and len(runtime_state.placed_objects) == len(OBJECT_NAMES),
        "no_object_collision": no_collision,
        "reachable_placements": reachable_all and len(runtime_state.placed_objects) == len(OBJECT_NAMES),
    }


def goal_satisfied(state: dict[str, Any], runtime_state: RuntimeState, verbose: bool = True) -> bool:
    details = evaluate_goal(state, runtime_state)
    if verbose:
        for name in OBJECT_NAMES:
            print(f"[GOAL] {name} placed: {str(details[f'{name}_placed']).lower()}")
        print(f"[GOAL] inside target area: {str(details['inside_target_area']).lower()}")
        print(f"[GOAL] no object collision: {str(details['no_object_collision']).lower()}")
        print(f"[GOAL] reachable placements: {str(details['reachable_placements']).lower()}")
    return all(details.values())


def runtime_state_to_state(state: dict[str, Any], runtime_state: RuntimeState) -> dict[str, Any]:
    final_state = copy.deepcopy(state)
    placed_by_name = {placed["object"]: placed for placed in runtime_state.placed_objects}
    for obj in final_state["objects"]:
        name = obj["name"]
        obj["status"] = runtime_state.object_status.get(name, obj.get("status", "on_table"))
        if name in placed_by_name:
            placed = placed_by_name[name]
            obj["pose"] = [
                round(float(placed["x"]), 4),
                round(float(placed["y"]), 4),
                round(float(placed["theta"]), 4),
            ]
    return final_state


class MotionValidator:
    def __init__(
        self,
        state: dict[str, Any],
        runtime_state: RuntimeState | None = None,
        verbose: bool = True,
        model: Any | None = None,
        data: Any | None = None,
    ) -> None:
        self.state = copy.deepcopy(state)
        self.runtime_state = runtime_state or RuntimeState.from_state(self.state)
        self.verbose = verbose
        self.model = model
        self.data = data

    def validate_pick(self, object_name: str, step: int | None = None) -> dict[str, Any]:
        return validate_pick(object_name, self.state, self.runtime_state, step=step, verbose=self.verbose)

    def validate_place(self, object_name: str, x: float, y: float, theta: float, step: int | None = None) -> dict[str, Any]:
        return validate_place(
            object_name,
            x,
            y,
            theta,
            self.state,
            self.runtime_state,
            step=step,
            verbose=self.verbose,
        )

    def validate_and_execute(self, action_spec: dict[str, Any]) -> dict[str, Any]:
        action = action_spec.get("action")
        if action == "pick":
            return self.validate_pick(action_spec["object"], step=action_spec.get("step"))
        if action == "place":
            parameters = action_spec.get("parameters", {})
            return self.validate_place(
                action_spec["object"],
                float(parameters["x"]),
                float(parameters["y"]),
                float(parameters["theta"]),
                step=action_spec.get("step"),
            )
        return _failure(action_spec, "invalid_action_sequence", f"Unsupported action {action}.")

    def rollout(self, plan: list[dict[str, Any]]) -> RolloutResult:
        executed: list[dict[str, Any]] = []
        for action_spec in plan:
            result = self.validate_and_execute(action_spec)
            if result["result"] == "success":
                executed.append(result)
                continue
            return RolloutResult(
                success=False,
                executed_successfully=executed,
                failed_action=result,
                final_state=runtime_state_to_state(self.state, self.runtime_state),
                goal_details=evaluate_goal(self.state, self.runtime_state),
            )

        details = evaluate_goal(self.state, self.runtime_state)
        if not all(details.values()):
            return RolloutResult(
                success=False,
                executed_successfully=executed,
                failed_action={
                    "step": None,
                    "action": "goal_check",
                    "object": None,
                    "parameters": {},
                    "result": "failed",
                    "failure_type": "goal_not_satisfied",
                    "failure_reason": "The full plan executed, but the final goal is not satisfied.",
                },
                final_state=runtime_state_to_state(self.state, self.runtime_state),
                goal_details=details,
            )

        return RolloutResult(
            success=True,
            executed_successfully=executed,
            failed_action=None,
            final_state=runtime_state_to_state(self.state, self.runtime_state),
            goal_details=details,
        )


def _success(action_spec: dict[str, Any]) -> dict[str, Any]:
    record = {
        "step": action_spec.get("step"),
        "action": action_spec.get("action"),
        "object": action_spec.get("object"),
        "result": "success",
    }
    if action_spec.get("action") == "place":
        record["parameters"] = action_spec.get("parameters", {})
    return record


def _failure(action_spec: dict[str, Any], failure_type: str, failure_reason: str) -> dict[str, Any]:
    return {
        "step": action_spec.get("step"),
        "action": action_spec.get("action"),
        "object": action_spec.get("object"),
        "parameters": action_spec.get("parameters", {}),
        "result": "failed",
        "failure_type": failure_type,
        "failure_reason": failure_reason,
    }
