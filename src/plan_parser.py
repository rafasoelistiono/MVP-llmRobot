from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


ALLOWED_ACTIONS = {"pick", "place"}
ALLOWED_OBJECTS = {"red_box", "blue_box", "green_box"}
REQUIRED_TOP_LEVEL_KEYS = ("failure_analysis", "strategy", "plan")


@dataclass
class PlanParseResult:
    success: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    raw_response: str = ""
    data: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.success


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the first decodable JSON object, tolerating prose around it."""
    candidates = [strip_markdown_fences(text)]

    raw = text.strip()
    fence_start = raw.find("```")
    while fence_start >= 0:
        content_start = raw.find("\n", fence_start)
        fence_end = raw.find("```", content_start + 1 if content_start >= 0 else fence_start + 3)
        if content_start >= 0 and fence_end >= 0:
            candidates.append(raw[content_start + 1 : fence_end].strip())
            fence_start = raw.find("```", fence_end + 3)
        else:
            break

    decoder = json.JSONDecoder()
    last_error: str | None = None
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError as exc:
                last_error = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
                continue
            if isinstance(obj, dict):
                return obj, None
            last_error = "First decoded JSON value was not an object."

    return None, last_error or "No JSON object was found in the LLM response."


def _structured_error(error_type: str, message: str, step: int | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"error_type": error_type, "message": message}
    if step is not None:
        error["step"] = step
    return error


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _failure(raw_response: str, errors: list[dict[str, Any]], data: dict[str, Any] | None = None) -> PlanParseResult:
    return PlanParseResult(
        success=False,
        plan=[],
        error="; ".join(error["message"] for error in errors),
        raw_response=raw_response,
        data=data,
        errors=errors,
    )


def parse_llm_plan(llm_output: str) -> PlanParseResult:
    data, extraction_error = extract_first_json_object(llm_output)
    if data is None:
        return _failure(
            llm_output,
            [_structured_error("invalid_json", f"LLM output does not contain valid JSON: {extraction_error}")],
        )

    errors: list[dict[str, Any]] = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            errors.append(_structured_error("invalid_schema", f'Top-level key "{key}" is required.'))

    plan = data.get("plan")
    if not isinstance(plan, list):
        errors.append(_structured_error("invalid_schema", '"plan" must be a list.'))
        return _failure(llm_output, errors, data)

    previous_step = 0
    for index, action_spec in enumerate(plan, start=1):
        if not isinstance(action_spec, dict):
            errors.append(_structured_error("invalid_action", "Every plan entry must be an object.", index))
            continue

        step = action_spec.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            errors.append(_structured_error("invalid_step", "Every action must have an integer step.", index))
            step_for_error = index
        else:
            step_for_error = step
            if step != previous_step + 1:
                errors.append(_structured_error("invalid_step", "Plan steps must be ordered and sequential.", step))
            previous_step = step

        action = action_spec.get("action")
        if action not in ALLOWED_ACTIONS:
            errors.append(_structured_error("invalid_action", 'Action must be either "pick" or "place".', step_for_error))

        object_name = action_spec.get("object")
        if object_name not in ALLOWED_OBJECTS:
            errors.append(
                _structured_error("invalid_object", "Object must be red_box, blue_box, or green_box.", step_for_error)
            )

        if "parameters" not in action_spec:
            errors.append(_structured_error("invalid_parameters", 'Action key "parameters" is required.', step_for_error))
            parameters = {}
        else:
            parameters = action_spec["parameters"]
        if not isinstance(parameters, dict):
            errors.append(_structured_error("invalid_parameters", "Action parameters must be an object.", step_for_error))
            continue

        if action == "pick" and parameters != {}:
            errors.append(_structured_error("invalid_parameters", "pick actions must have empty parameters.", step_for_error))

        if action == "place":
            missing = [name for name in ("x", "y", "theta") if name not in parameters]
            if missing:
                errors.append(
                    _structured_error(
                        "invalid_parameters",
                        f"place action is missing numeric parameters: {', '.join(missing)}.",
                        step_for_error,
                    )
                )
                continue

            for parameter_name in ("x", "y", "theta"):
                if not _is_number(parameters[parameter_name]):
                    errors.append(
                        _structured_error(
                            "invalid_parameters",
                            f'place parameter "{parameter_name}" must be a finite number.',
                            step_for_error,
                        )
                    )

    if errors:
        return _failure(llm_output, errors, data)

    return PlanParseResult(
        success=True,
        plan=plan,
        error=None,
        raw_response=llm_output,
        data=data,
        errors=[],
    )
