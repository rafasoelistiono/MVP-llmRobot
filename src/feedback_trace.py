from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validation_feedback_attempt(
    attempt: int,
    executed_successfully: list[dict[str, Any]],
    failed_action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "executed_successfully": executed_successfully,
        "failed_action": failed_action,
    }


def invalid_json_feedback_attempt(attempt: int, errors: list[dict[str, Any]]) -> dict[str, Any]:
    reason = "; ".join(error.get("message", str(error)) for error in errors) or "Invalid JSON response."
    return {
        "attempt": attempt,
        "executed_successfully": [],
        "failed_action": {
            "step": None,
            "action": "parse_plan",
            "object": None,
            "parameters": {},
            "result": "failed",
            "failure_type": "invalid_json",
            "failure_reason": reason,
            "parser_errors": errors,
        },
    }


def prompt_feedback_trace(feedback_trace: list[dict[str, Any]], max_attempts: int = 3) -> list[dict[str, Any]]:
    return feedback_trace[-max_attempts:]


def feedback_trace_json(feedback_trace: list[dict[str, Any]], max_attempts: int = 3) -> str:
    return json.dumps(prompt_feedback_trace(feedback_trace, max_attempts), indent=2)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
