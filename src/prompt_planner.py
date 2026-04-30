"""Rule-based Indonesian and English prompt parser."""

from __future__ import annotations

from typing import Any

from llm_schema import validate_plan


def parse_prompt(prompt: str) -> dict[str, Any]:
    """Parse a short user prompt into a validated 4-drone plan."""
    text = prompt.lower().strip()
    plan: dict[str, Any] = {"num_drones": 4}

    formation_set = True
    if _contains_any(text, ["circle", "lingkaran", "circular"]):
        plan["formation"] = "circle"
    elif _contains_any(text, ["square", "kotak", "persegi"]):
        plan["formation"] = "square"
    elif _contains_any(text, ["line", "garis"]):
        plan["formation"] = "line"
    elif _contains_any(text, ["diamond", "berlian", "belah ketupat"]):
        plan["formation"] = "diamond"
    else:
        formation_set = False
        plan["formation"] = "square"

    if _contains_any(text, ["rotate", "berputar", "putar"]):
        plan["primitive"] = "rotate"
    elif _contains_any(text, ["rise", "naik", "up"]):
        plan["primitive"] = "rise"
    elif _contains_any(text, ["wave", "gelombang"]):
        plan["primitive"] = "wave"
    elif "spiral" in text:
        plan["primitive"] = "spiral"
        if not formation_set:
            plan["formation"] = "circle"
    elif _contains_any(text, ["forward", "maju"]):
        plan["primitive"] = "move_forward"
    else:
        plan["primitive"] = "hover"

    if plan["primitive"] == "spiral" and not formation_set:
        plan["formation"] = "circle"
    if plan["primitive"] == "wave" and not formation_set:
        plan["formation"] = "line"

    if _contains_any(text, ["slow", "pelan", "perlahan"]):
        plan["speed"] = "slow"
    elif _contains_any(text, ["fast", "cepat"]):
        plan["speed"] = "fast"
    else:
        plan["speed"] = "normal"

    return validate_plan(plan)


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return true when any keyword appears in text."""
    return any(keyword in text for keyword in keywords)
