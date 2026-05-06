from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm3_app import (
    DEFAULT_ACTION_DELAY_SECONDS,
    DEFAULT_LOG_DIR,
    DEFAULT_PROMPT,
    DemoConfig,
    LLM3PickPlaceApp,
    auto_detect_scene,
    build_prompt,
    mock_llm_response,
)


def parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def run_llm3_loop(
    scene_path: Path | None = None,
    prompt_path: Path = DEFAULT_PROMPT,
    log_dir: Path = DEFAULT_LOG_DIR,
    max_attempts: int = 5,
    mock_llm: bool = False,
    viewer: bool = True,
    action_delay: float = DEFAULT_ACTION_DELAY_SECONDS,
    arm_motion: bool = True,
    live_after_planning: bool = True,
) -> int:
    config = DemoConfig(
        scene_path=scene_path,
        prompt_path=prompt_path,
        log_dir=log_dir,
        max_attempts=max_attempts,
        mock_llm=mock_llm,
        viewer=viewer,
        action_delay=action_delay,
        arm_motion=arm_motion,
        live_after_planning=live_after_planning,
    )
    return LLM3PickPlaceApp(config).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the UR5e LLM3 TAMP tabletop pick-and-place demo.")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--mock-llm", type=parse_bool, default=False)
    parser.add_argument("--viewer", type=parse_bool, default=True)
    parser.add_argument("--action-delay", type=float, default=DEFAULT_ACTION_DELAY_SECONDS)
    parser.add_argument("--arm-motion", type=parse_bool, default=True)
    parser.add_argument("--live-after-planning", type=parse_bool, default=True)
    args = parser.parse_args(argv)

    try:
        return run_llm3_loop(
            scene_path=args.scene,
            prompt_path=args.prompt,
            log_dir=args.log_dir,
            max_attempts=args.max_attempts,
            mock_llm=args.mock_llm,
            viewer=args.viewer,
            action_delay=args.action_delay,
            arm_motion=args.arm_motion,
            live_after_planning=args.live_after_planning,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI should report a concise failure.
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


__all__ = [
    "DEFAULT_ACTION_DELAY_SECONDS",
    "DEFAULT_LOG_DIR",
    "DEFAULT_PROMPT",
    "auto_detect_scene",
    "build_prompt",
    "main",
    "mock_llm_response",
    "parse_bool",
    "run_llm3_loop",
]
