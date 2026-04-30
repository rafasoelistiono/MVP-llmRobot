"""Command-line runner for Mini SwarmGPT MuJoCo trajectory playback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from trajectory_player import TrajectoryPlayer

DEFAULT_BODY_NAMES = ["drone_0", "drone_1", "drone_2", "drone_3"]


def main() -> None:
    """Validate inputs and run trajectory playback."""
    args = _parse_args()

    if not args.trajectory.exists():
        raise SystemExit(f"Trajectory file not found: {args.trajectory}")
    if not args.scene.exists():
        raise SystemExit(f"Scene XML not found: {args.scene}")

    body_names = [name.strip() for name in args.body_names.split(",") if name.strip()]
    try:
        player = TrajectoryPlayer(
            scene_path=args.scene,
            trajectory_path=args.trajectory,
            body_names=body_names,
            playback_speed=args.playback_speed,
        )
        player.load_trajectory()
        player.setup_mujoco()
        print("Mini SwarmGPT MuJoCo Trajectory Execution")
        print(f"- trajectory: {args.trajectory}")
        print(f"- scene: {args.scene}")
        print(f"- body names: {', '.join(body_names)}")
        print(f"- render: {str(not args.no_render).lower()}")
        player.play(render=not args.no_render)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Play Mini SwarmGPT trajectory in MuJoCo")
    parser.add_argument("--trajectory", type=Path, default=PROJECT_ROOT / "outputs" / "trajectory.csv")
    parser.add_argument("--scene", type=Path, default=CURRENT_DIR / "swarm_scene.xml")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--body-names", default=",".join(DEFAULT_BODY_NAMES))
    parser.add_argument("--no-render", action="store_true", help="Validate and play without opening the viewer")
    return parser.parse_args()


if __name__ == "__main__":
    main()
