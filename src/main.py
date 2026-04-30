from __future__ import annotations

import argparse
from pathlib import Path

from motion_primitives import generate_trajectory
from prompt_planner import parse_prompt
from simulator import save_trajectory_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    """Parse a prompt and save only trajectory.csv for MuJoCo execution."""
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = parse_prompt(args.prompt)
    trajectory, timestamps = generate_trajectory(plan, dt=args.dt)
    trajectory_path = OUTPUT_DIR / "trajectory.csv"
    save_trajectory_csv(trajectory, timestamps, trajectory_path)

    _print_summary(args.prompt, plan, trajectory_path, len(timestamps))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate 4-drone trajectory.csv for MuJoCo")
    parser.add_argument("--prompt", required=True, help="Indonesian or English swarm prompt")
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory timestep in seconds")
    return parser.parse_args()


def _print_summary(prompt: str, plan: dict, trajectory_path: Path, num_steps: int) -> None:
    """Print a compact trajectory generation summary."""
    print("Mini SwarmGPT 4-Drone Trajectory Generator")
    print()
    print("Prompt:")
    print(prompt)
    print()
    print("Generated Plan:")
    print(f"- formation: {plan['formation']}")
    print(f"- primitive: {plan['primitive']}")
    print(f"- speed: {plan['speed']}")
    print(f"- height: {plan['height']}")
    print(f"- duration: {plan['duration']}")
    print(f"- timesteps: {num_steps}")
    print()
    print("File generated:")
    print(f"- {trajectory_path.relative_to(PROJECT_ROOT).as_posix()}")
    print()
    print("Run MuJoCo trajectory playback:")
    print("python mujoco_execution/run_mujoco_trajectory.py --trajectory outputs/trajectory.csv --scene mujoco_execution/swarm_scene.xml")


if __name__ == "__main__":
    main()
