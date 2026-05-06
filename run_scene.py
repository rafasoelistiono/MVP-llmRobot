import time
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a passive MuJoCo viewer for a scene XML.")
    parser.add_argument("scene", nargs="?", default="scene.xml", help="Scene XML path to load.")
    args = parser.parse_args()

    xml_path = Path(args.scene)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
