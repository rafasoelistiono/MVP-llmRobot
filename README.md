# Mini SwarmGPT 4-Drone Planner with MuJoCo Trajectory Execution

Mini SwarmGPT adalah MVP sederhana untuk lab robot planning. Project ini terinspirasi dari SwarmGPT, tetapi dibuat tanpa Crazyflow, tanpa API berbayar, dan tanpa LLM asli. Versi minimal ini fokus pada satu kebutuhan utama: prompt masuk, lalu `outputs/trajectory.csv` dibuat untuk MuJoCo execution.

## Scope

- 4 drone only.
- Drone ID selalu `0`, `1`, `2`, dan `3`.
- Rule-based LLM-style planner.
- Trajectory generation dengan Python dan NumPy.
- Output utama hanya `outputs/trajectory.csv`.
- Optional MuJoCo trajectory playback memakai `mujoco_execution/swarm_scene.xml`.

File lama Progress 2 seperti `main.py`, `solution.py`, `swarm_scene.xml`, `logs/`, dan `mujoco_menagerie-main/` tetap ada di repository. Project Mini SwarmGPT dijalankan melalui `src/main.py`.

## Architecture

```txt
User prompt
-> Prompt Planner
-> 4-Drone JSON Plan
-> Motion Primitive Generator
-> Trajectory Generator
-> outputs/trajectory.csv
-> Optional MuJoCo Trajectory Playback
```

## Install Main Project

Linux, WSL, atau macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Dependency utama sengaja minimal:

```txt
numpy
```

## Optional MuJoCo

MuJoCo tidak masuk `requirements.txt` utama agar planner tetap ringan. Untuk trajectory playback di MuJoCo:

```bash
pip install mujoco
```

## Run Planner

```bash
python src/main.py --prompt "buat 4 drone membentuk lingkaran dan berputar pelan"
python src/main.py --prompt "make the drones form a circle and rotate slowly"
python src/main.py --prompt "buat drone bergerak seperti gelombang" --dt 0.05
```

## Run MuJoCo Playback

```bash
python mujoco_execution/run_mujoco_trajectory.py --trajectory outputs/trajectory.csv --scene mujoco_execution/swarm_scene.xml
```

Jika body name di scene berbeda:

```bash
python mujoco_execution/run_mujoco_trajectory.py --body-names drone0,drone1,drone2,drone3
```

Untuk validasi tanpa membuka viewer:

```bash
python mujoco_execution/run_mujoco_trajectory.py --trajectory outputs/trajectory.csv --scene mujoco_execution/swarm_scene.xml --no-render
```

## Output

Output planner masuk ke `outputs/`.

- `outputs/trajectory.csv`: trajectory dengan format `time,drone_id,x,y,z`.

File ini adalah input utama untuk MuJoCo execution. File report dan plot tidak dibuat oleh pipeline minimal ini.

## Supported Plans

Formations:

- `square`
- `circle`
- `line`
- `diamond`

Primitives:

- `hover`
- `rotate`
- `rise`
- `wave`
- `spiral`
- `move_forward`

Speeds:

- `slow`
- `normal`
- `fast`
