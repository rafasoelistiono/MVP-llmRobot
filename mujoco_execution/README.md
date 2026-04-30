# MuJoCo Trajectory Execution

Folder ini berisi runner sederhana untuk memainkan `outputs/trajectory.csv` di MuJoCo.

Alur kerja:

```txt
outputs/trajectory.csv
-> load trajectory
-> load mujoco_execution/swarm_scene.xml
-> cari body drone_0 sampai drone_3
-> tulis posisi ke freejoint qpos
-> mujoco.mj_forward()
-> viewer.sync()
```

Cara menjalankan:

```bash
python mujoco_execution/run_mujoco_trajectory.py --trajectory outputs/trajectory.csv --scene mujoco_execution/swarm_scene.xml
```

Validasi tanpa viewer:

```bash
python mujoco_execution/run_mujoco_trajectory.py --trajectory outputs/trajectory.csv --scene mujoco_execution/swarm_scene.xml --no-render
```

Scene harus punya 4 body dengan freejoint. Default body names:

```txt
drone_0,drone_1,drone_2,drone_3
```

Jika nama body berbeda:

```bash
python mujoco_execution/run_mujoco_trajectory.py --body-names drone0,drone1,drone2,drone3
```
