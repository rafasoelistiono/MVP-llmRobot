# Spesifikasi Aplikasi LLM3 TAMP UR5e Pick-and-Place

## 1. Ringkasan

Aplikasi ini adalah demo Task and Motion Planning berbasis LLM3 untuk simulasi pick-and-place tabletop memakai MuJoCo. Sistem membaca state scene UR5e, membangun prompt planning, memanggil Gemini atau mock LLM, memvalidasi plan secara deterministik, mengeksekusi hasil valid di MuJoCo, memberi feedback jika gagal, lalu melakukan replanning sampai goal tercapai atau batas attempt habis.

Versi saat ini sudah mendukung loop LLM3 end-to-end pada level task planning dan geometric motion validation. Alur dibuat dua fase: sistem membuat log attempt dan final plan terlebih dahulu, lalu membuka MuJoCo viewer untuk live replay final plan. Pada live replay, UR5e menjalankan gerakan joint-space heuristik supaya arm terlihat berkolaborasi dengan plan pick-and-place. Gerakan ini belum memakai IK numerik penuh dan belum merupakan physical grasp yang presisi.

## 1.1 Model Mental Sederhana

Aplikasi dapat dipahami sebagai 5 tahap:

1. **Plan**: baca state MuJoCo, buat prompt, panggil Gemini atau mock LLM.
2. **Validate**: parser mengecek JSON, validator mengecek urutan action, target area, reachability, dan collision.
3. **Feedback**: jika gagal, failure disimpan ke feedback trace dan dikirim ke attempt berikutnya.
4. **Log**: jika plan valid, sistem menyimpan `attempts.json`, `feedback_trace.json`, `final_plan.json`, dan `final_state.json`.
5. **Replay**: viewer dibuka, final plan dijalankan ulang; arm bergerak dan object mengikuti gripper sampai dilepas di goal.

Entry utama sekarang sederhana:

```text
src/run_llm3_ur5e_demo.py
  -> src/llm3_tamp_loop.py
      -> LLM3PickPlaceApp.run()
```

`LLM3PickPlaceApp` berisi fase yang mudah dibaca:

```text
run()
  setup runtime
  planning phase
  save logs
  live replay
```

## 2. Tujuan Aplikasi

Tujuan utama:

1. Menyediakan loop LLM3 TAMP lengkap untuk tabletop pick-and-place.
2. Menggunakan Gemini sebagai task planner dan continuous parameter sampler.
3. Memvalidasi output LLM sebelum dieksekusi.
4. Menghasilkan feedback terstruktur saat plan gagal.
5. Memakai feedback tersebut untuk replanning dari initial state.
6. Menampilkan perubahan object pose di MuJoCo viewer.
7. Menyimpan log attempt, response LLM, feedback, final plan, dan final state.

Tujuan lanjutan:

1. Menambahkan IK UR5e untuk pose pre-grasp, grasp, lift, pre-place, dan place.
2. Menambahkan trajectory generation dan joint controller.
3. Membuat objek mengikuti end-effector saat held.
4. Mengaktifkan gripper secara fisik jika scene mendukung finger joints dan actuator.

## 3. Scope

In scope versi saat ini:

1. Load MuJoCo scene UR5e tabletop.
2. Extract state robot, table, object pose, object size, dan goal.
3. Build prompt dari state dan feedback trace.
4. Call Gemini memakai `GEMINI_API_KEY` dan `GEMINI_MODEL`.
5. Mock LLM mode untuk test tanpa API call.
6. Parse JSON output LLM, termasuk fenced JSON dan teks tambahan.
7. Validate pick-place sequence.
8. Validate target area, reachability, dan collision antar placed objects.
9. Kinematic object pose update di MuJoCo setelah valid place.
10. Replanning berbasis feedback.
11. Goal checker dan log file.
12. Live replay setelah planning selesai.
13. Visual joint motion UR5e berbasis heuristic waypoint untuk pick, carry, place, dan retreat.

Out of scope versi saat ini:

1. IK solver aktual.
2. Collision-free trajectory planner aktual.
3. Dynamic grasp constraint fisik antara gripper dan objek.
4. Physical gripper open-close.
5. Full physics-based manipulation.
6. Contact-accurate grasp dan release.

## 4. Arsitektur Modul

| Modul | Tanggung jawab |
| --- | --- |
| `src/run_llm3_ur5e_demo.py` | Entry point CLI untuk menjalankan demo. |
| `src/llm3_tamp_loop.py` | Facade CLI tipis yang membangun config dan menjalankan app. |
| `src/llm3_app.py` | Orkestrasi utama yang dibagi per fase: setup, planning, attempt, validation, logging, live replay. |
| `src/gemini_llm.py` | Client Gemini, baca env var, fallback REST API, error handling. |
| `src/state_extractor.py` | Extract state dari MuJoCo, save initial qpos, reset qpos. |
| `src/plan_parser.py` | Extract dan validasi JSON plan dari response LLM. |
| `src/motion_validator.py` | Validasi pick/place, runtime symbolic state, collision, reachability, goal check. |
| `src/mujoco_executor.py` | Passive viewer session, object pose update, viewer sync. |
| `src/arm_motion_executor.py` | Visual UR5e joint-motion replay untuk pick, carry, place, dan home. |
| `src/feedback_trace.py` | Format feedback, prompt trace limit, save JSON. |
| `prompts/llm3_ur5e_prompt.txt` | Template prompt planning untuk Gemini. |
| `logs/` | Output attempt, raw response, feedback, final plan, final state. |

Compatibility wrappers:

1. `src/llm3_loop.py` meneruskan ke loop baru.
2. `src/validator.py` meneruskan ke `motion_validator.py`.
3. `src/feedback.py` meneruskan ke `feedback_trace.py`.
4. `src/run_ur5e_llm3_demo.py` tetap bisa dipakai sebagai entry lama.

## 5. Alur Kerja End-to-End

Alur implementasi saat ini sengaja dibuat linear:

```text
load scene
save initial qpos
for each attempt:
  reset to initial qpos
  extract state
  build prompt
  call LLM/mock
  parse JSON plan
  validate full plan
  save attempt log
  if failed: append feedback and continue
  if success: save final plan/state and stop planning
if viewer enabled:
  reset to initial qpos
  replay final plan live with arm-object attachment
```

### 5.1 Startup

1. User menjalankan:

```bash
python -m src.run_llm3_ur5e_demo --mock-llm true --viewer true
```

atau:

```bash
python -m src.run_llm3_ur5e_demo --viewer true
```

2. CLI membaca argumen:
   - `--scene`
   - `--max-attempts`
   - `--mock-llm`
   - `--viewer`
   - `--action-delay`
   - `--prompt`
   - `--log-dir`

3. Jika `--scene` kosong, sistem auto-detect scene:
   - `scene/ur5e_llm3_pick_place.xml`
   - fallback ke `scene.xml`

4. Sistem load MuJoCo model dan data.
5. Sistem menyimpan `initial_qpos`.
6. Jika viewer enabled, passive MuJoCo viewer dibuka.

### 5.2 State Extraction

1. Sistem reset data ke `initial_qpos`.
2. Sistem menjalankan `mujoco.mj_forward`.
3. Sistem membaca pose object:
   - `red_box`
   - `blue_box`
   - `green_box`
4. Sistem membuat state JSON:

```json
{
  "robot": {
    "name": "UR5e",
    "base": [0.0, 0.0, 0.0],
    "max_reach": 0.85,
    "effective_reach": 0.75,
    "workspace": "tabletop",
    "gripper": "two_finger_gripper"
  },
  "table": {
    "name": "target_table",
    "height": 0.4,
    "valid_placement_area": {
      "x_min": 0.35,
      "x_max": 0.70,
      "y_min": -0.25,
      "y_max": 0.25
    }
  },
  "objects": [
    {
      "name": "red_box",
      "size": [0.08, 0.08, 0.04],
      "pose": [0.3, -0.3, 0.0],
      "status": "on_table"
    }
  ],
  "goal": "Place all objects inside the valid placement area on the target table without collision and with feasible UR5e motions."
}
```

### 5.3 Prompt Construction

1. Sistem membaca `prompts/llm3_ur5e_prompt.txt`.
2. Placeholder `{STATE_JSON}` diganti dengan state JSON.
3. Placeholder `{FEEDBACK_TRACE_JSON}` diganti dengan feedback trace.
4. Hanya 3 feedback attempt terakhir dimasukkan ke prompt.
5. Full feedback trace tetap disimpan di disk.

### 5.4 LLM Planning

Mode Gemini:

1. Sistem membaca `.env` atau environment.
2. Required:
   - `GEMINI_API_KEY`
3. Optional:
   - `GEMINI_MODEL`, default `gemini-1.5-pro`
4. Sistem memanggil Gemini.
5. Response mentah disimpan ke `logs/raw_llm_responses/attempt_XXX.txt`.

Mode mock:

1. Attempt 1 sengaja menghasilkan collision.
2. Attempt 2 menghasilkan plan valid.
3. Mode ini wajib dipakai untuk regression test tanpa API.

### 5.5 Plan Parsing

Parser menerima response LLM yang bisa berupa:

1. Pure JSON.
2. Markdown fenced JSON.
3. JSON dengan teks tambahan di luar object JSON.

Parser wajib memvalidasi:

1. Top-level object JSON.
2. Key wajib:
   - `failure_analysis`
   - `strategy`
   - `plan`
3. `plan` harus list.
4. Step harus integer berurutan mulai dari 1.
5. Action hanya:
   - `pick`
   - `place`
6. Object hanya:
   - `red_box`
   - `blue_box`
   - `green_box`
7. `pick` wajib punya `parameters: {}`.
8. `place` wajib punya numeric:
   - `x`
   - `y`
   - `theta`

Jika parsing gagal:

1. Sistem membuat feedback `invalid_json`.
2. Feedback ditambahkan ke trace.
3. Attempt berikutnya meminta LLM replan.

### 5.6 Reset Sebelum Rollout

Sebelum setiap full-plan rollout:

1. MuJoCo data direset.
2. `data.qpos` dikembalikan ke `initial_qpos`.
3. `data.qvel` direset ke nol.
4. `mujoco.mj_forward` dipanggil.

Tujuannya: setiap LLM attempt dievaluasi dari initial state yang sama.

### 5.7 Motion Validation

Runtime symbolic state:

```json
{
  "held_object": null,
  "object_status": {
    "red_box": "on_table",
    "blue_box": "on_table",
    "green_box": "on_table"
  },
  "placed_objects": []
}
```

Validasi `pick(object)`:

1. Object harus ada.
2. Object tidak boleh sudah placed.
3. Tidak boleh ada object lain yang sedang held.
4. Object harus reachable dari base:

```text
distance = sqrt((x - base_x)^2 + (y - base_y)^2)
distance <= effective_reach
```

Jika sukses:

1. `held_object = object`.
2. `object_status[object] = "held"`.
3. Executor mencetak bahwa object held.

Validasi `place(object, x, y, theta)`:

1. Object harus sedang held.
2. Footprint object harus fully inside target area.
3. Target harus reachable.
4. Target tidak boleh collision dengan placed objects.
5. IK check dilewati jika tidak ada utility.
6. Path check dilewati jika tidak ada planner.

Footprint:

```text
footprint_x = abs(length * cos(theta)) + abs(width * sin(theta))
footprint_y = abs(length * sin(theta)) + abs(width * cos(theta))
```

Collision antar object:

```text
abs(x_i - x_j) < (footprint_x_i + footprint_x_j) / 2 + clearance
AND
abs(y_i - y_j) < (footprint_y_i + footprint_y_j) / 2 + clearance
```

Jika sukses:

1. `object_status[object] = "placed"`.
2. `held_object = None`.
3. Object ditambahkan ke `placed_objects`.
4. MuJoCo object pose diupdate via freejoint.

### 5.8 MuJoCo Execution

Planning phase:

1. `pick` hanya mengubah symbolic state dan log.
2. `place` mengubah pose freejoint object:
   - `x = target x`
   - `y = target y`
   - `z = table_height + object_height / 2`
   - `quat = yaw_to_quat(theta)`
3. Viewer di-sync setelah action.
4. Object terlihat berpindah posisi.

Live replay phase:

1. Viewer dibuka setelah attempt log dan final plan disimpan.
2. Final plan dieksekusi ulang dari initial state.
3. Arm UR5e bergerak melalui 6 joint utama:
   - `shoulder_pan_joint`
   - `shoulder_lift_joint`
   - `elbow_joint`
   - `wrist_1_joint`
   - `wrist_2_joint`
   - `wrist_3_joint`
4. Pick replay:
   - arm bergerak ke pre-grasp
   - arm turun ke grasp
   - gripper close masih symbolic
   - object diangkat secara kinematik
5. Place replay:
   - arm membawa object ke pre-place
   - object mengikuti lintasan carry sederhana
   - arm menurunkan object ke table
   - gripper open masih symbolic
   - arm retreat

Catatan penting:

Gerakan arm saat ini adalah heuristic visual joint planning, bukan IK penuh. Sistem belum memiliki:

1. IK solver.
2. Joint trajectory planner.
3. Controller execution.
4. Grasp attachment.
5. Physical gripper actuation.

### 5.9 Feedback dan Replanning

Jika action gagal:

1. Rollout berhenti.
2. Sistem mencatat action yang sukses sebelum gagal.
3. Sistem mencatat failed action:

```json
{
  "step": 4,
  "action": "place",
  "object": "blue_box",
  "parameters": {
    "x": 0.46,
    "y": -0.06,
    "theta": 0.0
  },
  "result": "failed",
  "failure_type": "collision",
  "failure_reason": "blue_box placement is in collision with red_box."
}
```

4. Feedback ditambahkan ke trace:

```json
{
  "attempt": 1,
  "executed_successfully": [],
  "failed_action": {}
}
```

5. Attempt berikutnya memakai initial state yang sama dan feedback trace.

### 5.10 Goal Checking

Goal satisfied jika:

1. Semua object status `placed`.
2. Semua footprint berada di dalam target area.
3. Tidak ada object-object collision.
4. Semua placement reachable.

Sistem mencetak:

```text
[GOAL] red_box placed: true
[GOAL] blue_box placed: true
[GOAL] green_box placed: true
[GOAL] inside target area: true
[GOAL] no object collision: true
[GOAL] reachable placements: true
```

### 5.11 Logging

File log wajib:

1. `logs/attempts.json`
2. `logs/feedback_trace.json`
3. `logs/final_plan.json`
4. `logs/final_state.json`
5. `logs/raw_llm_responses/attempt_XXX.txt`
6. `logs/live_execution.json`

Setiap attempt mencatat:

1. Attempt number.
2. Prompt template path.
3. Prompt SHA-256.
4. Raw LLM response path.
5. Raw LLM response.
6. Parsed plan.
7. Validation result.
8. Feedback jika gagal.
9. Final object poses setelah attempt.

## 6. Functional Requirements

### FR-001 Scene Loading

Sistem harus dapat load scene MuJoCo dari path CLI atau auto-detect default scene.

Acceptance:

1. `python -m src.run_llm3_ur5e_demo --mock-llm true --viewer false` berhasil load scene.
2. Error scene path harus jelas jika file tidak ditemukan atau XML invalid.

### FR-002 Viewer

Sistem harus dapat membuka passive MuJoCo viewer jika `--viewer true`.

Acceptance:

1. Viewer terbuka dan tetap terlihat selama demo.
2. Object pose update terlihat setelah place sukses.
3. Terminal tetap mencetak log.

### FR-003 State Extraction

Sistem harus mengekstrak robot, table, objects, pose, status, dan goal.

Acceptance:

1. State JSON berisi `red_box`, `blue_box`, `green_box`.
2. Pose object berbentuk `[x, y, theta]`.
3. Jika theta tidak tersedia, theta default `0.0`.

### FR-004 Prompt Builder

Sistem harus membangun prompt dari template, state, dan feedback trace.

Acceptance:

1. `{STATE_JSON}` terganti.
2. `{FEEDBACK_TRACE_JSON}` terganti.
3. Prompt hanya memuat 3 feedback attempt terakhir.

### FR-005 Gemini Client

Sistem harus bisa memanggil Gemini.

Acceptance:

1. `GEMINI_API_KEY` wajib ada untuk non-mock.
2. `GEMINI_MODEL` dipakai jika ada.
3. Jika env var hilang, error harus eksplisit.
4. Empty response dianggap error.
5. API error ditampilkan jelas.

### FR-006 Mock LLM

Sistem harus mendukung mock mode untuk test.

Acceptance:

1. Attempt 1 gagal collision.
2. Attempt 2 sukses.
3. Tidak membutuhkan API key.

### FR-007 Plan Parser

Sistem harus robust terhadap output LLM yang tidak bersih.

Acceptance:

1. Pure JSON diterima.
2. Markdown fenced JSON diterima.
3. JSON dengan teks sebelum atau sesudah diterima.
4. Invalid schema menghasilkan feedback `invalid_json`.

### FR-008 Motion Validation

Sistem harus memvalidasi plan action-by-action.

Acceptance:

1. Invalid sequence gagal dengan `invalid_action_sequence`.
2. Outside area gagal dengan `outside_table_area`.
3. Unreachable gagal dengan `unreachable`.
4. Collision gagal dengan `collision`.
5. IK dan path planner skipped jika belum ada utility.

### FR-009 MuJoCo Object Execution

Sistem harus memindahkan object setelah place valid.

Acceptance:

1. Object freejoint qpos berubah.
2. Object z dihitung dari table height dan object height.
3. Viewer sync setelah update.
4. Jika object tidak punya freejoint, sistem memberi warning jelas.

### FR-010 Feedback Trace

Sistem harus membuat feedback terstruktur saat gagal.

Acceptance:

1. Feedback mencatat executed actions.
2. Feedback mencatat failed action.
3. Feedback disimpan ke `logs/feedback_trace.json`.
4. Attempt berikutnya menerima feedback tersebut.

### FR-011 Goal Checker

Sistem harus mengecek final goal setelah rollout sukses.

Acceptance:

1. Semua object harus placed.
2. Semua object harus inside target area.
3. Tidak boleh ada collision.
4. Semua placement harus reachable.

### FR-012 Logging

Sistem harus menyimpan semua log utama.

Acceptance:

1. `attempts.json` dibuat.
2. `feedback_trace.json` dibuat.
3. `final_plan.json` dibuat saat sukses.
4. `final_state.json` dibuat saat sukses.
5. Raw response disimpan per attempt.

## 7. Non-Functional Requirements

### NFR-001 Determinism

Validator harus deterministik. Untuk input state dan plan yang sama, hasil validasi harus sama.

### NFR-002 Observability

Terminal log harus menjelaskan:

1. Attempt.
2. State object.
3. LLM call.
4. Parser result.
5. Action.
6. Validator checks.
7. Success atau failure.
8. Feedback.
9. Goal checks.
10. Log paths.

### NFR-003 Modularity

Setiap concern harus terpisah:

1. LLM client.
2. Parser.
3. State extractor.
4. Validator.
5. Executor.
6. Loop orchestration.
7. Feedback/logging.

### NFR-004 Robustness

Sistem harus tidak crash untuk:

1. LLM response invalid.
2. Missing keys.
3. Wrong action.
4. Wrong object name.
5. Missing parameters.
6. Failed validation.

### NFR-005 Reproducibility

Setiap attempt harus reset ke initial qpos agar feedback-based replanning bisa dibandingkan secara fair.

### NFR-006 Minimal External Dependency

Core demo harus bisa berjalan mock mode hanya dengan MuJoCo dan standard Python dependency yang sudah ada.

## 8. System Requirements

### 8.1 Software

Minimum:

1. Python 3.10 atau lebih baru.
2. MuJoCo Python package.
3. Network access untuk Gemini non-mock mode.

Optional:

1. `python-dotenv`, jika ingin load `.env`.
2. `google-genai` atau `google-generativeai`.

Catatan:

Jika Gemini SDK tidak tersedia, client memakai REST API fallback via Python standard library.

### 8.2 Environment Variables

Untuk Gemini mode:

```text
GEMINI_API_KEY=<api-key>
GEMINI_MODEL=<model-name>
```

`GEMINI_MODEL` optional. Default saat ini `gemini-1.5-pro`.

Untuk mock mode:

```text
GEMINI_API_KEY tidak diperlukan.
```

### 8.3 Scene Requirements

Scene harus memiliki:

1. UR5e robot arm.
2. Two-finger gripper body atau visual.
3. Table body.
4. Target placement area marker.
5. Object bodies:
   - `red_box`
   - `blue_box`
   - `green_box`
6. Object freejoint:
   - `red_box_freejoint`
   - `blue_box_freejoint`
   - `green_box_freejoint`

### 8.4 CLI Requirements

Command utama:

```bash
python -m src.run_llm3_ur5e_demo
```

Command direct script:

```bash
python src/run_llm3_ur5e_demo.py
```

Arguments:

| Argumen | Default | Deskripsi |
| --- | --- | --- |
| `--scene` | auto-detect | Path scene XML. |
| `--prompt` | `prompts/llm3_ur5e_prompt.txt` | Path prompt template. |
| `--log-dir` | `logs` | Folder output log. |
| `--max-attempts` | `5` | Batas replanning attempt. |
| `--mock-llm` | `false` | Pakai mock response. |
| `--viewer` | `true` | Buka MuJoCo viewer. |
| `--action-delay` | `0.75` | Delay setelah action agar terlihat di viewer. |

## 9. Data Contracts

### 9.1 LLM Output Schema

```json
{
  "failure_analysis": "No previous failure.",
  "strategy": "Brief strategy.",
  "plan": [
    {
      "step": 1,
      "action": "pick",
      "object": "green_box",
      "parameters": {}
    },
    {
      "step": 2,
      "action": "place",
      "object": "green_box",
      "parameters": {
        "x": 0.50,
        "y": 0.14,
        "theta": 1.57
      }
    }
  ]
}
```

### 9.2 Feedback Trace Schema

```json
{
  "attempt": 1,
  "executed_successfully": [
    {
      "step": 1,
      "action": "pick",
      "object": "red_box",
      "result": "success"
    }
  ],
  "failed_action": {
    "step": 4,
    "action": "place",
    "object": "blue_box",
    "parameters": {
      "x": 0.46,
      "y": -0.06,
      "theta": 0.0
    },
    "result": "failed",
    "failure_type": "collision",
    "failure_reason": "blue_box placement is in collision with red_box."
  }
}
```

### 9.3 Failure Types

Supported failure types:

1. `invalid_json`
2. `invalid_action_sequence`
3. `outside_table_area`
4. `unreachable`
5. `collision`
6. `no_ik_solution`
7. `no_collision_free_path`
8. `executor_pose_update_failed`
9. `goal_not_satisfied`

## 10. Acceptance Criteria End-to-End

### AC-001 Mock Demo

Command:

```bash
python -m src.run_llm3_ur5e_demo --mock-llm true --viewer true
```

Expected:

1. Viewer terbuka.
2. Attempt 1 berjalan.
3. Mock response attempt 1 diparse.
4. Red box berhasil placed.
5. Blue box place gagal collision dengan red box.
6. Feedback trace dibuat.
7. Attempt 2 berjalan.
8. Green, red, blue placed tanpa collision.
9. Goal satisfied.
10. Logs tersimpan.

### AC-002 Gemini Demo

Command:

```bash
python -m src.run_llm3_ur5e_demo --viewer true
```

Precondition:

1. `GEMINI_API_KEY` valid.
2. `GEMINI_MODEL` valid atau default model tersedia.

Expected:

1. Gemini menerima prompt.
2. Response diparse.
3. Plan divalidasi.
4. Jika gagal, feedback dikirim pada attempt berikutnya.
5. Loop berhenti saat sukses atau max attempts.

### AC-003 Direct Script

Command:

```bash
python src/run_llm3_ur5e_demo.py --mock-llm true --viewer false
```

Expected:

1. Script berjalan dari path langsung.
2. Exit code 0 saat mock attempt 2 sukses.

## 11. Testing Requirements

Regression tests minimal:

1. Parser accepts pure JSON.
2. Parser accepts markdown fenced JSON.
3. Parser rejects missing `parameters`.
4. Validator rejects collision placement.
5. Validator rejects outside target area.
6. Validator rejects unreachable placement.
7. Mock loop succeeds in 2 attempts.
8. Log files created.

Manual tests:

1. Run viewer enabled.
2. Confirm object positions visibly update.
3. Run Gemini mode with `.env`.
4. Inspect `logs/attempts.json`.
5. Inspect `logs/feedback_trace.json`.

## 12. Requirements Untuk Real Arm Motion

Bagian ini adalah requirement lanjutan agar robot arm benar-benar bergerak, bukan hanya object teleport.

### ARM-FR-001 End-Effector Site

Scene harus punya site end-effector yang jelas, misalnya:

```text
attachment_site
```

Sistem harus bisa membaca world pose site tersebut dari MuJoCo.

### ARM-FR-002 IK Solver

Sistem harus memiliki IK solver untuk UR5e.

Input:

1. Target position `[x, y, z]`.
2. Target orientation.
3. Initial qpos seed.
4. Joint limits.

Output:

1. Joint qpos solution.
2. Success/failure.
3. Residual error.

Required IK targets:

1. Home.
2. Pre-grasp.
3. Grasp.
4. Lift.
5. Pre-place.
6. Place.
7. Retreat.

### ARM-FR-003 Trajectory Generation

Sistem harus membuat trajectory joint-space antar qpos.

Minimum:

1. Linear interpolation joint-space.
2. Velocity limit.
3. Acceleration smoothing optional.
4. Collision check optional pada setiap waypoint.

### ARM-FR-004 Controller Execution

Sistem harus mengeksekusi trajectory ke actuator UR5e.

Minimum:

1. Set `data.ctrl[:]` mengikuti target qpos.
2. Step MuJoCo sampai target tracking error kecil.
3. Viewer sync selama motion.
4. Timeout jika robot tidak mencapai target.

### ARM-FR-005 Object Attachment

Saat object held, object harus mengikuti gripper.

Opsi implementasi:

1. Update object pose kinematik mengikuti end-effector setiap simulation step.
2. Tambah equality weld constraint runtime.
3. Pakai mocap body sebagai grasp frame.

Acceptance:

1. Object bergerak bersama end-effector setelah grasp.
2. Object dilepas di target pose saat place.

### ARM-FR-006 Gripper Actuation

Jika scene memiliki finger joints:

1. Gripper open sebelum approach.
2. Gripper close saat grasp.
3. Gripper open saat release.

Jika scene belum punya finger joints:

1. Sistem boleh memakai symbolic gripper state.
2. Log harus menjelaskan gripper actuation skipped.

### ARM-FR-007 Arm Motion Failure Feedback

Failure baru harus masuk feedback trace:

1. `no_ik_solution`
2. `joint_limit_violation`
3. `trajectory_timeout`
4. `path_collision`
5. `grasp_failed`
6. `place_release_failed`

Feedback harus cukup spesifik agar Gemini bisa replan, misalnya:

```json
{
  "failure_type": "no_ik_solution",
  "failure_reason": "green_box place target at x=0.72, y=0.22 has no IK solution. Move closer to base and away from positive y boundary."
}
```

## 13. Known Limitations Saat Ini

1. Arm UR5e bergerak secara visual dengan heuristic joint waypoint, belum IK numerik.
2. Planning/validation masih geometric; IK check dan path check masih skipped.
3. Gripper open-close masih symbolic.
4. Object mengikuti arm secara kinematik saat live replay, belum weld/contact fisik.
5. Collision validator hanya footprint 2D antar placed objects.
6. Tidak ada collision checking robot-object atau robot-table aktual.
7. Tidak ada physical grasp stability.

## 14. Roadmap Implementasi

Phase 1 - Current:

1. LLM3 loop.
2. Gemini/mock planner.
3. Parser robust.
4. Geometric validator.
5. Object pose execution.
6. Feedback replanning.
7. Logs.

Phase 2 - Arm kinematic motion:

1. Add IK solver.
2. Add joint interpolation.
3. Move UR5e arm to pre-grasp/place poses.
4. Object follows gripper kinematically.
5. Keep geometric validation.

Phase 3 - Motion planning:

1. Collision check along trajectory.
2. Planner fallback if straight joint interpolation fails.
3. Better feedback for IK/path failures.

Phase 4 - Physical manipulation:

1. Gripper joints and actuator.
2. Contact-aware grasp.
3. Release stability.
4. Full robot-object collision simulation.

## 15. Definition of Done

Versi saat ini dianggap done jika:

1. Mock demo succeeds in 2 attempts.
2. Viewer opens when enabled.
3. Object visibly moves after valid place.
4. Attempt 1 mock fails with collision feedback.
5. Attempt 2 mock succeeds.
6. Gemini mode can run with valid env vars.
7. All required log files are written.
8. Direct script and module execution both work.

Versi real arm motion dianggap done jika:

1. UR5e joint qpos changes visibly during pick and place.
2. End-effector reaches pre-grasp and place poses via IK.
3. Object follows gripper while held.
4. Object is released at target pose.
5. Motion failures become structured feedback for Gemini.
6. Viewer shows continuous robot motion, not object teleport only.
