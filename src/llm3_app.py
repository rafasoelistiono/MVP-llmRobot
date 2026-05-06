from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from .feedback_trace import (
    feedback_trace_json,
    invalid_json_feedback_attempt,
    prompt_feedback_trace,
    save_json,
    validation_feedback_attempt,
)
from .gemini_llm import GeminiLLM, load_dotenv_if_present
from .motion_validator import (
    MotionValidator,
    RuntimeState,
    evaluate_goal,
    goal_satisfied,
    runtime_state_to_state,
)
from .mujoco_executor import MujocoExecutor
from .plan_parser import parse_llm_plan
from .state_extractor import (
    OBJECT_NAMES,
    extract_state_from_mujoco,
    reset_to_initial_qpos,
    save_initial_qpos,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "prompts" / "llm3_ur5e_prompt.txt"
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_ACTION_DELAY_SECONDS = 1.25
LIVE_BETWEEN_ACTION_PAUSE_SECONDS = 0.85
DEFAULT_SCENE_CANDIDATES = (
    ROOT / "scene" / "ur5e_llm3_pick_place.xml",
    ROOT / "scene.xml",
)


MOCK_ATTEMPT_1: dict[str, Any] = {
    "failure_analysis": "No previous failure.",
    "strategy": "Initial plan that accidentally places blue_box too close to red_box.",
    "plan": [
        {"step": 1, "action": "pick", "object": "red_box", "parameters": {}},
        {"step": 2, "action": "place", "object": "red_box", "parameters": {"x": 0.45, "y": -0.08, "theta": 0.00}},
        {"step": 3, "action": "pick", "object": "blue_box", "parameters": {}},
        {"step": 4, "action": "place", "object": "blue_box", "parameters": {"x": 0.46, "y": -0.06, "theta": 0.00}},
        {"step": 5, "action": "pick", "object": "green_box", "parameters": {}},
        {"step": 6, "action": "place", "object": "green_box", "parameters": {"x": 0.58, "y": 0.12, "theta": 1.57}},
    ],
}


MOCK_ATTEMPT_2: dict[str, Any] = {
    "failure_analysis": "The previous plan failed because blue_box was placed too close to red_box.",
    "strategy": "Separate the boxes and rotate the long green_box.",
    "plan": [
        {"step": 1, "action": "pick", "object": "green_box", "parameters": {}},
        {"step": 2, "action": "place", "object": "green_box", "parameters": {"x": 0.50, "y": 0.14, "theta": 1.57}},
        {"step": 3, "action": "pick", "object": "red_box", "parameters": {}},
        {"step": 4, "action": "place", "object": "red_box", "parameters": {"x": 0.42, "y": -0.10, "theta": 0.00}},
        {"step": 5, "action": "pick", "object": "blue_box", "parameters": {}},
        {"step": 6, "action": "place", "object": "blue_box", "parameters": {"x": 0.62, "y": -0.10, "theta": 0.00}},
    ],
}


@dataclass
class DemoConfig:
    scene_path: Path | None = None
    prompt_path: Path = DEFAULT_PROMPT
    log_dir: Path = DEFAULT_LOG_DIR
    max_attempts: int = 5
    mock_llm: bool = False
    viewer: bool = True
    action_delay: float = DEFAULT_ACTION_DELAY_SECONDS
    arm_motion: bool = True
    live_after_planning: bool = True


@dataclass
class DemoRuntime:
    model: mujoco.MjModel
    data: mujoco.MjData
    initial_qpos: Any
    prompt_template: str
    raw_response_dir: Path
    llm: GeminiLLM | None


@dataclass
class AttemptOutcome:
    success: bool
    plan_payload: dict[str, Any] | None = None
    plan_actions: list[dict[str, Any]] = field(default_factory=list)
    final_state: dict[str, Any] | None = None
    should_continue: bool = True


def auto_detect_scene() -> Path:
    for candidate in DEFAULT_SCENE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not auto-detect a MuJoCo scene XML.")


def build_prompt(prompt_template: str, state: dict[str, Any], feedback_trace: list[dict[str, Any]]) -> str:
    return (
        prompt_template.replace("{STATE_JSON}", json.dumps(state, indent=2))
        .replace("{FEEDBACK_TRACE_JSON}", feedback_trace_json(feedback_trace))
    )


def mock_llm_response(_prompt: str, attempt: int, feedback_trace: list[dict[str, Any]]) -> str:
    if attempt == 1 and not feedback_trace:
        return json.dumps(MOCK_ATTEMPT_1, indent=2)
    return json.dumps(MOCK_ATTEMPT_2, indent=2)


class LLM3PickPlaceApp:
    def __init__(self, config: DemoConfig) -> None:
        load_dotenv_if_present()
        self.config = self._resolve_config(config)
        self.feedback_trace: list[dict[str, Any]] = []
        self.attempts_log: list[dict[str, Any]] = []

    def run(self) -> int:
        runtime = self._setup_runtime()
        self._print_startup()

        outcome = self._planning_phase(runtime)
        if not outcome.success:
            self._save_failure_logs()
            return 1

        self._save_success_logs(outcome.plan_payload, outcome.final_state)
        if self.config.viewer and self.config.live_after_planning:
            live_ok = self._live_replay(runtime, outcome.plan_actions)
            if not live_ok:
                print("[FAILED] Live MuJoCo replay failed after a valid planning phase.")
                return 1

        return 0

    def _resolve_config(self, config: DemoConfig) -> DemoConfig:
        scene = (config.scene_path or auto_detect_scene()).resolve()
        return DemoConfig(
            scene_path=scene,
            prompt_path=config.prompt_path.resolve(),
            log_dir=config.log_dir.resolve(),
            max_attempts=config.max_attempts,
            mock_llm=config.mock_llm,
            viewer=config.viewer,
            action_delay=config.action_delay,
            arm_motion=config.arm_motion,
            live_after_planning=config.live_after_planning,
        )

    def _setup_runtime(self) -> DemoRuntime:
        assert self.config.scene_path is not None
        raw_response_dir = self.config.log_dir / "raw_llm_responses"
        raw_response_dir.mkdir(parents=True, exist_ok=True)

        model = mujoco.MjModel.from_xml_path(str(self.config.scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        llm = None if self.config.mock_llm else GeminiLLM()

        return DemoRuntime(
            model=model,
            data=data,
            initial_qpos=save_initial_qpos(data),
            prompt_template=self.config.prompt_path.read_text(encoding="utf-8"),
            raw_response_dir=raw_response_dir,
            llm=llm,
        )

    def _planning_phase(self, runtime: DemoRuntime) -> AttemptOutcome:
        planning_viewer = self.config.viewer and not self.config.live_after_planning
        executor = MujocoExecutor(
            runtime.model,
            runtime.data,
            viewer_enabled=planning_viewer,
            action_delay=self.config.action_delay,
            use_arm_motion=self.config.arm_motion and planning_viewer,
        )

        with executor.viewer_session():
            for attempt in range(1, self.config.max_attempts + 1):
                outcome = self._run_single_attempt(runtime, executor, attempt)
                if outcome.success:
                    return outcome

        print("[FAILED] Max attempts reached.")
        print(f"[LOG] feedback trace saved to {self.config.log_dir / 'feedback_trace.json'}")
        return AttemptOutcome(success=False, should_continue=False)

    def _run_single_attempt(
        self,
        runtime: DemoRuntime,
        executor: MujocoExecutor,
        attempt: int,
    ) -> AttemptOutcome:
        print(f"[ATTEMPT {attempt}/{self.config.max_attempts}] Extracting state...")
        initial_state = self._reset_and_extract_state(runtime, executor)
        _print_state_summary(initial_state)

        prompt = build_prompt(runtime.prompt_template, initial_state, self.feedback_trace)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        print(f"[PROMPT] Built prompt with feedback attempts: {len(prompt_feedback_trace(self.feedback_trace))}")

        raw_response = self._call_llm(runtime, prompt, attempt)
        raw_response_path = runtime.raw_response_dir / f"attempt_{attempt:03d}.txt"
        raw_response_path.write_text(raw_response, encoding="utf-8")

        parse_result = parse_llm_plan(raw_response)
        attempt_log = self._new_attempt_log(attempt, prompt_hash, raw_response_path, raw_response, parse_result.data)

        if not parse_result.success:
            feedback = self._handle_parse_failure(attempt, parse_result, attempt_log, initial_state)
            self._append_attempt(attempt_log, feedback)
            return AttemptOutcome(success=False)

        print(f"[PARSER] Plan parsed successfully. Number of actions: {len(parse_result.plan)}")
        rollout = self._validate_plan(runtime, executor, parse_result.plan)

        if rollout["failed_action"] is not None:
            feedback = validation_feedback_attempt(attempt, rollout["executed"], rollout["failed_action"])
            print("[TRACE] Feedback appended. Replanning...")
            attempt_log["validation_result"] = {
                "success": False,
                "executed_successfully": rollout["executed"],
                "failed_action": rollout["failed_action"],
            }
            attempt_log["feedback"] = feedback
            attempt_log["final_object_poses_after_attempt"] = _object_poses_for_log(rollout["final_state"])
            self._append_attempt(attempt_log, feedback)
            return AttemptOutcome(success=False)

        if not rollout["goal_ok"]:
            failed_action = _goal_failure_record(rollout["goal_details"])
            feedback = validation_feedback_attempt(attempt, rollout["executed"], failed_action)
            print("[RESULT] failed")
            print(f"[FAILURE] type={failed_action['failure_type']}")
            print(f"[FAILURE] reason={failed_action['failure_reason']}")
            print("[TRACE] Feedback appended. Replanning...")
            attempt_log["validation_result"] = {
                "success": False,
                "executed_successfully": rollout["executed"],
                "failed_action": failed_action,
                "goal_details": rollout["goal_details"],
            }
            attempt_log["feedback"] = feedback
            attempt_log["final_object_poses_after_attempt"] = _object_poses_for_log(rollout["final_state"])
            self._append_attempt(attempt_log, feedback)
            return AttemptOutcome(success=False)

        attempt_log["validation_result"] = {
            "success": True,
            "executed_successfully": rollout["executed"],
            "goal_details": rollout["goal_details"],
        }
        attempt_log["feedback"] = None
        attempt_log["final_object_poses_after_attempt"] = _object_poses_for_log(rollout["mujoco_final_state"])
        self.attempts_log.append(attempt_log)
        self._save_attempt_progress()

        print("[SUCCESS] LLM3 TAMP completed.")
        print(f"[LOG] final plan saved to {self.config.log_dir / 'final_plan.json'}")
        print(f"[LOG] feedback trace saved to {self.config.log_dir / 'feedback_trace.json'}")
        print(f"[LOG] attempts saved to {self.config.log_dir / 'attempts.json'}")
        executor.sync(max(1.0, self.config.action_delay))

        return AttemptOutcome(
            success=True,
            plan_payload=parse_result.data,
            plan_actions=parse_result.plan,
            final_state=rollout["mujoco_final_state"],
        )

    def _validate_plan(
        self,
        runtime: DemoRuntime,
        executor: MujocoExecutor,
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        reset_to_initial_qpos(runtime.model, runtime.data, runtime.initial_qpos)
        executor.sync(0.15)

        rollout_state = extract_state_from_mujoco(runtime.model, runtime.data)
        runtime_state = RuntimeState.from_state(rollout_state)
        validator = MotionValidator(rollout_state, runtime_state=runtime_state, verbose=True)
        executed: list[dict[str, Any]] = []
        failed_action: dict[str, Any] | None = None

        for action_spec in plan:
            result = self._validate_and_execute_action(validator, executor, action_spec)
            if result["result"] == "success":
                executed.append(result)
                print("[RESULT] success")
                continue

            failed_action = result
            print("[RESULT] failed")
            print(f"[FAILURE] type={failed_action['failure_type']}")
            print(f"[FAILURE] reason={failed_action['failure_reason']}")
            break

        final_state = runtime_state_to_state(rollout_state, runtime_state)
        goal_details = evaluate_goal(rollout_state, runtime_state)
        goal_ok = False if failed_action is not None else goal_satisfied(rollout_state, runtime_state, verbose=True)
        mujoco_final_state = extract_state_from_mujoco(runtime.model, runtime.data, runtime_state.object_status)

        return {
            "executed": executed,
            "failed_action": failed_action,
            "final_state": final_state,
            "goal_details": goal_details,
            "goal_ok": goal_ok,
            "mujoco_final_state": mujoco_final_state,
        }

    def _live_replay(self, runtime: DemoRuntime, plan: list[dict[str, Any]]) -> bool:
        print("[LIVE] Opening MuJoCo viewer and replaying final plan.")
        print(f"[LIVE] UR5e joint arm motion: {'enabled' if self.config.arm_motion else 'disabled'}")
        reset_to_initial_qpos(runtime.model, runtime.data, runtime.initial_qpos)

        executor = MujocoExecutor(
            runtime.model,
            runtime.data,
            viewer_enabled=True,
            action_delay=self.config.action_delay,
            use_arm_motion=self.config.arm_motion,
        )
        live_records: list[dict[str, Any]] = []
        live_success = False
        failed_action: dict[str, Any] | None = None
        final_state: dict[str, Any] | None = None

        with executor.viewer_session():
            executor.move_arm_home()
            rollout_state = extract_state_from_mujoco(runtime.model, runtime.data)
            runtime_state = RuntimeState.from_state(rollout_state)
            validator = MotionValidator(rollout_state, runtime_state=runtime_state, verbose=True)

            for action_spec in plan:
                result = self._validate_and_execute_live_action(validator, executor, action_spec)
                live_records.append(result)
                if result["result"] == "success":
                    print("[LIVE RESULT] success")
                    print("[LIVE] pausing before next task")
                    executor.sync(max(self.config.action_delay, LIVE_BETWEEN_ACTION_PAUSE_SECONDS))
                    continue

                failed_action = result
                print("[LIVE RESULT] failed")
                print(f"[LIVE FAILURE] type={failed_action['failure_type']}")
                print(f"[LIVE FAILURE] reason={failed_action['failure_reason']}")
                break

            if failed_action is None:
                live_success = goal_satisfied(rollout_state, runtime_state, verbose=True)
                if live_success:
                    print("[LIVE] Final plan executed in MuJoCo viewer.")
                else:
                    failed_action = _goal_failure_record(evaluate_goal(rollout_state, runtime_state))

            executor.move_arm_home()
            final_state = extract_state_from_mujoco(runtime.model, runtime.data, runtime_state.object_status)
            executor.sync(max(1.0, self.config.action_delay))

        save_json(
            self.config.log_dir / "live_execution.json",
            {
                "success": live_success,
                "arm_motion": self.config.arm_motion,
                "executed_actions": live_records,
                "failed_action": failed_action,
                "final_state": final_state,
            },
        )
        print(f"[LOG] live execution saved to {self.config.log_dir / 'live_execution.json'}")
        return live_success

    def _validate_and_execute_action(
        self,
        validator: MotionValidator,
        executor: MujocoExecutor,
        action_spec: dict[str, Any],
    ) -> dict[str, Any]:
        print(f"[ACTION {action_spec['step']}] {_format_action(action_spec)}")
        return self._validate_and_execute_common(validator, executor, action_spec)

    def _validate_and_execute_live_action(
        self,
        validator: MotionValidator,
        executor: MujocoExecutor,
        action_spec: dict[str, Any],
    ) -> dict[str, Any]:
        print(f"[LIVE ACTION {action_spec['step']}] {_format_action(action_spec)}")
        return self._validate_and_execute_common(validator, executor, action_spec, live=True)

    def _validate_and_execute_common(
        self,
        validator: MotionValidator,
        executor: MujocoExecutor,
        action_spec: dict[str, Any],
        live: bool = False,
    ) -> dict[str, Any]:
        if action_spec["action"] == "pick":
            result = validator.validate_pick(action_spec["object"], step=action_spec["step"])
            if result["result"] == "success":
                executor.pick_object(action_spec["object"])
            return result

        parameters = action_spec["parameters"]
        x = float(parameters["x"])
        y = float(parameters["y"])
        theta = float(parameters["theta"])
        result = validator.validate_place(action_spec["object"], x, y, theta, step=action_spec["step"])
        if result["result"] != "success":
            return result

        updated = executor.place_object(action_spec["object"], x, y, theta)
        if updated:
            return result

        scope = " live replay" if live else ""
        return {
            "step": action_spec["step"],
            "action": action_spec["action"],
            "object": action_spec["object"],
            "parameters": action_spec["parameters"],
            "result": "failed",
            "failure_type": "executor_pose_update_failed",
            "failure_reason": f"Could not update {action_spec['object']} pose in MuJoCo{scope}.",
        }

    def _reset_and_extract_state(self, runtime: DemoRuntime, executor: MujocoExecutor) -> dict[str, Any]:
        reset_to_initial_qpos(runtime.model, runtime.data, runtime.initial_qpos)
        executor.sync(0.15)
        return extract_state_from_mujoco(runtime.model, runtime.data)

    def _call_llm(self, runtime: DemoRuntime, prompt: str, attempt: int) -> str:
        if self.config.mock_llm:
            print("[LLM] Calling Gemini model: mock-llm")
            raw_response = mock_llm_response(prompt, attempt, self.feedback_trace)
        else:
            assert runtime.llm is not None
            raw_response = runtime.llm.generate(prompt)

        print(f"[LLM] Response received, length: {len(raw_response)}")
        print(f"[LLM] Raw response:\n{raw_response}")
        return raw_response

    def _handle_parse_failure(
        self,
        attempt: int,
        parse_result: Any,
        attempt_log: dict[str, Any],
        initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        print("[PARSER] Plan parsing failed.")
        print("[FAILURE] type=invalid_json")
        print(f"[FAILURE] reason={parse_result.error}")
        feedback = invalid_json_feedback_attempt(attempt, parse_result.errors)
        attempt_log["validation_result"] = {"success": False, "failure_type": "invalid_json"}
        attempt_log["feedback"] = feedback
        attempt_log["final_object_poses_after_attempt"] = _object_poses_for_log(initial_state)
        print("[TRACE] Feedback appended. Replanning...")
        return feedback

    def _new_attempt_log(
        self,
        attempt: int,
        prompt_hash: str,
        raw_response_path: Path,
        raw_response: str,
        parsed_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "attempt": attempt,
            "prompt_template": str(self.config.prompt_path),
            "prompt_sha256": prompt_hash,
            "raw_llm_response_path": str(raw_response_path),
            "raw_llm_response": raw_response,
            "parsed_plan": parsed_plan,
            "validation_result": None,
            "feedback": None,
            "final_object_poses_after_attempt": None,
        }

    def _append_attempt(self, attempt_log: dict[str, Any], feedback: dict[str, Any]) -> None:
        self.feedback_trace.append(feedback)
        self.attempts_log.append(attempt_log)
        self._save_attempt_progress()

    def _save_attempt_progress(self) -> None:
        save_json(self.config.log_dir / "attempts.json", self.attempts_log)
        save_json(self.config.log_dir / "feedback_trace.json", self.feedback_trace)

    def _save_success_logs(self, final_plan: dict[str, Any] | None, final_state: dict[str, Any] | None) -> None:
        save_json(self.config.log_dir / "attempts.json", self.attempts_log)
        save_json(self.config.log_dir / "feedback_trace.json", self.feedback_trace)
        if final_plan is not None:
            save_json(self.config.log_dir / "final_plan.json", final_plan)
        if final_state is not None:
            save_json(self.config.log_dir / "final_state.json", final_state)

    def _save_failure_logs(self) -> None:
        save_json(self.config.log_dir / "attempts.json", self.attempts_log)
        save_json(self.config.log_dir / "feedback_trace.json", self.feedback_trace)

    def _print_startup(self) -> None:
        model_name = "mock-llm" if self.config.mock_llm else os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        print("[LLM3] Starting UR5e tabletop pick-and-place demo")
        print(f"[LLM3] Scene: {self.config.scene_path}")
        print(f"[LLM3] Viewer: {'enabled' if self.config.viewer else 'disabled'}")
        print(f"[LLM3] Live mode: {'after planning logs' if self.config.live_after_planning else 'during attempts'}")
        print(f"[LLM3] Arm motion: {'enabled' if self.config.arm_motion else 'disabled'}")
        print(f"[LLM3] Model: {model_name}")
        print(f"[LLM3] Max attempts: {self.config.max_attempts}")


def _print_state_summary(state: dict[str, Any]) -> None:
    objects = {obj["name"]: obj for obj in state["objects"]}
    for name in OBJECT_NAMES:
        pose = objects[name]["pose"]
        print(f"[STATE] {name} pose: x={pose[0]:.3f}, y={pose[1]:.3f}, theta={pose[2]:.3f}")


def _format_action(action_spec: dict[str, Any]) -> str:
    if action_spec["action"] == "pick":
        return f"pick({action_spec['object']})"
    parameters = action_spec["parameters"]
    return (
        f"place({action_spec['object']}, x={float(parameters['x']):.2f}, "
        f"y={float(parameters['y']):.2f}, theta={float(parameters['theta']):.2f})"
    )


def _object_poses_for_log(state: dict[str, Any]) -> dict[str, Any]:
    return {obj["name"]: {"pose": obj["pose"], "status": obj["status"]} for obj in state["objects"]}


def _goal_failure_record(details: dict[str, bool]) -> dict[str, Any]:
    failed_checks = [name for name, passed in details.items() if not passed]
    return {
        "step": None,
        "action": "goal_check",
        "object": None,
        "parameters": {},
        "result": "failed",
        "failure_type": "goal_not_satisfied",
        "failure_reason": f"Final goal checks failed: {', '.join(failed_checks)}.",
    }
