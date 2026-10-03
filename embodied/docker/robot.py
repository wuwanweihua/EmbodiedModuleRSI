"""Robot API for LIBERO-Pro harness tasks (stable layer, baked into the image).

A policy program never talks to MuJoCo directly. It receives a `Robot` instance
whose methods are the control primitives. Every call is blocking, returns
``{"ok": bool, "detail": str}``, appends one JSON line to the structured trace,
and (rate-limited) saves a camera frame that the harness attaches to the
agent's next observation.

CLI (installed on PATH by the image):

    robot-run <policy.py>   reset to the task's fixed initial state, import the
                            policy file, call ``run(robot)``, print the summary
    robot-scene             print the initial scene JSON (no policy)
    robot-selftest          run a canned smoke script against the API

Environment (set per task by task.toml / test.sh):

    LIBERO_BDDL          path to the task's BDDL file
    LIBERO_INIT_FILE     path to the init-state array (.npy / .pruned_init)
    LIBERO_INIT_STATE    index into the init-state array (default 0)
    ROBOT_FRAMES_DIR     where frames are written (default /logs/agent/frames)
    ROBOT_BUDGET         environment-step budget per run (default 600)
    ROBOT_CAMERAS        comma-separated camera names (default agentview)
    MUJOCO_GL            osmesa | egl (set by the image; osmesa = CPU rendering)

Deliberately NOT exposed: the task's success predicate. Success is judged by
the verifier on a separate pristine run; policies verify progress with
``scene()``/``object_pose()`` instead.

NOTE (server-side verification): this file is the fixed API layer the harness
evolves *around*. Its primitive implementations use a proportional OSC servo
(no external IK dependency); verify the workspace bounds and grasp heights
against the actual LIBERO-Pro scenes before trusting the numbers.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FRAMES_DIR = Path(os.environ.get("ROBOT_FRAMES_DIR", "/logs/agent/frames"))
BUDGET = int(os.environ.get("ROBOT_BUDGET", "600"))
CAMERAS = [
    c.strip()
    for c in os.environ.get("ROBOT_CAMERAS", "agentview").split(",")
    if c.strip()
]

# Servo step clamp for `goto` (metres per env step) and tolerance.
_MAX_STEP = 0.05
_TOL = 0.01
_GRASP_DOWN_RPY = (np.pi, 0.0, 0.0)  # end-effector pointing down (LIBERO convention)

# Approximate reachable workspace of the Franka in LIBERO tabletop scenes,
# in world coordinates. `check_reach` uses it as a fast pre-filter.
_WORKSPACE = {"x": (-0.45, 0.55), "y": (-0.55, 0.55), "z": (0.02, 0.75)}


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------


def load_env():
    """Build the LIBERO env for this task and reset to the fixed init state."""
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.environ.get("LIBERO_BDDL")
    if not bddl:
        raise RuntimeError("LIBERO_BDDL is not set")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        camera_names=CAMERAS,
    )
    env.seed(0)
    env.reset()

    init_file = os.environ.get("LIBERO_INIT_FILE")
    if init_file:
        states = np.load(init_file, allow_pickle=True)
        # Some LIBERO releases wrap the array in a 0-d object array.
        if states.dtype == object and states.shape == ():
            states = states.item()
        idx = int(os.environ.get("LIBERO_INIT_STATE", "0"))
        if idx >= len(states):
            raise RuntimeError(f"init state {idx} out of range (n={len(states)})")
        env.set_init_state(states[idx])
        env.reset()
    return env


# ---------------------------------------------------------------------------
# Robot API
# ---------------------------------------------------------------------------


class Robot:
    """Control primitives for one episode. All motion calls are blocking."""

    def __init__(self, env, frames_dir: Path = FRAMES_DIR, budget: int = BUDGET):
        self._env = env
        self._frames_dir = Path(frames_dir)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._budget = budget
        self._steps = 0
        self._frames_saved = 0
        self._last_frame_step = -10
        self._trace: list[dict] = []

    # ---------- trace ----------

    def _record(self, call: str, ok: bool, detail: str = "") -> dict:
        row = {"call": call, "ok": bool(ok), "detail": detail, "step": self._steps}
        self._trace.append(row)
        print(json.dumps(row), flush=True)
        return row

    def log(self, msg: str) -> None:
        """Add a free-form note to the trace."""
        self._record("log", True, str(msg))

    # ---------- scene ----------

    def instruction(self) -> str:
        return os.environ.get("LIBERO_INSTRUCTION", "")

    def _object_names(self) -> list[str]:
        names = list(self._env.sim.model.body_names)
        skip = {
            "world",
            "table",
            "robot0_base",
            "robot0_base_main",
            "robot0_link0",
        }
        return [n for n in names if n not in skip and not n.startswith("robot0_")]

    def object_pose(self, name: str) -> dict:
        """Pose and size of one object. Raises KeyError when not present."""
        body_id = self._env.sim.model.body_name2id(name)
        pos = np.array(self._env.sim.data.body_xpos[body_id])
        quat = np.array(self._env.sim.data.body_xquat[body_id])
        size = np.array(self._env.sim.model.geom_size[self._env.sim.model.body_geomadr[body_id]])
        return {"name": name, "pos": pos.round(4).tolist(), "quat": quat.round(4).tolist(),
                "size": size.round(4).tolist()}

    def objects(self) -> list[str]:
        return self._object_names()

    def scene(self) -> dict:
        """Full scene snapshot: objects, gripper state, step counter, budget."""
        objects = []
        for name in self._object_names():
            try:
                objects.append(self.object_pose(name))
            except Exception:
                continue
        eef = self._eef_pos()
        gripper = self._gripper_width()
        return {
            "instruction": self.instruction(),
            "objects": objects,
            "gripper": {"pos": eef.round(4).tolist(), "width": round(gripper, 4)},
            "step": self._steps,
            "budget": self._budget,
        }

    # ---------- low-level state helpers ----------

    def _eef_pos(self) -> np.ndarray:
        return np.array(self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id])

    def _gripper_width(self) -> float:
        try:
            return float(self._env.robots[0].gripper.current_action[0])
        except Exception:
            return float("nan")

    def _is_grasped(self, name: str) -> bool:
        """Object lifted off the table while the gripper is closed."""
        try:
            pose = self.object_pose(name)
            eef = self._eef_pos()
            return bool(np.linalg.norm(np.array(pose["pos"]) - eef) < 0.08)
        except Exception:
            return False

    # ---------- stepping ----------

    def _step(self, action: np.ndarray, save_frame: bool = False) -> None:
        if self._steps >= self._budget:
            raise RuntimeError(f"step budget exhausted ({self._budget})")
        self._env.step(action)
        self._steps += 1
        if save_frame and self._steps - self._last_frame_step >= 5:
            self.save_frame()

    def _servo_to(self, target: np.ndarray, tol: float = _TOL, timeout: int = 120) -> dict:
        """Proportional OSC servo: step the EEF toward an absolute position."""
        deadline = self._steps + timeout
        while self._steps < deadline:
            current = self._eef_pos()
            delta = np.asarray(target, dtype=float) - current
            if np.linalg.norm(delta) <= tol:
                return {"ok": True, "detail": f"reached within {tol}"}
            action = np.zeros(7)
            action[:3] = np.clip(delta, -_MAX_STEP, _MAX_STEP)
            action[6] = -1.0  # keep the gripper closed-state unchanged
            self._step(action, save_frame=True)
        return {"ok": False, "detail": f"timeout after {timeout} steps"}

    # ---------- motion primitives ----------

    def goto(self, pos, tol: float = _TOL, timeout: int = 120) -> dict:
        """Move the end-effector to an absolute xyz position."""
        result = self._servo_to(np.asarray(pos, dtype=float), tol=tol, timeout=timeout)
        return self._record("goto", result["ok"], result["detail"])

    def move_rel(self, delta, timeout: int = 60) -> dict:
        """Relative end-effector move."""
        target = self._eef_pos() + np.asarray(delta, dtype=float)
        result = self._servo_to(target, timeout=timeout)
        return self._record("move_rel", result["ok"], result["detail"])

    def rotate_eef(self, rpy_delta) -> dict:
        """Rotate the end-effector by an axis-angle delta (radians)."""
        action = np.zeros(7)
        action[3:6] = np.asarray(rpy_delta, dtype=float)
        action[6] = -1.0
        try:
            self._step(action, save_frame=True)
        except Exception as exc:
            return self._record("rotate_eef", False, str(exc))
        return self._record("rotate_eef", True, "applied")

    def open_gripper(self) -> dict:
        action = np.zeros(7)
        action[6] = 1.0
        try:
            for _ in range(5):
                self._step(action)
        except Exception as exc:
            return self._record("open_gripper", False, str(exc))
        return self._record("open_gripper", True, "opened")

    def close_gripper(self) -> dict:
        action = np.zeros(7)
        action[6] = -1.0
        try:
            for _ in range(10):
                self._step(action)
        except Exception as exc:
            return self._record("close_gripper", False, str(exc))
        return self._record("close_gripper", True, "closed")

    def check_reach(self, pos) -> dict:
        """Approximate IK feasibility: is the target inside the workspace box?"""
        p = np.asarray(pos, dtype=float)
        ok = all(
            _WORKSPACE[axis][0] <= p[i] <= _WORKSPACE[axis][1]
            for i, axis in enumerate(("x", "y", "z"))
        )
        return self._record("check_reach", ok, "inside workspace" if ok else "outside workspace")

    def pick(self, name: str, approach_h: float = 0.10, grasp_h: float = 0.005) -> dict:
        """Approach, grasp and lift an object by name."""
        try:
            pose = self.object_pose(name)
        except Exception as exc:
            return self._record("pick", False, f"unknown object {name!r}: {exc}")
        target = np.array(pose["pos"], dtype=float)
        approach = target + np.array([0.0, 0.0, approach_h])
        down = target + np.array([0.0, 0.0, grasp_h])

        if not self._servo_to(approach)["ok"]:
            return self._record("pick", False, "unreachable approach pose")
        self.rotate_eef(_GRASP_DOWN_RPY)
        if not self._servo_to(down, tol=0.005)["ok"]:
            return self._record("pick", False, "unreachable grasp pose")
        self.close_gripper()
        self._servo_to(approach, tol=0.02)
        if not self._is_grasped(name):
            return self._record("pick", False, "grasp_failed")
        return self._record("pick", True, f"grasped {name}")

    def place(self, target, release_h: float = 0.05) -> dict:
        """Move above a target (name or xyz) and release the object."""
        if isinstance(target, str):
            try:
                pos = np.array(self.object_pose(target)["pos"], dtype=float)
            except Exception as exc:
                return self._record("place", False, f"unknown target {target!r}: {exc}")
        else:
            pos = np.asarray(target, dtype=float)
        above = pos + np.array([0.0, 0.0, release_h])
        if not self._servo_to(above, tol=0.02)["ok"]:
            return self._record("place", False, "unreachable release pose")
        self.open_gripper()
        self._servo_to(above + np.array([0.0, 0.0, 0.08]), tol=0.02)
        return self._record("place", True, "released")

    def push(self, name: str, delta) -> dict:
        """Push an object by a relative xyz delta (contact-free approximation)."""
        try:
            pos = np.array(self.object_pose(name)["pos"], dtype=float)
        except Exception as exc:
            return self._record("push", False, f"unknown object {name!r}: {exc}")
        delta = np.asarray(delta, dtype=float)
        behind = pos - delta / max(np.linalg.norm(delta), 1e-6) * 0.05
        if not self._servo_to(behind + np.array([0.0, 0.0, 0.03]))["ok"]:
            return self._record("push", False, "unreachable push start")
        return self.goto(pos + delta + np.array([0.0, 0.0, 0.03]))

    # ---------- frames ----------

    def save_frame(self, tag: str = "") -> str:
        """Render the configured cameras and write a JPEG/PNG frame."""
        try:
            obs = self._env.sim.render(
                camera_name=CAMERAS[0], height=256, width=256, depth=False
            )
        except Exception as exc:
            self._record("save_frame", False, str(exc))
            return ""
        self._frames_saved += 1
        self._last_frame_step = self._steps
        name = f"f{self._frames_saved:04d}_step{self._steps:04d}"
        if tag:
            name = f"{name}_{tag}"
        path = self._frames_dir / f"{name}.jpg"
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, obs[::-1])
        except Exception:
            np.save(path.with_suffix(".npy"), obs)
            path = path.with_suffix(".npy")
        self._record("save_frame", True, str(path))
        return str(path)

    # ---------- summary ----------

    def summary(self) -> dict:
        ok_calls = sum(1 for r in self._trace if r["ok"])
        return {
            "steps": self._steps,
            "budget": self._budget,
            "calls": len(self._trace),
            "ok_calls": ok_calls,
            "frames": self._frames_saved,
            "failed": [r for r in self._trace if not r["ok"]][:10],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _load_policy(path: str):
    spec = importlib.util.spec_from_file_location("harness_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_policy(policy_path: str) -> int:
    started = time.time()
    try:
        env = load_env()
    except Exception as exc:
        print(json.dumps({"call": "load_env", "ok": False, "detail": str(exc)}))
        return 1
    robot = Robot(env)
    try:
        policy = _load_policy(policy_path)
        entry = getattr(policy, "run", None)
        if entry is None:
            print(json.dumps({"call": "load_policy", "ok": False,
                              "detail": "policy must define run(robot)"}))
            return 1
        print(json.dumps({"call": "start", "ok": True, "detail": policy_path}))
        entry(robot)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(json.dumps({"call": "policy_error", "ok": False, "detail": str(exc)}))
        robot.save_frame("error")
        print(json.dumps({"summary": robot.summary(),
                          "wall_sec": round(time.time() - started, 1)}))
        return 1
    robot.save_frame("final")
    print(json.dumps({"summary": robot.summary(),
                      "wall_sec": round(time.time() - started, 1)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "robot-run":
        return run_policy(rest[0] if rest else "/workspace/policy.py")
    if command == "robot-scene":
        env = load_env()
        robot = Robot(env)
        print(json.dumps(robot.scene(), indent=2))
        return 0
    if command == "robot-selftest":
        env = load_env()
        robot = Robot(env)
        robot.save_frame("selftest")
        scene = robot.scene()
        print(json.dumps(scene, indent=2)[:2000])
        if scene["objects"]:
            first = scene["objects"][0]["name"]
            robot.log(f"selftest: first object {first}")
            robot.goto(scene["objects"][0]["pos"])
        print(json.dumps(robot.summary(), indent=2))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
