"""Embodied Tool Use module — the `tools` baseline for LIBERO-Pro tasks.

The agent is still a terminal agent: it writes `/workspace/policy.py` with the
file helpers and runs it in the sandbox. What changes versus the terminal
baseline is the CONTRACT this module puts in front of the model:

- `format_initial_prompt` documents the robot API (the stable, image-baked
  `robot` package), the policy entry point, the structured trace, the episode
  rules, and the failure signatures with their repair guidance;
- the API IMPLEMENTATION is fixed in the image — the analogue of `bash` being
  fixed in the terminal setting — so this module evolves the *presentation,
  validation and recovery surface*, which is what the paper's tools-module
  evolution changes (prompt format, helper tools, parser robustness, error
  handling).

Inherited from `terminal_baseline`: tmux session lifecycle, Terminus response
parsers, helper-tool dispatch, and the tmux execution path.

NOTE for the evolution editor: `baseline.py` is the fixed gen_0 seed and is
never writable. Write NEW sibling variants under `modules/tools/<name>.py`
(and helpers under `modules/tool_helper/<name>.py`). A variant may subclass
`BaselineTools` from this module; the staged module tree is not an importable
package, so import the base class from
`harbor.agents.terminus_2_modular.modules.tools.baseline`.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.modules.tools.terminal_baseline import (
    TerminalBaselineTools,
)
from harbor.agents.terminus_2_modular.protocols import ModuleCtx

POLICY_PATH = "/workspace/policy.py"
API_PATH = "/opt/robot/robot.py"
FRAMES_DIR = "/logs/agent/frames"

_EMBODIED_CONTRACT = f"""
## Embodied task contract (LIBERO-Pro)

You control a Franka Panda arm in a MuJoCo tabletop scene. You do NOT drive
joints directly: you write a Python program that calls the `robot` API and run
it inside the sandbox.

### Deliverable
Write your program to `{POLICY_PATH}`. It must define:

    def run(robot):
        ...  # your policy; return when the task is finished or the budget is out

Run it with:

    robot-run {POLICY_PATH}

The run prints a structured trace (one line per API call, then a summary).
Frames are captured automatically after each primitive and are attached to
your next observation.

### Robot API (module `robot`; full docstrings in `{API_PATH}`)

Scene:
    robot.instruction()            task instruction (same as this prompt)
    robot.scene()                  objects (name/pos/quat/size/is_grasped),
                                   gripper state, step counter, budget
    robot.objects()                names of objects currently in the scene
    robot.object_pose(name)        pose and size of one object

Motion (blocking; each returns {{"ok": bool, "detail": str}}):
    robot.goto(pos, tol=0.01)      move end-effector to absolute xyz
    robot.move_rel(delta)          relative end-effector move
    robot.rotate_eef(rpy_delta)    rotate end-effector (radians)
    robot.open_gripper() / robot.close_gripper()
    robot.pick(name)               approach + grasp an object by name
    robot.place(pos_or_name)       move above a target and release
    robot.push(name, delta)        push an object
    robot.check_reach(pos)         IK feasibility for a target position

Utility:
    robot.save_frame(tag)          force a frame capture (returns its path)
    robot.log(msg)                 add a note to the trace

### Rules
- One episode per run. The scene is reset to a FIXED initial state before your
  program starts; re-running your program replays the same episode.
- Budget: `robot.scene()["budget"]` environment steps per run. Exceeding it
  ends the episode as a failure.
- Your program must terminate on its own; never wait for input.
- Do not read or modify `/tests`, `/logs/verifier`, or the simulator sources.
  The scored run is executed separately on a pristine copy, and tampering is
  detected.
- Success is judged by the task's goal predicate after your program exits. You
  cannot query it; verify progress with `robot.scene()` instead.

### Failure signatures and what they mean
- `ModuleNotFoundError: robot` → run it with `robot-run {POLICY_PATH}`
  (plain `python3 {POLICY_PATH}` has no environment bound).
- `{{"ok": false, "detail": "unreachable"}}` → the target is outside the arm's
  reachable workspace or blocked. Re-plan the approach; do not repeat the same
  motion.
- `{{"ok": false, "detail": "grasp_failed"}}` → the gripper closed on nothing.
  Re-read `robot.object_pose(name)`, align above the object centre, then lower
  in small steps before closing.
- Traceback ending in `TimeoutError` → a primitive exceeded its time budget;
  split the motion into smaller `move_rel` steps.
- Program exits but the goal is not reached → check subtask ordering and
  confirm each subtask's effect with `robot.scene()` before continuing.

### Working style
1. Inspect first: `robot-scene` prints the initial scene without writing code.
2. Build incrementally: one primitive, run, read the trace and frames, extend.
3. Re-verify with `robot.scene()` after every pick and place.
"""


class BaselineTools(TerminalBaselineTools):
    """Embodied tools baseline: terminal transport + robot task contract."""

    NAME = "baseline"
    NICHE = {"transport": "tmux", "encoding": "both", "domain": "embodied"}
    DESCRIPTION = (
        "Embodied (LIBERO-Pro) tools module: tmux transport and Terminus "
        "parsers inherited from the terminal baseline, plus the robot task "
        "contract (policy entry point, robot API docs, episode rules, failure "
        "signatures with repair guidance)."
    )
    PARAMS_SCHEMA = {
        **TerminalBaselineTools.PARAMS_SCHEMA,
        "contract_extra": "str: appended to the embodied contract (default '')",
    }

    def __init__(
        self,
        parser_name: str = "json",
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        session_name: str = "terminus-2-modular",
        helper_tools: list | None = None,
        contract_extra: str = "",
    ):
        super().__init__(
            parser_name=parser_name,
            tmux_pane_width=tmux_pane_width,
            tmux_pane_height=tmux_pane_height,
            session_name=session_name,
            helper_tools=helper_tools,
        )
        self._contract_extra = contract_extra or ""

    def format_initial_prompt(
        self,
        instruction: str,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str:
        """Task instruction + embodied contract, then the inherited template."""
        augmented = f"{instruction}\n{_EMBODIED_CONTRACT}"
        if self._contract_extra:
            augmented = f"{augmented}\n{self._contract_extra}"
        return super().format_initial_prompt(augmented, terminal_state, ctx)


def register(library):
    library.register(
        type_="tools",
        name=BaselineTools.NAME,
        factory=lambda params: BaselineTools(
            parser_name=str(params.get("parser_name", "json")),
            tmux_pane_width=int(params.get("tmux_pane_width", 160)),
            tmux_pane_height=int(params.get("tmux_pane_height", 40)),
            session_name=str(params.get("session_name", "terminus-2-modular")),
            helper_tools=params.get("helper_tools") or [],
            contract_extra=str(params.get("contract_extra", "")),
        ),
        description=BaselineTools.DESCRIPTION,
        params_schema=BaselineTools.PARAMS_SCHEMA,
        niche=BaselineTools.NICHE,
    )
