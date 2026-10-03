"""run_policy — run a policy program in the sandbox and return a compact trace.

Usage:  run_policy [policy_path]        (default /workspace/policy.py)

Why a helper instead of plain `robot-run` through tmux: a helper's output is
returned DIRECTLY to the agent (bypassing the pane), so the agent gets the tail
of the structured trace plus any traceback even when the program prints a lot,
without waiting for a pane poll. It is also the natural place to classify
failures and attach repair hints — the tools-module surface this experiment
evolves.

This helper is intentionally small and self-contained: the evolution editor may
rewrite it (or add siblings) under `modules/tool_helper/`.
"""

from __future__ import annotations

import shlex

NAME = "run_policy"
USAGE = (
    "run_policy [policy.py] — run the policy in the sandbox and print the tail "
    "of the structured trace plus any traceback (default /workspace/policy.py)"
)
DESCRIPTION = (
    "Execute a policy program through the sandbox runner and return a compact "
    "trace summary with failure classification."
)
NICHE = {"kind": "execute"}

DEFAULT_POLICY = "/workspace/policy.py"
RUN_TIMEOUT_SEC = 1200
MAX_OUTPUT_CHARS = 6000

_FAILURE_HINTS = (
    ("ModuleNotFoundError: robot", "run it with `robot-run <policy.py>`"),
    ('"detail": "unreachable"', "target outside the reachable workspace — re-plan the approach"),
    ('"detail": "grasp_failed"', "gripper closed on nothing — re-read object_pose and align above the centre"),
    ("TimeoutError", "a primitive exceeded its time budget — split the motion into smaller steps"),
)


async def run(args: list[str], ctx):
    policy = args[0] if args else DEFAULT_POLICY
    env = ctx.state.env
    if env is None:
        return "[run_policy] no environment on ctx"

    command = f"robot-run {shlex.quote(policy)}"
    try:
        result = await env.exec(
            command, timeout_sec=RUN_TIMEOUT_SEC, user=env.default_user
        )
    except Exception as exc:
        return f"[run_policy] exec failed: {exc}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = f"{stdout}\n{stderr}"

    tail = combined[-MAX_OUTPUT_CHARS:]
    if len(combined) > MAX_OUTPUT_CHARS:
        tail = "[... earlier output truncated ...]\n" + tail

    hints = [msg for marker, msg in _FAILURE_HINTS if marker in combined]

    lines = [
        f"[run_policy] exit={result.return_code} policy={policy}",
        tail or "(no output)",
    ]
    if hints:
        lines.append("[run_policy] repair hints:")
        lines.extend(f"- {h}" for h in hints)
    return "\n".join(lines)


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
