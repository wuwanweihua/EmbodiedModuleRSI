"""Generate LIBERO-Pro task instances for the embodied harness experiment.

One task instance = one (task cell, initial state) pair. The evolution run uses
15 instances, evaluation uses 15 different instances of the same cell.

    python embodied/gen_tasks.py \
        --cell libero_spatial_task0 \
        --bddl /opt/libero_pro/bddl/libero_spatial_task0.bddl \
        --init-file /opt/libero_pro/init/libero_spatial_task0_init.npy \
        --instruction "pick up the black bowl and put it in the basket"

Writes:
    embodied/tasks/<cell>_seed_00 .. _29/{instruction.md,task.toml,environment/,tests/}
    embodied/manifest.json          (split="train" for seeds 00..14)

The `environment/` directory holds a stub Dockerfile: tasks use the prebuilt
`libero-pro-harbor:latest` image via `[environment] docker_image`, so Harbor
never builds per task (Harbor's TaskPaths still requires the directory).

SERVER-SIDE VERIFICATION: the LIBERO-Pro bddl/init layout inside the image must
be confirmed (`/opt/libero_pro`), then this generator is re-run with the real
paths. The init-state index is the only per-seed parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBODIED = Path(__file__).resolve().parent
TASKS_DIR = EMBODIED / "tasks"
MANIFEST = EMBODIED / "manifest.json"

N_TRAIN = 15
N_EVAL = 15
DOCKER_IMAGE = "libero-pro-harbor:latest"
ROBOT_PY = EMBODIED / "docker" / "robot.py"

_TASK_TOML = """schema_version = "1.2"

[metadata]
cell = "{cell}"
seed_index = {seed}
split = "{split}"

[environment]
docker_image = "{image}"
cpus = 2
memory_mb = 4096
storage_mb = 8192
build_timeout_sec = 60.0
allow_internet = {allow_internet}

[environment.env]
LIBERO_BDDL = "{bddl}"
LIBERO_INIT_FILE = "{init_file}"
LIBERO_INIT_STATE = "{seed}"
LIBERO_INSTRUCTION = "{instruction}"
ROBOT_BUDGET = "600"
MUJOCO_GL = "osmesa"
{extra_env}

[verifier]
timeout_sec = 1800.0

[verifier.env]
LIBERO_BDDL = "{bddl}"
LIBERO_INIT_FILE = "{init_file}"
LIBERO_INIT_STATE = "{seed}"
LIBERO_INSTRUCTION = "{instruction}"
{extra_env}

[agent]
timeout_sec = 3600.0
"""

_ENV_DOCKERFILE = """# Stub: this task uses the prebuilt image via `[environment] docker_image`
# in task.toml, so Harbor never builds from this file. It exists because
# TaskPaths requires an environment/ directory.
FROM {image}
"""

_TEST_SH = """#!/bin/bash
# Scored run: execute the agent's policy on a FRESH simulator instance at this
# task's fixed initial state and write the binary reward.
#
# The integrity guard below refuses to score a run whose robot API was modified
# (the agent has a shell in this container; full isolation would need a separate
# verifier container — tracked as a phase-2 upgrade).
set -uo pipefail
mkdir -p /logs/verifier

python3 /tests/score.py
status=$?

if [ ! -f /logs/verifier/reward.txt ]; then
    echo "score.py produced no reward (exit=$status)" >&2
    echo 0 > /logs/verifier/reward.txt
fi

exit 0
"""

_SCORE_PY = '''"""Verifier-side scoring: run the policy, then query the task's goal predicate.

This file is uploaded at verify time and is NOT present in the container while
the agent works.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "{robot_dir}")

POLICY = os.environ.get("ROBOT_POLICY", "/workspace/policy.py")
EXPECTED_ROBOT_SHA256 = "{robot_sha}"
REWARD_PATH = Path("/logs/verifier/reward.txt")


def write_reward(value: float) -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))


def guard() -> str | None:
    """Return a rejection reason, or None when the environment looks pristine."""
    robot_py = Path("/opt/robot/robot.py")
    if not robot_py.is_file():
        return "robot API missing"
    digest = hashlib.sha256(robot_py.read_bytes()).hexdigest()
    if EXPECTED_ROBOT_SHA256 and digest != EXPECTED_ROBOT_SHA256:
        return f"robot API was modified (sha256={{digest[:12]}}…)"
    return None


def goal_reached(env) -> bool:
    """Ask the simulator for the task's success predicate."""
    for name in ("check_success", "_check_success"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                continue
    raise RuntimeError("no success predicate found on the environment")


def main() -> int:
    reason = guard()
    if reason is not None:
        print(json.dumps({{"scored": False, "reason": reason}}))
        write_reward(0)
        return 0

    if not Path(POLICY).is_file():
        print(json.dumps({{"scored": False, "reason": "no policy at " + POLICY}}))
        write_reward(0)
        return 0

    try:
        import robot as robot_api

        env = robot_api.load_env()
        policy = robot_api._load_policy(POLICY)
        entry = getattr(policy, "run", None)
        if entry is None:
            print(json.dumps({{"scored": False, "reason": "policy has no run(robot)"}}))
            write_reward(0)
            return 0

        r = robot_api.Robot(env)
        entry(r)
        reward = 1.0 if goal_reached(env) else 0.0
        print(json.dumps({{"scored": True, "reward": reward, "summary": r.summary()}}))
        write_reward(reward)
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({{"scored": False, "reason": f"{{type(exc).__name__}}: {{exc}}"}}))
        write_reward(0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_task(
    cell: str,
    seed: int,
    split: str,
    bddl: str,
    init_file: str,
    instruction: str,
    robot_sha: str,
    allow_internet: bool,
    robot_dir: str,
    extra_env: str,
) -> Path:
    task_dir = TASKS_DIR / f"{cell}_seed_{seed:02d}"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)

    (task_dir / "instruction.md").write_text(instruction.strip() + "\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        _TASK_TOML.format(
            cell=cell,
            seed=seed,
            split=split,
            image=DOCKER_IMAGE,
            bddl=bddl,
            init_file=init_file,
            instruction=instruction.replace('"', '\\"'),
            allow_internet="true" if allow_internet else "false",
            extra_env=extra_env,
        ),
        encoding="utf-8",
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        _ENV_DOCKERFILE.format(image=DOCKER_IMAGE), encoding="utf-8"
    )
    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(_TEST_SH, encoding="utf-8")
    test_sh.chmod(0o755)
    (task_dir / "tests" / "score.py").write_text(
        _SCORE_PY.format(robot_sha=robot_sha, robot_dir=robot_dir), encoding="utf-8"
    )
    return task_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="task cell id, e.g. libero_spatial_task0")
    parser.add_argument("--bddl", required=True, help="in-image path to the task's BDDL file")
    parser.add_argument("--init-file", required=True, help="in-image path to the init-state array")
    parser.add_argument("--instruction", required=True, help="task instruction (LIBERO-Pro wording)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument(
        "--local-root",
        default="~/harbor-root",
        help=(
            "local mode only: the writable root that HARBOR_LOCAL_ROOT points "
            "at. Its workspace/ and logs/ replace /workspace and /logs."
        ),
    )
    parser.add_argument(
        "--robot-dir",
        default="/opt/robot",
        help=(
            "Directory holding robot.py that score.py imports. docker mode: the "
            "in-image path /opt/robot; local mode: <repo>/embodied/docker"
        ),
    )
    parser.add_argument(
        "--environment",
        choices=("docker", "local"),
        default="docker",
        help=(
            "Target environment backend. docker -> allow_internet=false (the "
            "container can be isolated); local -> allow_internet=true, because "
            "the LocalEnvironment cannot enforce network isolation."
        ),
    )
    args = parser.parse_args()

    # The local backend cannot cut the network, and Harbor's validator rejects
    # allow_internet=false for it, so local tasks must declare internet on.
    allow_internet = args.environment == "local"
    extra_env = ""
    if allow_internet:
        local_root = os.path.expanduser(args.local_root).rstrip("/")
        workspace = f"{local_root}/workspace"
        extra_env = "\n".join(
            [
                f'WORKSPACE = "{workspace}"',
                f'ROBOT_POLICY = "{workspace}/policy.py"',
                f'ROBOT_FRAMES_DIR = "{local_root}/logs/agent/frames"',
                f'ROBOT_API_PATH = "{args.robot_dir}/robot.py"',
            ]
        )
        print(
            "note: --environment local -> allow_internet=true "
            "(no network isolation) and paths under "
            f"{local_root} (set HARBOR_LOCAL_ROOT to match)"
        )

    if not ROBOT_PY.is_file():
        print(f"missing {ROBOT_PY}", file=sys.stderr)
        return 2
    robot_sha = hashlib.sha256(ROBOT_PY.read_bytes()).hexdigest()

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.n_train + args.n_eval):
        split = "train" if seed < args.n_train else "eval"
        task_dir = _write_task(
            args.cell,
            seed,
            split,
            args.bddl,
            args.init_file,
            args.instruction,
            robot_sha,
            allow_internet,
            args.robot_dir,
            extra_env,
        )
        rows.append({"name": task_dir.name, "split": split, "seed_index": seed})
        print(f"wrote {task_dir.relative_to(REPO_ROOT)}")

    MANIFEST.write_text(
        json.dumps(
            {
                "cell": args.cell,
                "bddl": args.bddl,
                "init_file": args.init_file,
                "instruction": args.instruction,
                "robot_sha256": robot_sha,
                "tasks": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)} ({args.n_train} train / {args.n_eval} eval)")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
