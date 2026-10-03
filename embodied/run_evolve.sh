#!/usr/bin/env bash
# Embodied evolution launcher: LIBERO-Pro tasks, single-module lock on `tools`.
#
# Everything can be overridden from the environment, e.g.
#   PROFILE=smoke bash embodied/run_evolve.sh
#   PROFILE=train EPOCHS=3 TASK_CONCURRENCY=4 SOLVER_TEMPERATURE=0.7 bash embodied/run_evolve.sh
#
# Requires: .env with MODEL / API_BASE / HARBOR_EVO_API_KEY (see .env.example),
# a built `libero-pro-harbor:latest` image, and generated task instances
# (python embodied/gen_tasks.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export SUPPORT_DATASET_DIR="${SUPPORT_DATASET_DIR:-$SCRIPT_DIR}"
export SUPPORT_SPLIT="${SUPPORT_SPLIT:-train}"
export PARENT_MODULES_FOR_GEN0="${PARENT_MODULES_FOR_GEN0:-$SCRIPT_DIR/modules}"
export LOCKED_MODULE="${LOCKED_MODULE:-tools}"
export ENVIRONMENT="${ENVIRONMENT:-docker}"

# K-roll same-task contrast needs sampling diversity: at temperature 0 the K
# rolls of one task are identical and there is no pass/fail pair to learn from.
export SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0.7}"

export PROFILE="${PROFILE:-smoke}"
export EPOCHS="${EPOCHS:-3}"
export TASK_CONCURRENCY="${TASK_CONCURRENCY:-4}"
export ATTEMPTS="${ATTEMPTS:-3}"
export MAX_LANES="${MAX_LANES:-2}"

if [[ ! -d "$SUPPORT_DATASET_DIR/tasks" ]]; then
    echo "error: no task instances at $SUPPORT_DATASET_DIR/tasks" >&2
    echo "run: python embodied/gen_tasks.py --cell ... --bddl ... --init-file ... --instruction ..." >&2
    exit 2
fi

python "$SCRIPT_DIR/sync_modules.py" --check

exec bash "$REPO_ROOT/scripts/evolve.sh"
