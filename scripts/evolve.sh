#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Parameters: edit this block, then run `bash scripts/evolve.sh`.
# Keep API keys in .env, not in this file.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-openai/model-name}"
API_BASE="${API_BASE:-https://api.example.com/v1}"
ENVIRONMENT="${ENVIRONMENT:-docker}"           # docker | e2b

SUPPORT_DATASET_DIR="${SUPPORT_DATASET_DIR:-$REPO_ROOT/../ModularRSI_2000_Instances/tb}"
SUPPORT_SPLIT="${SUPPORT_SPLIT:-train}"         # train (120) | all (1000)
LOCKED_MODULE="${LOCKED_MODULE:-tools}"          # observation | tools | context_mgmt | agent_loop | verification
PARENT_MODULES_FOR_GEN0="${PARENT_MODULES_FOR_GEN0:-}"  # empty = package modules (terminal baseline); point at a domain tree to change the gen_0 seed
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0.0}"  # K-roll same-task contrast needs >0 (e.g. 0.7)

PROFILE="${PROFILE:-train}"                     # train | smoke
REFLECT_EVERY="${REFLECT_EVERY:-}"             # empty = profile default (train: 10, smoke: 2)
TASK_CONCURRENCY="${TASK_CONCURRENCY:-}"       # empty = profile default (train: 6, smoke: 2)
EPOCHS="${EPOCHS:-}"                           # empty = profile default (train: 3, smoke: 1)
MAX_TASKS="${MAX_TASKS:-}"                      # empty = profile default (train: all, smoke: 2)
TASK_SEED="${TASK_SEED:-0}"

ATTEMPTS="${ATTEMPTS:-3}"
MAX_LANES="${MAX_LANES:-2}"
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-2}"
RUN_ID="${RUN_ID:-}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-}"
DRY_RUN="${DRY_RUN:-false}"
# ---------------------------------------------------------------------------

fail() {
    echo "error: $*" >&2
    exit 2
}

case "$PROFILE" in
    train)
        REFLECT_EVERY="${REFLECT_EVERY:-10}"
        TASK_CONCURRENCY="${TASK_CONCURRENCY:-6}"
        EPOCHS="${EPOCHS:-3}"
        ;;
    smoke)
        REFLECT_EVERY="${REFLECT_EVERY:-2}"
        TASK_CONCURRENCY="${TASK_CONCURRENCY:-2}"
        EPOCHS="${EPOCHS:-1}"
        ;;
    *) fail "PROFILE must be train or smoke" ;;
esac

[[ "$MODEL" != "openai/model-name" ]] || fail "set MODEL in scripts/evolve.sh"
[[ "$API_BASE" != "https://api.example.com/v1" ]] || fail "set API_BASE in .env or scripts/evolve.sh"
[[ -d "$SUPPORT_DATASET_DIR/tasks" ]] || fail "dataset must contain tasks/: $SUPPORT_DATASET_DIR"
case "$ENVIRONMENT" in docker|e2b) ;; *) fail "ENVIRONMENT must be docker or e2b" ;; esac
case "$SUPPORT_SPLIT" in train|all) ;; *) fail "SUPPORT_SPLIT must be train or all" ;; esac
case "$LOCKED_MODULE" in observation|tools|context_mgmt|agent_loop|verification) ;;
    *) fail "invalid LOCKED_MODULE: $LOCKED_MODULE" ;;
esac

API_KEY="${HARBOR_EVO_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$API_KEY" && "$API_KEY" != "change-me" ]] || fail "set HARBOR_EVO_API_KEY in .env"
export HARBOR_EVO_API_KEY="$API_KEY"
export OPENAI_API_KEY="$API_KEY"
export PYTHONDONTWRITEBYTECODE=1
DEFAULT_MODEL_INFO='{"max_input_tokens":176000,"max_output_tokens":16000,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
export HARBOR_MODEL_INFO="${HARBOR_MODEL_INFO:-$DEFAULT_MODEL_INFO}"

if [[ "$ENVIRONMENT" == "e2b" && -z "${E2B_API_KEY:-${EVO_E2B_ACCOUNTS:-}}" ]]; then
    fail "set E2B_API_KEY or EVO_E2B_ACCOUNTS in .env"
fi
if [[ -n "${REMOTE_DOCKER_HOST:-}" ]]; then
    export DOCKER_HOST="${DOCKER_HOST:-$REMOTE_DOCKER_HOST}"
fi

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date -u +%Y%m%d_%H%M%S)__${LOCKED_MODULE}"
fi
if [[ -z "$ARCHIVE_ROOT" ]]; then
    ARCHIVE_ROOT="$REPO_ROOT/self_evo_runs/runs/$RUN_ID"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi
[[ -x "$PYTHON_BIN" ]] || fail "activate the installed Python environment or set PYTHON_BIN"

"$PYTHON_BIN" - "$REPO_ROOT" <<'PY_CHECK' || fail "selected Python must import Harbor from this release; run python -m pip install -e . in the selected environment"
from pathlib import Path
import sys
import harbor

expected = (Path(sys.argv[1]) / "src/harbor/__init__.py").resolve()
actual = Path(harbor.__file__).resolve()
if actual != expected:
    raise SystemExit("Harbor import path does not match this release")
PY_CHECK
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

ARGS=(
    --archive-root "$ARCHIVE_ROOT"
    --support-dataset-dir "$SUPPORT_DATASET_DIR"
    --support-split "$SUPPORT_SPLIT"
    --locked-module "$LOCKED_MODULE"
    --model "$MODEL"
    --api-base "$API_BASE"
    --environment "$ENVIRONMENT"
    --profile "$PROFILE"
    --reflect-every "$REFLECT_EVERY"
    --task-concurrency "$TASK_CONCURRENCY"
    --epochs "$EPOCHS"
    --task-seed "$TASK_SEED"
    --attempts "$ATTEMPTS"
    --max-lanes "$MAX_LANES"
    --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER"
    --solver-temperature "$SOLVER_TEMPERATURE"
)
[[ -z "$MAX_TASKS" ]] || ARGS+=(--max-tasks "$MAX_TASKS")
[[ -z "$PARENT_MODULES_FOR_GEN0" ]] || ARGS+=(--parent-modules-for-gen0 "$PARENT_MODULES_FOR_GEN0")

echo "model       : $MODEL"
echo "dataset     : $SUPPORT_DATASET_DIR ($SUPPORT_SPLIT)"
echo "module      : $LOCKED_MODULE"
echo "gen0 seed   : ${PARENT_MODULES_FOR_GEN0:-<package modules>}"
echo "temperature : $SOLVER_TEMPERATURE"
echo "workload    : epochs=$EPOCHS reflect_every=$REFLECT_EVERY concurrency=$TASK_CONCURRENCY attempts=$ATTEMPTS"
echo "environment : $ENVIRONMENT"
echo "output      : $ARCHIVE_ROOT"

COMMAND=("$PYTHON_BIN" -m harbor.agents.terminus_2_modular.self_evo.phase0 "${ARGS[@]}")
if [[ "$DRY_RUN" == "true" ]]; then
    printf 'command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
