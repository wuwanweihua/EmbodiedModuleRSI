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
# Parameters: edit this block, then run `bash scripts/evaluate.sh`.
# Keep API keys in .env, not in this file.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-openai/model-name}"
API_BASE="${API_BASE:-https://api.example.com/v1}"
ENVIRONMENT="${ENVIRONMENT:-docker}"           # docker | e2b

MODULES_ROOT="${MODULES_ROOT:-$REPO_ROOT/generations/merged_active/gen_0/modules}"
DATASET="${DATASET:-terminal-bench@2.0}"
TASK_NAME="${TASK_NAME-fix-git}"              # empty = every task in DATASET
OUTPUT_DIR="${OUTPUT_DIR:-}"

N_CONCURRENT="${N_CONCURRENT:-1}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
MAX_TURNS="${MAX_TURNS:-200}"
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0}"

COMPOSER_NAME="${COMPOSER_NAME:-llm_dynamic}"  # llm_dynamic | static
COMPOSER_SCOPE="${COMPOSER_SCOPE:-all}"
LOCKED_MODULE="${LOCKED_MODULE:-tools}"
DRY_RUN="${DRY_RUN:-false}"
# ---------------------------------------------------------------------------

fail() {
    echo "error: $*" >&2
    exit 2
}

[[ "$MODEL" != "openai/model-name" ]] || fail "set MODEL in scripts/evaluate.sh"
[[ "$API_BASE" != "https://api.example.com/v1" ]] || fail "set API_BASE in .env or scripts/evaluate.sh"
[[ -d "$MODULES_ROOT" ]] || fail "modules directory does not exist: $MODULES_ROOT"
case "$ENVIRONMENT" in docker|e2b) ;; *) fail "ENVIRONMENT must be docker or e2b" ;; esac

API_KEY="${HARBOR_EVO_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$API_KEY" && "$API_KEY" != "change-me" ]] || fail "set HARBOR_EVO_API_KEY in .env"
export HARBOR_EVO_API_KEY="$API_KEY"
export OPENAI_API_KEY="$API_KEY"
export PYTHONDONTWRITEBYTECODE=1
DEFAULT_MODEL_INFO='{"max_input_tokens":176000,"max_output_tokens":16000,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
export HARBOR_MODEL_INFO="${HARBOR_MODEL_INFO:-$DEFAULT_MODEL_INFO}"

if [[ "$ENVIRONMENT" == "e2b" && -z "${E2B_API_KEY:-}" ]]; then
    fail "set E2B_API_KEY in .env"
fi
if [[ -n "${REMOTE_DOCKER_HOST:-}" ]]; then
    export DOCKER_HOST="${DOCKER_HOST:-$REMOTE_DOCKER_HOST}"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/results/evaluation/$(date -u +%Y%m%d_%H%M%S)"
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
    run --yes
    -d "$DATASET"
    -a terminus-2-modular
    -m "$MODEL"
    -e "$ENVIRONMENT"
    -n "$N_CONCURRENT"
    -k "$N_ATTEMPTS"
    -o "$OUTPUT_DIR"
    --ak "modules_root=$MODULES_ROOT"
    --ak "api_base=$API_BASE"
    --ak "max_turns=$MAX_TURNS"
    --ak "model_info=$HARBOR_MODEL_INFO"
    --ak "temperature=$SOLVER_TEMPERATURE"
    --ak "locked_module_type=$LOCKED_MODULE"
    --ak "composer_scope=$COMPOSER_SCOPE"
    --ak "composer_name=$COMPOSER_NAME"
)
[[ -z "$TASK_NAME" ]] || ARGS+=(-i "$TASK_NAME")

echo "model       : $MODEL"
echo "modules     : $MODULES_ROOT"
echo "dataset     : $DATASET"
echo "task        : ${TASK_NAME:-all}"
echo "composer    : $COMPOSER_NAME ($COMPOSER_SCOPE)"
echo "environment : $ENVIRONMENT"
echo "output      : $OUTPUT_DIR"

COMMAND=("$PYTHON_BIN" -m harbor.cli.main "${ARGS[@]}")
if [[ "$DRY_RUN" == "true" ]]; then
    printf 'command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
