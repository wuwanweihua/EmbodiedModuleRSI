#!/usr/bin/env bash
# Prepare this machine as a LocalEnvironment host — works WITHOUT root.
#
#   bash embodied/setup_local_host.sh
#
# Everything lives under $HOME (default ~/harbor-root). In every shell that runs
# the experiment, export the two variables printed at the end:
#
#   export HARBOR_LOCAL_ROOT="$HOME/harbor-root"
#   export PATH="$HOME/.local/bin:$PATH"
#
# and generate tasks with the SAME root:
#   python embodied/gen_tasks.py --environment local --local-root "$HARBOR_LOCAL_ROOT" ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_DIR="$REPO_ROOT/embodied/docker"
BIN_DIR="$HOME/.local/bin"
LOCAL_ROOT="${HARBOR_LOCAL_ROOT:-$HOME/harbor-root}"

echo "== 1/4 environment root (no root needed) =="
mkdir -p "$LOCAL_ROOT"/workspace \
         "$LOCAL_ROOT"/logs/agent \
         "$LOCAL_ROOT"/logs/verifier \
         "$LOCAL_ROOT"/logs/artifacts \
         "$LOCAL_ROOT"/tests \
         "$LOCAL_ROOT"/solution
echo "   HARBOR_LOCAL_ROOT=$LOCAL_ROOT"

echo "== 2/4 agent terminal: tmux (+ optional asciinema) =="
if ! command -v tmux >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        echo "   installing tmux from conda-forge (no root)…"
        conda install -y -c conda-forge tmux
    else
        cat >&2 <<'EOF'
   tmux is missing and conda is not available. Options without root:
     a) install miniconda, then: conda install -y -c conda-forge tmux
     b) drop a static tmux binary into ~/.local/bin and chmod +x it
     c) ask the admin for: apt-get install -y tmux asciinema
EOF
        exit 2
    fi
fi
command -v asciinema >/dev/null 2>&1 || pip install --quiet asciinema \
    || echo "   (asciinema unavailable: terminal recording will be skipped)"
tmux -V

echo "== 3/4 robot-run / robot-scene / robot-selftest =="
mkdir -p "$BIN_DIR"
for cmd in robot-run robot-scene robot-selftest; do
    printf '#!/bin/sh\nexec python3 %s/robot.py %s "$@"\n' "$ROBOT_DIR" "$cmd" > "$BIN_DIR/$cmd"
    chmod +x "$BIN_DIR/$cmd"
done
echo "   installed into $BIN_DIR"

echo "== 4/4 use these in every experiment shell =="
cat <<EOF
   export HARBOR_LOCAL_ROOT="$LOCAL_ROOT"
   export PATH="$BIN_DIR:\$PATH"

   Then make LIBERO(-Pro) importable in the env that runs harbor, and verify:
     pip uninstall -y libero
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git ~/libero
     pip install -e ~/libero && pip install -r ~/libero/extra_requirements.txt
     git lfs install && git clone https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro ~/libero_pro
     export LIBERO_BDDL=<...>/<task>.bddl LIBERO_INIT_FILE=<...>/<task>_init.npy LIBERO_INIT_STATE=0
     robot-scene
EOF
echo "host preparation done"
