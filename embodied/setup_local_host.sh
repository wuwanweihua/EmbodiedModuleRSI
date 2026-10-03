#!/usr/bin/env bash
# Prepare this machine as a LocalEnvironment host (run once; needs sudo).
#
# Local mode runs the simulator, the agent's terminal and the verifier directly
# on this machine — no container runtime. Run this from the repo root:
#
#     bash embodied/setup_local_host.sh
#
# It installs the agent's terminal (tmux/asciinema), creates the fixed
# environment paths, and installs the robot-* CLIs pointing at this repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_DIR="$REPO_ROOT/embodied/docker"
BIN_DIR="$HOME/.local/bin"

echo "== 1/4 installing tmux + asciinema (the agent's terminal) =="
sudo apt-get update
sudo apt-get install -y tmux asciinema

echo "== 2/4 creating the environment's fixed paths =="
sudo mkdir -p /logs /tests /solution /workspace
sudo chown -R "$USER":"$(id -gn)" /logs /tests /solution /workspace

echo "== 3/4 installing robot-run / robot-scene / robot-selftest =="
mkdir -p "$BIN_DIR"
for cmd in robot-run robot-scene robot-selftest; do
    printf '#!/bin/sh\nexec python3 %s/robot.py %s "$@"\n' "$ROBOT_DIR" "$cmd" > "$BIN_DIR/$cmd"
    chmod +x "$BIN_DIR/$cmd"
done
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "   NOTE: add $BIN_DIR to PATH (export PATH=\"$BIN_DIR:\$PATH\")" ;;
esac

echo "== 4/4 next: make LIBERO(-Pro) importable in the env that runs harbor =="
cat <<'EOF'
   # in the venv/conda env you use for harbor:
   #   pip install torch --index-url https://download.pytorch.org/whl/cpu
   #   git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git ~/libero
   #   pip install -e ~/libero && pip install -r ~/libero/extra_requirements.txt
   #   git lfs install && git clone https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro ~/libero_pro
   #
   # then verify (LIBERO_BDDL/INIT_FILE from ~/libero_pro):
   #   robot-scene
EOF
echo "host preparation done"
