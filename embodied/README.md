# Embodied harness experiment (LIBERO-Pro, code-as-policy)

Single-module evolution (`tools`) of the ModularRSI harness on an embodied
code-as-policy task: the agent writes `/workspace/policy.py`, runs it against a
robot API inside the sandbox, and the harness's Tool Use module is what evolves.

## Layout

```
embodied/
├── modules/                 gen_0 module tree (--parent-modules-for-gen0)
│   ├── tools/baseline.py    embodied Tool Use seed (synced from the package)
│   ├── tool_helper/run_policy.py   structured policy execution helper
│   ├── observation/baseline.py     pane text + newly rendered frames
│   └── {agent_loop,context_mgmt,verification}/baseline.py   package shims
├── docker/
│   ├── Dockerfile           LIBERO(-Pro) + osmesa + robot API
│   └── robot.py             stable Robot API + robot-run/robot-scene/robot-selftest
├── gen_tasks.py             writes tasks/ + manifest.json
├── sync_modules.py          keeps tools/baseline.py identical to the package
└── run_evolve.sh            embodied defaults on top of scripts/evolve.sh
```

## Server steps

```bash
# 0. install the repo (once)
python3 -m venv .venv && . .venv/bin/activate && pip install -e .

# 1. build the image (verification item #1: LIBERO-Pro deps)
docker build -f embodied/docker/Dockerfile -t libero-pro-harbor:latest embodied/docker

# 2. verify the image end to end (no agent involved)
docker run --rm libero-pro-harbor:latest bash -lc '
  export LIBERO_BDDL=<...>/<task>.bddl LIBERO_INIT_FILE=<...>/<task>_init.npy LIBERO_INIT_STATE=0
  robot-scene && robot-selftest'

# 3. generate the 15+15 task instances for ONE cell
python embodied/gen_tasks.py --cell <cell> --bddl <in-image bddl> \
  --init-file <in-image init npy> --instruction "<LIBERO-Pro instruction>"
python embodied/sync_modules.py --check

# 4. vision sanity: confirm the served model accepts image content parts
#    (litellm image_url + base64 data URI). One minimal call is enough; if the
#    endpoint rejects images, the harness degrades to text-only observations and
#    frames are still recorded in the trajectory.
python - <<'PY'
import base64, litellm, os
url = "data:image/png;base64," + base64.b64encode(open("frame.png","rb").read()).decode()
r = litellm.completion(model=os.environ["MODEL"], api_base=os.environ["API_BASE"],
    messages=[{"role":"user","content":[{"type":"text","text":"what colour dominates?"},
                                        {"type":"image_url","image_url":{"url":url}}]}])
print(r.choices[0].message.content[:200])
PY

# 5. baseline sanity: success rate must land in a middle band before evolving
PROFILE=smoke bash embodied/run_evolve.sh        # 2 tasks, 1 epoch, end-to-end

# 6. the real run
PROFILE=train EPOCHS=3 TASK_CONCURRENCY=4 bash embodied/run_evolve.sh
```

Artifacts land in `self_evo_runs/runs/<RUN_ID>/` (gen_0, gen_N, staging,
archive.json, evolution_log.jsonl, editor_memory.jsonl).

## What evolves

Locked to `tools`, so the editor may write `modules/tools/*.py` and
`modules/tool_helper/*.py` only; every other module stays on its baseline.
`tools/baseline.py` itself is the fixed seed and is never writable — evolution
adds sibling variants. The robot API implementation is fixed in the image (the
analogue of `bash` being fixed in the terminal setting); evolution changes the
contract, validation, helpers and failure recovery around it.

## Model endpoint

`.env` needs `MODEL`, `API_BASE`, `HARBOR_EVO_API_KEY`, `HARBOR_MODEL_INFO`
(note `MODEL` is deliberately not in `.env.example`; the launcher's placeholder
default fails fast). `MODEL` uses litellm naming: `openai/<model>` for an
OpenAI-compatible endpoint, or `deepseek/<model>` for DeepSeek's own API.

Gateways that require their own routing headers work without code changes —
they are applied to every request (solver, editor, summarizer subagents):

```bash
# .env — example: opencode go requires a stable session id
HARBOR_LLM_EXTRA_HEADERS='{"x-opencode-session": "embodied-harness"}'
HARBOR_LLM_USER_AGENT=embodied-harness/1.0
# Endpoint rejects multimodal messages? Keep frames out of the LLM request
# (they are still recorded in the trajectory):
# HARBOR_DISABLE_IMAGES=1
```

## Local mode (no container runtime)

If the machine has no docker/podman/apptainer (for example, it *is* a container
and cannot start `dockerd`), run the whole experiment natively with the `local`
environment backend: the simulator, the agent's terminal and the verifier all
run on this machine. The isolation boundary is the machine itself.

```bash
# once: everything under $HOME, no root required
bash embodied/setup_local_host.sh
# then, in every experiment shell:
export HARBOR_LOCAL_ROOT="$HOME/harbor-root"      # must match --local-root below
export PATH="$HOME/.local/bin:$PATH"
# make LIBERO(-Pro) importable in the env that runs harbor (see the script's
# closing instructions), and verify:
robot-scene

# generate tasks for local mode: allow_internet=true (the local backend cannot
# cut the network) and every fixed path moved under HARBOR_LOCAL_ROOT
python embodied/gen_tasks.py --environment local --local-root "$HARBOR_LOCAL_ROOT" \
  --robot-dir "$PWD/embodied/docker" \
  --cell <cell> --bddl <path> --init-file <path> --instruction "<...>"

ENVIRONMENT=local TASK_CONCURRENCY=1 bash embodied/run_evolve.sh
```

Caveats of local mode:
- **One trial at a time** (`TASK_CONCURRENCY=1`, editor/sanity concurrency 1):
  the fixed `/logs` paths are shared. The backend takes a lock and fails fast
  if another trial is running.
- **No network isolation**: tasks must set `allow_internet = true`.
- The agent's shell sees this machine's filesystem as your user — including the
  repo and `.env`. Use a scoped, revocable API key for these runs.
- No container image is built, so the LIBERO install must be reproducible by
  hand (record the exact commands / `conda env export` for the paper).

## Known limitations (milestone 1)

- 15 "seeds" are 15 separate task directories, so the paper's cross-task vote
  becomes a cross-seed vote. Scale up by generating more cells.
- CPU rendering (osmesa). Frames are keyframes only; Harbor's docker backend
  does not pass GPUs through.
- The verifier runs in the same container as the agent with a hash guard on
  `/opt/robot/robot.py`. Full isolation (a separate verifier container via
  `[verifier] environment_mode = "separate"`) is a phase-2 upgrade.
- `robot.py`'s primitives (servo `goto`, grasp heights, workspace bounds) are a
  first draft: verify them on the server with `robot-selftest` before the run.
