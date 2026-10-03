# LIBERO-Pro harness image (simulator + stable robot API layer).
#
# Build (on the Linux server, from the repo root):
#   docker build -f embodied/docker/Dockerfile -t libero-pro-harbor:latest embodied/docker
#
# This image is referenced by every embodied task via `[environment] docker_image`
# in task.toml, so Harbor never builds per task.
#
# Contents:
#   /opt/libero           LIBERO(-Pro) source, installed editable
#   /opt/libero_pro       LIBERO-Pro bddl + init-state files (HuggingFace dataset)
#   /opt/robot/robot.py   stable Robot API + robot-run / robot-scene / robot-selftest
#   /workspace            where the agent writes policy.py
#   /logs/agent/frames    where frames are written (bind-mounted to the trial dir)
#
# Rendering: CPU (osmesa). Harbor's docker backend does not pass GPUs through, and
# the harness only renders keyframes, so CPU rendering is sufficient. Switch to
# MUJOCO_GL=egl only together with GPU passthrough support in the environment.
#
# SERVER VERIFICATION ITEM #1: the LIBERO-Pro dependency set is the largest
# integration risk. If `pip install -e .` fails, pin the versions LIBERO-Pro's
# own README/extra_requirements.txt asks for and rebuild.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    LIBERO_PRO_ROOT=/opt/libero_pro \
    ROBOT_FRAMES_DIR=/logs/agent/frames \
    ROBOT_BUDGET=600 \
    ROBOT_CAMERAS=agentview

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3-venv \
        git git-lfs wget unzip ca-certificates \
        libosmesa6 libgl1 libglib2.0-0 libegl1 libgles2 libglew2.2 \
        libexpat1 libfontconfig1-dev libmagickwand-dev libpython3-stdlib \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && python3 -m pip install --no-cache-dir --upgrade pip

# CPU torch keeps the image small; the simulator does not need CUDA.
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch

# --- simulator -----------------------------------------------------------------
# Point LIBERO_REPO at the LIBERO-Pro fork for perturbation tasks. The fork is a
# drop-in replacement for `libero` (install exactly one of them).
ARG LIBERO_REPO=https://github.com/Zxy-MLlab/LIBERO-PRO.git
ARG LIBERO_REF=master

RUN git clone --depth 1 --branch "${LIBERO_REF}" "${LIBERO_REPO}" /opt/libero \
    && python3 -m pip install --no-cache-dir -e /opt/libero \
    && if [ -f /opt/libero/extra_requirements.txt ]; then \
           python3 -m pip install --no-cache-dir -r /opt/libero/extra_requirements.txt; \
       fi

# LIBERO-Plus needs its assets.zip placed inside the package tree; LIBERO-PRO
# ships its scenes through the repo. Uncomment and adapt when using LIBERO-Plus:
# COPY assets.zip /opt/libero/libero/libero/
# RUN cd /opt/libero/libero/libero && unzip -o assets.zip && rm assets.zip

# --- task assets (bddl + init states) ------------------------------------------
# LIBERO-Pro publishes bddl/init files as a HuggingFace dataset (small files;
# git-lfs is installed in case the repo uses it).
RUN git lfs install --system \
    && git clone --depth 1 https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro /opt/libero_pro \
    && (cd /opt/libero_pro && git lfs pull || true)

# --- stable robot API + CLIs ---------------------------------------------------
COPY robot.py /opt/robot/robot.py
RUN printf '#!/bin/sh\nexec python3 /opt/robot/robot.py robot-run "$@"\n' > /usr/local/bin/robot-run \
    && printf '#!/bin/sh\nexec python3 /opt/robot/robot.py robot-scene "$@"\n' > /usr/local/bin/robot-scene \
    && printf '#!/bin/sh\nexec python3 /opt/robot/robot.py robot-selftest "$@"\n' > /usr/local/bin/robot-selftest \
    && chmod +x /usr/local/bin/robot-run /usr/local/bin/robot-scene /usr/local/bin/robot-selftest

RUN mkdir -p /workspace /logs/agent/frames /logs/verifier
WORKDIR /workspace

# Smoke check at build time: the API module must import without a live env.
RUN python3 -c "import importlib.util as u; s=u.spec_from_file_location('r','/opt/robot/robot.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('robot.py imports OK')"
