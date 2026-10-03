"""Embodied gen_0 shim — reuse the package agent_loop baseline verbatim.

The staged module tree is not an importable package: discovery loads each file
by path. Re-exporting the package implementation keeps this tree and the
installed package in sync (including the multimodal prompt assembly).
"""

from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (  # noqa: F401
    BaselineAgentLoop,
    register,
)
