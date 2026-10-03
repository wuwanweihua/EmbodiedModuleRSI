"""Embodied gen_0 shim — reuse the package context_mgmt baseline verbatim.

The 3-step QA summarizer is text-only and the agent_loop projects multimodal
history to text before the subagent calls (see
`harbor.agents.terminus_2_modular.image_utils.project_text_only`), so the
stock implementation is safe for image-carrying histories.
"""

from harbor.agents.terminus_2_modular.modules.context_mgmt.baseline import (  # noqa: F401
    BaselineContextMgmt,
    register,
)
