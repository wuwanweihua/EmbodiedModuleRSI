"""End-to-end test of the multimodal wiring: observation images -> LLM call.

Exercises `BaselineAgentLoop._query_llm` with a stub model: images must reach
the LLM as content parts, older turns must be flattened on later calls, and the
text-only path must keep its exact string wire format.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (
    BaselineAgentLoop,
)
from harbor.agents.terminus_2_modular.protocols import ImageRef
from harbor.llms.base import LLMResponse
from harbor.llms.chat import Chat


class StubModel:
    def __init__(self) -> None:
        self.prompts: list[object] = []
        self.histories: list[list] = []

    async def call(self, prompt, message_history=None, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        # Snapshot the dicts: the loop flattens image parts in place afterwards.
        self.histories.append([dict(m) for m in (message_history or [])])
        return LLMResponse(content='{"analysis": "ok", "commands": [], "task_complete": true}')


def _ctx() -> SimpleNamespace:
    logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return SimpleNamespace(services=SimpleNamespace(logger=logger))


def _image(tmp_path: Path) -> ImageRef:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return ImageRef(path=str(frame), media_type="image/jpeg")


async def _call(loop, chat, model, tmp_path, prompt, images):
    return await loop._query_llm(
        chat=chat,
        prompt=prompt,
        ctx=_ctx(),
        context_mgmt=SimpleNamespace(),
        tools=SimpleNamespace(),
        original_instruction="task",
        logging_paths=(None, None, None),
        images=images,
    )


@pytest.mark.asyncio
async def test_images_reach_the_llm_as_content_parts(tmp_path):
    model = StubModel()
    chat = Chat(model)
    loop = BaselineAgentLoop()

    await _call(loop, chat, model, tmp_path, "look", [_image(tmp_path)])

    sent = model.prompts[0]
    assert isinstance(sent, list)
    assert sent[0]["type"] == "text" and sent[0]["text"] == "look"
    assert sent[1]["type"] == "image_url"
    assert sent[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # The live history keeps the multimodal user turn.
    assert isinstance(chat.messages[0]["content"], list)


@pytest.mark.asyncio
async def test_text_only_path_is_unchanged(tmp_path):
    model = StubModel()
    chat = Chat(model)
    loop = BaselineAgentLoop()

    await _call(loop, chat, model, tmp_path, "plain", [])
    assert model.prompts[0] == "plain"
    assert chat.messages[0]["content"] == "plain"


@pytest.mark.asyncio
async def test_older_images_are_evicted_from_history(tmp_path):
    model = StubModel()
    chat = Chat(model)
    loop = BaselineAgentLoop()

    await _call(loop, chat, model, tmp_path, "first", [_image(tmp_path)])
    await _call(loop, chat, model, tmp_path, "second", [])

    # Second call's history still carried the first turn's images...
    assert isinstance(model.histories[1][0]["content"], list)
    # ...but after the call they are flattened, so later calls don't re-upload.
    assert isinstance(chat.messages[0]["content"], str)
    assert "1 image(s) omitted" in chat.messages[0]["content"]
