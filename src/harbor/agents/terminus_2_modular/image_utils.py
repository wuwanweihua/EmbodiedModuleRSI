"""Image helpers for multimodal observations.

Modules that attach frames to an observation return host-side file references
(`ImageRef`). These helpers turn them into OpenAI-style content parts for the
LLM call, keep conversation history from re-sending every past frame, and give
summarization subagents a text-only projection of a multimodal history.

Design notes:
- Images live on disk (host side) and are base64-encoded at request time. Only
  the most recent multimodal turn keeps its images; older turns are flattened
  to text by `evict_images` so a long rollout does not re-upload every frame on
  every call.
- Every failure path degrades to text: an unreadable or oversized frame is
  skipped with a note rather than raising into the agent loop.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from harbor.agents.terminus_2_modular.protocols import ImageRef

# Hard ceiling for one attached image. Larger frames are skipped with a text
# note; a single 4MB frame is already ~1.3M base64 chars, which is well past a
# sane per-step budget for the models this harness targets.
MAX_IMAGE_BYTES = 4 * 1024 * 1024

_IMAGE_PART_TYPES = ("image_url", "image")
_TRUTHY = {"1", "true", "yes", "on"}


def images_disabled() -> bool:
    """True when the deployment must not send image content parts.

    Set `HARBOR_DISABLE_IMAGES=1` for endpoints that reject multimodal
    messages. Frames are still produced and recorded in the trajectory; they
    just never enter the LLM request.
    """
    return os.environ.get("HARBOR_DISABLE_IMAGES", "").strip().lower() in _TRUTHY


def data_url(ref: ImageRef, max_bytes: int = MAX_IMAGE_BYTES) -> str | None:
    """Return an OpenAI `image_url` data URI for `ref`, or None if unusable."""
    try:
        raw = Path(ref.path).read_bytes()
    except OSError:
        return None
    if not raw or len(raw) > max_bytes:
        return None
    media_type = ref.media_type or (
        mimetypes.guess_type(ref.path)[0] or "image/jpeg"
    )
    return f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii")


def build_user_content(
    prompt: str,
    refs: Iterable[ImageRef],
    max_images: int = 2,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> str | list[dict[str, Any]]:
    """Build the user-message content for the next LLM call.

    Returns `prompt` unchanged when no image can be read (so text-only flows
    keep their exact wire format). Otherwise returns a content-parts list:
    the text first, then up to `max_images` frames (newest last).
    """
    if images_disabled():
        return prompt
    refs = list(refs)[-max_images:] if max_images > 0 else []
    parts: list[dict[str, Any]] = []
    skipped = 0
    for ref in refs:
        url = data_url(ref, max_bytes=max_bytes)
        if url is None:
            skipped += 1
            continue
        parts.append({"type": "image_url", "image_url": {"url": url}})
    if not parts:
        return prompt

    text = prompt
    if skipped:
        text = f"{text}\n[{skipped} image(s) could not be attached]"
    return [{"type": "text", "text": text}, *parts]


def _flatten_message_content(content: Any) -> Any:
    """Replace a multimodal content list with a text-only string."""
    if not isinstance(content, list):
        return content
    texts = [
        str(p.get("text", ""))
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    n_images = sum(
        1
        for p in content
        if isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES
    )
    text = "\n".join(t for t in texts if t)
    if n_images:
        text = f"{text}\n[{n_images} image(s) omitted]".strip()
    return text


def evict_images(messages: Sequence[dict[str, Any]], keep_last: int = 2) -> int:
    """Flatten image parts in multimodal user turns older than a message window.

    `keep_last` counts MESSAGES, not image turns: only the trailing `keep_last`
    messages may still carry images. With the default (2 = the last
    user/assistant pair) a frame survives exactly one extra call, so the model
    can refer back to what it just saw, and is dropped afterwards instead of
    being re-uploaded for the rest of the rollout. `keep_last=0` flattens
    everything (used when the context overflows).

    Mutates the message dicts in place (they are the live history list) and
    returns the number of turns flattened. Text-only messages are untouched.
    """
    keep_from = max(0, len(messages) - max(0, keep_last))
    flattened = 0
    for i, m in enumerate(messages):
        if i >= keep_from:
            break
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), list)
        ):
            m["content"] = _flatten_message_content(m["content"])
            flattened += 1
    return flattened


def project_text_only(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` with every multimodal content list flattened to text.

    Used before summarization subagent calls: those prompts must stay text-only
    (cheap, and the subagent model may not accept images at all).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            m = {**m, "content": _flatten_message_content(m["content"])}
        out.append(m)
    return out
