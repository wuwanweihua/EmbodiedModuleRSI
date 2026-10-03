"""Unit tests for the multimodal observation helpers."""

from __future__ import annotations

import base64
from pathlib import Path

from harbor.agents.terminus_2_modular.image_utils import (
    MAX_IMAGE_BYTES,
    build_user_content,
    evict_images,
    project_text_only,
)
from harbor.agents.terminus_2_modular.protocols import ImageRef, ObsResult


def _png(path: Path, payload: bytes = b"\x89PNG\r\n\x1a\nfake") -> Path:
    path.write_bytes(payload)
    return path


def test_obs_result_defaults_to_no_images():
    obs = ObsResult(text="hello")
    assert obs.images == []


def test_build_user_content_text_only_when_unreadable(tmp_path):
    ref = ImageRef(path=str(tmp_path / "missing.jpg"))
    assert build_user_content("prompt", [ref]) == "prompt"


def test_build_user_content_attaches_data_uri(tmp_path):
    ref = ImageRef(path=str(_png(tmp_path / "f.png")), media_type="image/png")
    content = build_user_content("prompt", [ref])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "prompt"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")


def test_build_user_content_skips_oversized(tmp_path):
    big = _png(tmp_path / "big.jpg", b"x" * (MAX_IMAGE_BYTES + 1))
    ref = ImageRef(path=str(big))
    assert build_user_content("prompt", [ref]) == "prompt"


def test_images_disabled_env_switch(tmp_path, monkeypatch):
    ref = ImageRef(path=str(_png(tmp_path / "f.png")), media_type="image/png")
    monkeypatch.setenv("HARBOR_DISABLE_IMAGES", "1")
    assert build_user_content("prompt", [ref]) == "prompt"
    monkeypatch.setenv("HARBOR_DISABLE_IMAGES", "off")
    assert isinstance(build_user_content("prompt", [ref]), list)


def test_build_user_content_keeps_newest(tmp_path):
    refs = [
        ImageRef(path=str(_png(tmp_path / f"f{i}.jpg", b"a" * 2048))) for i in range(3)
    ]
    content = build_user_content("p", refs, max_images=2)
    assert len(content) == 3  # text + 2 images


def test_evict_images_flattens_old_turns():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "one"},
                                     {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "two"},
                                     {"type": "image_url", "image_url": {"url": "data:y"}}]},
    ]
    assert evict_images(messages, keep_last=1) == 1
    assert isinstance(messages[0]["content"], str)
    assert "1 image(s) omitted" in messages[0]["content"]
    assert isinstance(messages[2]["content"], list)


def test_evict_images_keep_zero_flattens_all():
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:y"}}]},
    ]
    assert evict_images(messages, keep_last=0) == 1
    assert messages[0]["content"] == "[1 image(s) omitted]"


def test_project_text_only_does_not_mutate():
    original = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": "d"}}]}]
    projected = project_text_only(original)
    assert isinstance(original[0]["content"], list)
    assert isinstance(projected[0]["content"], str)
