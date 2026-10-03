"""Tests for deployment-wide LLM request headers.

Gateways may require their own headers (a stable session id for routing, a
client-identifying user agent). They are configured via
HARBOR_LLM_EXTRA_HEADERS / HARBOR_LLM_USER_AGENT and applied to every call.
"""

from __future__ import annotations

import asyncio

import pytest

import harbor.llms.lite_llm as mod


class _Captured(Exception):
    """Raised by the fake completion so we can inspect the request kwargs."""


def _llm() -> mod.LiteLLM:
    return mod.LiteLLM(model_name="openai/test-model", api_base="http://localhost:1/v1")


def test_deployment_headers_from_env(monkeypatch):
    monkeypatch.setenv("HARBOR_LLM_EXTRA_HEADERS", '{"x-opencode-session": "s-1"}')
    monkeypatch.setenv("HARBOR_LLM_USER_AGENT", "embodied/1.0")
    assert _llm()._deployment_headers() == {
        "x-opencode-session": "s-1",
        "User-Agent": "embodied/1.0",
    }


def test_deployment_headers_ignore_bad_json(monkeypatch):
    monkeypatch.setenv("HARBOR_LLM_EXTRA_HEADERS", "not-json")
    monkeypatch.delenv("HARBOR_LLM_USER_AGENT", raising=False)
    assert _llm()._deployment_headers() == {}


def test_headers_reach_the_request(monkeypatch):
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("HARBOR_LLM_EXTRA_HEADERS", '{"x-opencode-session": "s-1"}')
    monkeypatch.delenv("HARBOR_LLM_USER_AGENT", raising=False)

    with pytest.raises(Exception):
        asyncio.run(_llm().call("hi"))
    assert captured["extra_headers"]["x-opencode-session"] == "s-1"


def test_caller_headers_win_over_env(monkeypatch):
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(mod.litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("HARBOR_LLM_EXTRA_HEADERS", '{"x-opencode-session": "env"}')

    with pytest.raises(Exception):
        asyncio.run(_llm().call("hi", extra_headers={"x-opencode-session": "caller"}))
    assert captured["extra_headers"]["x-opencode-session"] == "caller"
