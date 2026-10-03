"""Tests for the embodied module tree: task contract and frame attachment."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied.modules.observation.baseline import BaselineObservation
from embodied.modules.tools.baseline import POLICY_PATH, BaselineTools
from harbor.agents.terminus_2_modular.protocols import ModuleCtx, ObsState


def _ctx(tmp_path: Path) -> ModuleCtx:
    state = SimpleNamespace(logs_dir=tmp_path, mcp_servers=[], skills_dir=None)
    services = SimpleNamespace(logger=SimpleNamespace(warning=lambda *a, **k: None))
    shared = SimpleNamespace(tmux_session=None)
    return ModuleCtx(state=state, services=services, shared=shared)


def test_tools_contract_is_in_the_initial_prompt(tmp_path):
    tools = BaselineTools()
    prompt = tools.format_initial_prompt("pick up the bowl", "", _ctx(tmp_path))
    assert POLICY_PATH in prompt
    assert "robot-run" in prompt
    assert "grasp_failed" in prompt
    assert "pick up the bowl" in prompt


def test_observation_attaches_new_frames_only(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    obs = BaselineObservation(max_frames=2, min_age_sec=0.0, min_frame_bytes=1)

    old = frames / "f0001.jpg"
    old.write_bytes(b"a" * 4096)
    time.sleep(0.01)

    import asyncio

    first, _ = asyncio.run(obs.capture(ObsState(), _ctx(tmp_path)))
    assert len(first.images) == 1
    assert first.images[0].path == str(old)
    assert "1 new frame(s) attached" in first.text

    # Same file is not re-attached on the next capture.
    second, _ = asyncio.run(obs.capture(ObsState(), _ctx(tmp_path)))
    assert second.images == []

    new = frames / "f0002.jpg"
    new.write_bytes(b"b" * 4096)
    time.sleep(0.01)
    third, _ = asyncio.run(obs.capture(ObsState(), _ctx(tmp_path)))
    assert [Path(i.path).name for i in third.images] == ["f0002.jpg"]


def test_observation_skips_partial_frames(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "tiny.jpg").write_bytes(b"x")
    obs = BaselineObservation(min_age_sec=30.0, min_frame_bytes=1024)

    import asyncio

    result, _ = asyncio.run(obs.capture(ObsState(), _ctx(tmp_path)))
    assert result.images == []


def test_observation_keeps_newest_when_more_than_max(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(4):
        (frames / f"f{i}.jpg").write_bytes(b"x" * 2048)
        time.sleep(0.01)
    obs = BaselineObservation(max_frames=2, min_age_sec=0.0)

    import asyncio

    result, _ = asyncio.run(obs.capture(ObsState(), _ctx(tmp_path)))
    assert [Path(i.path).name for i in result.images] == ["f2.jpg", "f3.jpg"]


@pytest.mark.asyncio
async def test_run_policy_helper_classifies_failures(tmp_path):
    from embodied.modules.tool_helper import run_policy as helper

    class FakeEnv:
        default_user = "root"

        async def exec(self, command, timeout_sec=None, user=None):
            return SimpleNamespace(
                return_code=1,
                stdout='{"call": "pick", "ok": false, "detail": "grasp_failed"}',
                stderr="",
            )

    ctx = ModuleCtx(
        state=SimpleNamespace(env=FakeEnv()),
        services=SimpleNamespace(),
        shared=SimpleNamespace(),
    )
    out = await helper.run(["/workspace/policy.py"], ctx)
    assert "exit=1" in out
    assert "grasp_failed" in out
    assert "repair hints" in out
