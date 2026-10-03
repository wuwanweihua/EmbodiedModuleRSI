"""Tests for the LocalEnvironment backend (no container runtime)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.environments.local import LocalEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


def _env(tmp_path: Path, monkeypatch) -> LocalEnvironment:
    monkeypatch.setenv("HARBOR_LOCAL_ROOT", str(tmp_path / "root"))
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    return LocalEnvironment(
        environment_dir=tmp_path,
        environment_name="local-test",
        session_id="local-test__1",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        logger=SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            error=lambda *a, **k: None,
            getChild=lambda *a, **k: SimpleNamespace(
                info=lambda *x, **y: None,
                warning=lambda *x, **y: None,
                debug=lambda *x, **y: None,
                error=lambda *x, **y: None,
            ),
        ),
    )


def test_type_and_capabilities(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    assert env.type() == "local"
    caps = env.capabilities
    assert caps.mounted is True
    assert caps.disable_internet is False
    assert caps.gpus is False


def test_start_links_logs_to_trial_dir(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    asyncio.run(env.start(force_build=False))
    root = Path(os.environ["HARBOR_LOCAL_ROOT"])
    agent = root / "logs" / "agent"
    assert agent.exists()
    if os.name == "posix":
        # The link target is this trial's agent dir, so harness-side readers
        # see exactly what the environment writes.
        assert agent.is_symlink()
        assert agent.resolve() == (tmp_path / "trial" / "agent").resolve()
    asyncio.run(env.stop(delete=False))


def test_exec_captures_output_and_cwd(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    result = asyncio.run(env.exec("echo hello", cwd=str(tmp_path)))
    assert result.return_code == 0
    assert "hello" in (result.stdout or "")


@pytest.mark.skipif(os.name != "posix", reason="uses POSIX shell variable syntax")
def test_exec_env_and_failure(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    result = asyncio.run(env.exec("echo $FOO"))
    assert "bar" not in (result.stdout or "")  # per-call env only
    result = asyncio.run(env.exec("echo $FOO", env={"FOO": "bar"}))
    assert "bar" in (result.stdout or "")


def test_exec_timeout_and_failure(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    result = asyncio.run(env.exec("exit 3"))
    assert result.return_code == 3

    result = asyncio.run(
        env.exec("python -c \"import time; time.sleep(5)\"", timeout_sec=1)
    )
    assert result.return_code == 124
    assert "timed out" in (result.stderr or "")


def test_upload_download_roundtrip(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    source = tmp_path / "policy.py"
    source.write_text("print('hi')")
    asyncio.run(env.upload_file(source, "/workspace/policy.py"))

    root = Path(os.environ["HARBOR_LOCAL_ROOT"])
    assert (root / "workspace" / "policy.py").read_text() == "print('hi')"
    assert asyncio.run(env.is_file("/workspace/policy.py")) is True
    assert asyncio.run(env.is_dir("/workspace")) is True

    target = tmp_path / "out.py"
    asyncio.run(env.download_file("/workspace/policy.py", target))
    assert target.read_text() == "print('hi')"


def test_reset_dirs_clears_contents_but_keeps_links(tmp_path, monkeypatch):
    env = _env(tmp_path, monkeypatch)
    asyncio.run(env.start(force_build=False))

    root = Path(os.environ["HARBOR_LOCAL_ROOT"])
    verifier = root / "logs" / "verifier"
    (verifier / "reward.txt").write_text("1")

    asyncio.run(
        env.reset_dirs(
            remove_dirs=[EnvironmentPaths.verifier_dir, EnvironmentPaths.tests_dir],
            create_dirs=[EnvironmentPaths.verifier_dir, EnvironmentPaths.tests_dir],
            chmod_dirs=[EnvironmentPaths.verifier_dir],
        )
    )

    # Contents cleared, and the trial-side dir still reachable.
    assert not (verifier / "reward.txt").exists()
    assert (tmp_path / "trial" / "verifier").is_dir()
    assert (root / "tests").is_dir()
    asyncio.run(env.stop(delete=False))


@pytest.mark.skipif(os.name != "posix", reason="lock uses fcntl (POSIX)")
def test_single_trial_lock(tmp_path, monkeypatch):
    env_a = _env(tmp_path / "a", monkeypatch)
    asyncio.run(env_a.start(force_build=False))
    env_b = _env(tmp_path / "b", monkeypatch)
    with pytest.raises(RuntimeError, match="one trial at a time"):
        asyncio.run(env_b.start(force_build=False))
    asyncio.run(env_a.stop(delete=False))
    asyncio.run(env_b.start(force_build=False))
    asyncio.run(env_b.stop(delete=False))
