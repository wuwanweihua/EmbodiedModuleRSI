"""Local (no-container) environment backend.

Runs tasks directly on the current machine, for hosts where no container
runtime is available — e.g. a machine that is itself a container and therefore
cannot start ``dockerd``. The isolation boundary is the machine you are
already on; nothing else is sandboxed.

Behaviour differences from the container backends:

- **Log dirs are host dirs.** ``start()`` links ``/logs/{agent,verifier,
  artifacts}`` to this trial's directory, the equivalent of Docker's bind
  mounts, and ``capabilities.mounted`` is True so the harness reads them
  directly instead of downloading. ``reset_dirs`` clears their *contents*
  rather than removing the links.
- **One trial at a time.** The ``/logs`` links are fixed paths, so concurrent
  trials would clobber each other. ``start()`` takes an exclusive lock and
  fails fast with a clear message; run with ``TASK_CONCURRENCY=1``.
- **No internet isolation.** ``allow_internet = false`` cannot be enforced, so
  tasks must set it to true (the base validator rejects the combination
  otherwise).
- ``[environment].env`` values are exported for every command.

Host preparation (once, needs root):

    sudo mkdir -p /logs /tests /solution /workspace
    sudo chown -R "$USER" /logs /tests /solution /workspace

Then run with ``-e local`` (or ``ENVIRONMENT=local`` in the embodied launcher).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from harbor.environments.base import BaseEnvironment, EnvironmentPath, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import EnvironmentPaths

try:  # POSIX only; the backend is not meant for Windows hosts.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_LOCK_PATH = Path("/tmp/harbor-local-environment.lock")
_MOUNTED_SUBDIRS = ("agent", "verifier", "artifacts")


def _root() -> Path:
    """Root used for the environment's fixed paths (default `/`)."""
    return Path(os.environ.get("HARBOR_LOCAL_ROOT", "/"))


def _host_path(env_path: EnvironmentPath | str) -> Path:
    """Map an in-environment path (``/logs/agent``) to a host path."""
    raw = str(env_path)
    root = _root()
    return raw if root == Path("/") else root / raw.lstrip("/")


def _current_uid() -> int | None:
    return os.getuid() if hasattr(os, "getuid") else None


class LocalEnvironment(BaseEnvironment):
    """No container: commands run on this machine as the current user."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._lock_file: Any = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return EnvironmentType.LOCAL.value

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=False,
            disable_internet=False,
            windows=False,
            mounted=True,
        )

    def _validate_definition(self) -> None:
        """No Dockerfile or compose file is needed: the environment is the host."""
        return None

    # ---------- lifecycle ----------

    def _acquire_lock(self) -> None:
        if fcntl is None:
            return
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        handle = open(_LOCK_PATH, "w")  # noqa: SIM115 - held for the trial
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError(
                "LocalEnvironment runs one trial at a time (fixed /logs paths); "
                f"another trial holds {_LOCK_PATH}. Use TASK_CONCURRENCY=1 and "
                "keep editor/sanity concurrency at 1 as well."
            ) from None
        self._lock_file = handle

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            try:
                self._lock_file.close()
            finally:
                self._lock_file = None

    async def start(self, force_build: bool) -> None:
        self._acquire_lock()

        # /logs is a real directory owned by the user; each subdir is a link to
        # this trial's directory (Docker bind-mount equivalent).
        logs_root = _host_path(EnvironmentPaths.logs_dir)
        logs_root.mkdir(parents=True, exist_ok=True)
        trial_dir = Path(self.trial_paths.trial_dir)
        for name in _MOUNTED_SUBDIRS:
            target = trial_dir / name
            target.mkdir(parents=True, exist_ok=True)
            link = logs_root / name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                # A real directory left over from an earlier run: keep it aside.
                backup = link.with_name(f"{link.name}.prev")
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                link.rename(backup)
                self.logger.warning("LocalEnvironment moved %s aside to %s", link, backup)
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                # Filesystems without symlink support: fall back to real dirs.
                # Agent logs then stay inside the environment (downloaded at the
                # end of the trial instead of being read live).
                self.logger.warning(
                    "LocalEnvironment cannot link %s -> %s (%s); using a real directory",
                    link,
                    target,
                    exc,
                )
                link.mkdir(parents=True, exist_ok=True)

        for env_path in (EnvironmentPaths.tests_dir, EnvironmentPaths.solution_dir):
            _host_path(env_path).mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "LocalEnvironment ready (logs -> %s, one trial at a time)", trial_dir
        )

    async def stop(self, delete: bool) -> None:
        if delete:
            logs_root = _host_path(EnvironmentPaths.logs_dir)
            for name in _MOUNTED_SUBDIRS:
                link = logs_root / name
                if link.is_symlink():
                    link.unlink()
        self._release_lock()

    async def reset_dirs(
        self,
        *,
        remove_dirs: Sequence[EnvironmentPath],
        create_dirs: Sequence[EnvironmentPath],
        chmod_dirs: Sequence[EnvironmentPath] | None = None,
    ) -> ExecResult:
        """Clear directory contents; never unlink the trial links."""
        for env_path in remove_dirs:
            host = _host_path(env_path)
            target = host.resolve() if host.is_symlink() else host
            if target.is_dir():
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            elif target.exists():
                target.unlink(missing_ok=True)
        for env_path in create_dirs:
            host = _host_path(env_path)
            if not host.exists():
                host.mkdir(parents=True, exist_ok=True)
        for env_path in chmod_dirs or ():
            host = _host_path(env_path)
            try:
                host.chmod(0o777)
            except OSError:
                pass
        return ExecResult(stdout="", stderr="", return_code=0)

    # ---------- file transfer ----------

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return _host_path(path).is_dir()

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return _host_path(path).is_file()

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = _host_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = _host_path(target_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = _host_path(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = _host_path(source_dir)
        target = Path(target_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)

    # ---------- execution ----------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        merged = self._merge_env(env)
        proc_env = {**os.environ}
        if merged:
            proc_env.update(merged)

        requested_user = self._resolve_user(user)
        if requested_user is not None:
            uid = _current_uid()
            if str(requested_user) not in {"0", "root"} and uid is not None:
                if str(requested_user) != str(uid):
                    self.logger.warning(
                        "LocalEnvironment ignores user=%r; running as uid %s",
                        requested_user,
                        uid,
                    )

        workdir = str(_host_path(cwd)) if cwd else None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=workdir,
                env=proc_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # bad cwd, missing shell, ...
            return ExecResult(stdout="", stderr=str(exc), return_code=127)

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout_sec}s",
                return_code=124,
            )

        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=process.returncode if process.returncode is not None else -1,
        )
