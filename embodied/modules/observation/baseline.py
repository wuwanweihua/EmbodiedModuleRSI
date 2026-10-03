"""Embodied observation: incremental terminal output + newly rendered frames.

Terminal text comes from the tmux pane exactly as in the terminal baseline
(`get_incremental_output` + byte-accurate head/tail truncation). On top of
that, frames written by the sandbox-side runner into `/logs/agent/frames`
(bind-mounted to this trial's agent dir) are attached to the observation as
`ImageRef`s: newest last, at most `max_frames` per capture, and never the same
file twice.

A frame is attached only when it is complete — older than `min_age_sec` and
larger than `min_frame_bytes` — so a frame that is still being written is
skipped this round and picked up on the next capture.
"""

from __future__ import annotations

import time
from pathlib import Path

from harbor.agents.terminus_2_modular.protocols import (
    ImageRef,
    ModuleCtx,
    ObsResult,
    ObsState,
)

DEFAULT_MAX_BYTES = 10_000
FRAMES_REL = "frames"
_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


def _smart_truncate(output: str, max_bytes: int) -> str:
    """Byte-accurate head+tail truncation; same marker format as
    `Terminus2._limit_output_length` (terminus_2.py:529-565)."""
    output_bytes = output.encode("utf-8")
    if len(output_bytes) <= max_bytes:
        return output

    portion = max_bytes // 2
    first = output_bytes[:portion].decode("utf-8", errors="ignore")
    last = output_bytes[-portion:].decode("utf-8", errors="ignore")

    omitted = len(output_bytes) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return (
        f"{first}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{last}"
    )


class BaselineObservation:
    NAME = "baseline"
    NICHE = {"capture": "incremental", "capacity": "low", "staleness": "none"}
    DESCRIPTION = (
        "Incremental tmux pane capture (byte-accurate truncation) plus newly "
        "rendered frames from /logs/agent/frames attached as images."
    )
    PARAMS_SCHEMA = {
        "max_bytes": "int (default 10000)",
        "max_frames": "int (default 2)",
        "min_frame_bytes": "int (default 1024)",
        "min_age_sec": "float (default 0.5)",
    }

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_frames: int = 2,
        min_frame_bytes: int = 1024,
        min_age_sec: float = 0.5,
    ):
        self.max_bytes = max_bytes
        self.max_frames = max_frames
        self.min_frame_bytes = min_frame_bytes
        self.min_age_sec = min_age_sec
        self._seen_frames: set[str] = set()

    def _new_frames(self, ctx: ModuleCtx) -> list[ImageRef]:
        logs_dir = getattr(ctx.state, "logs_dir", None)
        if not logs_dir:
            return []
        frames_dir = Path(logs_dir) / FRAMES_REL
        try:
            entries = sorted(
                (p for p in frames_dir.glob("*") if p.suffix.lower() in _FRAME_SUFFIXES),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return []

        now = time.time()
        picked: list[ImageRef] = []
        for path in entries:
            key = str(path)
            if key in self._seen_frames:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size < self.min_frame_bytes:
                continue
            if now - stat.st_mtime < self.min_age_sec:
                continue
            self._seen_frames.add(key)
            media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            picked.append(ImageRef(path=key, media_type=media_type))

        if self.max_frames <= 0:
            return []
        return picked[-self.max_frames :]

    async def capture(
        self,
        prev: ObsState,
        ctx: ModuleCtx,
    ) -> tuple[ObsResult, ObsState]:
        session = ctx.shared.tmux_session
        raw = ""
        if session is not None:
            try:
                raw = await session.get_incremental_output()
            except Exception as exc:
                ctx.services.logger.warning(
                    "embodied_obs: get_incremental_output failed: %s", exc
                )
                raw = ""

        text = _smart_truncate(raw or "", self.max_bytes)
        images = self._new_frames(ctx)
        if images:
            text = f"{text}\n[{len(images)} new frame(s) attached]".strip()
        return ObsResult(text=text, images=images), ObsState(prev_terminal=raw or "")


def register(library):
    library.register(
        type_="observation",
        name=BaselineObservation.NAME,
        factory=lambda params: BaselineObservation(
            max_bytes=int(params.get("max_bytes", DEFAULT_MAX_BYTES)),
            max_frames=int(params.get("max_frames", 2)),
            min_frame_bytes=int(params.get("min_frame_bytes", 1024)),
            min_age_sec=float(params.get("min_age_sec", 0.5)),
        ),
        description=BaselineObservation.DESCRIPTION,
        params_schema=BaselineObservation.PARAMS_SCHEMA,
        niche=BaselineObservation.NICHE,
    )
