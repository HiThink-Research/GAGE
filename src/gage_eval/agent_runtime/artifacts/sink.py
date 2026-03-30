"""Artifact sink for local filesystem writes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactSink(Protocol):
    """Write interface for runtime artifacts."""

    def write_text(self, path: str, content: str) -> None: ...

    def write_bytes(self, path: str, content: bytes) -> None: ...

    def record_event(self, name: str, payload: dict) -> None: ...


class FileArtifactSink:
    """Write runtime artifacts to the local filesystem."""

    def write_text(self, path: str, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def write_bytes(self, path: str, content: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def record_event(self, name: str, payload: dict) -> None:
        """Record an event. The default implementation is a no-op."""
        return None

