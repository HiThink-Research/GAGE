"""ArtifactSink — write artifacts to disk."""

from __future__ import annotations

import os
from typing import Protocol


class ArtifactSink(Protocol):
    """Protocol for artifact writers."""

    def write_text(self, path: str, content: str) -> None:
        """Write text content to an artifact path."""

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary content to an artifact path."""

    def record_event(self, name: str, payload: dict) -> None:
        """Record an event alongside artifact outputs."""


class FileArtifactSink:
    """Writes artifacts to the local filesystem."""

    def write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def write_bytes(self, path: str, content: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)

    def record_event(self, name: str, payload: dict) -> None:
        return None
