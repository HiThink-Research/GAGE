"""Helpers for stable run identity generation and projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

RUN_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Stable identity metadata associated with a single evaluation run."""

    run_id: str
    source: str
    schema_version: int
    created_at_iso: str


def build_run_identity(run_id: str | None = None, *, now: datetime | None = None) -> RunIdentity:
    """Return a stable run identity from user input or generated defaults."""

    current = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    effective_run_id = str(run_id) if run_id else _generate_run_id(current)
    source = "provided" if run_id else "generated"
    return RunIdentity(
        run_id=effective_run_id,
        source=source,
        schema_version=RUN_IDENTITY_SCHEMA_VERSION,
        created_at_iso=current.isoformat().replace("+00:00", "Z"),
    )


def build_run_identity_metadata(identity: RunIdentity) -> dict[str, str | int]:
    """Project a RunIdentity into stable summary metadata fields."""

    return {
        "run_id": identity.run_id,
        "source": identity.source,
        "schema_version": identity.schema_version,
        "created_at_iso": identity.created_at_iso,
    }


def _generate_run_id(now: datetime) -> str:
    timestamp = now.strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run-{timestamp}-{suffix}"
