"""Shared helpers for backend configuration models."""

from __future__ import annotations

from pydantic import BaseModel

try:  # Pydantic v2+
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - v1 fallback
    ConfigDict = None


class BackendConfigBase(BaseModel):
    """Base class for backend configuration models.

    Design goals:
    - Disable Pydantic's protected namespaces for `model_*` to avoid collisions
      with backend parameters.
    - Allow extra fields by default to support the "typed + passthrough" mode:
      - Declared fields benefit from validation and defaults.
      - Undeclared fields are preserved and can still be passed to the backend.
    """

    if ConfigDict is not None:  # pragma: no branch - runtime evaluated once
        # NOTE: Pydantic v2 config: allow extra fields + relax protected namespaces.
        model_config = ConfigDict(  # type: ignore[assignment]
            protected_namespaces=(),
            extra="allow",
        )
    else:  # pragma: no cover - Pydantic v1

        class Config:
            protected_namespaces = ()
            extra = "allow"
